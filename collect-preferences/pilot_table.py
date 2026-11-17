#!/usr/bin/env python3
"""One row per pilot run: quality, failures, length bias and measured compute cost.

  collect-preferences/run.sh pilot_table.py [--round v1v2|v3]

Quality, with analyze_pilot.py's definitions: position consistency (both orders give the same winner) and
agreement with Opus on pairs where both are consistent; a pair with a single valid verdict counts as that verdict.
No verdict: calls that ended without a parseable verdict (length limit, broken JSON, prompt too long for the
context). Length: win rate of an answer twice as long as an equal one between equal responders, from
length_bias.py's Bradley-Terry fit on clean English pairs (50% = neutral; one standard error is about 4 points).
Cost: generation wall time from the worker logs (model loading excluded, 2-14 min per node and job), as LUMI-G
node-hours (8 GCDs) per 1,000 pairs in both orders, and multiplied out to the 128,000 pairs of the full dataset
for one judge. Opus: Message Batches usage at $2 / $10 per million input / output tokens.
Rounds: v1v2 (prompts v1 and v2, mostly greedy; judged on the answers before the truncated responses were
continued, pairs.before_continuation.parquet; writes RUNS.md) and v3 (prompt v3, thinking on wherever a judge
has it, vendor sampling with one retry; the current pairs.parquet; votes, Opus labels and v3/RUNS.md under
v3/; generation times are read from the worker logs). Retried: calls whose first attempt gave no verdict.
"""
import argparse
import collections
import glob
import math
import os
import re
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PILOT = os.path.join(ROOT, "data", "preferences", "pilot")
sys.path.insert(0, HERE)
from analyze_pilot import agreement, load_votes, preference  # noqa: E402
from judge_prompt import same_lang  # noqa: E402
from length_bias import fit  # noqa: E402

FULL_PAIRS = 128000
OPUS_PRICE = (2.0, 10.0)          # $ per million input / output tokens, Message Batches (half the $4 / $20 list price)

# run -> (label, variant, tensor parallel size, generation seconds per worker, source of the timing)
# A worker with TP=2 holds a quarter of a node, one with TP=8 the whole node; a run split over two jobs sums its parts.
RUNS = [
    ("gemma-4-31B-it|v1|off", "Gemma 4 31B", "v1, no thinking, grammar", 2, [649, 630, 644, 654], "pilot-gemma-4-31B-it-off-23744510"),
    ("gemma-4-31B-it|v2|off", "Gemma 4 31B", "v2, no thinking, grammar", 2, [656, 638, 639, 664], "pilot-gemma-4-31B-it-v2-off-23744510"),
    ("gemma-4-31B-it|v2|free", "Gemma 4 31B", "v2, no thinking, no grammar (replicate)", 2, [641, 651, 625, 650],
     "identical re-run pilot-gemma-4-31B-it-v2-on-23746250.*.bogus (the original log was overwritten)"),
    ("gemma-4-31B-it|v2|on", "Gemma 4 31B", "v2, thinking on", 2, [1609, 1727, 1617, 1715], "pilot-gemma-4-31B-it-v2-on-23746250"),
    ("qwen3.5-27b|v1|off", "Qwen3.5-27B", "v1, grammar-suppressed thinking", 2, [829, 848, 880, 898], "pilot-qwen3.5-27b-off-23744510"),
    ("qwen3.5-27b|v2|off", "Qwen3.5-27B", "v2, grammar-suppressed thinking", 2, [900, 913, 936, 967], "pilot-qwen3.5-27b-v2-off-23744510"),
    ("qwen3.5-27b|v2|off@rendered", "Qwen3.5-27B", "v2, no-think template, grammar", 2, [834, 811, 848, 864],
     "pilot-qwen3.5-27b-v2-off-23746250 (votes_rendered/)"),
    ("qwen3.5-27b|v2|on", "Qwen3.5-27B", "v2, thinking on", 2, [2158, 2369, 2205, 2162], "pilot-qwen3.5-27b-v2-on-23745465"),
    ("qwen3.5-122b-a10b|v1|off", "Qwen3.5-122B", "v1, grammar-suppressed thinking", 8, [1533], "pilot-qwen3.5-122b-a10b-off-23744510"),
    ("qwen3.5-122b-a10b|v2|off", "Qwen3.5-122B", "v2, grammar-suppressed thinking", 8, [782 + 752],
     "pilot-qwen3.5-122b-a10b-v2-off-23744510 (1,024 calls) + -23745465 (992 calls)"),
    ("qwen3.5-122b-a10b|v2|on", "Qwen3.5-122B", "v2, thinking on", 8, [4720], "pilot-qwen3.5-122b-a10b-v2-on-23746250"),
    ("gpt-oss-120b-bf16|v1|off", "gpt-oss-120b", "v1, reasoning medium, 1,500-token budget", 8, [865], "pilot-gpt-oss-120b-bf16-off-23744510 (restart)"),
    ("gpt-oss-120b-bf16|v2|off", "gpt-oss-120b", "v2, reasoning medium", 8, [899], "pilot-gpt-oss-120b-bf16-v2-off-23745465"),
    ("gpt-oss-120b-bf16|v2|rep", "gpt-oss-120b", "v2, reasoning medium (replicate)", 8, [942],
     "identical re-run pilot-gpt-oss-120b-bf16-v2-on-23746250.w0.bogus (the original log was overwritten)"),
    ("gpt-oss-120b-bf16|v2|on", "gpt-oss-120b", "v2, reasoning high", 8, [2239], "pilot-gpt-oss-120b-bf16-v2-on-23746250"),
    ("glm-4.5-air|v1|off", "GLM-4.5-Air", "v1, grammar-suppressed thinking", 8, [916], "pilot-glm-4.5-air-off-23744510"),
    ("glm-4.5-air|v2|off", "GLM-4.5-Air", "v2, grammar-suppressed thinking", 8, [963], "pilot-glm-4.5-air-v2-off-23744510"),
    ("glm-4.5-air|v2|on", "GLM-4.5-Air", "v2, thinking on", 8, [3281], "pilot-glm-4.5-air-v2-on-23746250"),
    ("mistral-large-2411|v1|off", "Mistral Large 2411", "v1 (no thinking mode)", 8, [2894], "pilot-mistral-large-2411-off-23744510"),
    ("mistral-large-2411|v2|off", "Mistral Large 2411", "v2", 8, [2540 + 342],
     "pilot-mistral-large-2411-v2-off-23745465 (1,792 calls) + -23746250 (224 calls)"),
]

# round 3: run -> (label, variant, tensor parallel size, worker logs); the generation seconds come from the logs
RUNS_V3 = [
    ("gemma-4-31B-it|v3|on", "Gemma 4 31B", "v3, thinking on", 2, "pilot-gemma-4-31B-it-v3-on-*.w*.log"),
    ("qwen3.5-27b|v3|on", "Qwen3.5-27B", "v3, thinking on", 2, "pilot-qwen3.5-27b-v3-on-*.w*.log"),
    ("qwen3.5-122b-a10b|v3|on", "Qwen3.5-122B", "v3, thinking on", 8, "pilot-qwen3.5-122b-a10b-v3-on-*.w*.log"),
    ("qwen3.8-27b|v3|on", "Qwen3.8-27B", "v3, thinking on (effort xhigh)", 2, "pilot-qwen3.8-27b-v3-on-*.w*.log"),
    ("command-a-plus-2605|v3|on", "Command A+", "v3, thinking on", 8, "pilot-command-a-plus-2605-v3-on-*.w*.log"),
    ("mistral-small-4-119b|v3|on", "Mistral Small 4 119B", "v3, reasoning high", 8, "pilot-mistral-small-4-119b-v3-on-*.w*.log"),
    ("apertus-1.5-70b|v3|on", "Apertus v1.5 70B", "v3, thinking on", 4, "pilot-apertus-1.5-70b-v3-on-*.w*.log"),
    ("gpt-oss-120b-bf16|v3|on", "gpt-oss-120b", "v3, reasoning high", 8, "pilot-gpt-oss-120b-bf16-v3-on-*.w*.log"),
    ("gpt-oss-120b-bf16|v3|off", "gpt-oss-120b", "v3, reasoning medium", 8, "pilot-gpt-oss-120b-bf16-v3-off-*.w*.log"),
    ("glm-4.5-air|v3|on", "GLM-4.5-Air", "v3, thinking on", 8, "pilot-glm-4.5-air-v3-on-*.w*.log"),
    ("mistral-large-2411|v3|off", "Mistral Large 2411", "v3 (no thinking mode)", 8, "pilot-mistral-large-2411-v3-off-*.w*.log"),
]
LOGS = os.path.join(ROOT, "data", "logs")
DONE_RE = re.compile(r"(\d+) new in (\d+)s")
TERM_RE = re.compile(r"SIGTERM after (\d+)s")
PROG_RE = re.compile(r": \d+/\d+ calls, (\d+)s, ")


def log_seconds(pattern):
    """generation seconds per worker log (judge.py's final line, its SIGTERM line when a job ran out of time, or
    else its last progress line: a cancelled step's log can end without the SIGTERM line, and the minutes after
    the last progress line are then not counted) and the log names"""
    secs, names = [], []
    for f in sorted(glob.glob(os.path.join(LOGS, pattern))):
        text = open(f, errors="replace").read()
        m = DONE_RE.findall(text) or TERM_RE.findall(text) or PROG_RE.findall(text)
        if m:
            secs.append(int(m[-1][1] if isinstance(m[-1], tuple) else m[-1]))
            names.append(os.path.basename(f))
    return secs, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", choices=["v1v2", "v3"], default="v3")
    args = ap.parse_args()
    v3 = args.round == "v3"
    base = os.path.join(PILOT, "v3") if v3 else PILOT
    pairs = {p["id"]: p for p in pq.read_table(os.path.join(PILOT, "pairs.parquet" if v3 else "pairs.before_continuation.parquet")).to_pylist()}
    runs = load_votes(os.path.join(base, "votes", "*.parquet"))
    if not v3:
        for run, d in load_votes(os.path.join(PILOT, "votes_rendered", "*.parquet")).items():
            runs[run + "@rendered"] = d
    opus = load_votes(os.path.join(base, "opus", "votes.parquet"))
    opus_run = next(iter(opus))
    runs.update(opus)
    prefs = {run: {i: preference(rows) for i, rows in d.items()} for run, d in runs.items()}
    ref = prefs[opus_run]
    models = sorted({p["model_x"] for p in pairs.values()} | {p["model_y"] for p in pairs.values()})

    def flagged(p, s):
        if p["task_type"] == "translation" or p["locale_dependence"] == "language_specific":
            return False
        return bool(p["lid_" + s] and p["lid_conf_" + s] >= 0.5 and len(p["response_" + s].strip()) >= 20
                    and not same_lang(p["lid_" + s], p["language"]))

    clean_en = [i for i, p in pairs.items() if p["tier"] == "en" and min(p["chars_x"], p["chars_y"]) >= 200
                and p["finish_x"] != "length" and p["finish_y"] != "length" and not flagged(p, "x") and not flagged(p, "y")]

    def win2x(run):
        ids, k, n = [], [], []
        for i in clean_en:
            v = [runs[run][i][o]["prefers"] for o in (0, 1) if o in runs[run].get(i, {}) and runs[run][i][o].get("prefers") in ("x", "y")]
            if v:
                ids.append(i)
                k.append(sum(x == "x" for x in v))
                n.append(len(v))
        X = np.zeros((len(ids), len(models) + 3))
        for r, i in enumerate(ids):
            p = pairs[i]
            X[r, models.index(p["model_x"])] += 1
            X[r, models.index(p["model_y"])] -= 1
            X[r, -3] = math.log(p["chars_x"]) - math.log(p["chars_y"])
        b, se = fit(X[:, 1:], np.array(k, float), np.array(n, float))
        return 1 / (1 + math.exp(-b[-3] * math.log(2)))

    def quality(run):
        d, pr = runs[run], prefs[run]
        calls = [r for rows in d.values() for r in rows.values()]
        fail = sum(not r["status"].startswith("ok") for r in calls)
        toks = sum(r.get("completion_tokens") or 0 for r in calls) / len(calls)
        decided = [i for i in pr if pr[i][0] is not None]
        cons = sum(pr[i][0] in ("x", "y") for i in decided) / len(decided)
        same, n = agreement(pr, ref, list(pairs)) if run != opus_run else (0, 0)
        retried = sum((r.get("attempt") or 0) > 0 for r in calls)
        return dict(calls=len(calls), fail=fail / len(calls), retried=retried / len(calls), toks=toks, cons=cons,
                    agr=same / n if n else None, n=n,
                    win2x=win2x(run), prompt_toks=sum(r.get("prompt_tokens") or 0 for r in calls), out_toks=sum(r.get("completion_tokens") or 0 for r in calls))

    retry_col = " Retried |" if v3 else ""
    L = ["| Judge | Variant | Consistency | Agreement with Opus | No verdict |" + retry_col + " Output tokens per call | "
         "2× longer wins (en) | Node-h per 1k pairs | Node-h, all 128k pairs |",
         "|---|---|---|---|---|" + ("---|" if v3 else "") + "---|---|---|---|"]
    notes = []
    todo = [(run, label, variant, tp) + log_seconds(pattern) for run, label, variant, tp, pattern in RUNS_V3] if v3 else RUNS
    for run, label, variant, tp, secs, src in todo:
        if run not in runs or not secs:
            notes.append("{}: no votes or no finished worker log yet".format(run))
            continue
        q = quality(run)
        node_h = tp * sum(secs) / 8 / 3600 * 1000 / len(pairs)
        partial = q["calls"] < 0.9 * 2 * len(pairs)     # a stopped run: its seconds are mostly ramp-up, so no cost
        if partial:
            variant += "; stopped after {:,} of {:,} calls".format(q["calls"], 2 * len(pairs))
        L.append("| {} | {} | {:.1f}% | {:.1f}% | {:.1f}% |{} {:,.0f} | {:.0f}% | {} | {} |".format(
            label, variant, 100 * q["cons"], 100 * q["agr"], 100 * q["fail"],
            " {:.1f}% |".format(100 * q["retried"]) if v3 else "", q["toks"], 100 * q["win2x"],
            "-" if partial else "{:.2f}".format(node_h), "-" if partial else "{:.0f}".format(node_h * FULL_PAIRS / 1000)))
        notes.append("{}: {} worker(s) at TP={}, generation {} s; {}".format(run, len(secs), tp, "+".join(map(str, secs)),
                                                                            src if isinstance(src, str) else ", ".join(src)))
    q = quality(opus_run)
    usd = (q["prompt_toks"] * OPUS_PRICE[0] + q["out_toks"] * OPUS_PRICE[1]) / 1e6
    per_k = usd * 1000 / len(pairs)
    L.append("| Claude Opus 5.5 | {}, adaptive thinking, effort high | {:.1f}% | reference | {:.1f}% |{} {:,.0f} | {:.0f}% | ${:,.0f} | ${:,.0f} |".format(
        "v3" if v3 else "v1", 100 * q["cons"], 100 * q["fail"], " - |" if v3 else "", q["toks"], 100 * q["win2x"], per_k,
        per_k * FULL_PAIRS / 1000))
    notes.append("{}: {:.2f}M input, {:.2f}M output tokens (thinking included) = ${:.0f} for {} pairs".format(
        opus_run, q["prompt_toks"] / 1e6, q["out_toks"] / 1e6, usd, len(pairs)))
    text = "\n".join(L + [""] + notes)
    print(text)
    with open(os.path.join(base, "RUNS.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
