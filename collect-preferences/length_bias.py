#!/usr/bin/env python3
"""Length bias of the judges: do longer answers win because they are longer?

  collect-preferences/run.sh length_bias.py [--votes data/preferences/pilot/v3/votes] [--opus data/preferences/pilot/v3/opus/votes.parquet]
                                            [--runs judge|prompt|thinking,...] [--pool run,run,run] [--examples N]

Lengths are response characters. A "clean" pair has both answers >= 200 characters, neither cut by its length
limit and neither flagged by language ID: the comparisons where length could plausibly be the deciding factor.

A. share of consistent pairs won by the longer answer, by length-ratio band, each judge against Opus on the same pairs
B. disagreements with Opus: how often the judge took the longer answer (50% = no bias relative to Opus)
C. length-controlled fit per run: P(x wins) = sigmoid(theta[model_x] - theta[model_y] + b_len * ln(chars_x / chars_y)
   + b_trunc * d_trunc + b_lid * d_lid), binomial over the votes of a pair (2 per judge, 2 per judge for the pool).
   The responder terms absorb "stronger models write longer answers"; b_len is the residual effect of length
   between equal responders, and win2x the implied win rate of an answer twice as long as an equal one.
   Fitted on all pairs (with the truncation and LID terms), on clean pairs, and on clean pairs in / outside English.
D. what the judges cite in their "comparison" field when they take the longer or the shorter answer
E. the pool's majority label against length: longer-wins share overall, by tier and task type, and the accuracy a
   length-only classifier would reach on the labels; mean pool vote share for the longer answer by band
--examples N prints N clean pairs where the pool majority and Opus disagree, in each direction, with the rationales.
Writes data/preferences/pilot/LENGTH_BIAS.txt.

Full run (three judges per pair, which differ by pair, plus a tie-breaker on even splits): the pool label uses
every vote a pair has (--pool-any); A and B only cover the pilot pairs, the ones with Opus votes:
  collect-preferences/run.sh length_bias.py --pairs data/preferences/pairs.parquet \
      --votes data/preferences/votes/pass1,data/preferences/votes/substitutes,data/preferences/votes/tiebreak \
      --runs <judge>|v3|on,... --pool <the same> --pool-any --out data/preferences/LENGTH_BIAS.txt
"""
import argparse
import collections
import glob
import json
import math
import os
import random
import re
import sys

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PILOT = os.path.join(ROOT, "data", "preferences", "pilot")
sys.path.insert(0, HERE)
from judge_prompt import same_lang  # noqa: E402

DEFAULT_RUNS = ["gemma-4-31B-it|v3|on", "qwen3.8-27b|v3|on", "gpt-oss-120b-bf16|v3|on", "qwen3.5-122b-a10b|v3|on",
                "mistral-small-4-119b|v3|on", "glm-4.5-air|v3|on"]
DEFAULT_POOL = ["gemma-4-31B-it|v3|on", "qwen3.8-27b|v3|on", "gpt-oss-120b-bf16|v3|on"]
BANDS = [(1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 4.0), (4.0, float("inf"))]
LONG_KW = re.compile(r"more (comprehensive|detailed|thorough|complete|in-depth|extensive|elaborate|nuanced|depth)|"
                     r"comprehensive|thorough|in-depth|covers more|greater depth|more context|richer", re.I)
SHORT_KW = re.compile(r"concise|succinct|brevity|to the point|verbose|verbosity|padded|padding|repetiti|rambl|"
                      r"unnecessar|bloat|filler|overly long|too long|excessive", re.I)


def band_name(a, b):
    return "{}-{}".format(a, b if b != float("inf") else "inf")


VOTE_COLS = ["pair_id", "order", "judge", "prompt_version", "thinking", "prefers", "comparison"]


def load_votes(path_globs):
    runs = collections.defaultdict(dict)        # run -> pair_id -> {order: row}
    for p in sorted(f for g in path_globs for f in glob.glob(g)):
        for r in pq.read_table(p, columns=VOTE_COLS).to_pylist():
            run = "{}|{}|{}".format(r["judge"], r["prompt_version"], r["thinking"])
            runs[run].setdefault(r["pair_id"], {})[r["order"]] = r
    return runs


def votes(d, i):
    """the valid votes ('x'/'y') of pair i in run dict d, one per order"""
    rows = d.get(i, {})
    return [rows[o]["prefers"] for o in (0, 1) if o in rows and rows[o].get("prefers") in ("x", "y")]


def pct(a, b):
    return "{:5.1f}%".format(100.0 * a / b) if b else "   - "


def fit(X, k, n, lam=0.05):
    """binomial logistic regression; ridge lam on all but the last three columns (length, truncation, LID)"""
    d = X.shape[1]

    def nll(b):
        z = X @ b
        return -(k * -np.logaddexp(0, -z) + (n - k) * -np.logaddexp(0, z)).sum() + lam * np.sum(b[:-3] ** 2)

    def grad(b):
        pr = 1 / (1 + np.exp(-(X @ b)))
        g = -X.T @ (k - n * pr)
        g[:-3] += 2 * lam * b[:-3]
        return g
    b = minimize(nll, np.zeros(d), jac=grad, method="L-BFGS-B").x
    pr = 1 / (1 + np.exp(-(X @ b)))
    H = X.T @ (X * (n * pr * (1 - pr))[:, None])
    H[np.arange(d - 3), np.arange(d - 3)] += 2 * lam
    return b, np.sqrt(np.diag(np.linalg.pinv(H)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(PILOT, "pairs.parquet"))
    ap.add_argument("--votes", default=os.path.join(PILOT, "v3", "votes"), help="vote directories, comma-separated")
    ap.add_argument("--opus", default=os.path.join(PILOT, "v3", "opus", "votes.parquet"))
    ap.add_argument("--out", default=os.path.join(PILOT, "v3", "LENGTH_BIAS.txt"))
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="runs (judge|prompt|thinking) to analyse")
    ap.add_argument("--pool", default=",".join(DEFAULT_POOL), help="runs whose votes form the pool")
    ap.add_argument("--examples", type=int, default=0, help="print N disputed clean pairs per direction with rationales")
    ap.add_argument("--pool-any", action="store_true", help="pool label from every vote a pair has (the full run), not only pairs with all pool runs")
    args = ap.parse_args()
    for k in ("pairs", "opus", "out"):
        setattr(args, k, os.path.join(ROOT, getattr(args, k)))
    cols = [c for c in pq.read_schema(args.pairs).names if c != "messages" or args.examples]
    pairs = {p["id"]: p for p in pq.read_table(args.pairs, columns=cols).to_pylist()}
    runs = load_votes([os.path.join(ROOT, d, "*.parquet") for d in args.votes.split(",")])
    opus = load_votes([args.opus])
    opus_run = next(iter(opus))
    runs.update(opus)
    run_names = [r for r in args.runs.split(",") if r in runs]
    pool_runs = [r for r in args.pool.split(",") if r in runs]
    models = sorted({p["model_x"] for p in pairs.values()} | {p["model_y"] for p in pairs.values()})
    L = []

    def flagged(p, s):
        if p["task_type"] == "translation" or p["locale_dependence"] == "language_specific":
            return False
        return bool(p["lid_" + s] and p["lid_conf_" + s] >= 0.5 and len(p["response_" + s].strip()) >= 20
                    and not same_lang(p["lid_" + s], p["language"]))

    def clean(p):
        return (min(p["chars_x"], p["chars_y"]) >= 200 and p["finish_x"] != "length" and p["finish_y"] != "length"
                and not flagged(p, "x") and not flagged(p, "y"))

    def ratio(p):
        return max(p["chars_x"], p["chars_y"]) / max(1, min(p["chars_x"], p["chars_y"]))

    def longer(p):
        return "x" if p["chars_x"] > p["chars_y"] else "y"

    # per run and pair: (votes for x, valid votes); the pool concatenates its runs' votes
    kn = {run: {i: (sum(v == "x" for v in vs), len(vs)) for i in pairs for vs in [votes(runs[run], i)]} for run in run_names + [opus_run]}
    pool_votes = {i: sum((votes(runs[r], i) for r in pool_runs), []) for i in pairs}
    kn["POOL"] = {i: (sum(v == "x" for v in vs), len(vs)) for i, vs in pool_votes.items()}
    full = {run: 2 for run in run_names + [opus_run]}
    full["POOL"] = 2 if args.pool_any else 2 * len(pool_runs)

    def pref(run, i):
        """'x' / 'y' when all votes agree (judges) or a majority exists (pool); None otherwise"""
        k, n = kn[run][i]
        if n < full[run]:
            return None
        if run == "POOL":
            return "x" if k > n / 2 else "y" if k < n / 2 else None
        return "x" if k == n else "y" if k == 0 else None

    ref = {i: pref(opus_run, i) for i in pairs}
    shown = run_names + ["POOL"]
    L.append("LENGTH BIAS: {} pairs, {} clean (both >= 200 chars, neither truncated, no LID flag); reference {}; pool = {}".format(
        len(pairs), sum(clean(p) for p in pairs.values()), opus_run, " + ".join(pool_runs)))
    L.append("length-ratio bands, all pairs:   " + "  ".join("{}: {}".format(band_name(a, b), sum(a <= ratio(p) < b for p in pairs.values())) for a, b in BANDS))
    L.append("                   clean pairs: " + "  ".join("{}: {}".format(band_name(a, b), sum(a <= ratio(p) < b for p in pairs.values() if clean(p))) for a, b in BANDS))
    L.append("")

    # ---- A
    for title, keep in (("all pairs", lambda p: True), ("clean pairs", clean)):
        L.append("A. LONGER ANSWER WINS by length-ratio band, {} (judge / Opus, pairs where both are consistent; count)".format(title))
        L.append("  {:28s} ".format("run") + " ".join("{:>19s}".format(band_name(a, b)) for a, b in BANDS) + "   {:>19s}".format("ratio >= 1.2"))
        for run in shown:
            cells = []
            for a, b in BANDS + [(1.2, float("inf"))]:
                lw = ow = n = 0
                for i, p in pairs.items():
                    if not keep(p) or not (a <= ratio(p) < b) or pref(run, i) is None or ref[i] is None:
                        continue
                    lw += pref(run, i) == longer(p)
                    ow += ref[i] == longer(p)
                    n += 1
                cells.append("{}/{}({})".format(pct(lw, n).strip(), pct(ow, n).strip(), n))
            L.append("  {:28s} ".format(run) + " ".join("{:>19s}".format(c) for c in cells[:-1]) + "   {:>19s}".format(cells[-1]))
        L.append("")

    # ---- B
    L.append("B. DISAGREEMENTS WITH OPUS (both consistent, ratio >= 1.2): share where the judge took the LONGER answer (50% = no bias relative to Opus)")
    for title, keep in (("all pairs", lambda p: True), ("clean pairs", clean)):
        L.append("  " + title)
        for run in shown:
            ids = [i for i, p in pairs.items() if keep(p) and ratio(p) >= 1.2 and pref(run, i) and ref[i]]
            dis = [i for i in ids if pref(run, i) != ref[i]]
            jl = sum(pref(run, i) == longer(pairs[i]) for i in dis)
            L.append("    {:28s} disagreements {:4d} of {:4d}: judge longer {} ({}), Opus longer {} ({})".format(
                run, len(dis), len(ids), pct(jl, len(dis)), jl, pct(len(dis) - jl, len(dis)), len(dis) - jl))
    L.append("")

    # ---- C
    def design(ids):
        X = np.zeros((len(ids), len(models) + 3))
        for r, i in enumerate(ids):
            p = pairs[i]
            X[r, models.index(p["model_x"])] += 1
            X[r, models.index(p["model_y"])] -= 1
            X[r, -3] = math.log(max(1, p["chars_x"])) - math.log(max(1, p["chars_y"]))
            X[r, -2] = (p["finish_x"] == "length") - (p["finish_y"] == "length")
            X[r, -1] = flagged(p, "x") - flagged(p, "y")
        return X[:, 1:]                                              # first responder is the reference

    L.append("C. LENGTH-CONTROLLED FIT: P(x wins) = sigmoid(theta[model_x] - theta[model_y] + b_len*ln(chars_x/chars_y) + b_trunc*d_trunc + b_lid*d_lid)")
    L.append("   binomial over a pair's votes; b_len > 0 means longer answers win between equal responders; win2x = implied win rate of an answer")
    L.append("   twice as long as an equal one. SE treats the two orders (and the pool's judges) as independent votes, so it is optimistic.")
    for title, keep in (("all pairs (with truncation and LID terms)", lambda p: True), ("clean pairs", clean),
                        ("clean pairs, English", lambda p: clean(p) and p["tier"] == "en"),
                        ("clean pairs, not English", lambda p: clean(p) and p["tier"] != "en")):
        L.append("  " + title)
        for run in shown + [opus_run]:
            ids = [i for i, p in pairs.items() if keep(p) and kn[run][i][1] > 0]
            k = np.array([kn[run][i][0] for i in ids], float)
            n = np.array([kn[run][i][1] for i in ids], float)
            b, se = fit(design(ids), k, n)
            win2x = 1 / (1 + math.exp(-b[-3] * math.log(2)))
            extra = "  b_trunc {:+.2f}  b_lid {:+.2f}".format(b[-2], b[-1]) if title.startswith("all") else ""
            L.append("    {:28s} pairs {:4d}  b_len {:+.3f} (SE {:.3f}, z {:+.1f})  win2x {:5.1f}%{}".format(
                run, len(ids), b[-3], se[-3], b[-3] / se[-3], 100 * win2x, extra))
    L.append("")

    # ---- D
    L.append("D. WHAT THE JUDGES CITE in 'comparison' (consistent clean pairs, ratio >= 1.2); long-kw = comprehensive/thorough/detailed...,")
    L.append("   short-kw = concise/verbose/padding/repetitive...")
    L.append("  {:28s} {:>24s} {:>10s} | {:>24s} {:>10s}".format("run", "took longer: long-kw (n)", "short-kw", "took shorter: long-kw (n)", "short-kw"))
    for run in run_names + [opus_run]:
        c = collections.Counter()
        for i, p in pairs.items():
            if not clean(p) or ratio(p) < 1.2 or pref(run, i) is None:
                continue
            side = "L" if pref(run, i) == longer(p) else "S"
            c[side] += 1
            txt = " ".join((runs[run][i][o].get("comparison") or "") for o in (0, 1) if o in runs[run][i])
            c[side + "long"] += bool(LONG_KW.search(txt))
            c[side + "short"] += bool(SHORT_KW.search(txt))
        L.append("  {:28s} {:>24s} {:>10s} | {:>24s} {:>10s}".format(
            run, "{} ({})".format(pct(c["Llong"], c["L"]).strip(), c["L"]), pct(c["Lshort"], c["L"]).strip(),
            "{} ({})".format(pct(c["Slong"], c["S"]).strip(), c["S"]), pct(c["Sshort"], c["S"]).strip()))
    L.append("")

    # ---- E
    L.append("E. POOL MAJORITY LABEL against length ({}):".format(" + ".join(pool_runs)))
    for title, keep in (("all pairs", lambda p: True), ("clean pairs", clean)):
        ids = [i for i, p in pairs.items() if keep(p) and pref("POOL", i)]
        lw = sum(pref("POOL", i) == longer(pairs[i]) for i in ids)
        oids = [i for i in ids if ref[i]]
        ow = sum(ref[i] == longer(pairs[i]) for i in oids)
        L.append("  {:12s} pairs {:4d}: longer wins {} -> a length-only classifier reaches {}; Opus on the same pairs {} ({})".format(
            title, len(ids), pct(lw, len(ids)), pct(max(lw, len(ids) - lw), len(ids)), pct(ow, len(oids)), len(oids)))
        for key, order in (("tier", ("en", "large", "mid", "small")), ("task_type", None)):
            c = collections.defaultdict(lambda: [0, 0])
            for i in ids:
                c[pairs[i][key]][0] += pref("POOL", i) == longer(pairs[i])
                c[pairs[i][key]][1] += 1
            keys = order or [k for k, v in sorted(c.items(), key=lambda kv: -kv[1][1]) if v[1] >= 20]
            L.append("    by {}: ".format(key) + "  ".join("{}={} ({})".format(k, pct(*c[k]), c[k][1]) for k in keys))
    L.append("  mean vote share for the LONGER answer by band (clean pairs with all votes), pool / Opus:")
    for a, b in BANDS:
        fp = fo = []
        fp = [(k if longer(pairs[i]) == "x" else n - k) / n for i, (k, n) in kn["POOL"].items() if n >= full["POOL"] and clean(pairs[i]) and a <= ratio(pairs[i]) < b]
        fo = [(k if longer(pairs[i]) == "x" else n - k) / n for i, (k, n) in kn[opus_run].items() if n == 2 and clean(pairs[i]) and a <= ratio(pairs[i]) < b]
        L.append("    {:8s} {:.3f} ({})  /  {:.3f} ({})".format(band_name(a, b), sum(fp) / max(1, len(fp)), len(fp), sum(fo) / max(1, len(fo)), len(fo)))
    L.append("")

    # ---- examples
    if args.examples:
        ex = collections.defaultdict(list)
        for i, p in pairs.items():
            m = pref("POOL", i)
            if m and ref[i] and m != ref[i] and clean(p) and ratio(p) >= 1.2:
                ex["pool took the LONGER answer" if m == longer(p) else "pool took the SHORTER answer"].append(i)
        random.seed(1)
        for d, ids in sorted(ex.items()):
            L.append("EXAMPLES: {} ({} clean disputed pairs; tiers {})".format(d, len(ids), dict(collections.Counter(pairs[i]["tier"] for i in ids))))
            for i in random.sample(ids, min(args.examples, len(ids))):
                p = pairs[i]
                L.append("  -- {} [{} / {}] chars x={} y={}; pool votes for x {}/{}; Opus {}".format(
                    i, p["language"], p["task_type"], p["chars_x"], p["chars_y"], kn["POOL"][i][0], kn["POOL"][i][1], ref[i]))
                L.append("     PROMPT: " + json.loads(p["messages"])[-1]["content"][:300].replace("\n", " "))
                for run in pool_runs + [opus_run]:
                    row = runs[run].get(i, {}).get(0) or runs[run].get(i, {}).get(1)
                    if row is None:
                        continue
                    L.append("     {:18s} ({}): {}".format(run.split("|")[0], row["prefers"], (row.get("comparison") or "")[:450].replace("\n", " ")))
            L.append("")

    text = "\n".join(L)
    print(text)
    with open(args.out, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
