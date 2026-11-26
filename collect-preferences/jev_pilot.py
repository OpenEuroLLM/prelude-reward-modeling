#!/usr/bin/env python3
"""Jev, TypeSafe's decision model, as a judge of the pilot pairs (through OpenRouter's Decisions API), compared
with Opus and the open judges. Evaluation only: these votes never become labels.

  collect-preferences/run.sh jev_pilot.py show [--pair N] [--order 0|1]   # one request as sent, and sizes; sends nothing
  collect-preferences/run.sh jev_pilot.py run [--limit N] [--workers 8]  # calls the API, appends to <out>/calls.jsonl
  collect-preferences/run.sh jev_pilot.py collect                         # -> <out>/votes.parquet
  collect-preferences/run.sh jev_pilot.py report                          # -> <out>/REPORT.txt

Jev writes no text: it answers typed questions about a `state` with a probability for each option. The request
keeps the open judges' prompt (judge_prompt.py, v3) wherever the API allows. `state` is the judges' user message,
the conversation and the two responses in the same tags, without its last line. The one question is a Choice
whose `instructions` are the judges' system prompt (judge_prompt.txt) without its last paragraph, the format of
the JSON reply, followed by the question of that last line, "Which response is better, A or B?". Its options are
"A" and "B", described as "Response A is better." and "Response B is better.". Both orders, as for every judge.
Jev takes no sampling parameters. Requests over about 32,000 bytes run in TypeSafe's long-context mode, which
TypeSafe calls slower and less accurate; the report shows them separately.

The key is read from OPENROUTER_API_KEY or ~/.openrouter_api_key and never printed. Requests ask OpenRouter for
providers that do not collect data (`data_collection: deny`). Every call is appended to <out>/calls.jsonl (default
<out>: data/preferences/pilot/v3/jev), so an interrupted run resumes where it stopped; `run` stops once the
amount spent (usage.cost) passes --max-cost, on a 402 (credits) and after 20 failed calls in a row.
"""
import argparse
import collections
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from judge_prompt import PROMPT_VERSION, SYSTEM, build  # noqa: E402

API = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.14"
PILOT = os.path.join(ROOT, "data", "preferences", "pilot")
OUT = os.path.join(PILOT, "v3", "jev")
LONG_BYTES = 32000                    # above this, OpenRouter runs Jev in TypeSafe's long-context mode
FULL_PAIRS = 128000
QUESTION = "Which response is better, A or B?"
CRITERIA = {"A": "Response A is better.", "B": "Response B is better."}
RETRY_HTTP = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


def instructions():
    """the judges' system prompt without its last paragraph (the JSON reply), then the question"""
    paras = SYSTEM.split("\n\n")
    if not paras[-1].startswith("Reply with a single JSON object"):
        raise SystemExit("judge_prompt.txt changed: its last paragraph is no longer the JSON reply format")
    return "\n\n".join(paras[:-1] + [QUESTION])


INSTRUCTIONS = instructions()


def request_for(pair, order, model):
    _, user = build(pair, order)
    state, last = user.rsplit("\n", 1)
    if not last.startswith(QUESTION):
        raise SystemExit("judge_prompt.build changed: the user message no longer ends with the question")
    return {"model": model, "state": state.rstrip("\n"),
            "questions": {"verdict": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}},
            "provider": {"data_collection": "deny"}}


def text_bytes(body):
    q = body["questions"]["verdict"]
    return len((body["state"] + q["instructions"] + "".join(k + v for k, v in q["criteria"].items())).encode("utf-8"))


def api_key():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        with open(os.path.expanduser("~/.openrouter_api_key")) as f:
            key = f.read().strip()
    return key


def load_pairs(path, limit=None):
    import pyarrow.parquet as pq
    pairs = pq.read_table(path).to_pylist()
    return pairs[:limit] if limit else pairs


def show(args):
    pairs = load_pairs(args.pairs)
    body = request_for(pairs[args.pair], args.order, args.model)
    print(json.dumps(body, ensure_ascii=False, indent=1))
    sizes = sorted(text_bytes(request_for(p, o, args.model)) for p in pairs for o in (0, 1))
    n = len(sizes)
    print("\n{} requests; text bytes: median {:,}, mean {:,.0f}, p95 {:,}, max {:,}; over {:,} bytes (long-context "
          "mode): {} ({:.1%}); instructions {:,} bytes".format(
              n, sizes[n // 2], sum(sizes) / n, sizes[int(0.95 * n)], sizes[-1], LONG_BYTES,
              sum(s > LONG_BYTES for s in sizes), sum(s > LONG_BYTES for s in sizes) / n, len(INSTRUCTIONS.encode())))


def read_calls(path):
    calls = []
    if os.path.exists(path):
        with open(path) as f:
            calls = [json.loads(line) for line in f if line.strip()]
    return calls


def call(session, body, max_attempts=6):
    """-> (attempts, http status or None, parsed JSON or error dict, seconds of the last attempt)"""
    wait = 2.0
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        try:
            r = session.post(API, json=body, timeout=900)
            status, retry_after = r.status_code, r.headers.get("retry-after")
            try:
                data = r.json()
            except ValueError:
                data = {"error_text": r.text[:4000]}
        except requests.RequestException as e:
            status, retry_after, data = None, None, {"error_text": repr(e)[:4000]}
        dt = time.time() - t0
        if status == 200 or (status not in RETRY_HTTP and status is not None) or attempt == max_attempts:
            return attempt, status, data, dt
        try:
            time.sleep(min(float(retry_after), 120) if retry_after else wait)
        except ValueError:
            time.sleep(wait)
        wait = min(2 * wait, 60)


def run(args):
    pairs = load_pairs(args.pairs, args.limit)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "calls.jsonl")
    old = read_calls(path)
    done = {(c["pair_id"], c["order"]) for c in old if c["status"] in ("ok", "refusal") and c["model_requested"] == args.model}
    spent = sum(c.get("cost") or 0 for c in old)
    todo = [(p, o) for p in pairs for o in (0, 1) if (p["id"], o) not in done]
    print("{}: {} pairs, {} calls done before, {} to make; spent so far ${:.4f} (stop at ${:.2f}); {} workers".format(
        args.model, len(pairs), len(done), len(todo), spent, args.max_cost, args.workers))
    if not todo:
        return
    key = api_key()
    local = threading.local()
    lock = threading.Lock()
    stop = threading.Event()
    state = {"spent": spent, "n": 0, "ok": 0, "fail": 0, "fail_run": 0, "secs": []}
    t_start = time.time()

    def session():
        if not hasattr(local, "s"):
            local.s = requests.Session()
            local.s.headers.update({"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        return local.s

    def work(pair, order):
        if stop.is_set():
            return
        body = request_for(pair, order, args.model)
        attempts, http, data, dt = call(session(), body)
        ans = ((data.get("answers") or {}).get("verdict") or {}) if http == 200 else {}
        status = {"choice": "ok", "refusal": "refusal"}.get(ans.get("type"), "error")
        usage = data.get("usage") or {}
        rec = {"pair_id": pair["id"], "order": order, "model_requested": args.model, "status": status, "http": http,
               "attempts": attempts, "seconds": round(dt, 3), "text_bytes": text_bytes(body),
               "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "cost": usage.get("cost"), "response": data}
        with lock:
            with open(path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            state["n"] += 1
            state["spent"] += usage.get("cost") or 0
            if status == "error":
                state["fail"] += 1
                state["fail_run"] += 1
                print("  error {}_o{}: HTTP {} {}".format(pair["id"], order, http, json.dumps(data)[:300]), flush=True)
            else:
                state["ok"] += status == "ok"
                state["fail_run"] = 0
                state["secs"].append(dt)
            if http == 402 or state["spent"] > args.max_cost or state["fail_run"] >= 20:
                if not stop.is_set():
                    print("STOPPING: {}".format("HTTP 402 (credits)" if http == 402 else "spent ${:.4f} > ${:.2f}".format(
                        state["spent"], args.max_cost) if state["spent"] > args.max_cost else "20 failed calls in a row"), flush=True)
                stop.set()
            if state["n"] % args.every == 0 or state["n"] == len(todo):
                secs = sorted(state["secs"]) or [0]
                el = time.time() - t_start
                print("{}/{} calls, {} ok, {} failed, ${:.4f} spent, {:.0f}s elapsed, median call {:.1f}s, ETA {:.0f}s".format(
                    state["n"], len(todo), state["ok"], state["fail"], state["spent"], el, secs[len(secs) // 2],
                    el / state["n"] * (len(todo) - state["n"])), flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(work, p, o) for p, o in todo]):
            fut.result()
    print("done: {} calls this run, {} ok, {} failed; total spent ${:.4f}".format(state["n"], state["ok"], state["fail"],
                                                                                   state["spent"]))


VOTE_SCHEMA = [("pair_id", "string"), ("order", "int8"), ("judge", "string"), ("judge_revision", "string"),
               ("prompt_version", "string"), ("thinking", "string"), ("language", "string"), ("status", "string"),
               ("verdict", "string"), ("prefers", "string"), ("request", "string"), ("response_a", "string"),
               ("response_b", "string"), ("comparison", "string"), ("raw", "string"), ("finish_reason", "string"),
               ("prompt_tokens", "int32"), ("completion_tokens", "int32"), ("prob_a", "float64"),
               ("confidence", "float64"), ("cost", "float64"), ("text_bytes", "int32"), ("seconds", "float64")]


def collect(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pairs = {p["id"]: p for p in load_pairs(args.pairs)}
    last = {}
    for c in read_calls(os.path.join(args.out, "calls.jsonl")):
        if c["model_requested"] != args.model:
            continue
        k = (c["pair_id"], c["order"])
        if k not in last or last[k]["status"] == "error":      # a final answer is never replaced by a later error
            last[k] = c
    rows = []
    for (pid, order), c in sorted(last.items()):
        data = c["response"]
        ans = (data.get("answers") or {}).get("verdict") or {}
        usage = data.get("usage") or {}
        verdict = ans.get("choice") if c["status"] == "ok" else None
        prob = ans.get("probabilities") or {}
        rows.append({"pair_id": pid, "order": order, "judge": args.model, "judge_revision": data.get("model"),
                     "prompt_version": PROMPT_VERSION, "thinking": "none", "language": pairs[pid]["language"],
                     "status": c["status"], "verdict": verdict,
                     "prefers": None if verdict not in ("A", "B") else ("x" if (verdict == "A") == (order == 0) else "y"),
                     "raw": json.dumps(ans or data, ensure_ascii=False)[:4000], "prompt_tokens": usage.get("input_tokens"),
                     "completion_tokens": usage.get("output_tokens"), "prob_a": prob.get("A"),
                     "confidence": ans.get("confidence"), "cost": usage.get("cost"), "text_bytes": c["text_bytes"],
                     "seconds": c["seconds"]})
    schema = pa.schema([(n, getattr(pa, t)()) for n, t in VOTE_SCHEMA])
    path = os.path.join(args.out, "votes.parquet")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    st = collections.Counter(r["status"] for r in rows)
    print("{} calls of {} expected ({}) -> {}".format(len(rows), 2 * len(pairs), dict(st), path))


def report(args):
    import numpy as np
    import pyarrow.parquet as pq
    from analyze_pilot import agreement, load_votes, preference, soft_p
    from judge_prompt import same_lang
    from length_bias import fit
    from pilot_table import RUNS_V3

    pairs = {p["id"]: p for p in pq.read_table(args.pairs).to_pylist()}
    v3 = os.path.dirname(args.out)
    runs = load_votes(os.path.join(v3, "votes", "*.parquet"))
    opus = load_votes(os.path.join(v3, "opus", "votes.parquet"))
    opus_run = next(iter(opus))
    jev = load_votes(os.path.join(args.out, "votes.parquet"))
    jev_run = next(iter(jev))
    runs.update(opus)
    runs.update(jev)
    prefs = {run: {i: preference(rows) for i, rows in d.items()} for run, d in runs.items()}
    ref = prefs[opus_run]
    softp = {run: {i: soft_p(rows) for i, rows in d.items()} for run, d in runs.items()}
    refp = softp[opus_run]
    opus_ids = [i for i in pairs if refp.get(i) in (0.0, 1.0)]       # Opus gives the same winner in both orders
    def variant(v):                       # "v3, thinking on" -> "thinking on"; "v3 (no thinking mode)" -> "no thinking mode"
        v = v.replace("v3, ", "").replace("v3 ", "")
        return v[1:-1] if v.startswith("(") and v.endswith(")") else v

    labels = {r[0]: "{}, {}".format(r[1], variant(r[2])) for r in RUNS_V3}
    labels[jev_run] = "Jev 1.14"
    labels[opus_run] = "Claude Opus 5.5"
    models = sorted({p["model_x"] for p in pairs.values()} | {p["model_y"] for p in pairs.values()})

    def flagged(p, s):
        if p["task_type"] == "translation" or p["locale_dependence"] == "language_specific":
            return False
        return bool(p["lid_" + s] and p["lid_conf_" + s] >= 0.5 and len(p["response_" + s].strip()) >= 20
                    and not same_lang(p["lid_" + s], p["language"]))

    clean_en = [i for i, p in pairs.items() if p["tier"] == "en" and min(p["chars_x"], p["chars_y"]) >= 200
                and p["finish_x"] != "length" and p["finish_y"] != "length" and not flagged(p, "x") and not flagged(p, "y")]

    def win2x(run):
        """as pilot_table.py: win rate of an answer twice as long as an equal one, clean English pairs"""
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
        return 1 / (1 + math.exp(-b[-3] * math.log(2))), se[-3] * math.log(2) / 4

    def quality(run, ids=None):
        ids = list(pairs) if ids is None else ids
        d, pr = runs[run], prefs[run]
        calls = [r for i in ids for r in d.get(i, {}).values()]
        verdicts = [r["verdict"] for r in calls if r.get("verdict") in ("A", "B")]
        decided = [i for i in ids if pr.get(i, (None,))[0] is not None]
        same, n = agreement(softp[run], refp, ids, ref=True)        # an inconsistent pair scores half
        return dict(calls=len(calls), fail=sum(not r["status"].startswith("ok") for r in calls) / max(len(calls), 1),
                    cons=sum(pr[i][0] in ("x", "y") for i in decided) / max(len(decided), 1),
                    agr=same / n if n else float("nan"), n=n,
                    first=sum(v == "A" for v in verdicts) / max(len(verdicts), 1))

    L = ["Jev on the pilot pairs: {} pairs in both orders, request built from judge prompt {} (see jev_pilot.py).".format(
        len(pairs), PROMPT_VERSION), "",
        "Consistency: both orders give the same winner. Agreement: with Opus on the {} pairs where Opus gives the same".format(len(opus_ids)),
        "winner in both orders (n = those the judge judged), each of the judge's two calls scoring 1 if it matches, 0 if",
        "not and 1/2 without a verdict, so a pair the judge is inconsistent on scores 1/2 (as in RUNS.md). First: share",
        "of calls that chose the response shown first (A). 2x longer: win rate of an answer twice as long as an equal one",
        "(clean English pairs, 50% = neutral, +- 1 s.e.).", "",
        "| Judge | Consistency | Agreement with Opus | First | No verdict | 2x longer wins (en) |",
        "|---|---|---|---|---|---|"]
    order = [r[0] for r in RUNS_V3 if r[0] in runs] + [jev_run]
    for run in order:
        q = quality(run)
        w, s = win2x(run)
        partial = "" if q["calls"] >= 0.9 * 2 * len(pairs) else ", stopped after {:,} calls".format(q["calls"])
        L.append("| {} | {:.1f}% | {:.1f}% (n={}) | {:.1f}% | {:.1f}% | {:.0f}% +- {:.0f} |".format(
            labels.get(run, run) + partial, 100 * q["cons"],
            100 * q["agr"], q["n"], 100 * q["first"], 100 * q["fail"], 100 * w, 100 * s))
    q = quality(opus_run)
    L.append("| Claude Opus 5.5 | {:.1f}% | reference | {:.1f}% | {:.1f}% | {:.0f}% |".format(
        100 * q["cons"], 100 * q["first"], 100 * q["fail"], 100 * win2x(opus_run)[0]))

    # Jev's probabilities: per pair, the probability that x is better, averaged over the two orders
    jd = jev[jev_run]

    def p_x(i):
        ps = [(r["prob_a"] if o == 0 else 1 - r["prob_a"]) for o, r in jd.get(i, {}).items() if r.get("prob_a") is not None]
        return sum(ps) / len(ps) if ps else None

    px = {i: p_x(i) for i in pairs}
    have = [i for i in opus_ids if px[i] is not None]
    y = np.array([ref[i][0] == "x" for i in have], float)
    pv = np.array([px[i] for i in have])
    tie = pv == 0.5
    hit = np.where(tie, 0.5, ((pv > 0.5) == (y == 1)).astype(float))      # a probability of exactly 0.5 counts as half
    acc = float(hit.mean())
    L += ["", "PROBABILITIES (Jev's P(x better), averaged over the two orders; on the {} pairs where Opus is consistent)".format(len(have)),
          "  side of 0.5 agrees with Opus: {:.1%}; mean probability given to Opus's side: {:.1%}; Brier score {:.3f} "
          "(0.25 = a coin)".format(acc, float(np.mean(np.where(y == 1, pv, 1 - pv))), float(np.mean((pv - y) ** 2)))]
    L.append("  by margin |2p - 1|:   share of pairs   agreement with Opus")
    marg = np.abs(2 * pv - 1)
    for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]:
        m = (marg >= lo) & (marg < hi)
        if m.sum():
            L.append("    {:.2f}-{:.2f}            {:5.1%}            {:5.1%}  (n={})".format(
                lo, min(hi, 1), m.mean(), hit[m].mean(), int(m.sum())))
    calls = [r for d in jd.values() for r in d.values() if r.get("prob_a") is not None]
    pmax = np.array([max(r["prob_a"], 1 - r["prob_a"]) for r in calls])
    L.append("  per call, probability of the chosen option: median {:.3f}; share >= 0.99: {:.1%}; share < 0.6: {:.1%}".format(
        float(np.median(pmax)), float(np.mean(pmax >= 0.99)), float(np.mean(pmax < 0.6))))

    # request size: TypeSafe's long-context mode above about 32,000 bytes
    size = {i: max(r["text_bytes"] for r in jd[i].values()) for i in jd}
    L += ["", "REQUEST SIZE (text bytes of state + question; long-context mode above {:,})".format(LONG_BYTES),
          "  slice          pairs   Jev consistency   Jev agreement   Qwen3.8-27B agreement   Gemma 4 31B agreement"]
    for name, ids in [("<= 32,000", [i for i in pairs if size.get(i, 0) <= LONG_BYTES]),
                      ("> 32,000", [i for i in pairs if size.get(i, 0) > LONG_BYTES])]:
        qj = quality(jev_run, ids)
        other = ["{:.1%} (n={})".format(quality(r, ids)["agr"], quality(r, ids)["n"]) if r in runs else "-"
                 for r in ("qwen3.8-27b|v3|on", "gemma-4-31B-it|v3|on")]
        L.append("  {:<12} {:>6}   {:>15.1%}   {:>8.1%} (n={})   {:>20}   {:>20}".format(name, len(ids), qj["cons"], qj["agr"], qj["n"], *other))

    # by language tier and by category: Jev against the best open judge on the same pairs
    for field in ("tier", "category"):
        L += ["", "BY {} (agreement with Opus as above; n = pairs where Opus is consistent)".format(field.upper()),
              "  {:<28} {:>6}   {:>16}   {:>16}   {:>16}".format(field, "pairs", "Jev", "Qwen3.8-27B", "gpt-oss-120b high")]
        for val in sorted({p[field] for p in pairs.values()}):
            ids = [i for i, p in pairs.items() if p[field] == val]
            cells = []
            for r in (jev_run, "qwen3.8-27b|v3|on", "gpt-oss-120b-bf16|v3|on"):
                q = quality(r, ids) if r in runs else None
                cells.append("{:.1%} (n={})".format(q["agr"], q["n"]) if q and q["n"] else "-")
            L.append("  {:<28} {:>6}   {:>16}   {:>16}   {:>16}".format(str(val), len(ids), *cells))

    # cost and latency
    cost = sum(r.get("cost") or 0 for d in jd.values() for r in d.values())
    toks = [r["prompt_tokens"] for d in jd.values() for r in d.values() if r.get("prompt_tokens")]
    secs = sorted(r["seconds"] for d in jd.values() for r in d.values() if r.get("seconds") is not None)
    revs = collections.Counter(r.get("judge_revision") for d in jd.values() for r in d.values())
    L += ["", "COST AND LATENCY",
          "  {} calls answered by {}; usage.cost ${:.4f} in all = ${:.3f} per 1,000 pairs (both orders), "
          "${:.0f} for {:,} pairs".format(sum(revs.values()), dict(revs), cost, cost * 1000 / len(pairs),
                                           cost * FULL_PAIRS / len(pairs), FULL_PAIRS),
          "  input tokens per call: mean {:,.0f}, max {:,}; seconds per call: median {:.1f}, p95 {:.1f}, max {:.1f}".format(
              sum(toks) / max(len(toks), 1), max(toks or [0]), secs[len(secs) // 2], secs[int(0.95 * len(secs))], secs[-1])]
    text = "\n".join(L)
    print(text)
    with open(os.path.join(args.out, "REPORT.txt"), "w") as f:
        f.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["show", "run", "collect", "report"])
    ap.add_argument("--pairs", default=os.path.join(PILOT, "pairs.parquet"))
    ap.add_argument("--out", default=OUT, help="directory for calls.jsonl, votes.parquet and REPORT.txt")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, help="first N pairs only")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-cost", type=float, default=3.0, help="stop once usage.cost adds up to more than this (USD)")
    ap.add_argument("--every", type=int, default=100, help="progress line every N calls")
    ap.add_argument("--pair", type=int, default=0, help="show: index of the pair")
    ap.add_argument("--order", type=int, default=0, choices=[0, 1], help="show: presentation order")
    args = ap.parse_args()
    args.pairs, args.out = os.path.join(ROOT, args.pairs), os.path.join(ROOT, args.out)   # relative to the repo root
    {"show": show, "run": run, "collect": collect, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
