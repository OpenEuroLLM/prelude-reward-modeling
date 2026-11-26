#!/usr/bin/env python3
"""A chat model behind OpenRouter as a judge of the pilot pairs, compared with Opus and the open judges.
Evaluation only: these votes never become labels. Written for Inkling 2 Mini (Thinking Machines, 2026-11-20), whose
weights (NVFP4/MXFP8, a new looped architecture) the MI250X and vLLM 0.26 cannot run.

  collect-preferences/run.sh api_judge_pilot.py show [--pair N] [--order 0|1]   # one request as sent; sends nothing
  collect-preferences/run.sh api_judge_pilot.py run [--limit N] [--workers 8]  # calls the API -> <out>/calls.jsonl
  collect-preferences/run.sh api_judge_pilot.py collect                         # -> <out>/votes.parquet
  collect-preferences/run.sh api_judge_pilot.py report                          # -> <out>/REPORT.txt

The request is exactly what the open judges get (judge_prompt.py v3: system prompt and user message, both orders).
Decoding follows the judges' rules: reasoning on at the model's default effort (Inkling's chat template defaults to
0.9, "high"), the model's own sampling (Inkling's generation_config: temperature 1.0, nothing else), a recorded
seed per call, a budget of 16,384 new tokens and, for a call without a verdict, one retry with twice the budget.
The verdict is the JSON object of the final answer (status ok), or a single verdict letter in it (ok_regex).
OpenRouter returns the reasoning separately from the answer; both are kept.

The key is read from OPENROUTER_API_KEY or ~/.openrouter_api_key and never printed. Requests ask OpenRouter for
providers that do not collect data (`data_collection: deny`). Every call is appended to <out>/calls.jsonl, so an
interrupted run resumes where it stopped; `run` stops once the amount spent (usage.cost) passes --max-cost, on a
402 (credits) and after 20 failed calls in a row.
"""
import argparse
import collections
import glob
import hashlib
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
from judge_prompt import PROMPT_VERSION, build, extract  # noqa: E402

API = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "thinkingmachines/inkling-2-mini"
PILOT = os.path.join(ROOT, "data", "preferences", "pilot")
OUT = os.path.join(PILOT, "v3", "inkling-2-mini")
BUDGET = 16384
FULL_PAIRS = 128000
RETRY_HTTP = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
VERDICT_RE = None                     # set from judge.py on first use (it imports vLLM-free helpers only)


def seed_for(pair_id, order, attempt):
    return int(hashlib.sha256("{}|{}|{}".format(pair_id, order, attempt).encode()).hexdigest()[:8], 16)


def request_for(pair, order, model, effort, max_tokens, attempt=0):
    system, user = build(pair, order)
    return {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "reasoning": {"effort": effort}, "temperature": 1.0, "seed": seed_for(pair["id"], order, attempt),
            "max_tokens": max_tokens, "provider": {"data_collection": "deny"}, "usage": {"include": True}}


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


def verdict_of(content):
    """-> (status, obj, verdict) from the final answer text"""
    global VERDICT_RE
    if VERDICT_RE is None:
        import re
        VERDICT_RE = re.compile(r'"verdict"\s*:\s*"([AB])"')
    obj = extract(content or "")
    if obj and obj.get("verdict") in ("A", "B"):
        return "ok", obj, obj["verdict"]
    found = set(VERDICT_RE.findall(content or ""))
    if len(found) == 1:
        return "ok_regex", None, found.pop()
    return "parse_error", None, None


def show(args):
    pairs = load_pairs(args.pairs)
    body = request_for(pairs[args.pair], args.order, args.model, args.effort, BUDGET)
    shown = dict(body, messages=[dict(m, content=m["content"][:1500] + (" [...]" if len(m["content"]) > 1500 else ""))
                                 for m in body["messages"]])
    print(json.dumps(shown, ensure_ascii=False, indent=1))
    chars = sorted(sum(len(m["content"]) for m in request_for(p, o, args.model, args.effort, BUDGET)["messages"])
                   for p in pairs for o in (0, 1))
    print("\n{} requests; characters: median {:,}, mean {:,.0f}, max {:,}".format(
        len(chars), chars[len(chars) // 2], sum(chars) / len(chars), chars[-1]))


def read_calls(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def call(session, body, max_attempts=6):
    """-> (http attempts, status or None, parsed JSON or error dict, seconds of the last attempt)"""
    wait = 2.0
    for attempt in range(1, max_attempts + 1):
        t0 = time.time()
        try:
            r = session.post(API, json=body, timeout=1800)
            status, retry_after = r.status_code, r.headers.get("retry-after")
            try:
                data = r.json()
            except ValueError:
                data = {"error_text": r.text[:4000]}
        except requests.RequestException as e:
            status, retry_after, data = None, None, {"error_text": repr(e)[:4000]}
        dt = time.time() - t0
        if status == 200 and "error" in data and not data.get("choices"):
            status = (data["error"] or {}).get("code") or 502           # provider error inside a 200
            status = status if isinstance(status, int) else 502
        if status == 200 or (status not in RETRY_HTTP and status is not None) or attempt == max_attempts:
            return attempt, status, data, dt
        try:
            time.sleep(min(float(retry_after), 120) if retry_after else wait)
        except ValueError:
            time.sleep(wait)
        wait = min(2 * wait, 60)


def message_of(data):
    ch = (data.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    return msg.get("content") or "", msg.get("reasoning") or "", ch.get("finish_reason") or ch.get("native_finish_reason")


def run(args):
    pairs = load_pairs(args.pairs, args.limit)
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "calls.jsonl")
    old = read_calls(path)
    final = {(c["pair_id"], c["order"]) for c in old if c["model_requested"] == args.model and c.get("final")}
    spent = sum(c.get("cost") or 0 for c in old)
    todo = [(p, o) for p in pairs for o in (0, 1) if (p["id"], o) not in final]
    print("{}: {} pairs, {} calls done before, {} to make; spent so far ${:.4f} (stop at ${:.2f}); effort {}; {} workers"
          .format(args.model, len(pairs), len(final), len(todo), spent, args.max_cost, args.effort, args.workers))
    if not todo:
        return
    key = api_key()
    local, lock, stop = threading.local(), threading.Lock(), threading.Event()
    state = {"spent": spent, "n": 0, "ok": 0, "nov": 0, "fail": 0, "fail_run": 0, "secs": []}
    t_start = time.time()

    def session():
        if not hasattr(local, "s"):
            local.s = requests.Session()
            local.s.headers.update({"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        return local.s

    def one(pair, order, attempt, budget):
        body = request_for(pair, order, args.model, args.effort, budget, attempt)
        tries, http, data, dt = call(session(), body)
        content, reasoning, finish = message_of(data) if http == 200 else ("", "", None)
        status = verdict_of(content)[0] if http == 200 else "error"
        usage = data.get("usage") or {}
        return {"pair_id": pair["id"], "order": order, "model_requested": args.model, "attempt": attempt,
                "max_tokens": budget, "seed": body["seed"], "effort": args.effort, "status": status, "http": http,
                "http_tries": tries, "seconds": round(dt, 3), "finish_reason": finish,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "cost": usage.get("cost"), "response": data}

    def work(pair, order):
        if stop.is_set():
            return
        recs = [one(pair, order, 0, BUDGET)]
        if recs[0]["status"] == "parse_error":                 # no verdict: one retry with twice the budget
            recs.append(one(pair, order, 1, 2 * BUDGET))
        if recs[-1]["status"] != "error":
            recs[-1]["final"] = True
        with lock:
            with open(path, "a") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            state["n"] += 1
            state["spent"] += sum(r.get("cost") or 0 for r in recs)
            last = recs[-1]
            if last["status"] == "error":
                state["fail"] += 1
                state["fail_run"] += 1
                print("  error {}_o{}: HTTP {} {}".format(pair["id"], order, last["http"], json.dumps(last["response"])[:300]),
                      flush=True)
            else:
                state["ok"] += last["status"] in ("ok", "ok_regex")
                state["nov"] += last["status"] == "parse_error"
                state["fail_run"] = 0
                state["secs"].append(sum(r["seconds"] for r in recs))
            if last["http"] == 402 or state["spent"] > args.max_cost or state["fail_run"] >= 20:
                if not stop.is_set():
                    print("STOPPING: {}".format("HTTP 402 (credits)" if last["http"] == 402 else "spent ${:.4f} > ${:.2f}".format(
                        state["spent"], args.max_cost) if state["spent"] > args.max_cost else "20 failed calls in a row"), flush=True)
                stop.set()
            if state["n"] % args.every == 0 or state["n"] == len(todo):
                secs = sorted(state["secs"]) or [0]
                el = time.time() - t_start
                print("{}/{} calls, {} with a verdict, {} without, {} failed, ${:.4f} spent, {:.0f}s elapsed, median call {:.0f}s, "
                      "ETA {:.0f}s".format(state["n"], len(todo), state["ok"], state["nov"], state["fail"], state["spent"], el,
                                          secs[len(secs) // 2], el / state["n"] * (len(todo) - state["n"])), flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        for fut in as_completed([ex.submit(work, p, o) for p, o in todo]):
            fut.result()
    print("done: {} calls this run, {} with a verdict, {} without, {} failed; total spent ${:.4f}".format(
        state["n"], state["ok"], state["nov"], state["fail"], state["spent"]))


VOTE_SCHEMA = [("pair_id", "string"), ("order", "int8"), ("judge", "string"), ("judge_revision", "string"),
               ("prompt_version", "string"), ("thinking", "string"), ("language", "string"), ("status", "string"),
               ("verdict", "string"), ("prefers", "string"), ("request", "string"), ("response_a", "string"),
               ("response_b", "string"), ("comparison", "string"), ("raw", "string"), ("reasoning", "string"),
               ("finish_reason", "string"), ("prompt_tokens", "int32"), ("completion_tokens", "int32"),
               ("reasoning_tokens", "int32"), ("attempt", "int8"), ("first_status", "string"), ("seed", "int64"),
               ("max_tokens", "int32"), ("sampling", "string"), ("provider", "string"), ("cost", "float64"),
               ("seconds", "float64")]


def collect(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pairs = {p["id"]: p for p in load_pairs(args.pairs)}
    by = collections.defaultdict(list)
    for c in read_calls(os.path.join(args.out, "calls.jsonl")):
        if c["model_requested"] == args.model:
            by[(c["pair_id"], c["order"])].append(c)
    rows = []
    for (pid, order), cs in sorted(by.items()):
        fin = [c for c in cs if c.get("final")]
        c = fin[-1] if fin else cs[-1]
        first = [x for x in cs if x["attempt"] == 0]
        data = c["response"]
        content, reasoning, finish = message_of(data) if c["http"] == 200 else ("", "", None)
        status, obj, verdict = verdict_of(content) if c["http"] == 200 else ("error", None, None)
        usage = data.get("usage") or {}
        rows.append({"pair_id": pid, "order": order, "judge": args.model, "judge_revision": data.get("model"),
                     "prompt_version": PROMPT_VERSION, "thinking": "on", "language": pairs[pid]["language"],
                     "status": status, "verdict": verdict,
                     "prefers": None if verdict not in ("A", "B") else ("x" if (verdict == "A") == (order == 0) else "y"),
                     "request": (obj or {}).get("request"), "response_a": (obj or {}).get("response_a"),
                     "response_b": (obj or {}).get("response_b"), "comparison": (obj or {}).get("comparison"),
                     "raw": content, "reasoning": reasoning, "finish_reason": finish,
                     "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                     "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                     "attempt": c["attempt"], "first_status": first[-1]["status"] if first else None, "seed": c["seed"],
                     "max_tokens": c["max_tokens"], "sampling": json.dumps({"temperature": 1.0, "effort": c["effort"]}),
                     "provider": data.get("provider"), "cost": sum(x.get("cost") or 0 for x in cs), "seconds": c["seconds"]})
    schema = pa.schema([(n, getattr(pa, t)()) for n, t in VOTE_SCHEMA])
    path = os.path.join(args.out, "votes.parquet")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    print("{} calls of {} expected ({}) -> {}".format(len(rows), 2 * len(pairs), dict(collections.Counter(r["status"] for r in rows)), path))


def report(args):
    import numpy as np
    import pyarrow.parquet as pq
    from analyze_pilot import agreement, load_votes, soft_p
    from length_bias import fit

    pairs = {p["id"]: p for p in pq.read_table(args.pairs).to_pylist()}
    v3 = os.path.dirname(args.out)
    runs = load_votes(os.path.join(v3, "votes", "*.parquet"))
    opus = load_votes(os.path.join(v3, "opus", "votes.parquet"))
    opus_run = next(iter(opus))
    new = load_votes(os.path.join(args.out, "votes.parquet"))
    new_run = next(iter(new))
    runs.update(new)
    if os.path.exists(os.path.join(v3, "jev", "votes.parquet")):
        runs.update(load_votes(os.path.join(v3, "jev", "votes.parquet")))
    runs.update(opus)
    softp = {run: {i: soft_p(rows) for i, rows in d.items()} for run, d in runs.items()}
    refp = softp[opus_run]
    opus_ids = [i for i in pairs if refp.get(i) in (0.0, 1.0)]
    jev_run = next((r for r in runs if "jev" in r.lower()), None)
    show = [("Inkling 2 Mini, thinking (effort {})".format(args.effort), new_run),
            ("Gemma 4 31B, thinking", "gemma-4-31B-it|v3|on"), ("Qwen3.8-27B, thinking (xhigh)", "qwen3.8-27b|v3|on"),
            ("Jev 1.14", jev_run), ("Qwen3.5-122B, thinking", "qwen3.5-122b-a10b|v3|on"),
            ("Qwen3.5-27B, thinking", "qwen3.5-27b|v3|on"), ("gpt-oss-120b, reasoning high", "gpt-oss-120b-bf16|v3|on")]
    show = [(n, r) for n, r in show if r in runs]
    models = sorted({p["model_x"] for p in pairs.values()} | {p["model_y"] for p in pairs.values()})

    def flagged(p, s):
        from judge_prompt import same_lang
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
        d = runs[run]
        calls = [r for i in ids for r in d.get(i, {}).values()]
        verdicts = [r["verdict"] for r in calls if r.get("verdict") in ("A", "B")]
        both = [i for i in ids if i in d and all((d[i].get(o) or {}).get("prefers") in ("x", "y") for o in (0, 1))]
        cons = sum(d[i][0]["prefers"] == d[i][1]["prefers"] for i in both)
        s, n = agreement(softp[run], refp, ids, ref=True)
        toks = [r.get("completion_tokens") or 0 for r in calls]
        return dict(calls=len(calls), nov=sum(r.get("prefers") not in ("x", "y") for r in calls) / max(len(calls), 1),
                    cons=cons / max(len(both), 1), agr=s / n if n else float("nan"), n=n,
                    first=sum(v == "A" for v in verdicts) / max(len(verdicts), 1), toks=sum(toks) / max(len(toks), 1))

    L = ["{} on the pilot pairs: {} pairs in both orders, judge prompt {} as for the open judges (api_judge_pilot.py)."
         .format(new_run.split("|")[0], len(pairs), PROMPT_VERSION), "",
         "Agreement: with Opus on the {} pairs where Opus gives the same winner in both orders; each of the judge's two".format(len(opus_ids)),
         "calls scores 1 if it matches, 0 if not and 1/2 without a verdict (a pair it decides differently in the two orders",
         "scores 1/2). Consistency: both orders give the same winner (pairs with two verdicts). First: share of verdicts",
         "for the response shown first. 2x longer: win rate of an answer twice as long as an equal one (clean English",
         "pairs, 50% = neutral, +- 1 s.e.).", "",
         "| Judge | Agreement with Opus | Consistency | First | No verdict | Output tokens per call | 2x longer wins (en) |",
         "|---|---|---|---|---|---|---|"]
    for name, run in show:
        q = quality(run)
        w, se = win2x(run)
        L.append("| {} | {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}% | {:,.0f} | {:.0f}% +- {:.0f} |".format(
            name, 100 * q["agr"], 100 * q["cons"], 100 * q["first"], 100 * q["nov"], q["toks"], 100 * w, 100 * se))
    for field in ("tier", "category"):
        cols = [n for n, _ in show[:4]]
        L += ["", "BY {} (agreement with Opus as above; n = pairs where Opus is consistent)".format(field.upper()),
              "  {:<24} {:>5}  ".format(field, "n") + "  ".join("{:>14}".format(c.split(",")[0]) for c in cols)]
        for val in sorted({p[field] for p in pairs.values()}):
            ids = [i for i, p in pairs.items() if p[field] == val]
            n = sum(i in set(opus_ids) for i in ids)
            L.append("  {:<24} {:>5}  ".format(str(val), n) + "  ".join(
                "{:>13.1f}%".format(100 * quality(r, ids)["agr"]) for _, r in show[:4]))

    # the pool with this judge swapped in (Qwen3.5-122B breaks 3:3 ties, as in the labelling run; family rules ignored)
    J = {"Gemma": "gemma-4-31B-it|v3|on", "Qwen3.8": "qwen3.8-27b|v3|on", "gpt-oss": "gpt-oss-120b-bf16|v3|on", "new": new_run}

    def pool_acc(members):
        s = 0.0
        for i in opus_ids:
            v = [(runs[J[m]].get(i, {}).get(o) or {}).get("prefers") for m in members for o in (0, 1)]
            x, y = v.count("x"), v.count("y")
            if x == y:
                v2 = [(runs["qwen3.5-122b-a10b|v3|on"].get(i, {}).get(o) or {}).get("prefers") for o in (0, 1)]
                x, y = x + v2.count("x"), y + v2.count("y")
            w = "x" if x > y else "y" if y > x else None
            s += 0.5 if w is None else float(w == ("x" if refp[i] == 1.0 else "y"))
        return s / len(opus_ids)

    name = new_run.split("|")[0]
    L += ["", "POOL (majority of 6 votes, Qwen3.5-122B on ties, a remaining tie counts half; own-family rules ignored)"]
    for members in (("Gemma", "Qwen3.8", "gpt-oss"), ("Gemma", "Qwen3.8", "new"), ("Gemma", "new", "gpt-oss"), ("new", "Qwen3.8", "gpt-oss")):
        L.append("  {:<40} {:.1f}%".format(" + ".join(name if m == "new" else m for m in members), 100 * pool_acc(members)))

    calls = [r for d in new.values() for rows in d.values() for r in rows.values()]
    cost = sum(r.get("cost") or 0 for r in calls)
    secs = sorted(r["seconds"] for r in calls if r.get("seconds") is not None) or [0]
    toks = [r.get("completion_tokens") or 0 for r in calls]
    prov = collections.Counter(r.get("provider") for r in calls)
    L += ["", "COST AND LATENCY",
          "  {} calls; providers {}; retried (no verdict at {:,} tokens) {}; finished by the length limit {}".format(
              len(calls), dict(prov), BUDGET, sum((r.get("attempt") or 0) > 0 for r in calls),
              sum(r.get("finish_reason") == "length" for r in calls)),
          "  usage.cost ${:.3f} in all = ${:.2f} per 1,000 pairs (both orders), ${:.0f} for {:,} pairs".format(
              cost, cost * 1000 / len(pairs), cost * FULL_PAIRS / len(pairs), FULL_PAIRS),
          "  output tokens per call: mean {:,.0f}, p99 {:,.0f}, max {:,}; seconds per call: median {:.0f}, p95 {:.0f}, max {:.0f}".format(
              sum(toks) / max(len(toks), 1), float(np.quantile(toks, 0.99)) if toks else 0, max(toks or [0]),
              secs[len(secs) // 2], secs[int(0.95 * len(secs))], secs[-1])]
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
    ap.add_argument("--effort", default="high", help="reasoning effort (Inkling's template default 0.9 = high)")
    ap.add_argument("--limit", type=int, help="first N pairs only")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-cost", type=float, default=5.0, help="stop once usage.cost adds up to more than this (USD)")
    ap.add_argument("--every", type=int, default=100, help="progress line every N calls")
    ap.add_argument("--pair", type=int, default=0, help="show: index of the pair")
    ap.add_argument("--order", type=int, default=0, choices=[0, 1], help="show: presentation order")
    args = ap.parse_args()
    args.pairs, args.out = os.path.join(ROOT, args.pairs), os.path.join(ROOT, args.out)   # relative to the repo root
    {"show": show, "run": run, "collect": collect, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
