#!/usr/bin/env python3
"""Per-model statistics of the responses: data/responses/STATS.txt.

  collect-responses/run.sh stats.py [--responses data/responses/responses.parquet] [--out data/responses/STATS.txt]

Per group, model and mode: n, mean completion tokens (reasoning + answer), cut (stopped by the token budget),
closed (thinking ended before the budget), no_rsn (empty reasoning), empty (empty answer), salad (more than 1%
of the answer's letters in a script the prompt's language does not use, prompts that contain such letters
themselves and translation tasks excluded; or the answer loops: compresses below 0.15), leaks (special tokens
in the text), and, once continue_truncated.py has run, cont (share of responses that were continued).
The first STATS.txt (2026-11-14) came from an unsaved script whose salad rule differed slightly (up to 2 points
higher); this script is the reference from 2026-11-15 on.
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from smoke_compare import foreign_share, loop  # noqa: E402
from smoke_test import pool  # noqa: E402

ORDER = ["prelude", "strong", "mid", "weak"]


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default="data/responses/responses.parquet")
    ap.add_argument("--out", default="data/responses/STATS.txt")
    args = ap.parse_args()
    p, models = pool()
    t = pq.read_table(os.path.join(ROOT, args.responses))
    cols = [c for c in ["id", "slot", "model", "mode", "model_group", "reasoning", "response", "finish_reason", "think_closed",
                        "leaks", "completion_tokens", "continued"] if c in t.schema.names]
    rows = t.select(cols).to_pylist()
    prompts = {}
    with open(os.path.join(ROOT, "data", "compiled", "prompts.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            prompts[o["id"]] = (o["language"], o.get("task_type"),
                                foreign_share(" ".join(m["content"] or "" for m in o["messages"]), o["language"]) > 0.01)
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["model_group"], r["model"], r["mode"])].append(r)
    has_cont = "continued" in cols
    hdr = "{:8s} {:36s} {:6s} {:>6s} {:>7s} {:>6s} {:>7s} {:>7s} {:>6s} {:>6s} {:>6s}".format(
        "group", "model", "mode", "n", "tokens", "cut", "closed", "no_rsn", "empty", "salad", "leaks")
    L = [hdr + ("  {:>5s}".format("cont") if has_cont else "")]
    pct = lambda a, b: "{:5.1f}%".format(100.0 * a / b) if b else "    -"  # noqa: E731
    for (g, m, mode), rs in sorted(by.items(), key=lambda kv: (ORDER.index(kv[0][0]), kv[0][1], kv[0][2])):
        n = len(rs)
        think = mode == "think"

        def salad(r):
            lang, task, prompt_foreign = prompts[r["id"]]
            if task == "translation" or prompt_foreign:
                return loop(r["response"] or "")
            return foreign_share(r["response"] or "", lang) > 0.01 or loop(r["response"] or "")
        line = "{:8s} {:36s} {:6s} {:6d} {:7.0f} {:>6s} {:>7s} {:>7s} {:>6s} {:>6s} {:>6s}".format(
            g, m, mode, n, sum(r["completion_tokens"] for r in rs) / n, pct(sum(r["finish_reason"] == "length" for r in rs), n),
            pct(sum(bool(r["think_closed"]) for r in rs), n) if think else "      -",
            pct(sum(not (r["reasoning"] or "").strip() for r in rs), n) if think else "      -",
            pct(sum(not (r["response"] or "").strip() for r in rs), n), pct(sum(salad(r) for r in rs), n),
            pct(sum(bool(r["leaks"]) for r in rs), n))
        if has_cont:
            line += "  {:>5s}".format(pct(sum(bool(r["continued"]) for r in rs), n))
        L.append(line)
    fin = collections.Counter(r["finish_reason"] for r in rows)
    same = collections.defaultdict(set)
    for r in rows:
        same[r["id"]].add(r["model"])
    L.append("")
    L.append("{} responses, {} prompts; {:.1f}M generated tokens; finish reasons {}; same model for both responses: {:.1f}% of prompts".format(
        len(rows), len(same), sum(r["completion_tokens"] for r in rows) / 1e6, dict(fin),
        100.0 * sum(len(v) == 1 for v in same.values()) / len(same)))
    L.append("tokens: mean completion tokens (reasoning + answer); cut: stopped by the token budget; closed: thinking ended "
             "before the budget; no_rsn: empty reasoning; empty: empty answer; salad: answer has >1% letters in a script the "
             "prompt language does not use (prompts that themselves contain such letters, and translation tasks, excluded) "
             "or loops; leaks: special tokens in the text" + ("; cont: continued to the larger budget by continue_truncated.py" if has_cont else ""))
    text = "\n".join(L)
    print(text)
    with open(os.path.join(ROOT, args.out), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
