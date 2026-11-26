#!/usr/bin/env python3
"""Length probes from the validation answers, in the benchmark format of prepare_bench.py.

  train-reward-model/run.sh prepare_probes.py            # -> data/reward_model/bench/probes_meta.parquet
  train-reward-model/run.sh prepare_bench.py tokenize --model M --benches probes

Every probe pairs a good answer (it won its validation pair with at least 5 of 6 votes, ended normally and has at
least three paragraphs) with a copy that differs only in length. The original should win both kinds:
  padded      the original with its last paragraph repeated once more at the end (longer, no new content)
  truncated   the original cut after the paragraph where it first passes half its length (shorter, unfinished)
A length-biased RM prefers padded copies; one that rewards brevity for its own sake prefers truncated ones. The
paragraphs are the answer's own text, so the probes work in every language.
"""
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prepare_data import OUT  # noqa: E402


def paragraphs(text):
    return [p for p in text.split("\n\n") if p.strip()]


def main():
    t = pq.read_table(os.path.join(OUT, "pairs.parquet"),
                      columns=["id", "split", "language", "messages", "response_a", "response_b", "p_a", "finish_a",
                               "finish_b", "rule"]).to_pylist()
    rows = []
    for r in t:
        if r["split"] != "val" or r["rule"]:
            continue
        for side, win in (("a", r["p_a"] >= 5 / 6), ("b", r["p_a"] <= 1 / 6)):
            text = r["response_" + side]
            ps = paragraphs(text)
            if not win or r["finish_" + side] != "stop" or len(ps) < 3:
                continue
            msgs = json.loads(r["messages"])
            if len(msgs) != 1:
                continue                      # single-turn prompts only (the benchmark format)
            prompt = msgs[0]["content"]
            item = "{}:{}".format(r["id"], side)
            padded = text.rstrip() + "\n\n" + ps[-1]
            half, acc, cut = len(text) / 2, 0, len(ps)
            for k, p in enumerate(ps):
                acc += len(p) + 2
                if acc >= half:
                    cut = k + 1
                    break
            truncated = "\n\n".join(ps[:max(1, min(cut, len(ps) - 1))])
            for group, other in (("padded", padded), ("truncated", truncated)):
                it = "{}:{}".format(item, group)
                rows.append({"id": it + ":c", "item": it, "role": "chosen", "k": 0, "group": group,
                             "language": r["language"], "prompt": prompt, "response": text})
                rows.append({"id": it + ":r", "item": it, "role": "rejected", "k": 0, "group": group,
                             "language": r["language"], "prompt": prompt, "response": other})
    os.makedirs(os.path.join(OUT, "bench"), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(OUT, "bench", "probes_meta.parquet"))
    langs = len({x["language"] for x in rows})
    print("{} probes ({} answers) in {} languages".format(len(rows) // 2, len(rows) // 4, langs))


if __name__ == "__main__":
    main()
