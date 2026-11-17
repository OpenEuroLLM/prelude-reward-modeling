#!/usr/bin/env python3
"""All pairs for the full run, one row per prompt in the pilot's format (select_pilot.pair_row): the two
answers in slot order (x = slot 0, y = slot 1) with their models, modes, finish reasons and OpenLID languages.
A pair with one empty answer is judged like any other (the judge sees "(empty response)": an empty answer can
still beat a harmful one); the pairs where both answers are empty are left out, as there is nothing to compare.

  collect-preferences/run.sh build_pairs.py [--out data/preferences/pairs.parquet]   (login node, a few minutes)

The pilot pairs are rows of this file, field for field (checked at the end).
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from select_pilot import pair_row  # noqa: E402  (also puts collect-prompts/compile on the path)


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    import build_pool as B
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "preferences", "pairs.parquet"))
    args = ap.parse_args()
    out = os.path.join(ROOT, args.out)
    prompts = {}
    with open(os.path.join(ROOT, "data", "compiled", "prompts.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            prompts[o["id"]] = o
    resp = pq.read_table(os.path.join(ROOT, "data", "responses", "responses.parquet"),
                         columns=["id", "slot", "model", "model_group", "mode", "response", "reasoning", "finish_reason",
                                  "prompt_tokens", "completion_tokens"]).to_pylist()
    pairs = collections.defaultdict(dict)
    for r in resp:
        pairs[r["id"]][r["slot"]] = r
    del resp
    assert all(len(d) == 2 for d in pairs.values()) and set(pairs) == set(prompts), "every prompt needs both slots"
    ids = sorted(i for i, d in pairs.items() if d[0]["response"].strip() or d[1]["response"].strip())
    one_empty = sum(not pairs[i][0]["response"].strip() or not pairs[i][1]["response"].strip() for i in ids)
    print("{} prompts; {} pairs to judge ({} with one empty answer); {} left out with both answers empty".format(
        len(pairs), len(ids), one_empty, len(pairs) - len(ids)), flush=True)
    B.load_model(os.path.join(ROOT, "data", "models", "openlid-v3", "openlid-v3.bin"), {})
    rows = [pair_row(i, prompts[i], pairs[i][0], pairs[i][1], B) for i in ids]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out + ".tmp")
    os.replace(out + ".tmp", out)
    print("{} pairs -> {}".format(len(rows), out))
    print("languages", dict(sorted(collections.Counter(r["language"] for r in rows).items())))

    pilot = pq.read_table(os.path.join(ROOT, "data", "preferences", "pilot", "pairs.parquet")).to_pylist()
    full = {r["id"]: r for r in rows}
    diff = [(p["id"], k) for p in pilot for k, v in p.items() if full[p["id"]].get(k) != v]
    print("pilot pairs reproduced: {} of {}, {} differing fields{}".format(
        sum(p["id"] in full for p in pilot), len(pilot), len(diff), ": " + str(diff[:5]) if diff else ""))


if __name__ == "__main__":
    main()
