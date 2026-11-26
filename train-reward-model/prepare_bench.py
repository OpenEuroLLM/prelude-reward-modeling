#!/usr/bin/env python3
"""Public reward-model benchmarks in the scoring format of train_rm.py --score.

  train-reward-model/run.sh prepare_bench.py download        # -> data/reward_model/bench/<bench>_meta.parquet
  train-reward-model/run.sh prepare_bench.py tokenize --model qwen3.8-27b
      -> data/reward_model/bench/<model>/<bench>.parquet (id, prompt_ids, resp_ids)

Benchmarks (all single-turn; each answer becomes one sequence, rendered exactly like the training data, see
prepare_data.py):
  rb2       RewardBench 2 (allenai/reward-bench-2, 1,865 prompts): 1 chosen vs 3 rejected (Ties: several correct).
  rmbench   RM-Bench (THU-KEG/RM-Bench, 1,327 prompts): 3 chosen x 3 rejected in three styles
            (concise, detailed, detailed with markdown).
  mrb       M-RewardBench (CohereLabsCommunity/multilingual-reward-bench), the 11 European languages of our
            set (cs de el es fr it nl pl pt ro uk), 2,869 pairs each; mrb_small: 300 pairs per language
            (stratified by category), for quick comparisons.
Our prompts were decontaminated against all three (collect-prompts/).
"""
import argparse
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from prepare_data import OUT, load_tokenizer, render  # noqa: E402

BENCH = os.path.join(OUT, "bench")
MRB_LANGS = ["ces_Latn", "deu_Latn", "ell_Grek", "fra_Latn", "ita_Latn", "nld_Latn", "pol_Latn", "por_Latn",
             "ron_Latn", "spa_Latn", "ukr_Cyrl"]


def download(args):
    from datasets import load_dataset
    os.makedirs(BENCH, exist_ok=True)
    rows = []
    ds = load_dataset("allenai/reward-bench-2", split="test")
    for r in ds:
        for k, c in enumerate(r["chosen"]):
            rows.append({"id": "{}:c{}".format(r["id"], k), "item": r["id"], "role": "chosen", "k": k,
                         "group": r["subset"], "language": "en", "prompt": r["prompt"], "response": c})
        for k, c in enumerate(r["rejected"]):
            rows.append({"id": "{}:r{}".format(r["id"], k), "item": r["id"], "role": "rejected", "k": k,
                         "group": r["subset"], "language": "en", "prompt": r["prompt"], "response": c})
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(BENCH, "rb2_meta.parquet"))
    print("rb2", len(ds), "prompts", len(rows), "answers")
    rows = []
    ds = load_dataset("THU-KEG/RM-Bench", split="train")
    for r in ds:
        for role in ("chosen", "rejected"):
            for k, c in enumerate(r[role]):
                rows.append({"id": "{}:{}{}".format(r["id"], role[0], k), "item": r["id"], "role": role, "k": k,
                             "group": r["domain"], "language": "en", "prompt": r["prompt"], "response": c})
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(BENCH, "rmbench_meta.parquet"))
    print("rmbench", len(ds), "prompts", len(rows), "answers")
    rows, small = [], []
    rng = np.random.default_rng(20261120)
    for cfg in MRB_LANGS:
        ds = load_dataset("CohereLabsCommunity/multilingual-reward-bench", cfg, split="test")
        lang = ds[0]["language"]
        cats = np.array(ds["category"])
        pick = set()
        for c in sorted(set(cats)):                  # 300 per language, proportional to the categories
            idx = np.flatnonzero(cats == c)
            n = int(round(300 * len(idx) / len(cats)))
            pick.update(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())
        for j, r in enumerate(ds):
            item = "{}:{}".format(lang, r["id"])
            for role in ("chosen", "rejected"):
                row = {"id": "{}:{}".format(item, role[0]), "item": item, "role": role, "k": 0,
                       "group": r["category"], "language": lang, "prompt": r["prompt"], "response": r[role],
                       "source": r["source"]}
                rows.append(row)
                if j in pick:
                    small.append(row)
        print("mrb", cfg, len(ds), "pairs,", len(pick), "in mrb_small")
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(BENCH, "mrb_meta.parquet"))
    pq.write_table(pa.Table.from_pylist(small), os.path.join(BENCH, "mrb_small_meta.parquet"))


def tokenize(args):
    tok = load_tokenizer(args.model)
    d = os.path.join(BENCH, args.model)
    os.makedirs(d, exist_ok=True)
    for b in args.benches.split(","):
        t = pq.read_table(os.path.join(BENCH, b + "_meta.parquet"), columns=["id", "prompt", "response"]).to_pydict()
        prefixes = [render(tok, [{"role": "user", "content": p}])[0] for p in t["prompt"]]
        ids_p, ids_r = [], []
        for i in range(0, len(prefixes), 2048):
            ids_p += tok(prefixes[i:i + 2048], add_special_tokens=False)["input_ids"]
            ids_r += tok(t["response"][i:i + 2048], add_special_tokens=False, split_special_tokens=True)["input_ids"]
        i32 = pa.list_(pa.int32())
        pq.write_table(pa.table({"id": t["id"], "prompt_ids": pa.array(ids_p, type=i32),
                                 "resp_ids": pa.array(ids_r, type=i32)}), os.path.join(d, b + ".parquet"))
        n = np.array([len(a) + len(b) for a, b in zip(ids_p, ids_r)])
        print(args.model, b, len(n), "sequences, mean {:.0f} tokens, max {}".format(n.mean(), n.max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["download", "tokenize"])
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--benches", default="rb2,rmbench,mrb,mrb_small")
    args = ap.parse_args()
    {"download": download, "tokenize": tokenize}[args.step](args)


if __name__ == "__main__":
    main()
