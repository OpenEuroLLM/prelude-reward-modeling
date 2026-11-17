#!/usr/bin/env python3
"""Draw a stratified sample of the compiled set for manual review.

  ./run.sh compile/manual_sample.py [--per translated=3,native=2,source_translated=1]
                                    [--all-of ltg] [--seed 20261109]

Per language and fill: K random rows (all rows for the --all-of languages).
Translated rows are shown next to their source conversation; native and
source-translated rows with their first existing response. Messages are
truncated (final user turn longer than the rest).

Outputs data/compiled/manual_check/sample/<lang>.txt (readable) and
data/compiled/manual_check/sample.jsonl (the sampled ids, for the verdict log).
"""
import argparse
import collections
import json
import os
import random
import re

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
COLS = ["id", "lang", "fill", "context", "responses", "source", "subset", "source_uid",
        "source_lang", "purpose", "task", "turns", "safety", "instruction_ids", "if_kwargs"]


def cut(s, n):
    s = re.sub(r"\n{3,}", "\n\n", (s or "").replace("\r", "")).strip()
    return s if len(s) <= n else s[:n] + " …[+{} chars]".format(len(s) - n)


def show(ctx, big, small):
    last = max(i for i, m in enumerate(ctx) if m["role"] == "user")
    return "\n".join("[{}] {}".format(m["role"], cut(m["content"], big if i == last else small))
                     for i, m in enumerate(ctx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", default="translated=3,native=2,source_translated=1")
    ap.add_argument("--all-of", default="ltg")
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--out", default=os.path.join(DATA, "compiled", "manual_check"))
    args = ap.parse_args()
    per = {k: int(v) for k, v in (x.split("=") for x in args.per.split(","))}
    full = set(args.all_of.split(",")) if args.all_of else set()

    rows = pq.read_table(os.path.join(DATA, "compiled", "prompts.parquet"), columns=COLS).to_pylist()
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["lang"], r["fill"])].append(r)
    rng = random.Random(args.seed)
    sample = []
    for key in sorted(by):
        rs = sorted(by[key], key=lambda r: r["id"])
        k = len(rs) if key[0] in full else per.get(key[1], 0)
        sample += sorted(rng.sample(rs, min(k, len(rs))), key=lambda r: r["id"])
    need = {r["source_uid"] for r in sample if r["fill"] == "translated"}
    src = {}
    for b in pq.ParquetFile(os.path.join(DATA, "pool", "candidates_ctx.parquet")).iter_batches(
            batch_size=20000, columns=["uid", "context"]):
        for u, c in zip(b.column("uid").to_pylist(), b.column("context").to_pylist()):
            if u in need:
                src[u] = json.loads(c)

    os.makedirs(os.path.join(args.out, "sample"), exist_ok=True)
    texts = collections.defaultdict(list)
    for r in sample:
        big, small = (2000, 600) if r["lang"] in full else (1000, 280)
        t = ["=== {id} | {lang} | {fill} | {source} [{subset}] | src_lang={source_lang} | "
             "{purpose}/{task}/{turns}/{safety}".format(**r)]
        if r["instruction_ids"]:
            t.append("IF: {} {}".format(list(r["instruction_ids"]), cut(r["if_kwargs"], 300)))
        t.append(show(r["context"], big, small))
        if r["fill"] == "translated":
            t.append("--- SOURCE ({}):".format(r["source_lang"]))
            t.append(show(src[r["source_uid"]], big, small))
        elif r["responses"]:
            x = r["responses"][0]
            t.append("--- {} responses; [0] {}: {}".format(len(r["responses"]), x["model"],
                                                           cut(x["content"], 220)))
        texts[r["lang"]].append("\n".join(t))
    for lang, items in texts.items():
        with open(os.path.join(args.out, "sample", lang + ".txt"), "w") as f:
            f.write("\n\n".join(items) + "\n")
    with open(os.path.join(args.out, "sample.jsonl"), "w") as f:
        for r in sample:
            f.write(json.dumps({"id": r["id"], "lang": r["lang"], "fill": r["fill"],
                                "source": r["source"]}, ensure_ascii=False) + "\n")
    print("sampled", len(sample), "rows:", dict(collections.Counter(r["fill"] for r in sample)))
    print({lang: len(v) for lang, v in sorted(texts.items())})


if __name__ == "__main__":
    main()
