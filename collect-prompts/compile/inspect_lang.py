#!/usr/bin/env python3
"""Show sampled pool records for a language (optionally one source) with LID info.

  ./run.sh compile/inspect_lang.py nn [--source allenai/WildChat-4.8M] [-n 8]
"""
import argparse
import os
import random

import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX = os.path.join(ROOT, "data", "pool", "index")
EXT = os.path.join(ROOT, "data", "extracted")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lang")
    ap.add_argument("--source")
    ap.add_argument("-n", type=int, default=8)
    ap.add_argument("--chars", type=int, default=220)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    f = (ds.field("lang") == args.lang) & (ds.field("split") == "train")
    if args.source:
        f = f & (ds.field("source") == args.source)
    t = ds.dataset(INDEX, format="parquet").to_table(
        columns=["uid", "source", "part", "row", "lang_rule", "lid_last", "lid_last_p",
                 "lid_last2", "lid_last2_p", "lid_all", "lid_all_p", "lid_chars_last",
                 "lang_src", "resp_lang", "resp_lang_p"], filter=f).to_pylist()
    print("{} records".format(len(t)))
    random.Random(args.seed).shuffle(t)
    for r in t[:args.n]:
        pf = pq.ParquetFile(os.path.join(EXT, r["part"]))
        # locate row group
        acc = 0
        for i in range(pf.num_row_groups):
            nr = pf.metadata.row_group(i).num_rows
            if r["row"] < acc + nr:
                rec = pf.read_row_group(i, columns=["context"]).column("context")[r["row"] - acc].as_py()
                break
            acc += nr
        last = [m["content"] for m in rec if m["role"] == "user"][-1]
        print("--- {} | rule={} last={} {:.2f} 2nd={} {:.2f} all={} {:.2f} chars={} src_label={} resp={} {:.2f}".format(
            r["source"].split("/")[-1], r["lang_rule"], r["lid_last"], r["lid_last_p"],
            r["lid_last2"], r["lid_last2_p"], r["lid_all"], r["lid_all_p"],
            r["lid_chars_last"], r["lang_src"], r["resp_lang"], r["resp_lang_p"] or 0))
        print("    " + last[:args.chars].replace("\n", " | "))


if __name__ == "__main__":
    main()
