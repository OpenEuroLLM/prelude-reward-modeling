#!/usr/bin/env python3
"""Spot-check an allocation: random slots per language with cell, fill, source, gist.

  python3 compile/show_allocation.py [--dir data/compiled] [--langs de,lv,ltg] [-n 6]
(needs pyarrow: run via ./run.sh)
"""
import argparse
import os
import random

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "compiled"))
    ap.add_argument("--langs", default="en,de,lv,mt,ltg")
    ap.add_argument("-n", type=int, default=6)
    args = ap.parse_args()
    t = pq.read_table(os.path.join(args.dir, "allocation.parquet"),
                      columns=["slot", "lang", "purpose", "origin", "task", "domain", "turns",
                               "fill", "level", "source", "src_lang", "gist"]).to_pylist()
    rnd = random.Random(0)
    for lang in args.langs.split(","):
        rows = [r for r in t if r["lang"] == lang]
        print("== {} ({} slots)".format(lang, len(rows)))
        for r in rnd.sample(rows, min(args.n, len(rows))):
            print("  {:<17} {:<9} {:<18} {:<14} {:<6} L{} {:<17} {:<3}<- {} | {}".format(
                r["purpose"], r["origin"], r["task"], r["domain"], r["turns"], r["level"],
                r["fill"], r["src_lang"], (r["source"] or "").split("/")[-1][:22], r["gist"]))


if __name__ == "__main__":
    main()
