#!/usr/bin/env python3
"""Summarize annotation outputs (works on partial results).

  ./run.sh compile/annotation_stats.py [--examples-lang lv,fr] [-n 3]
Prints validity, label distributions overall and by candidate role, flag rates,
and a few examples for chosen languages.
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow.parquet as pq  # noqa: E402

import annotate_prompt as A  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
D = os.path.join(ROOT, "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples-lang", default="")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--source", help="restrict examples to this HF source id")
    args = ap.parse_args()
    meta = {r["uid"]: r for r in pq.read_table(os.path.join(D, "pool", "candidates.parquet"),
                                               columns=["uid", "lang", "role", "source"]).to_pylist()}
    outs = []
    for p in sorted(glob.glob(os.path.join(D, "llm", "annotate", "out", "*.parquet"))):
        outs += pq.read_table(p).to_pylist()
    fr = collections.Counter(o["finish_reason"] for o in outs)
    parsed = [(o["id"], A.parse(o["text"])) for o in outs if o["finish_reason"] == "stop"]
    ok = [(i, d) for i, d in parsed if d]
    print("outputs {:,}  finish {}  valid {:,} ({:.1%})".format(
        len(outs), dict(fr), len(ok), len(ok) / max(1, len(outs))))
    for f in ("task", "domain", "safety", "locale", "complexity"):
        c = collections.Counter(str(d.get(f)) for _, d in ok)
        tot = sum(c.values())
        print("  {:<11} ".format(f) + ", ".join("{} {:.1%}".format(k, v / tot) for k, v in c.most_common(20)))
    fc = collections.Counter(x for _, d in ok for x in d.get("flags") or [])
    print("  flags       " + ", ".join("{} {:.1%}".format(k, v / len(ok)) for k, v in fc.most_common()))
    cc = collections.Counter(x for _, d in ok for x in d.get("constraints") or [])
    print("  constraints " + ", ".join("{} {:.1%}".format(k, v / len(ok)) for k, v in cc.most_common()))
    by_role = collections.defaultdict(collections.Counter)
    for i, d in ok:
        by_role[meta[i]["role"]]["n"] += 1
        for x in d.get("flags") or []:
            by_role[meta[i]["role"]][x] += 1
    for role, c in by_role.items():
        n = c.pop("n")
        print("  flags[{}] n={:,}: ".format(role, n) + ", ".join(
            "{} {:.1%}".format(k, v / n) for k, v in c.most_common(6)))
    mism = collections.Counter()
    tot = collections.Counter()
    for i, d in ok:
        l = meta[i]["lang"]
        tot[l] += 1
        rl = str(d.get("response_language") or "").lower()
        if rl != l and d.get("task") != "translation":
            mism[l] += 1
    print("  response_language != candidate lang: " + ", ".join(
        "{} {:.0%}".format(l, mism[l] / tot[l]) for l in sorted(tot, key=lambda x: -mism[x] / tot[x])[:12]))
    rnd = random.Random(0)
    for lang in [x for x in args.examples_lang.split(",") if x]:
        items = [(i, d) for i, d in ok if meta[i]["lang"] == lang
                 and (not args.source or meta[i]["source"] == args.source)]
        print("\n-- {} ({} annotated)".format(lang, len(items)))
        for i, d in rnd.sample(items, min(args.n, len(items))):
            print("  [{}] {} | {}/{} safety={} locale={} flags={} lang={}".format(
                meta[i]["source"].split("/")[-1], d.get("gist"), d.get("task"), d.get("domain"),
                d.get("safety"), d.get("locale"), d.get("flags"), d.get("response_language")))


if __name__ == "__main__":
    main()
