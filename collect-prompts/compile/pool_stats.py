#!/usr/bin/env python3
"""Summarize the pool index (streaming, low memory).

  ./run.sh compile/pool_stats.py [--out data/pool/pool_stats.json]

Prints, per target language: eligible records by origin, purpose and
translated-by-source, plus eligibility drop reasons overall. "Eligible" =
not excluded, train split, licence allowed, target language, length bounds.
Exact duplicates are NOT removed here (that happens in candidate selection).
"""
import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow.parquet as pq  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
INDEX = os.path.join(ROOT, "data", "pool", "index")
COLS = ["source", "split", "origin", "translated", "purpose", "license_flag",
        "exclude", "lang", "last_user_chars", "user_chars", "n_resp", "pref_type",
        "prompt_harm"]
LICENSE_EXCLUDED = {"no_redistribution"}


def targets():
    out = []
    with open(os.path.join(HERE, "quotas.tsv"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("code\t") or not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            out.append((p[0], int(p[5])))
    return out


def drop_reason(r, tset):
    if r["exclude"]:
        return "excluded:" + r["exclude"]
    if r["split"] != "train":
        return "non_train_split"
    if r["license_flag"] in LICENSE_EXCLUDED:
        return "licence:" + r["license_flag"]
    if r["lang"] not in tset:
        return "non_target_lang"
    if not (10 <= r["last_user_chars"] <= 8000) or r["user_chars"] > 24000:
        return "length"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "pool", "pool_stats.json"))
    args = ap.parse_args()
    tq = targets()
    tset = {c for c, _ in tq}
    drops = collections.Counter()
    by = collections.defaultdict(collections.Counter)   # lang -> key -> n
    by_src = collections.defaultdict(collections.Counter)  # lang -> source -> n (native)
    files = sorted(glob.glob(os.path.join(INDEX, "*", "part-*.parquet")))
    total = 0
    for p in files:
        for r in pq.read_table(p, columns=COLS).to_pylist():
            total += 1
            why = drop_reason(r, tset)
            if why:
                drops[why] += 1
                continue
            l = r["lang"]
            tr = "translated" if r["translated"] else "native"
            by[l]["all"] += 1
            by[l][tr] += 1
            by[l][tr + ":" + r["origin"]] += 1
            by[l][tr + ":purpose:" + r["purpose"]] += 1
            if r["purpose"] == "safety":
                by[l][tr + ":harm:" + str(r["prompt_harm"])] += 1
            if r["n_resp"] > 0:
                by[l][tr + ":with_resp"] += 1
            if r["pref_type"]:
                by[l][tr + ":with_pref"] += 1
            if not r["translated"]:
                by_src[l][r["source"]] += 1
    print("index files {}, records {:,}".format(len(files), total))
    print("dropped:", dict(drops.most_common()))
    hdr = ("lang", "quota", "native", "real", "human", "synth", "web", "tmpl",
           "n_if", "n_safe", "transl", "t_if", "t_safe", "n_resp", "n_pref")
    print("\n" + " ".join("{:>8}".format(h) for h in hdr))
    for code, q in tq:
        c = by[code]
        vals = (code, q, c["native"], c["native:real_user"], c["native:human"],
                c["native:synthetic"], c["native:web"], c["native:template"],
                c["native:purpose:if"], c["native:purpose:safety"], c["translated"],
                c["translated:purpose:if"], c["translated:purpose:safety"],
                c["native:with_resp"], c["native:with_pref"])
        print(" ".join("{:>8}".format(v) for v in vals))
    with open(args.out, "w") as f:
        json.dump({"total": total, "dropped": drops, "by_lang": by,
                   "native_by_source": by_src}, f, indent=1, default=dict)
    print("->", args.out)


if __name__ == "__main__":
    main()
