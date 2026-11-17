#!/usr/bin/env python3
"""Stage 6e: drop EuroMultiSynth native rows in the languages where it is broken, for refilling.

  ./run.sh compile/ems_drop.py                 # counts per language
  ./run.sh compile/ems_drop.py --apply         # add them to data/compiled/force_fail.parquet

openeurollm/EuroMultiSynth-Instruct is Gemma-generated; the severity-aware blind review
(manual_check/quality_review2/FINDINGS.md) found several non-native errors per prompt in
8-13 of 13 items in et, eu, ga, lv, lt, is, mt, sl, and only isolated words elsewhere (kept,
per the user's bar). Refill with
  EXTRA_ALLOC="--refill-native --refill-exclude-sources openeurollm/EuroMultiSynth-Instruct".
force_fail entries are keyed by (slot, uid), so a refilled slot is not failed again.
"""
import argparse
import collections
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import mt_common as M  # noqa: E402

SOURCE = "openeurollm/EuroMultiSynth-Instruct"
LANGS = "et eu ga lv lt is mt sl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=LANGS)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    langs = set(args.langs.split())
    rows = [r for r in pq.read_table(os.path.join(M.DATA, "compiled", "prompts.parquet"),
                                     columns=["id", "lang", "fill", "source", "source_uid"]).to_pylist()
            if r["source"] == SOURCE and r["fill"] == "native" and r["lang"] in langs]
    print("EuroMultiSynth native rows to drop:", len(rows), dict(collections.Counter(r["lang"] for r in rows).most_common()))
    if args.apply:
        p = os.path.join(M.DATA, "compiled", "force_fail.parquet")
        old = pq.read_table(p).to_pylist() if os.path.exists(p) else []
        have = {(r["slot"], r["uid"]) for r in old}
        new = [{"slot": int(r["id"].split("-")[1]), "id": r["id"], "uid": r["source_uid"],
                "reason": "euromultisynth_low_quality"} for r in rows
               if (int(r["id"].split("-")[1]), r["source_uid"]) not in have]
        pq.write_table(pa.Table.from_pylist(old + new), p)
        print("force_fail: {} existing + {} new".format(len(old), len(new)))


if __name__ == "__main__":
    main()
