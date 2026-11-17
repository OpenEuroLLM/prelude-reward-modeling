#!/usr/bin/env python3
"""Print a few extracted records (truncated) and the meta keys of a source.

  ./run.sh peek.py <org>__<name> [-n 3] [--subset S] [--keys-only]
"""
import argparse
import collections
import glob
import json
import os

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def trunc(v, n):
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + "...[{}]".format(len(v))
    if isinstance(v, list):
        return [trunc(x, n) for x in v[:4]] + (["...[{}]".format(len(v))] if len(v) > 4 else [])
    if isinstance(v, dict):
        return {k: trunc(x, n) for k, x in v.items()}
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("-n", type=int, default=2)
    ap.add_argument("--subset")
    ap.add_argument("--chars", type=int, default=200)
    ap.add_argument("--keys-only", action="store_true")
    ap.add_argument("--scan", type=int, default=5000, help="rows scanned for meta keys")
    args = ap.parse_args()

    parts = sorted(glob.glob(os.path.join(ROOT, "data", "extracted", args.source, "part-*.parquet")))
    keys, rkeys, shown, scanned = collections.Counter(), collections.Counter(), 0, 0
    for p in parts:
        for batch in pq.ParquetFile(p).iter_batches(batch_size=1000):
            for r in batch.to_pylist():
                if args.subset and r["subset"] != args.subset:
                    continue
                scanned += 1
                m = json.loads(r["meta"]) if r["meta"] else {}
                keys.update(m.keys())
                for resp in r["responses"]:
                    if resp["meta"]:
                        rkeys.update(json.loads(resp["meta"]).keys())
                if shown < args.n and not args.keys_only:
                    r["meta"], r["preference"] = m, json.loads(r["preference"]) if r["preference"] else None
                    print(json.dumps(trunc(r, args.chars), ensure_ascii=False, default=str))
                    shown += 1
                if scanned >= args.scan:
                    break
            if scanned >= args.scan:
                break
        if scanned >= args.scan:
            break
    print("meta keys (in {} rows): {}".format(scanned, dict(keys.most_common())))
    print("response meta keys: {}".format(dict(rkeys.most_common())))


if __name__ == "__main__":
    main()
