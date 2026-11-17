#!/usr/bin/env python3
"""Print file layout, schema and one truncated example per data file group
for each downloaded dataset, to guide writing extractors.

  ./run.sh inspect_schema.py [--raw ../data/raw] [--only org__name ...]
"""
import argparse
import collections
import gzip
import json
import os
import re

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW = os.path.join(os.path.dirname(HERE), "data", "raw")
DATA_RE = re.compile(r"\.(parquet|jsonl|json|jsonl\.gz|json\.gz|csv|tsv)$")


def trunc(v, n=160):
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + "...[{} chars]".format(len(v))
    if isinstance(v, list):
        out = [trunc(x, n) for x in v[:3]]
        if len(v) > 3:
            out.append("...[{} items]".format(len(v)))
        return out
    if isinstance(v, dict):
        return {k: trunc(x, n) for k, x in v.items()}
    return v


def group_key(path):
    # collapse shard numbers so each logical file group is shown once
    return re.sub(r"\d{3,}", "#", path)


def first_json_record(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        if ".jsonl" in path:
            return json.loads(f.readline()), None
        data = json.load(f)
        if isinstance(data, list):
            return data[0], len(data)
        return data, None


def inspect(ds_dir):
    files = []
    for root, _, names in os.walk(ds_dir):
        if "/.cache" in root:
            continue
        for n in names:
            p = os.path.join(root, n)
            if DATA_RE.search(n) and n != "_download_info.json":
                files.append(os.path.relpath(p, ds_dir))
    groups = collections.OrderedDict()
    for f in sorted(files):
        groups.setdefault(group_key(f), []).append(f)
    print("=" * 100)
    print(os.path.basename(ds_dir), "|", len(files), "data files")
    for key, members in groups.items():
        path = os.path.join(ds_dir, members[0])
        print("-- {} ({} files)".format(key, len(members)))
        try:
            if path.endswith(".parquet"):
                rows = sum(pq.ParquetFile(os.path.join(ds_dir, m)).metadata.num_rows
                           for m in members)
                pf = pq.ParquetFile(path)
                print("   rows:", rows)
                print("   schema:", pf.schema_arrow.to_string(
                    show_field_metadata=False, show_schema_metadata=False
                ).replace("\n", "\n           "))
                batch = next(pf.iter_batches(batch_size=1))
                print("   example:", json.dumps(trunc(batch.to_pylist()[0]),
                                             ensure_ascii=False, default=str)[:3000])
            elif path.endswith((".csv", ".tsv")):
                with open(path, encoding="utf-8") as f:
                    print("   head:", f.readline()[:500].rstrip())
                    print("         ", f.readline()[:500].rstrip())
            else:
                rec, n = first_json_record(path)
                if n is not None:
                    print("   records (json list):", n)
                print("   example:", json.dumps(trunc(rec), ensure_ascii=False,
                                             default=str)[:3000])
        except Exception as e:
            print("   ERROR:", repr(e)[:300])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    for name in sorted(os.listdir(args.raw)):
        d = os.path.join(args.raw, name)
        if not os.path.isdir(d) or (args.only and name not in args.only):
            continue
        inspect(d)


if __name__ == "__main__":
    main()
