#!/usr/bin/env python3
"""Collect all user-prompt texts from the downloaded hold-out benchmarks.

  ./run.sh compile/holdout_prompts.py
Reads data/holdout/raw/<org>__<name>/** (json, jsonl, parquet), writes
data/holdout/prompts.parquet: id, bench, file, text, key (exact-match key as
in the pool index), plus data/llm/embed_holdout/input.parquet (id, text).

Generic: takes string values under prompt-like keys, user messages of
message lists (also when JSON-encoded), and per-language columns (M-ALERT);
never answer/response/judgment fields.
"""
import collections
import csv
import glob
import gzip
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import build_pool as B  # noqa: E402  (text_key)

ROOT = os.path.dirname(os.path.dirname(HERE))
RAW = os.path.join(ROOT, "data", "holdout", "raw")
OUT = os.path.join(ROOT, "data", "holdout")

PROMPT_KEY = re.compile(r"prompt|instruction|question|turn|input|goal|behavior|query|conversation", re.I)
BAD_KEY = re.compile(r"response|answer|chosen|rejected|completion|output|judg|label|score|"
                     r"model|category|_id$|^id$|reference|explanation|reason|kwargs|"
                     r"instruction_id|source", re.I)
LANG_KEYS = {"en", "de", "es", "fr", "it", "pt", "nl", "pl", "cs", "sv", "da", "fi",
             "el", "hu", "ro", "bg", "sk", "sl", "lt", "lv", "et", "hr", "ga", "mt"}


def messages_user(v):
    out = []
    for m in v:
        if isinstance(m, dict) and str(m.get("role", "")).lower() in ("user", "human") \
                and isinstance(m.get("content"), str):
            out.append(m["content"])
    return out


def texts_from(rec):
    out = []
    for k, v in rec.items():
        if k in LANG_KEYS and isinstance(v, str):
            out.append(v)
            continue
        if not PROMPT_KEY.search(k) or BAD_KEY.search(k):
            continue
        if isinstance(v, str):
            s = v.strip()
            if s[:1] in "[{" and '"role"' in s:
                try:
                    obj = json.loads(s)
                    out += messages_user(obj if isinstance(obj, list) else [obj])
                    continue
                except Exception:
                    pass
            out.append(v)
        elif isinstance(v, list):
            if v and all(isinstance(x, str) for x in v):
                out += v
            else:
                out += messages_user(v)
                for x in v:  # e.g. arena-hard turns: [{"content": ...}]
                    if isinstance(x, dict) and "role" not in x and isinstance(x.get("content"), str):
                        out.append(x["content"])
    return [t for t in out if isinstance(t, str) and len(t.strip()) >= 15]


def records(path):
    if path.endswith(".parquet"):
        for b in pq.ParquetFile(path).iter_batches(batch_size=2000):
            yield from b.to_pylist()
        return
    if path.endswith(".csv"):
        csv.field_size_limit(sys.maxsize)
        with open(path, encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
        return
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        if ".jsonl" in path:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except Exception:
                        pass
        else:
            try:
                d = json.load(f)
            except Exception:
                return
            if isinstance(d, list):
                yield from (x for x in d if isinstance(x, dict))
            elif isinstance(d, dict):
                vals = [x for x in d.values() if isinstance(x, list)]
                if vals:
                    for lst in vals:
                        yield from (x for x in lst if isinstance(x, dict))
                else:
                    yield d


def main():
    rows, seen = [], set()
    per = collections.Counter()
    for bench_dir in sorted(glob.glob(os.path.join(RAW, "*"))):
        bench = os.path.basename(bench_dir)
        for path in sorted(glob.glob(os.path.join(bench_dir, "**", "*"), recursive=True)):
            if "/.cache/" in path or not re.search(r"\.(parquet|jsonl|json|csv)(\.gz)?$", path) \
                    or path.endswith("_download_info.json"):
                continue
            for rec in records(path):
                if not isinstance(rec, dict):
                    continue
                for t in texts_from(rec):
                    key = B.text_key([t])
                    if (bench, key) in seen:
                        continue
                    seen.add((bench, key))
                    rid = "ho::" + hashlib.md5((bench + key).encode()).hexdigest()[:16]
                    rows.append({"id": rid, "bench": bench,
                                 "file": os.path.relpath(path, bench_dir),
                                 "text": t, "key": key})
                    per[bench] += 1
    os.makedirs(OUT, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(OUT, "prompts.parquet"))
    emb_dir = os.path.join(ROOT, "data", "llm", "embed_holdout")
    os.makedirs(emb_dir, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([{"id": r["id"], "text": r["text"][:2000]} for r in rows]),
                   os.path.join(emb_dir, "input.parquet"))
    for b, n in sorted(per.items()):
        print("{:<52} {:>7}".format(b, n))
    print("total unique (bench, prompt):", len(rows))


if __name__ == "__main__":
    main()
