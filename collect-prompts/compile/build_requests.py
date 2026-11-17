#!/usr/bin/env python3
"""Stage 2b: fetch candidate contexts and build annotation / embedding inputs.

Reads data/pool/candidates.parquet, fetches each candidate's context from
data/extracted/<part> (only the needed row groups), and writes
  data/pool/candidates_ctx.parquet     uid, role, lang, context (JSON)
  data/llm/annotate/requests.parquet   id=uid, messages, schema, max_tokens
  data/llm/embed/input.parquet         id=uid, text (for bge-m3)

  ./run.sh compile/build_requests.py [--workers 8]   (or via slurm/cpu_job.sbatch)
"""
import argparse
import bisect
import collections
import json
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import annotate_prompt as A  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
EXT = os.path.join(ROOT, "data", "extracted")
POOL = os.path.join(ROOT, "data", "pool")
LLM = os.path.join(ROOT, "data", "llm")
SCHEMA_JSON = json.dumps(A.SCHEMA)


def embed_text(context):
    users = [m["content"] or "" for m in context if m["role"] == "user"]
    last = users[-1][:1500]
    prev = [u[:300] for u in users[-3:-1]]
    return "\n".join(prev + [last])


def fetch_part(task):
    part, wanted = task  # wanted: list of (row, uid, role, lang)
    pf = pq.ParquetFile(os.path.join(EXT, part))
    starts, acc = [], 0
    for i in range(pf.num_row_groups):
        starts.append(acc)
        acc += pf.metadata.row_group(i).num_rows
    by_rg = collections.defaultdict(list)
    for w in wanted:
        rg = bisect.bisect_right(starts, w[0]) - 1
        by_rg[rg].append(w)
    out = []
    for rg, ws in by_rg.items():
        t = pf.read_row_group(rg, columns=["uid", "context"])
        uids = t.column("uid").to_pylist()
        ctxs = t.column("context").to_pylist()
        for row, uid, role, lang in ws:
            j = row - starts[rg]
            if uids[j] != uid:
                raise RuntimeError("pointer mismatch {} row {}: {} != {}".format(part, row, uids[j], uid))
            out.append((uid, role, lang, ctxs[j]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    cand = pq.read_table(os.path.join(POOL, "candidates.parquet"),
                         columns=["uid", "part", "row", "role", "lang"]).to_pylist()
    tasks = collections.defaultdict(list)
    for c in cand:
        tasks[c["part"]].append((c["row"], c["uid"], c["role"], c["lang"]))
    print("{} candidates in {} part files".format(len(cand), len(tasks)), flush=True)

    os.makedirs(os.path.join(LLM, "annotate"), exist_ok=True)
    os.makedirs(os.path.join(LLM, "embed"), exist_ok=True)
    w_ctx = pq.ParquetWriter(os.path.join(POOL, "candidates_ctx.parquet"), pa.schema(
        [("uid", pa.string()), ("role", pa.string()), ("lang", pa.string()),
         ("context", pa.string())]), compression="zstd")
    w_req = pq.ParquetWriter(os.path.join(LLM, "annotate", "requests.parquet"), pa.schema(
        [("id", pa.string()), ("messages", pa.string()), ("schema", pa.string()),
         ("max_tokens", pa.int32())]), compression="zstd")
    w_emb = pq.ParquetWriter(os.path.join(LLM, "embed", "input.parquet"), pa.schema(
        [("id", pa.string()), ("text", pa.string())]), compression="zstd")
    n = 0
    with mp.get_context("fork").Pool(args.workers) as pool:
        for rows in pool.imap_unordered(fetch_part, sorted(tasks.items())):
            ctx_rows, req_rows, emb_rows = [], [], []
            for uid, role, lang, ctx in rows:
                ctx_rows.append({"uid": uid, "role": role, "lang": lang,
                                 "context": json.dumps(ctx, ensure_ascii=False)})
                req_rows.append({"id": uid,
                                 "messages": json.dumps(A.messages(ctx), ensure_ascii=False),
                                 "schema": SCHEMA_JSON, "max_tokens": 300})
                emb_rows.append({"id": uid, "text": embed_text(ctx)})
            w_ctx.write_table(pa.Table.from_pylist(ctx_rows, schema=w_ctx.schema))
            w_req.write_table(pa.Table.from_pylist(req_rows, schema=w_req.schema))
            w_emb.write_table(pa.Table.from_pylist(emb_rows, schema=w_emb.schema))
            n += len(rows)
            print("  fetched {:,}".format(n), flush=True)
    for w in (w_ctx, w_req, w_emb):
        w.close()
    print("done:", n)


if __name__ == "__main__":
    main()
