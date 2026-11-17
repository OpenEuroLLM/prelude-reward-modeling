#!/usr/bin/env python3
"""Embed texts with a sentence-transformers model (default bge-m3) on GPU.

Input parquet: id (str), text (str). Output: <out>/shard<i>-<k>.parquet with
id and emb (list<float16>, L2-normalised). Resumable (done ids are skipped).
Run inside the LAIFS container, one process per GPU, e.g. via slurm/embed.sbatch:
  python compile/embed_runner.py --input X.parquet --out DIR --shard i --num-shards 8
"""
import argparse
import glob
import hashlib
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def shard_of(id_, n):
    return int(hashlib.md5(id_.encode("utf-8")).hexdigest(), 16) % n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "data", "models", "bge-m3"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=50_000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    done = set()
    for p in glob.glob(os.path.join(args.out, "shard*.parquet")):
        done.update(pq.read_table(p, columns=["id"]).column("id").to_pylist())
    t = pq.read_table(args.input, columns=["id", "text"]).to_pylist()
    todo = [r for r in t if shard_of(r["id"], args.num_shards) == args.shard
            and r["id"] not in done]
    print("shard {}/{}: {} texts ({} done)".format(args.shard, args.num_shards,
                                                   len(todo), len(done)), flush=True)
    if not todo:
        return
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device="cuda")
    model.max_seq_length = args.max_seq_len
    model.half()
    # sort by length within the shard for efficient batching
    todo.sort(key=lambda r: len(r["text"] or ""))
    t0 = time.time()
    k = len(glob.glob(os.path.join(args.out, "shard{}-*.parquet".format(args.shard))))
    for s in range(0, len(todo), args.chunk):
        chunk = todo[s:s + args.chunk]
        emb = model.encode([r["text"] or "" for r in chunk], batch_size=args.batch,
                           normalize_embeddings=True, convert_to_numpy=True,
                           show_progress_bar=False).astype(np.float16)
        tbl = pa.table({"id": [r["id"] for r in chunk],
                        "emb": pa.FixedSizeListArray.from_arrays(
                            pa.array(emb.reshape(-1), type=pa.float16()), emb.shape[1])})
        path = os.path.join(args.out, "shard{}-{:05d}.parquet".format(args.shard, k))
        pq.write_table(tbl, path + ".tmp")
        os.replace(path + ".tmp", path)
        k += 1
        n = s + len(chunk)
        print("shard {}: {}/{} ({:.0f}/s)".format(args.shard, n, len(todo),
                                                  n / (time.time() - t0)), flush=True)


if __name__ == "__main__":
    main()
