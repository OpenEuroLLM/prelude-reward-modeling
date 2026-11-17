#!/usr/bin/env python3
"""GPU similarity search over (L2-normalised) bge-m3 embeddings.

For each candidate (query rows of this shard):
  nn_ids / nn_sims   top-k most similar *other* candidates (any language)
  ho_sim / ho_id     highest cosine to any hold-out benchmark prompt

  python compile/similarity.py --shard i --num-shards 8     (one process per GPU)
Inputs:  data/llm/embed/out/*.parquet, data/llm/embed_holdout/out/*.parquet
Output:  data/pool/neighbors/shard<i>.parquet
"""
import argparse
import glob
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load(pattern):
    ids, embs = [], []
    for p in sorted(glob.glob(pattern)):
        t = pq.read_table(p)
        ids += t.column("id").to_pylist()
        col = t.column("emb").combine_chunks()
        dim = col.type.list_size
        embs.append(col.flatten().to_numpy(zero_copy_only=False).reshape(-1, dim))
    return ids, np.concatenate(embs).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--cand", default=os.path.join(ROOT, "data", "llm", "embed", "out", "*.parquet"))
    ap.add_argument("--holdout", default=os.path.join(ROOT, "data", "llm", "embed_holdout", "out", "*.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "pool", "neighbors"))
    ap.add_argument("--kmeans", type=int, default=8192, help="clusters (shard 0 only; 0 = off)")
    ap.add_argument("--kmeans-iters", type=int, default=25)
    args = ap.parse_args()

    t0 = time.time()
    cid, cemb = load(args.cand)
    hid, hemb = load(args.holdout)
    print("candidates {} x {}, holdout {} ({:.0f}s)".format(len(cid), cemb.shape[1], len(hid),
                                                           time.time() - t0), flush=True)
    dev = torch.device("cuda")
    C = torch.from_numpy(cemb).to(dev)
    H = torch.from_numpy(hemb).to(dev)
    n = len(cid)
    rows = np.arange(n)[np.arange(n) % args.num_shards == args.shard]
    out = {"id": [], "nn_ids": [], "nn_sims": [], "ho_sim": [], "ho_id": []}
    for s in range(0, len(rows), args.chunk):
        r = rows[s:s + args.chunk]
        Q = C[torch.from_numpy(r).to(dev)]
        sims = Q @ C.T
        sims[torch.arange(len(r), device=dev), torch.from_numpy(r).to(dev)] = -2.0  # exclude self
        v, i = torch.topk(sims.float(), args.k, dim=1)
        hs = Q @ H.T
        hv, hi = hs.float().max(dim=1)
        v, i, hv, hi = v.cpu().numpy(), i.cpu().numpy(), hv.cpu().numpy(), hi.cpu().numpy()
        for j, q in enumerate(r):
            out["id"].append(cid[q])
            out["nn_ids"].append([cid[x] for x in i[j]])
            out["nn_sims"].append([round(float(x), 4) for x in v[j]])
            out["ho_sim"].append(round(float(hv[j]), 4))
            out["ho_id"].append(hid[hi[j]])
    os.makedirs(args.out, exist_ok=True)
    pq.write_table(pa.table(out), os.path.join(args.out, "shard{}.parquet".format(args.shard)))
    print("shard {} done: {} rows, {:.0f}s".format(args.shard, len(rows), time.time() - t0), flush=True)

    if args.shard == 0 and args.kmeans > 0:
        del sims, hs
        torch.cuda.empty_cache()
        g = torch.Generator(device="cpu").manual_seed(0)
        cent = C[torch.randperm(n, generator=g)[:args.kmeans].to(dev)].float()
        for it in range(args.kmeans_iters):
            assign = torch.empty(n, dtype=torch.long, device=dev)
            for s in range(0, n, 65536):
                assign[s:s + 65536] = (C[s:s + 65536].float() @ cent.T).argmax(dim=1)
            new = torch.zeros_like(cent).index_add_(0, assign, C.float())
            cnt = torch.bincount(assign, minlength=args.kmeans).float().unsqueeze(1)
            empty = cnt.squeeze(1) == 0
            new = new / cnt.clamp(min=1)
            new[empty] = cent[empty]  # keep empty clusters' old centroids
            cent = torch.nn.functional.normalize(new, dim=1)  # spherical k-means
        a = assign.cpu().numpy()
        pq.write_table(pa.table({"id": cid, "cluster": a.astype(np.int32)}),
                       os.path.join(os.path.dirname(args.out), "clusters.parquet"))
        sizes = np.bincount(a, minlength=args.kmeans)
        print("kmeans k={}: non-empty {}, median size {}, max {} ({:.0f}s)".format(
            args.kmeans, int((sizes > 0).sum()), int(np.median(sizes)), int(sizes.max()),
            time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
