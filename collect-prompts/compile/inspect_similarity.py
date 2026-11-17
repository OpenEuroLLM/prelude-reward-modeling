#!/usr/bin/env python3
"""Calibrate similarity thresholds: hold-out max-sim and candidate kNN sims.

  ./run.sh compile/inspect_similarity.py [--per-bin 3]
Prints quantiles and example pairs per similarity bin (candidate text vs
hold-out prompt, and candidate vs nearest other candidate, incl. cross-lingual).
"""
import argparse
import glob
import os
import random

import numpy as np
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data")
BINS = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.01]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-bin", type=int, default=3)
    ap.add_argument("--chars", type=int, default=150)
    args = ap.parse_args()
    rnd = random.Random(0)
    nb = []
    for p in sorted(glob.glob(os.path.join(D, "pool", "neighbors", "*.parquet"))):
        nb += pq.read_table(p).to_pylist()
    lang = dict(zip(*[pq.read_table(os.path.join(D, "pool", "candidates.parquet"),
                                    columns=["uid", "lang"]).column(c).to_pylist()
                      for c in ("uid", "lang")]))
    ho = pq.read_table(os.path.join(D, "holdout", "prompts.parquet"), columns=["id", "bench", "text"]).to_pylist()
    ho = {r["id"]: r for r in ho}
    hs = np.array([r["ho_sim"] for r in nb])
    s1 = np.array([r["nn_sims"][0] for r in nb])
    q = [0.5, 0.9, 0.99, 0.999]
    print("hold-out max-sim quantiles", dict(zip(q, np.quantile(hs, q).round(3))))
    print("  share >= 0.85/0.88/0.90/0.92/0.95: {}".format(
        [round(float((hs >= t).mean()), 4) for t in (0.85, 0.88, 0.90, 0.92, 0.95)]))
    print("top-1 kNN sim quantiles", dict(zip(q, np.quantile(s1, q).round(3))))
    print("  share >= 0.90/0.92/0.95/0.98: {}".format(
        [round(float((s1 >= t).mean()), 4) for t in (0.90, 0.92, 0.95, 0.98)]))
    # sample ids needing text
    picks_ho, picks_nn = [], []
    for lo, hi in zip(BINS, BINS[1:]):
        a = [r for r in nb if lo <= r["ho_sim"] < hi]
        picks_ho.append((lo, hi, rnd.sample(a, min(args.per_bin, len(a))), len(a)))
        b = [r for r in nb if lo <= r["nn_sims"][0] < hi and lang.get(r["id"]) != lang.get(r["nn_ids"][0])]
        picks_nn.append((lo, hi, rnd.sample(b, min(args.per_bin, len(b))), len(b)))
    want = {r["id"] for _, _, rs, _ in picks_ho + picks_nn for r in rs}
    want |= {r["nn_ids"][0] for _, _, rs, _ in picks_nn for r in rs}
    text = {}
    for b in pq.ParquetFile(os.path.join(D, "llm", "embed", "input.parquet")).iter_batches(batch_size=50000):
        for i, t in zip(b.column("id").to_pylist(), b.column("text").to_pylist()):
            if i in want:
                text[i] = t
    cut = lambda s: (s or "")[:args.chars].replace("\n", " ")
    print("\n===== candidate vs hold-out")
    for lo, hi, rs, n in picks_ho:
        print("--- [{:.2f}, {:.2f}) n={}".format(lo, hi, n))
        for r in rs:
            h = ho[r["ho_id"]]
            print("  [{} {:.3f}] {}\n     HO {}: {}".format(lang.get(r["id"]), r["ho_sim"], cut(text.get(r["id"])),
                                                       h["bench"].split("__")[-1], cut(h["text"])))
    print("\n===== candidate vs nearest candidate in ANOTHER language")
    for lo, hi, rs, n in picks_nn:
        print("--- [{:.2f}, {:.2f}) n={}".format(lo, hi, n))
        for r in rs:
            o = r["nn_ids"][0]
            print("  [{} {:.3f}] {}\n     [{}] {}".format(lang.get(r["id"]), r["nn_sims"][0], cut(text.get(r["id"])),
                                                      lang.get(o), cut(text.get(o))))


if __name__ == "__main__":
    main()
