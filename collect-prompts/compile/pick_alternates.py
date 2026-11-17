#!/usr/bin/env python3
"""Stage 6c: choose among alternate donors of a refill round.

  ./run.sh compile/pick_alternates.py --round N [--out data/compiled]

After `allocate.py --refill ... --alternates K` and translating round N (the
primary and K-1 alternates per reopened slot), validate each candidate with the
same checks as assemble.py, in rank order (primary first), and keep the first
that passes: its row replaces the slot's row in <out>/allocation.parquet (the
previous file is kept as allocation.before_pick_roundN.parquet). Slots where
no candidate passes keep the primary and show up in failed_slots.parquet at the
next assembly. Unchosen alternates stay blocked for later rounds.
"""
import argparse
import collections
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import assemble as A  # noqa: E402
import build_pool as B  # noqa: E402

DATA = A.DATA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--out", default=os.path.join(DATA, "compiled"))
    args = ap.parse_args()
    alloc_path = os.path.join(args.out, "allocation.parquet")
    alt_path = os.path.join(args.out, "alternates_round{}.parquet".format(args.round))
    alloc = pq.read_table(alloc_path).to_pandas()
    alts = pq.read_table(alt_path).to_pandas()
    prim = alloc[alloc["round"] == args.round].assign(rank=0)
    cands = pd.concat([prim, alts[prim.columns]], ignore_index=True)
    cands = cands[cands["uid"].notna()].sort_values(["slot", "rank"])
    print("round {}: {} slots, {} candidates".format(args.round, len(prim), len(cands)), flush=True)

    B.load_model(B.ensure_model(), B.label_map())
    cand = {r["uid"]: r for r in pq.read_table(os.path.join(DATA, "pool", "candidates.parquet"),
                                               columns=["uid", "part", "row"]).to_pylist()}
    recs = A.fetch_records([cand[u] for u in cands["uid"]])
    ifm = {r["uid"]: r for r in pq.read_table(os.path.join(DATA, "pool", "if_meta.parquet")).to_pylist()
           if r["uid"] in recs}
    tr_meta, tr_out = A.load_translations()

    chosen, reasons = {}, collections.Counter()
    for r in cands.to_dict("records"):
        if r["slot"] in chosen:
            continue
        if r["fill"] != "translate":
            chosen[r["slot"]] = r
            continue
        im = ifm.get(r["uid"])
        kwargs = json.loads(im["kwargs"]) if im and im["kwargs"] else None
        ids = list(im["instruction_ids"]) if im else []
        m = tr_meta.get((r["slot"], r["uid"]))
        o = tr_out.get(m["id"]) if m else None
        reason = A.check_translation(recs[r["uid"]]["context"], m, o, r["lang"], kwargs, ids)[0]
        if reason is None:
            chosen[r["slot"]] = r
        else:
            reasons[reason.split(":")[0]] += 1

    ranks = collections.Counter(r["rank"] for r in chosen.values())
    swap = {s: r for s, r in chosen.items() if r["rank"] > 0}
    backup = os.path.join(args.out, "allocation.before_pick_round{}.parquet".format(args.round))
    shutil.copyfile(alloc_path, backup)
    if swap:   # replace whole rows (list-valued columns make cell-wise assignment fragile)
        new = pd.DataFrame(list(swap.values()))[alloc.columns]
        alloc = pd.concat([alloc[~alloc["slot"].isin(swap)], new], ignore_index=True)
        alloc = alloc.sort_values("slot").reset_index(drop=True)
    assert len(alloc) == alloc["slot"].nunique()
    alloc.to_parquet(alloc_path, index=False)
    print("chosen rank:", dict(sorted(ranks.items())), "| no candidate passed:",
          len(prim) - len(chosen), "| rejected candidates:", dict(reasons), flush=True)
    print("allocation updated ({} slots swapped to an alternate); backup {}".format(len(swap), backup))


if __name__ == "__main__":
    main()
