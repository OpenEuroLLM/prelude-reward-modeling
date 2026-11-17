#!/usr/bin/env python3
"""Stage 6b: drop low-quality every_prompt rows (judge of stage 5i/5j) for refilling.

  ./run.sh compile/ep_clean.py            # show counts and random examples per reason
  ./run.sh compile/ep_clean.py --apply    # add them to data/compiled/force_fail.parquet

A row is dropped if the judge says spam (betting/casino site or bonus, escort/
adult, psychic/occult service, advertising for a business), not a request (bare
keywords, names, titles, slugs) or not self-contained (needs the missing page).
force_fail entries are keyed by (slot, uid), so a refilled slot is not failed again.
"""
import argparse
import collections
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import mt_common as M  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--examples", type=int, default=15)
    args = ap.parse_args()
    judge = {r["id"]: r for r in pq.read_table(os.path.join(M.DATA, "llm", "bench", "every_prompt_judge.parquet")).to_pylist()}
    rows = {r["id"]: r for r in pq.read_table(os.path.join(M.DATA, "compiled", "prompts.parquet"),
                                              columns=["id", "lang", "source_uid", "context"]).to_pylist() if r["id"] in judge}
    drop, by_reason, by_lang = {}, collections.defaultdict(list), collections.Counter()
    for rid, v in judge.items():
        reasons = [k for k, bad in (("spam", v["spam"]), ("not_request", not v["is_request"]),
                                    ("not_self_contained", not v["self_contained"])) if bad]
        if reasons and rid in rows:
            drop[rid] = reasons
            by_lang[rows[rid]["lang"]] += 1
            for x in reasons:
                by_reason[x].append(rid)
    print("every_prompt rows judged:", len(judge), "| to drop:", len(drop),
          "| by reason:", {k: len(v) for k, v in by_reason.items()})
    print("by language:", dict(by_lang.most_common()))
    rng = random.Random(3)
    for reason, ids in by_reason.items():
        print("\n== {} (random {}):".format(reason, min(args.examples, len(ids))))
        for rid in rng.sample(sorted(ids), min(args.examples, len(ids))):
            print("  {} [{}] {}".format(rid, rows[rid]["lang"], rows[rid]["context"][-1]["content"].strip()[:110].replace("\n", " ")))
    kept = sorted(set(judge) - set(drop))
    print("\n== kept (random {}):".format(args.examples))
    for rid in rng.sample(kept, min(args.examples, len(kept))):
        print("  {} [{}] {}".format(rid, rows[rid]["lang"], rows[rid]["context"][-1]["content"].strip()[:110].replace("\n", " ")))
    if args.apply:
        p = os.path.join(M.DATA, "compiled", "force_fail.parquet")
        old = pq.read_table(p).to_pylist() if os.path.exists(p) else []
        have = {(r["slot"], r["uid"]) for r in old}
        new = [{"slot": int(rid.split("-")[1]), "id": rid, "uid": rows[rid]["source_uid"],
                "reason": "every_prompt:" + "+".join(drop[rid])} for rid in sorted(drop)
               if (int(rid.split("-")[1]), rows[rid]["source_uid"]) not in have]
        pq.write_table(pa.Table.from_pylist(old + new), p)
        print("\nforce_fail: {} existing + {} new".format(len(old), len(new)))


if __name__ == "__main__":
    main()
