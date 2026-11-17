#!/usr/bin/env python3
"""Which judges label which pair in the full run: three judges on every pair, and one more on a tie.

  collect-preferences/run.sh assign_judges.py substitutes   # after build_pairs.py
  collect-preferences/run.sh assign_judges.py tiebreak      # after the first pass and the substitutes

Pool: Gemma 4 31B, Qwen3.8-27B and gpt-oss-120b (reasoning high) on every pair, except where a pair has an
answer from the judge's family (families.py). A pool judge removed by that rule is replaced by the first
eligible substitute in the order Qwen3.5-122B-A10B, Mistral Small 4, GLM-4.5-Air, so that every pair has three
judges: with a Gemma 4 12B answer Qwen3.5-122B; with a Qwen answer Mistral Small 4, or GLM-4.5-Air when the other
answer is from a Mistral model; with both a Gemma and a Qwen answer Mistral Small 4 and GLM-4.5-Air.
Tie-break: a pair whose votes split evenly gets the next eligible judge not yet used, in the same order
(Qwen3.5-122B first). Every judge judges both orders.

substitutes: writes data/preferences/assigned/<judge>.parquet (rows of pairs.parquet) for each substitute and
data/preferences/assigned/judges.parquet (pair id, group, the three judges).
tiebreak: reads the votes in data/preferences/votes/{pass1,substitutes}/, checks that every assigned call is
there, and writes data/preferences/assigned/tiebreak_<judge>.parquet for the tied pairs.
"""
import argparse
import collections
import glob
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from families import RESPONDER_FAMILY, own_family  # noqa: E402

POOL = ["gemma-4-31B-it", "qwen3.8-27b", "gpt-oss-120b-bf16"]
SUBSTITUTES = ["qwen3.5-122b-a10b", "mistral-small-4-119b", "glm-4.5-air"]
PAIRS = os.path.join(ROOT, "data", "preferences", "pairs.parquet")
ASSIGNED = os.path.join(ROOT, "data", "preferences", "assigned")
VOTES = os.path.join(ROOT, "data", "preferences", "votes")


def families():
    return {j["name"]: j["family"] for j in yaml.safe_load(open(os.path.join(HERE, "judges.yaml")))["judges"]}


def group(p):
    f = {RESPONDER_FAMILY.get(p["model_x"]), RESPONDER_FAMILY.get(p["model_y"])}
    return "gemma+qwen" if {"gemma", "qwen"} <= f else "gemma" if "gemma" in f else "qwen" if "qwen" in f else "none"


def first_pass(p, fam):
    js = [j for j in POOL if not own_family(fam[j], p)]
    for s in SUBSTITUTES:
        if len(js) == 3:
            break
        if not own_family(fam[s], p):
            js.append(s)
    assert len(js) == 3, (p["id"], js)
    return js


def substitutes():
    import pyarrow as pa
    import pyarrow.parquet as pq
    fam = families()
    pairs = pq.read_table(PAIRS).to_pylist()
    os.makedirs(ASSIGNED, exist_ok=True)
    per = collections.defaultdict(list)
    table = []
    combos = collections.Counter()
    for p in pairs:
        js = first_pass(p, fam)
        table.append({"id": p["id"], "group": group(p), "judges": js})
        combos[(group(p), tuple(js))] += 1
        for j in js:
            if j in SUBSTITUTES:
                per[j].append(p)
    pq.write_table(pa.Table.from_pylist(table), os.path.join(ASSIGNED, "judges.parquet"))
    for j, rows in per.items():
        pq.write_table(pa.Table.from_pylist(rows), os.path.join(ASSIGNED, j + ".parquet"))
    print("{} pairs; judges per group:".format(len(pairs)))
    for (g, js), n in sorted(combos.items(), key=lambda kv: -kv[1]):
        print("  {:11s} {:6d} ({:4.1f}%)  {}".format(g, n, 100 * n / len(pairs), " + ".join(js)))
    for j in SUBSTITUTES:
        print("substitute {}: {} pairs -> {}".format(j, len(per[j]), os.path.join(ASSIGNED, j + ".parquet")))


def tiebreak():
    import pyarrow as pa
    import pyarrow.parquet as pq
    fam = families()
    pairs = {p["id"]: p for p in pq.read_table(PAIRS).to_pylist()}
    assigned = {r["id"]: r["judges"] for r in pq.read_table(os.path.join(ASSIGNED, "judges.parquet")).to_pylist()}
    votes = collections.defaultdict(dict)            # pair -> (judge, order) -> prefers
    for path in sorted(glob.glob(os.path.join(VOTES, "pass1", "*.parquet")) + glob.glob(os.path.join(VOTES, "substitutes", "*.parquet"))):
        for r in pq.read_table(path, columns=["pair_id", "order", "judge", "prefers"]).to_pylist():
            votes[r["pair_id"]][(r["judge"], r["order"])] = r["prefers"]
    missing = [(i, j, o) for i, js in assigned.items() for j in js for o in (0, 1) if (j, o) not in votes.get(i, {})]
    if missing:
        by = collections.Counter(j for _, j, _ in missing)
        raise SystemExit("{} assigned calls are missing ({}): finish the first pass and the substitutes first".format(len(missing), dict(by)))
    per = collections.defaultdict(list)
    ties = no_vote = 0
    for i, js in assigned.items():
        v = [votes[i][(j, o)] for j in js for o in (0, 1) if votes[i][(j, o)] in ("x", "y")]
        if not v:
            no_vote += 1                               # every call failed (e.g. too long): a tie-breaker would fail too
            continue
        if 2 * sum(x == "x" for x in v) != len(v):
            continue
        ties += 1
        for s in SUBSTITUTES:
            if s not in js and not own_family(fam[s], pairs[i]):
                per[s].append(pairs[i])
                break
    for s, rows in per.items():
        pq.write_table(pa.Table.from_pylist(rows), os.path.join(ASSIGNED, "tiebreak_" + s + ".parquet"))
    print("{} pairs: {} tied, {} without any vote".format(len(assigned), ties, no_vote))
    for s in SUBSTITUTES:
        print("tie-break {}: {} pairs".format(s, len(per[s])))
    print("tied pairs without an eligible judge left: {}".format(ties - sum(len(r) for r in per.values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["substitutes", "tiebreak"])
    args = ap.parse_args()
    substitutes() if args.step == "substitutes" else tiebreak()


if __name__ == "__main__":
    main()
