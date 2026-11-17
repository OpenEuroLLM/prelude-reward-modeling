#!/usr/bin/env python3
"""Stage 6a: slots to fail because the row is in the wrong language (trusted
source labels that turned out wrong), for the next refill round.

  ./run.sh compile/label_fixes.py

1. Rows confirmed mislabeled in the manual review (manual_check/label_review.txt).
2. Bokmål/Nynorsk check on every nb/nn row: per user turn, count variety-specific
   function words; a turn is nn/nb when one side has >= 2 markers and more than
   twice the other. A row fails if its turns mix nn and nb, or if all its
   decided turns contradict the slot (applied to ltg/normistral, the source with
   the label problem; other sources are only reported).
Output: data/compiled/force_fail.parquet (slot, id, uid, reason) + counts on stdout;
assemble.py fails a slot only while it still holds that uid.
"""
import collections
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import mt_common as M  # noqa: E402

NN = {"ikkje", "eg", "kva", "korleis", "kvifor", "kven", "kvar", "frå", "berre", "meir", "fleire",
      "noko", "nokon", "mykje", "sjølv", "vere", "vera", "heile", "kjem", "gjere", "mogleg",
      "dykk", "hjå", "eit"}
NB = {"ikke", "jeg", "hva", "hvordan", "hvorfor", "hvem", "hvor", "fra", "bare", "mer", "flere",
      "noe", "noen", "mye", "selv", "være", "hele", "kommer", "gjøre", "mulig"}
WORD = re.compile(r"[a-zæøåòéèô]+")
APPLY = {"ltg/normistral-11b-thinking-training"}


def variety(text):
    w = WORD.findall((text or "").lower())
    nn, nb = sum(x in NN for x in w), sum(x in NB for x in w)
    if nn >= 2 and nn > 2 * nb:
        return "nn"
    if nb >= 2 and nb > 2 * nn:
        return "nb"
    return None


def main():
    rows = pq.read_table(os.path.join(M.DATA, "compiled", "prompts.parquet"),
                         columns=["id", "lang", "source", "source_uid", "context"]).to_pylist()
    uid = {r["id"]: r["source_uid"] for r in rows}
    fail = {}
    review = os.path.join(M.DATA, "compiled", "manual_check", "label_review.txt")
    for ln in open(review):
        if ln.strip() and not ln.startswith("#"):
            rid, reason = ln.split(None, 1)
            fail[rid] = "manual_review: " + reason.strip()
    stats = collections.Counter()
    for r in rows:
        if r["lang"] not in ("nb", "nn"):
            continue
        v = [variety(m["content"]) for m in r["context"] if m["role"] == "user"]
        v = [x for x in v if x]
        mixed = len(set(v)) > 1
        wrong = bool(v) and all(x != r["lang"] for x in v)
        if not (mixed or wrong):
            continue
        stats[(r["source"], "mixed" if mixed else "wrong_variety")] += 1
        if r["source"] in APPLY and r["id"] not in fail:
            fail[r["id"]] = "nb_nn_" + ("mixed" if mixed else "wrong_variety")
    out = [{"slot": int(k.split("-")[1]), "id": k, "uid": uid[k], "reason": v}
           for k, v in sorted(fail.items())]
    pq.write_table(pa.Table.from_pylist(out), os.path.join(M.DATA, "compiled", "force_fail.parquet"))
    print("nb/nn heuristic hits by source:")
    for (s, k), n in sorted(stats.items(), key=lambda kv: -kv[1]):
        print("  {:5d}  {:14s} {}".format(n, k, s))
    print("force_fail:", len(out), dict(collections.Counter(v.split(":")[0] for v in fail.values())))


if __name__ == "__main__":
    main()
