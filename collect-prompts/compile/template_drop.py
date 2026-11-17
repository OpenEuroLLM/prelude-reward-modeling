#!/usr/bin/env python3
"""Stage 6f: drop rows of broken template families, for refilling, and keep the families out of
later refills.

  ./run.sh compile/template_drop.py                 # counts per family / language / fill
  ./run.sh compile/template_drop.py --apply         # add them to data/compiled/force_fail.parquet
                                                    # and write data/compiled/template_uids.txt

A family is a regex on the SOURCE text (first user turn's pool context, so translated rows are
found too) within one source; `fills` says which rows of the set are dropped. Found by the final
spot check of the rebuilt rows (manual_check/final_check/FINDINGS.md):
- lt_catalogue_faq: every_prompt's Lithuanian car-parts catalogue FAQ, 'Kokios <part> <car>
  Lietuvoje dalys yra kataloge?' / 'Kaip pateikti <part> <car> užsakymą?' / 'Ar galima pasirinkti
  atsarginių dalių analogus kategorijoje <part> <car> Lietuvoje?', mostly broken Lithuanian
  (machine-made SEO text, category names left in the nominative); 42 + 6 natives. Other large
  every_prompt families are fluent site FAQ and are kept.
- pt_legal_act_extraction: one scripted WildChat user, 2,017 pool items 'Usando o seguinte texto:
  ATO ... extract the personnel actions as JSON' over Brazilian legal acts. Refill donors at the
  relaxed cell levels (28 of the 42 Lithuanian refills in round 11); translations garble the legal
  titles or keep them Portuguese (failing language ID). Translated rows are dropped; the 51
  Portuguese natives (pre-existing) are kept.
template_uids.txt lists every pool uid of every family: refill with
  EXTRA_ALLOC="--refill-native --refill-exclude-sources openeurollm/EuroMultiSynth-Instruct
               --refill-exclude-uids $PWD/data/compiled/template_uids.txt"
force_fail is keyed by (slot, uid).
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import mt_common as M  # noqa: E402

TEMPLATES = {
    "lt_catalogue_faq": {"source": "lang-uk/every_prompt",      # all question forms of the site
                         "rx": r"Lietuvoje dalys yra kataloge|\b(Tvirtinimo elementai|Variklio dangčiai)\b",
                         "fills": ("native",)},
    "pt_legal_act_extraction": {"source": "allenai/WildChat-4.8M", "rx": r"^\s*Usando o seguinte texto:",
                                "fills": ("translated",)},
}


def pool_matches(names):
    """-> {template name: set of pool uids whose first user turn matches}"""
    rx = {n: re.compile(TEMPLATES[n]["rx"]) for n in names}
    prefix = {n: TEMPLATES[n]["source"] + "::" for n in names}
    out = {n: set() for n in names}
    for b in pq.ParquetFile(os.path.join(M.DATA, "pool", "candidates_ctx.parquet")).iter_batches(
            batch_size=20000, columns=["uid", "context"]):
        for u, c in zip(b.column("uid").to_pylist(), b.column("context").to_pylist()):
            todo = [n for n in names if u.startswith(prefix[n])]
            if not todo:
                continue
            first = next((m["content"] or "" for m in json.loads(c) if m["role"] == "user"), "")
            for n in todo:
                if rx[n].search(first):
                    out[n].add(u)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates", default=",".join(TEMPLATES))
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    names = args.templates.split(",")
    pool = pool_matches(names)
    rows = pq.read_table(os.path.join(M.DATA, "compiled", "prompts.parquet"),
                         columns=["id", "lang", "fill", "source_uid"]).to_pylist()
    hits = []
    for n in names:
        inset = [r for r in rows if r["source_uid"] in pool[n]]
        print("{}: {} pool items, {} in the set {}".format(n, len(pool[n]), len(inset),
              dict(collections.Counter((r["lang"], r["fill"]) for r in inset).most_common())))
        hits += [(n, r) for r in inset if r["fill"] in TEMPLATES[n]["fills"]]
    print("rows to drop:", len(hits), dict(collections.Counter(n for n, _ in hits)),
          dict(collections.Counter(r["lang"] for _, r in hits)))
    if args.apply:
        p = os.path.join(M.DATA, "compiled", "force_fail.parquet")
        old = pq.read_table(p).to_pylist() if os.path.exists(p) else []
        have = {(r["slot"], r["uid"]) for r in old}
        new = [{"slot": int(r["id"].split("-")[1]), "id": r["id"], "uid": r["source_uid"],
                "reason": "template_" + n} for n, r in hits
               if (int(r["id"].split("-")[1]), r["source_uid"]) not in have]
        pq.write_table(pa.Table.from_pylist(old + new), p)
        print("force_fail: {} existing + {} new".format(len(old), len(new)))
        allu = sorted(set().union(*pool.values()))
        with open(os.path.join(M.DATA, "compiled", "template_uids.txt"), "w") as f:
            f.write("\n".join(allu) + "\n")
        print("template_uids.txt:", len(allu), "uids")


if __name__ == "__main__":
    main()
