#!/usr/bin/env python3
"""Stage 5g: round-trip quality estimate, Gemma vs TranslateGemma, per re-translated row.

  ./run.sh compile/qe_requests.py

For every row whose current translation is TranslateGemma's (stage 5c), both
candidates -- "old" (the Gemma-4 conversation-level translation, from
prompts.before_retranslation.parquet) and "new" (TranslateGemma per message) --
are translated back into the row's source language by BOTH models, so each
candidate gets one back-translation from its own model and one from the other
(no self-preference). qe_select.py compares the back-translations with the
source (chrF++). Only rows whose old translation was valid (passed assembly
before stage 5c) have an "old" candidate; the rest keep TranslateGemma.
Outputs: data/llm/qe/{tg,gemma}/requests.parquet, data/llm/qe/layout.parquet
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import mt_common as M  # noqa: E402
import translate_prompt as T  # noqa: E402

OUT = os.path.join(M.DATA, "llm", "qe")
TG_MODEL = "google/translategemma-27b-it"


def gemma_messages(text, src, tgt):
    return [{"role": "user", "content": "Translate the following {} text into {}. Output only the "
             "translation.\n\n{}".format(T.LANG_NAMES.get(src, src), T.LANG_NAMES.get(tgt, tgt), text)}]


def main():
    cols = ["id", "lang", "translator", "source_uid", "source_lang", "context"]
    new = [r for r in pq.read_table(os.path.join(M.DATA, "compiled", "prompts.parquet"), columns=cols).to_pylist()
           if r["translator"] == TG_MODEL]
    old = {r["id"]: r for r in pq.read_table(os.path.join(M.DATA, "compiled", "prompts.before_retranslation.parquet"),
                                             columns=["id", "source_uid", "context"]).to_pylist()}
    flagged_path = os.path.join(M.DATA, "compiled", "manual_check", "flagged.txt")
    flagged = {ln.split()[0] for ln in open(flagged_path) if ln.strip() and not ln.startswith("#")}
    need = {r["source_uid"] for r in new}
    src = {}
    for b in pq.ParquetFile(os.path.join(M.DATA, "pool", "candidates_ctx.parquet")).iter_batches(
            batch_size=20000, columns=["uid", "context"]):
        for u, c in zip(b.column("uid").to_pylist(), b.column("context").to_pylist()):
            if u in need:
                src[u] = json.loads(c)
    reqs = {"tg": [], "gemma": []}
    layout = []
    n_old = 0
    for r in new:
        slot = int(r["id"].split("-")[1])
        cands = {"new": r["context"]}
        o = old.get(r["id"])
        # the old candidate competes only if it was a valid translation of the same donor,
        # has no script injection and was not flagged in the manual review
        if o and o["source_uid"] == r["source_uid"] and len(o["context"]) == len(r["context"]) \
                and r["id"] not in flagged and not M.injected(
                    "\n".join(m["content"] or "" for m in src[r["source_uid"]]),
                    "\n".join(m["content"] or "" for m in o["context"])):
            cands["old"] = o["context"]
            n_old += 1
        if "old" not in cands:
            continue          # nothing to compare against: TranslateGemma stays
        for cand, ctx in cands.items():
            for mi, m in enumerate(ctx):
                for pi, (kind, s) in enumerate(M.segments(m["content"] or "")):
                    rid = "{}::{}::{}::{}".format(cand, slot, mi, pi) if kind == "text" else None
                    layout.append({"slot": slot, "uid": r["source_uid"], "cand": cand, "msg": mi, "piece": pi,
                                   "kind": kind, "content": s, "rid": rid})
                    if not rid:
                        continue
                    mt = int(min(6144, len(s) * 1.0 + 128))
                    reqs["tg"].append({"id": "tg::" + rid, "prompt": None, "prompt_ids": None, "stop": None,
                                       "schema": None, "max_tokens": mt, "messages": json.dumps(
                                           M.tg_messages(s, r["lang"], r["source_lang"]), ensure_ascii=False)})
                    reqs["gemma"].append({"id": "gm::" + rid, "prompt": None, "prompt_ids": None, "stop": None,
                                          "schema": None, "max_tokens": mt, "messages": json.dumps(
                                              gemma_messages(s, r["lang"], r["source_lang"]), ensure_ascii=False)})
    for name, rs in reqs.items():
        os.makedirs(os.path.join(OUT, name), exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rs), os.path.join(OUT, name, "requests.parquet"))
    pq.write_table(pa.Table.from_pylist(layout), os.path.join(OUT, "layout.parquet"))
    print("TranslateGemma rows:", len(new), "| with an old candidate:", n_old,
          "| back-translation requests per model:", len(reqs["tg"]))


if __name__ == "__main__":
    main()
