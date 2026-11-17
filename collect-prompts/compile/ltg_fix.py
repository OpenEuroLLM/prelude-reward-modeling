#!/usr/bin/env python3
"""Stage 5e': targeted TildeOpen re-translations of single Latgalian messages.

  ./run.sh compile/ltg_fix.py      (writes data/ltg/eval/tilde_fix/requests.parquet)
  then: sbatch --export=ALL,PHASES=tilde,TILDE_REQ=...,TILDE_OUT=... retranslate_ltg.sbatch

Inputs: data/compiled/manual_check/ltg_fix_inputs.json, one entry per message:
  mode "direct"   the source message, re-segmented with max_chars (long messages
                  lost content when translated in 2,000-character chunks)
  mode "pivot_lv" a hand-checked Latvian pivot text (e.g. an ambiguous term that
                  every system mistranslated), translated lv -> ltg
fix_messages() reassembles the outputs: (row id, message index) -> text.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import mt_common as M  # noqa: E402
from ltg_requests import tilde_prompt  # noqa: E402

D = os.path.join(M.DATA, "ltg", "eval", "tilde_fix")
INPUTS = os.path.join(M.DATA, "compiled", "manual_check", "ltg_fix_inputs.json")


def source_contexts(ids):
    P = {r["id"]: r for r in pq.read_table(os.path.join(M.DATA, "compiled", "prompts.parquet"),
                                           columns=["id", "source_uid", "source_lang"]).to_pylist()
         if r["id"] in ids}
    need = {r["source_uid"] for r in P.values()}
    ctx = {}
    for b in pq.ParquetFile(os.path.join(M.DATA, "pool", "candidates_ctx.parquet")).iter_batches(
            batch_size=20000, columns=["uid", "context"]):
        for u, c in zip(b.column("uid").to_pylist(), b.column("context").to_pylist()):
            if u in need:
                ctx[u] = json.loads(c)
    return {i: (ctx[r["source_uid"]], r["source_lang"]) for i, r in P.items()}


def main():
    entries = json.load(open(INPUTS))
    src = source_contexts({e["id"] for e in entries})
    reqs, layout = [], []
    for e in entries:
        sctx, src_lang = src[e["id"]]
        if e["mode"] == "direct":
            pieces = M.segments(sctx[e["msg"]]["content"] or "", e.get("max_chars", 500))
        else:
            pieces, src_lang = [("text", e["text"])], "lv"
        for pi, (kind, s) in enumerate(pieces):
            rid = "FIX::{}::{}::{}".format(e["id"], e["msg"], pi) if kind == "text" else None
            layout.append({"id": e["id"], "msg": e["msg"], "piece": pi, "kind": kind, "content": s, "rid": rid})
            if rid:
                prompt = tilde_prompt(src_lang, s, e["k"])
                reqs.append({"id": rid, "messages": None, "prompt": prompt, "prompt_ids": M.tilde_encode(prompt),
                             "stop": json.dumps(["</Latgalian>"]), "schema": None,
                             "max_tokens": int(min(4096, len(s) * 0.8 + 128))})
    os.makedirs(D, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(reqs), os.path.join(D, "requests.parquet"))
    with open(os.path.join(D, "layout.json"), "w") as f:
        json.dump(layout, f, ensure_ascii=False, indent=1)
    print("fix requests:", len(reqs), "for", len(entries), "messages")


def fix_messages():
    """(row id, message index) -> reassembled Latgalian text (None if a piece is missing)."""
    out = {}
    for p in glob.glob(os.path.join(D, "out", "shard*.parquet")):
        for r in pq.read_table(p).to_pylist():
            if r["finish_reason"] == "stop" and r.get("token_ids") is not None:
                t = M.tilde_sp().decode([int(x) for x in r["token_ids"]]).split("</Latgalian>")[0]
                out[r["id"]] = t.split("\n<")[0].strip()
    res = {}
    for x in json.load(open(os.path.join(D, "layout.json"))):
        key = (x["id"], x["msg"])
        if key in res and res[key] is None:
            continue
        piece = out.get(x["rid"]) if x["kind"] == "text" else x["content"]
        res[key] = None if piece is None or (x["kind"] == "text" and not piece) else res.get(key, "") + piece
    return res


if __name__ == "__main__":
    main()
