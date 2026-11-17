#!/usr/bin/env python3
"""Stage 5a: translation requests for allocated slots with fill == "translate".

  ./run.sh compile/build_translation_requests.py
Inputs : data/compiled/allocation.parquet, data/pool/candidates_ctx.parquet,
         data/pool/if_meta.parquet
Outputs: data/llm/translate/requests.parquet       id, messages, schema, max_tokens
         data/llm/translate/requests_meta.parquet  id, slot, uid, src_lang, tgt_lang,
                                                   n_messages, params, kwargs
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import translate_prompt as T  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "llm", "translate")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0,
                    help="0 = initial allocation (writes data/llm/translate/); N > 0 = only "
                         "slots refilled in round N (writes data/llm/translate/roundN/)")
    args = ap.parse_args()
    out = OUT if args.round == 0 else os.path.join(OUT, "round{}".format(args.round))
    tbl = pq.read_table(os.path.join(DATA, "compiled", "allocation.parquet"))
    cols = ["slot", "uid", "lang", "src_lang", "fill"] + (["round"] if "round" in tbl.column_names else [])
    alloc = tbl.select(cols).to_pylist()
    todo = [a for a in alloc if a["fill"] == "translate" and a.get("round", 0) == args.round]
    # alternates (allocate.py --alternates): extra donors for the same slots
    alt_path = os.path.join(DATA, "compiled", "alternates_round{}.parquet".format(args.round))
    if args.round > 0 and os.path.exists(alt_path):
        alts = pq.read_table(alt_path).select(cols + ["rank"]).to_pylist()
        todo += [a for a in alts if a["fill"] == "translate"]
        print("with {} alternates".format(sum(a["fill"] == "translate" for a in alts)))
    need = {a["uid"] for a in todo}
    ctx = {}
    for b in pq.ParquetFile(os.path.join(DATA, "pool", "candidates_ctx.parquet")).iter_batches(
            batch_size=20000, columns=["uid", "context"]):
        for u, c in zip(b.column("uid").to_pylist(), b.column("context").to_pylist()):
            if u in need:
                ctx[u] = json.loads(c)
    ifm = {r["uid"]: r for r in pq.read_table(os.path.join(DATA, "pool", "if_meta.parquet")).to_pylist()
           if r["uid"] in need}
    reqs, meta = [], []
    n_if = 0
    for a in todo:
        c = ctx[a["uid"]]
        rid = "tr::{}".format(a["slot"]) if args.round == 0 else \
            "tr{}::{}".format(args.round, a["slot"])
        if a.get("rank"):
            rid = "tr{}.{}::{}".format(args.round, a["rank"], a["slot"])
        chars = sum(len(m["content"] or "") for m in c)
        # ~0.6 output tokens per source char covers scripts with low chars/token (ka, el)
        max_tokens = int(min(16000, max(512, chars * 0.6 + 300)))
        m = ifm.get(a["uid"])
        kwargs = json.loads(m["kwargs"]) if m and m["kwargs"] else None
        params = T.if_params(kwargs) if kwargs else []
        if params:
            msgs = T.messages_if(c, a["src_lang"], a["lang"], params)
            schema = T.schema_if(len(c), len(params))
            n_if += 1
        else:
            msgs = T.messages(c, a["src_lang"], a["lang"])
            schema = T.schema(len(c))
        reqs.append({"id": rid, "messages": json.dumps(msgs, ensure_ascii=False),
                     "schema": json.dumps(schema), "max_tokens": max_tokens})
        meta.append({"id": rid, "slot": a["slot"], "uid": a["uid"], "src_lang": a["src_lang"],
                     "tgt_lang": a["lang"], "n_messages": len(c),
                     "params": json.dumps(params, ensure_ascii=False),
                     "kwargs": json.dumps(kwargs, ensure_ascii=False) if kwargs else None})
    os.makedirs(out, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(reqs), os.path.join(out, "requests.parquet"))
    pq.write_table(pa.Table.from_pylist(meta), os.path.join(out, "requests_meta.parquet"))
    print("{} translation requests ({} with IF params) -> {}".format(len(reqs), n_if, out))


if __name__ == "__main__":
    main()
