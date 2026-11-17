#!/usr/bin/env python3
"""Stage 5c: TranslateGemma-27B requests.

  ./run.sh compile/tg_requests.py [--flagged data/compiled/manual_check/flagged.txt]

Three request groups, one requests file (one model pass):
  tg::<slot>::<m>::<p>   re-translation of the targeted set: every translated row
                         in the weak languages (eu, mt, is, ga), every translated
                         row with exotic-script injection, and manually flagged
                         rows. Rows with IF verifier params or code-heavy prose
                         stay with the conversation-level Gemma translation.
  lv::<slot>::<m>::<p>   Latgalian rows: source -> Latvian (pivot leg 1)
  flv::<flores id>       FLORES+ devtest eval sentences, English -> Latvian

Per message, prose is split into paragraph chunks (code blocks kept verbatim);
segments.parquet records the layout so outputs reassemble exactly.
Outputs: data/llm/translate_tg/{requests,segments}.parquet, selection.json

--round N (after refill round N and its assembly): only the slots refilled in
round N, request ids tgN::..., outputs under data/llm/translate_tg/roundN/ (ids
are per slot, so a refilled slot must never reuse an earlier round's output).
--langs L1,L2 --tag X: every row in these languages still translated by Gemma
(benchmark winners, stage 5j), ids tgX::..., outputs under translate_tg/roundX/.
The weak-language set is eu, mt, is, ga plus manual_check/tg_langs.txt.
"""
import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import mt_common as M  # noqa: E402

DATA = M.DATA
OUT = os.path.join(DATA, "llm", "translate_tg")
WEAK = {"eu", "mt", "is", "ga"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagged", default=os.path.join(DATA, "compiled", "manual_check", "flagged.txt"))
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--langs", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    flagged = set()
    if os.path.exists(args.flagged):
        flagged = {ln.split()[0] for ln in open(args.flagged) if ln.strip() and not ln.startswith("#")}
    weak = set(WEAK)
    lf = os.path.join(DATA, "compiled", "manual_check", "tg_langs.txt")
    if os.path.exists(lf):
        weak |= {ln.split()[0] for ln in open(lf) if ln.strip() and not ln.startswith("#")}
    langs = {x for x in args.langs.split(",") if x}
    if langs and not args.tag:
        sys.exit("--langs needs --tag")
    name = args.tag or (str(args.round) if args.round else "")
    out_dir = os.path.join(OUT, "round" + name) if name else OUT
    prefix = "tg" + name

    rows = pq.read_table(os.path.join(DATA, "compiled", "prompts.parquet"),
                         columns=["id", "lang", "fill", "source_uid", "source_lang", "context",
                                  "if_kwargs", "translator"]).to_pylist()
    rows = [r for r in rows if r["fill"] == "translated"]
    if langs:
        weak = langs
        rows = [r for r in rows if r["lang"] in langs and r["translator"] == "google/gemma-4-31B-it"]
    if args.round:
        alloc = pq.read_table(os.path.join(DATA, "compiled", "allocation.parquet"),
                              columns=["slot", "uid", "round"]).to_pylist()
        refilled = {(a["slot"], a["uid"]) for a in alloc if a["round"] == args.round}
        rows = [r for r in rows if (int(r["id"].split("-")[1]), r["source_uid"]) in refilled
                and r["lang"] != "ltg"]
    if name:   # never re-request a (slot, uid) another TranslateGemma run already covers
        done = set()
        for d in [OUT] + sorted(glob.glob(os.path.join(OUT, "round*"))):
            sp = os.path.join(d, "segments.parquet")
            if os.path.exists(sp) and os.path.abspath(d) != os.path.abspath(out_dir) \
                    and pq.ParquetFile(sp).metadata.num_rows > 0:          # an empty round has no columns
                done |= {(s["slot"], s["uid"]) for s in pq.read_table(sp, columns=["slot", "uid", "group"]).to_pylist()
                         if s["group"].startswith("tg")}
        n0 = len(rows)
        rows = [r for r in rows if (int(r["id"].split("-")[1]), r["source_uid"]) not in done]
        print("skipping {} rows already covered by earlier TranslateGemma runs".format(n0 - len(rows)))
    need = {r["source_uid"] for r in rows}
    src = {}
    for b in pq.ParquetFile(os.path.join(DATA, "pool", "candidates_ctx.parquet")).iter_batches(
            batch_size=20000, columns=["uid", "context"]):
        for u, c in zip(b.column("uid").to_pylist(), b.column("context").to_pylist()):
            if u in need:
                src[u] = json.loads(c)

    why = collections.Counter()
    skipped = collections.Counter()
    reqs, segs, chosen = [], [], {}
    for r in rows:
        slot = int(r["id"].split("-")[1])
        sctx = src[r["source_uid"]]
        if r["lang"] == "ltg":
            group, tgt = "lv", "lv"
            why["ltg_pivot"] += 1
        else:
            src_text = "\n".join(m["content"] or "" for m in sctx)
            tr_text = "\n".join(m["content"] or "" for m in r["context"])
            reasons = [x for x, ok in (("weak_lang", r["lang"] in weak),
                                       ("injection", bool(M.injected(src_text, tr_text))),
                                       ("flagged", r["id"] in flagged)) if ok]
            if not reasons:
                continue
            if r["if_kwargs"]:
                skipped["if_params"] += 1
                continue
            if any(M.codeish(m["content"]) for m in sctx):
                skipped["code_heavy"] += 1
                continue
            group, tgt = prefix, r["lang"]
            for x in reasons:
                why[x] += 1
            chosen[r["id"]] = {"lang": r["lang"], "reasons": reasons}
        for mi, m in enumerate(sctx):
            for pi, (kind, s) in enumerate(M.segments(m["content"] or "")):
                rid = "{}::{}::{}::{}".format(group, slot, mi, pi) if kind == "text" else None
                segs.append({"group": group, "slot": slot, "id": r["id"], "uid": r["source_uid"],
                             "msg": mi, "piece": pi,
                             "role": m["role"], "kind": kind, "content": s, "rid": rid,
                             "src_lang": r["source_lang"], "tgt_lang": r["lang"]})
                if rid:
                    reqs.append({"id": rid, "prompt": None, "stop": None, "schema": None,
                                 "messages": json.dumps(M.tg_messages(s, r["source_lang"], tgt),
                                                        ensure_ascii=False),
                                 "max_tokens": int(min(6144, len(s) * 1.0 + 128))})
    if not name:
        eng = M.flores("devtest", "en")
        for fid in M.eval_ids():
            reqs.append({"id": "flv::{}".format(fid), "prompt": None, "stop": None, "schema": None,
                         "messages": json.dumps(M.tg_messages(eng[fid], "en", "lv"), ensure_ascii=False),
                         "max_tokens": 512})
    os.makedirs(out_dir, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(reqs), os.path.join(out_dir, "requests.parquet"))
    pq.write_table(pa.Table.from_pylist(segs), os.path.join(out_dir, "segments.parquet"))
    with open(os.path.join(out_dir, "selection.json"), "w") as f:
        json.dump({"reasons": dict(why), "skipped": dict(skipped), "rows": chosen}, f, indent=1)
    n_rows = collections.Counter(g for g in (s["group"] for s in segs))
    print("rows selected for re-translation:", len(chosen), "| reasons:", dict(why),
          "| kept with Gemma:", dict(skipped))
    print("by lang:", dict(collections.Counter(v["lang"] for v in chosen.values()).most_common()))
    print("requests:", len(reqs), "| segment rows by group:", dict(n_rows))


if __name__ == "__main__":
    main()
