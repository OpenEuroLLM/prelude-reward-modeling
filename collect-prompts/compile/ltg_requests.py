#!/usr/bin/env python3
"""Stage 5d: Latgalian candidate systems -- requests for FLORES+ and the ltg rows.

  ./run.sh compile/ltg_requests.py      (after the TranslateGemma pass: pivots
                                         use its Latvian outputs)

Systems (Gemma = Gemma-4-31B-it chat, Tilde = TildeOpen-30b base, few-shot
exemplars from FLORES+ dev, evaluation on 300 FLORES+ devtest sentences):
  G0   Gemma, original pipeline prompt (zero-shot, JSON)      en  -> ltg
  G1   Gemma, guide + 8-shot                                  src -> ltg
  G3   Gemma, guide + 24-shot                                 src -> ltg
  G2   Gemma, guide + 8-shot from gold Latvian (upper bound)  lv  -> ltg
  G2p  Gemma, guide + 8-shot from TranslateGemma Latvian      lv  -> ltg (pivot)
  T1/T3   Tilde 8/24-shot                                     src -> ltg
  T2/T4   Tilde 8/24-shot from gold Latvian (upper bound)     lv  -> ltg
  T2p/T4p Tilde 8/24-shot from TranslateGemma Latvian         lv  -> ltg (pivot)
The ltg rows of the compiled set get every system that can run on them
(G1 G3 G2p T1 T3 T2p T4p), per text piece of each message (layout from
data/llm/translate_tg/segments.parquet), so the FLORES winner's output exists.
Outputs: data/ltg/eval/{gemma,tilde}/requests.parquet
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
import translate_prompt as T  # noqa: E402

EVAL = os.path.join(M.DATA, "ltg", "eval")
TG = os.path.join(M.DATA, "llm", "translate_tg")
ITEM_SYSTEMS = ["G1", "G3", "G2p", "T1", "T3", "T2p", "T4p"]


def gemma_fewshot(src_lang, text, k):
    S, L = M.flores("dev", src_lang), M.flores("dev", "ltg")
    msgs = [{"role": "system", "content": M.LTG_GUIDE + " Translate the user's {} text into "
             "Latgalian. Output only the Latgalian translation.".format(M.NAMES[src_lang])}]
    for e in M.exemplar_ids()[:k]:
        msgs += [{"role": "user", "content": S[e]}, {"role": "assistant", "content": L[e]}]
    return msgs + [{"role": "user", "content": text}]


def tilde_prompt(src_lang, text, k):
    S, L, name = M.flores("dev", src_lang), M.flores("dev", "ltg"), M.NAMES[src_lang]
    head = ("{} texts with their translations into Latgalian (latgalīšu volūda), the written "
            "language of Latgale in eastern Latvia.\n\n").format(name)
    ex = "".join("<{0}>\n{1}\n</{0}>\n<Latgalian>\n{2}\n</Latgalian>\n\n".format(name, S[e], L[e])
                 for e in M.exemplar_ids()[:k])
    return head + ex + "<{0}>\n{1}\n</{0}>\n<Latgalian>\n".format(name, text.strip())


def req(rid, system, src_lang, text, max_tokens):
    """One request for `system` translating `text` (in src_lang) into Latgalian."""
    k = 24 if system in ("G3", "T3", "T4", "T4p") else 8
    if system == "G0":
        return {"id": rid, "messages": json.dumps(T.messages([{"role": "user", "content": text}],
                                                             src_lang, "ltg"), ensure_ascii=False),
                "prompt": None, "prompt_ids": None, "stop": None, "schema": json.dumps(T.schema(1)),
                "max_tokens": max_tokens}
    if system.startswith("G"):
        return {"id": rid, "messages": json.dumps(gemma_fewshot(src_lang, text, k), ensure_ascii=False),
                "prompt": None, "prompt_ids": None, "stop": None, "schema": None,
                "max_tokens": max_tokens}
    prompt = tilde_prompt(src_lang, text, k)
    return {"id": rid, "messages": None, "prompt": prompt, "prompt_ids": M.tilde_encode(prompt),
            "stop": json.dumps(["</Latgalian>"]), "schema": None, "max_tokens": max_tokens}


def tg_outputs():
    out = {}
    for p in glob.glob(os.path.join(TG, "out", "shard*.parquet")):
        for r in pq.read_table(p).to_pylist():
            if r["finish_reason"] == "stop" and r["text"].strip():
                out[r["id"]] = r["text"].strip()
    return out


def main():
    tgo = tg_outputs()
    eng, lvs = M.flores("devtest", "en"), M.flores("devtest", "lv")
    reqs = {"gemma": [], "tilde": []}
    n_pivot_missing = 0
    for fid in M.eval_ids():
        for s in ["G0", "G1", "G3", "T1", "T3"]:
            reqs["gemma" if s[0] == "G" else "tilde"].append(req("{}::f::{}".format(s, fid), s, "en", eng[fid], 400))
        for s in ["G2", "T2", "T4"]:
            reqs["gemma" if s[0] == "G" else "tilde"].append(req("{}::f::{}".format(s, fid), s, "lv", lvs[fid], 400))
        lv = tgo.get("flv::{}".format(fid))
        if lv is None:
            n_pivot_missing += 1
            continue
        for s in ["G2p", "T2p", "T4p"]:
            reqs["gemma" if s[0] == "G" else "tilde"].append(req("{}::f::{}".format(s, fid), s, "lv", lv, 400))

    segs = [s for s in pq.read_table(os.path.join(TG, "segments.parquet")).to_pylist()
            if s["group"] == "lv" and s["kind"] == "text"]
    n_items = len({s["slot"] for s in segs})
    for s in segs:
        mt = int(min(6144, len(s["content"]) * 0.8 + 128))
        tail = "i::{}::{}::{}".format(s["slot"], s["msg"], s["piece"])
        for system in ITEM_SYSTEMS:
            pivot = system.endswith("p")
            if pivot:
                text = tgo.get(s["rid"])
                if text is None:
                    n_pivot_missing += 1
                    continue
                src_lang = "lv"
            else:
                text, src_lang = s["content"], s["src_lang"]
                if src_lang not in M.FLORES_CODE:
                    continue
            reqs["gemma" if system[0] == "G" else "tilde"].append(
                req("{}::{}".format(system, tail), system, src_lang, text, mt))
    for name, rs in reqs.items():
        os.makedirs(os.path.join(EVAL, name), exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rs), os.path.join(EVAL, name, "requests.parquet"))
        print(name, len(rs), "requests")
    print("ltg rows:", n_items, "| text pieces:", len(segs), "| pivot inputs missing:", n_pivot_missing)


if __name__ == "__main__":
    main()
