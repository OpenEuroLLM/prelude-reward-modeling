#!/usr/bin/env python3
"""Stage 5e: score the Latgalian candidate systems and reassemble the ltg rows.

  ./run.sh compile/ltg_score.py

FLORES+ (300 devtest sentences), per system:
  chrF++ vs the Latgalian reference (main metric) and vs the Latvian reference
  (a system that writes Latvian scores high there), and a Goldfish
  discriminator: bits/char under the Latvian Goldfish model minus under the
  Latgalian one (> 0 = more Latgalian-like), with the share of sentences > 0.
  Calibration rows: the two references, copying the gold Latvian, and the
  TranslateGemma Latvian (pivot leg 1).
ltg rows: each system's per-piece outputs are put back into the message layout
(code / whitespace verbatim); rows with a missing piece are incomplete.
Outputs: data/ltg/eval/report.md, report.json, items.parquet, items/<system>.txt
"""
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
import translate_prompt as T  # noqa: E402
from ltg_requests import ITEM_SYSTEMS  # noqa: E402

EVAL = os.path.join(M.DATA, "ltg", "eval")
TG = os.path.join(M.DATA, "llm", "translate_tg")
FLORES_SYSTEMS = ["G0", "G1", "G3", "G2", "G2p", "T1", "T3", "T2", "T4", "T2p", "T4p"]
DESC = {"G0": "Gemma, original zero-shot prompt", "G1": "Gemma, guide + 8-shot from the source language",
        "G3": "Gemma, guide + 24-shot from the source language", "G2": "Gemma, 8-shot from gold Latvian",
        "G2p": "Gemma, 8-shot via TranslateGemma Latvian", "T1": "TildeOpen 8-shot from the source language",
        "T3": "TildeOpen 24-shot from the source language", "T2": "TildeOpen 8-shot from gold Latvian",
        "T4": "TildeOpen 24-shot from gold Latvian", "T2p": "TildeOpen 8-shot via TranslateGemma Latvian",
        "T4p": "TildeOpen 24-shot via TranslateGemma Latvian"}


def outputs(d):
    out = {}
    for p in glob.glob(os.path.join(d, "out", "shard*.parquet")):
        for r in pq.read_table(p).to_pylist():
            out[r["id"]] = r
    return out


def clean(system, r):
    if r is None or r["finish_reason"] != "stop":
        return ""
    text = r["text"]
    if system.startswith("T") and r.get("token_ids") is not None:   # see mt_common.tilde_sp
        text = M.tilde_sp().decode([int(x) for x in r["token_ids"]])
    if system == "G0":
        t = T.parse(text, 1)
        return t[0].strip() if t else ""
    text = text.split("</Latgalian>")[0]
    if system.startswith("T"):
        text = text.split("\n<")[0]   # base model started another example
    return text.strip()


class Goldfish:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(16)
        self.torch, self.m = torch, {}
        for lang in ["ltg", "lvs"]:
            p = os.path.join(M.DATA, "ltg", "{}_latn_full".format(lang))
            tok = AutoTokenizer.from_pretrained(p)
            model = AutoModelForCausalLM.from_pretrained(p).eval()
            start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
            self.m[lang] = (tok, model, start)

    def bpc(self, lang, texts, bs=64):
        """Bits per character of each text (first 500 tokens) under one model; padded batches."""
        torch = self.torch
        tok, model, start = self.m[lang]
        res, items = [None] * len(texts), []
        for i, t in enumerate(texts):
            ids = tok(t, add_special_tokens=False)["input_ids"][:500] if t else []
            st = start
            if st is None:            # no BOS/EOS: condition on the first token instead
                st, ids = (ids[0], ids[1:]) if len(ids) > 1 else (None, [])
            if ids:
                items.append((i, [st] + ids, len(tok.decode(ids)) or 1))
        items.sort(key=lambda x: len(x[1]))
        pad = tok.pad_token_id if tok.pad_token_id is not None else 0
        for b in range(0, len(items), bs):
            chunk = items[b:b + bs]
            L = max(len(x[1]) for x in chunk)
            X = torch.full((len(chunk), L), pad, dtype=torch.long)
            A = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, (_, ids, _) in enumerate(chunk):
                X[j, :len(ids)] = torch.tensor(ids)
                A[j, :len(ids)] = 1
            with torch.no_grad():
                logp = torch.log_softmax(model(X, attention_mask=A).logits.float()[:, :-1], -1)
            nll = -logp.gather(-1, X[:, 1:].unsqueeze(-1)).squeeze(-1) * A[:, 1:]
            for j, (i, _, n_chars) in enumerate(chunk):
                res[i] = nll[j].sum().item() / n_chars / 0.6931471805599453
        return res

    def deltas(self, texts):
        a, b = self.bpc("lvs", texts), self.bpc("ltg", texts)
        return [None if x is None or y is None else x - y for x, y in zip(a, b)]


def corpus_stats(hyps, refs_ltg, refs_lvs, gf):
    import sacrebleu
    chrf = sacrebleu.metrics.CHRF(word_order=2)
    ok = [h for h in hyps if h]
    d = [x for x in gf.deltas(ok) if x is not None]
    return {"n": len(hyps), "empty": len(hyps) - len(ok),
            "chrf_ltg": round(chrf.corpus_score(hyps, [refs_ltg]).score, 2),
            "chrf_lvs": round(chrf.corpus_score(hyps, [refs_lvs]).score, 2),
            "dbpc": round(sum(d) / len(d), 3) if d else None,
            "ltg_like": round(sum(x > 0 for x in d) / len(d), 3) if d else None}


def main():
    gf = Goldfish()
    ids = M.eval_ids()
    ref_ltg, ref_lvs = M.flores("devtest", "ltg"), M.flores("devtest", "lv")
    R_ltg, R_lvs = [ref_ltg[i] for i in ids], [ref_lvs[i] for i in ids]
    go, to, tgo = outputs(os.path.join(EVAL, "gemma")), outputs(os.path.join(EVAL, "tilde")), outputs(TG)
    rows = {}
    rows["ref: Latgalian (gold)"] = corpus_stats(R_ltg, R_ltg, R_lvs, gf)
    rows["ref: Latvian (gold), i.e. copy Latvian"] = corpus_stats(R_lvs, R_ltg, R_lvs, gf)
    rows["TranslateGemma en->lv (pivot leg 1)"] = corpus_stats(
        [clean("TG", tgo.get("flv::{}".format(i))) for i in ids], R_ltg, R_lvs, gf)
    for s in FLORES_SYSTEMS:
        src = go if s.startswith("G") else to
        hyps = [clean(s, src.get("{}::f::{}".format(s, i))) for i in ids]
        rows["{} {}".format(s, DESC[s])] = corpus_stats(hyps, R_ltg, R_lvs, gf)

    lines = ["# Latgalian system selection (FLORES+ devtest, 300 sentences)", "",
             "chrF++ vs ltg = main metric. chrF++ vs lvs high + low vs ltg = writes Latvian. "
             "Δbpc = bits/char under the Latvian Goldfish LM minus under the Latgalian one "
             "(> 0: more Latgalian-like); ltg-like = share of sentences with Δbpc > 0.", "",
             "| system | chrF++ vs ltg | chrF++ vs lvs | Δbpc | ltg-like | empty |",
             "|---|---|---|---|---|---|"]
    for k, v in rows.items():
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            k, v["chrf_ltg"], v["chrf_lvs"], v["dbpc"], v["ltg_like"], v["empty"]))

    # ---- ltg rows: reassemble per system
    segs = [s for s in pq.read_table(os.path.join(TG, "segments.parquet")).to_pylist() if s["group"] == "lv"]
    by_row = collections.defaultdict(list)
    for s in segs:
        by_row[(s["slot"], s["id"])].append(s)
    items, os_dir = [], os.path.join(EVAL, "items")
    os.makedirs(os_dir, exist_ok=True)
    summary = {}
    for system in ITEM_SYSTEMS:
        src = go if system.startswith("G") else to
        built = []
        for (slot, rid_row), ss in sorted(by_row.items()):
            msgs = collections.OrderedDict()
            complete = True
            for s in sorted(ss, key=lambda x: (x["msg"], x["piece"])):
                m = msgs.setdefault(s["msg"], {"role": s["role"], "content": ""})
                if s["kind"] == "text":
                    out = clean(system, src.get("{}::i::{}::{}::{}".format(system, slot, s["msg"], s["piece"])))
                    if not out:
                        complete = False
                    m["content"] += out
                else:
                    m["content"] += s["content"]
            built.append((slot, rid_row, list(msgs.values()), complete))
        users = ["\n".join(m["content"] for m in ctx if m["role"] == "user") if complete else ""
                 for _, _, ctx, complete in built]
        n_ok, deltas, txt = 0, [], []
        for (slot, rid_row, ctx, complete), d in zip(built, gf.deltas(users)):
            d = d if complete else None
            n_ok += complete
            if d is not None:
                deltas.append(d)
            items.append({"system": system, "slot": slot, "id": rid_row, "complete": complete,
                          "dbpc": d, "context": json.dumps(ctx, ensure_ascii=False)})
            txt.append("=== {} | {} | complete={} | Δbpc={}\n".format(
                rid_row, system, complete, None if d is None else round(d, 3)) +
                "\n".join("[{}] {}".format(m["role"], m["content"]) for m in ctx))
        with open(os.path.join(os_dir, system + ".txt"), "w") as f:
            f.write("\n\n".join(txt) + "\n")
        summary[system] = {"complete": n_ok, "rows": len(by_row),
                           "mean_dbpc": round(sum(deltas) / len(deltas), 3) if deltas else None,
                           "ltg_like": round(sum(x > 0 for x in deltas) / len(deltas), 3) if deltas else None}
    lines += ["", "## The 35 ltg rows (per system)", "",
              "| system | complete | mean Δbpc (user text) | ltg-like |", "|---|---|---|---|"]
    for s, v in summary.items():
        lines.append("| {} {} | {}/{} | {} | {} |".format(s, DESC[s], v["complete"], v["rows"],
                                                         v["mean_dbpc"], v["ltg_like"]))
    pq.write_table(pa.Table.from_pylist(items), os.path.join(EVAL, "items.parquet"))
    with open(os.path.join(EVAL, "report.json"), "w") as f:
        json.dump({"flores": rows, "items": summary}, f, indent=1)
    with open(os.path.join(EVAL, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
