#!/usr/bin/env python3
"""Stage 5l: score the second translator benchmark (FLORES+ devtest, chrF++).

  ./run.sh compile/bench2_score.py

Per language: chrF++ of Gemma-4 and TranslateGemma (stage 5i outputs, same sentences)
and of each new candidate (bench2_requests.py); for the best candidate, the gain over
the better of Gemma/TranslateGemma with a paired bootstrap (1,000 resamples).
A system whose output is missing for a language is left blank.
Outputs: data/llm/bench2/report.md, report.json
"""
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow.parquet as pq  # noqa: E402
import sacrebleu  # noqa: E402

import mt_common as M  # noqa: E402
import translate_prompt as T  # noqa: E402
from bench2_requests import NAME  # noqa: E402

D1, D2 = os.path.join(M.DATA, "llm", "bench"), os.path.join(M.DATA, "llm", "bench2")
CHRF = sacrebleu.metrics.CHRF(word_order=2)
NEW = ["eurollm", "salta", "tilde", "tower"]
LABEL = {"gm": "Gemma-4", "tg": "TranslateGemma", "eurollm": "EuroLLM-22B", "salta": "SalamandraTA-7B",
         "tilde": "TildeOpen-30B", "tower": "Tower+-72B"}


def outputs(d):
    out = {}
    for p in glob.glob(os.path.join(d, "out", "shard*.parquet")):
        for r in pq.read_table(p).to_pylist():
            out[r["id"]] = r
    return out


def text(system, r, lang):
    if r is None or r["finish_reason"] != "stop":
        return ""
    t = r["text"]
    if system == "gm":
        p = T.parse(t, 1)
        return p[0].strip() if p else ""
    if system == "tilde" and r.get("token_ids") is not None:
        t = M.tilde_sp().decode([int(x) for x in r["token_ids"]])
        t = t.split("</{}>".format(NAME[lang]))[0].split("\n<")[0]
    return t.strip()


def stats(hyp, ref):
    return [list(x) for x in CHRF._extract_corpus_statistics(hyp, [ref])]


def main():
    ids = json.load(open(os.path.join(D2, "exemplars.json")))["eval_ids"]
    outs = {"gm": outputs(os.path.join(D1, "gemma")), "tg": outputs(os.path.join(D1, "tg"))}
    outs.update({s: outputs(os.path.join(D2, s)) for s in NEW})
    rng = random.Random(0)
    rows = []
    for lang in NAME:
        ref = M.flores("devtest", lang)
        R = [ref[i] for i in ids]
        row = {"lang": lang}
        hyps = {}
        for s, o in outs.items():
            got = [o.get("{}::{}::{}".format(s, lang, i)) for i in ids]
            if not any(got):
                continue
            hyps[s] = [text(s, g, lang) for g in got]
            row[s] = round(CHRF.corpus_score(hyps[s], [R]).score, 2)
            row[s + "_empty"] = sum(not h for h in hyps[s])
        old = max((s for s in ("gm", "tg") if s in row), key=lambda s: row[s], default=None)
        new = max((s for s in NEW if s in row), key=lambda s: row[s], default=None)
        if old and new:
            a, b = stats(hyps[old], R), stats(hyps[new], R)
            wins = 0
            for _ in range(1000):
                smp = [rng.randrange(len(ids)) for _ in ids]
                sa = [sum(a[j][k] for j in smp) for k in range(len(a[0]))]
                sb = [sum(b[j][k] for j in smp) for k in range(len(b[0]))]
                wins += CHRF._compute_score_from_stats(sb).score > CHRF._compute_score_from_stats(sa).score
            row.update(best_old=old, best_new=new, delta=round(row[new] - row[old], 2), p_new_better=wins / 1000)
        rows.append(row)
    cols = ["gm", "tg"] + NEW
    lines = ["# Translator benchmark 2: FLORES+ devtest, English -> X (chrF++, {} sentences)".format(len(ids)), "",
             "Gemma-4 / TranslateGemma from stage 5i (same sentences). Δ = best new candidate minus the better of "
             "Gemma/TranslateGemma; P = paired bootstrap P(new better). Blank = model card does not list the language.",
             "", "| lang | " + " | ".join(LABEL[c] for c in cols) + " | best new | Δ | P |",
             "|---|" + "---|" * (len(cols) + 3)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append("" if v is None else ("{}{}".format(v, " ({} empty)".format(r[c + "_empty"]) if r[c + "_empty"] else "")))
        lines.append("| {} | {} | {} | {} | {} |".format(
            r["lang"], " | ".join(cells), LABEL.get(r.get("best_new"), ""),
            "{:+.2f}".format(r["delta"]) if "delta" in r else "", r.get("p_new_better", "")))
    with open(os.path.join(D2, "report.json"), "w") as f:
        json.dump(rows, f, indent=1)
    with open(os.path.join(D2, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
