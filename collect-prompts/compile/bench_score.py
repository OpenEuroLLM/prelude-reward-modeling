#!/usr/bin/env python3
"""Stage 5j: score the translator benchmark and summarise the every_prompt judge.

  ./run.sh compile/bench_score.py [--min-delta 2.0] [--min-p 0.95]

FLORES+: chrF++ (corpus) per target language for Gemma-4 (pipeline prompt) and
TranslateGemma on the same sentences, with a paired bootstrap (1,000 resamples):
P(TranslateGemma > Gemma). A language is recommended for TranslateGemma if the
gain is >= --min-delta and P >= --min-p.
every_prompt: judge verdicts per row -> data/llm/bench/every_prompt_judge.parquet.
Outputs: data/llm/bench/report.md, report.json
"""
import argparse
import collections
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import sacrebleu  # noqa: E402

import mt_common as M  # noqa: E402
import translate_prompt as T  # noqa: E402

D = os.path.join(M.DATA, "llm", "bench")
CHRF = sacrebleu.metrics.CHRF(word_order=2)


def outputs(name):
    out = {}
    for p in glob.glob(os.path.join(D, name, "out", "shard*.parquet")):
        for r in pq.read_table(p).to_pylist():
            out[r["id"]] = r
    return out


def gm_text(r):
    if not r or r["finish_reason"] != "stop":
        return ""
    t = T.parse(r["text"], 1)
    return t[0].strip() if t else ""


def tg_text(r):
    return r["text"].strip() if r and r["finish_reason"] == "stop" else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-delta", type=float, default=2.0)
    ap.add_argument("--min-p", type=float, default=0.95)
    args = ap.parse_args()
    go, to = outputs("gemma"), outputs("tg")
    langs = sorted({k.split("::")[1] for k in to if k.startswith("tg::")})
    rng = random.Random(0)
    rows = []
    for lang in langs:
        ref = M.flores("devtest", lang)
        ids = sorted(int(k.split("::")[2]) for k in to if k.startswith("tg::{}::".format(lang)))
        R = [ref[i] for i in ids]
        G = [gm_text(go.get("gm::{}::{}".format(lang, i))) for i in ids]
        Tg = [tg_text(to.get("tg::{}::{}".format(lang, i))) for i in ids]
        g, t = CHRF.corpus_score(G, [R]).score, CHRF.corpus_score(Tg, [R]).score
        # paired bootstrap on per-sentence chrF statistics (corpus chrF = f(sum of stats))
        sg = [list(x) for x in CHRF._extract_corpus_statistics(G, [R])]
        st = [list(x) for x in CHRF._extract_corpus_statistics(Tg, [R])]
        wins = 0
        for _ in range(1000):
            s = [rng.randrange(len(ids)) for _ in ids]
            a = [sum(sg[j][k] for j in s) for k in range(len(sg[0]))]
            b = [sum(st[j][k] for j in s) for k in range(len(st[0]))]
            wins += CHRF._compute_score_from_stats(b).score > CHRF._compute_score_from_stats(a).score
        rows.append({"lang": lang, "n": len(ids), "gemma": round(g, 2), "translategemma": round(t, 2),
                     "delta": round(t - g, 2), "p_tg_better": wins / 1000,
                     "gemma_empty": sum(not x for x in G), "tg_empty": sum(not x for x in Tg),
                     "recommend_tg": (t - g) >= args.min_delta and wins / 1000 >= args.min_p})
    rows.sort(key=lambda r: -r["delta"])
    lines = ["# Translator benchmark: FLORES+ devtest (chrF++)", "",
             "Gemma-4-31B-it with the pipeline's translation prompt vs TranslateGemma-27B, same sentences "
             "(English source; French for English). Recommend TranslateGemma: gain >= {} and paired-bootstrap "
             "P >= {}.".format(args.min_delta, args.min_p), "",
             "| lang | Gemma | TranslateGemma | Δ | P(TG better) | empty G/T | TranslateGemma? |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {lang} | {gemma} | {translategemma} | {delta:+.2f} | {p_tg_better:.3f} | {gemma_empty}/{tg_empty} | {rec} |".format(
            rec="**yes**" if r["recommend_tg"] else "", **r))

    # every_prompt judge
    verdicts, bad = [], 0
    for k, r in go.items():
        if not k.startswith("ep::"):
            continue
        try:
            v = json.loads(r["text"]) if r["finish_reason"] == "stop" else None
        except Exception:
            v = None
        if not v:
            bad += 1
            continue
        verdicts.append({"id": k[4:], **{f: v.get(f) == "yes" for f in ("is_request", "self_contained", "spam")}})
    c = collections.Counter()
    for v in verdicts:
        c["spam"] += v["spam"]
        c["not_request"] += not v["is_request"]
        c["not_self_contained"] += not v["self_contained"]
        c["drop_any"] += v["spam"] or not v["is_request"] or not v["self_contained"]
    pq.write_table(pa.Table.from_pylist(verdicts), os.path.join(D, "every_prompt_judge.parquet"))
    lines += ["", "## every_prompt judge", "", "rows judged: {} (unparsable: {}) | {}".format(len(verdicts), bad, dict(c))]
    with open(os.path.join(D, "report.json"), "w") as f:
        json.dump({"flores": rows, "every_prompt": dict(c), "unparsable": bad}, f, indent=1)
    with open(os.path.join(D, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
