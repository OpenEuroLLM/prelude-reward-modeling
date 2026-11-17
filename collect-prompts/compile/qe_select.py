#!/usr/bin/env python3
"""Stage 5h: choose Gemma or TranslateGemma per re-translated row (round-trip QE).

  ./run.sh compile/qe_select.py [--margin 2.0] [--show id,id,...]

Per candidate and back-translator (TranslateGemma, Gemma-4): messages are
reassembled from the back-translated pieces and compared with the source
message (chrF++). Candidate score = mean over the two back-translators of
0.5 * length-weighted mean over messages + 0.5 * last user message (the prompt
the reward model answers). Round-trip chrF rewards words left untranslated
(they come back verbatim), so each candidate is penalised by --leftover x the
share of its words (>= 4 letters) that are copied from the source but
translated by the other candidate (names and terms both keep cancel out).
The old (Gemma) translation is kept only if its score is more than --margin
above TranslateGemma's: round-trip chrF measures adequacy, not fluency, which
is where TranslateGemma is better.
Output: data/llm/translate_tg/choice.parquet (slot, uid, choice, score_old, score_new)
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import sacrebleu  # noqa: E402

import mt_common as M  # noqa: E402

QE = os.path.join(M.DATA, "llm", "qe")
CHRF = sacrebleu.metrics.CHRF(word_order=2)
WORDS = re.compile(r"[^\W\d_]{4,}")


def words(text):
    return {w.lower() for w in WORDS.findall(M.FENCE.sub(" ", text or ""))}


def leftovers(src_text, cand_text, other_text):
    """Share of the candidate's words copied from the source but not by the other candidate."""
    s, c, o = words(src_text), words(cand_text), words(other_text)
    return len((c & s) - o) / max(1, len(c))


def outputs(d):
    out = {}
    for p in glob.glob(os.path.join(QE, d, "out", "shard*.parquet")):
        for r in pq.read_table(p).to_pylist():
            if r["finish_reason"] == "stop" and r["text"].strip():
                out[r["id"]] = r["text"].strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--margin", type=float, default=2.0)
    ap.add_argument("--leftover", type=float, default=100.0)
    ap.add_argument("--guard", type=float, default=None,
                    help="narrow mode: keep the old translation only if TranslateGemma's LAST USER TURN "
                         "scores at least this much below the old one's (gross meaning loss); "
                         "overall scores and leftovers are reported but not used")
    ap.add_argument("--show", default="")
    args = ap.parse_args()
    show = set(x for x in args.show.split(",") if x)
    bt = {"tg": outputs("tg"), "gm": outputs("gemma")}
    layout = collections.defaultdict(list)
    for x in pq.read_table(os.path.join(QE, "layout.parquet")).to_pylist():
        layout[(x["slot"], x["uid"], x["cand"])].append(x)
    keys = {(s, u) for s, u, _ in layout}
    need = {u for _, u in keys}
    src = {}
    for b in pq.ParquetFile(os.path.join(M.DATA, "pool", "candidates_ctx.parquet")).iter_batches(
            batch_size=20000, columns=["uid", "context"]):
        for u, c in zip(b.column("uid").to_pylist(), b.column("context").to_pylist()):
            if u in need:
                src[u] = json.loads(c)
    lang = {int(r["id"].split("-")[1]): r["lang"] for r in pq.read_table(
        os.path.join(M.DATA, "compiled", "prompts.parquet"), columns=["id", "lang"]).to_pylist()}

    def score(slot, uid, cand, last_only=False):
        sctx = src[uid]
        res = []
        for b, out in bt.items():
            msgs, ok = collections.defaultdict(str), True
            for x in sorted(layout[(slot, uid, cand)], key=lambda x: (x["msg"], x["piece"])):
                if x["kind"] == "text":
                    t = out.get("{}::{}".format(b, x["rid"]))
                    if t is None:
                        ok = False
                        break
                    msgs[x["msg"]] += t
                else:
                    msgs[x["msg"]] += x["content"]
            if not ok:
                continue
            per = [(len(m["content"] or ""), CHRF.sentence_score(msgs.get(i, ""), [m["content"] or ""]).score)
                   for i, m in enumerate(sctx) if (m["content"] or "").strip()]
            last = max(i for i, m in enumerate(sctx) if m["role"] == "user")
            last_s = CHRF.sentence_score(msgs.get(last, ""), [sctx[last]["content"] or ""]).score
            w = sum(n for n, _ in per) or 1
            res.append(last_s if last_only else 0.5 * sum(n * s for n, s in per) / w + 0.5 * last_s)
        return sum(res) / len(res) if res else None

    def text(slot, uid, cand):
        return "".join(x["content"] for x in sorted(layout[(slot, uid, cand)], key=lambda x: (x["msg"], x["piece"])))

    rows, stats = [], collections.Counter()
    by_lang = collections.defaultdict(collections.Counter)
    for slot, uid in sorted(keys):
        so, sn = score(slot, uid, "old"), score(slot, uid, "new")
        s_text = "\n".join(m["content"] or "" for m in src[uid])
        t_old, t_new = text(slot, uid, "old"), text(slot, uid, "new")
        lo, ln = leftovers(s_text, t_old, t_new), leftovers(s_text, t_new, t_old)
        lso, lsn = score(slot, uid, "old", True), score(slot, uid, "new", True)
        if args.guard is not None:
            gap = None if lso is None or lsn is None else lso - lsn
            choice = "old" if gap is not None and gap >= args.guard else "new"
        else:
            ao = None if so is None else so - args.leftover * lo
            an = None if sn is None else sn - args.leftover * ln
            choice = "old" if ao is not None and (an is None or ao > an + args.margin) else "new"
        stats[choice] += 1
        by_lang[lang.get(slot)][choice] += 1
        rows.append({"slot": slot, "uid": uid, "choice": choice, "score_old": so, "score_new": sn,
                     "last_old": lso, "last_new": lsn, "leftover_old": lo, "leftover_new": ln})
        rid = "mlp128k-{:06d}".format(slot)
        if rid in show:
            print("{} {}: overall old {} new {} | last turn old {} new {} | leftovers {:.3f}/{:.3f} -> {}".format(
                rid, lang.get(slot), *[None if x is None else round(x, 1) for x in (so, sn, lso, lsn)], lo, ln, choice))
    gaps = sorted(r["last_old"] - r["last_new"] for r in rows if r["last_old"] is not None and r["last_new"] is not None)
    for g in (10, 15, 20, 25, 30, 40):
        print("last-turn gap (old - new) >= {}: {} rows".format(g, sum(x >= g for x in gaps)))
    out_name = "choice.parquet" if args.guard is not None else "choice_overall.parquet"
    # only the narrow guard writes the file assemble.py reads; the overall-score mode is diagnostic
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(M.DATA, "llm", "translate_tg", out_name))
    diffs = sorted(r["score_new"] - r["score_old"] for r in rows if r["score_old"] is not None and r["score_new"] is not None)
    if diffs:
        q = lambda f: diffs[min(len(diffs) - 1, int(f * len(diffs)))]
        print("score_new - score_old: p10 {:.1f}, median {:.1f}, p90 {:.1f}".format(q(0.1), q(0.5), q(0.9)))
    print("choice:", dict(stats), "| margin", args.margin)
    print("by lang:", {k: dict(v) for k, v in sorted(by_lang.items(), key=lambda kv: str(kv[0]))})


if __name__ == "__main__":
    main()
