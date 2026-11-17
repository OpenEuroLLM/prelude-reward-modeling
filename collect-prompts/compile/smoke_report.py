#!/usr/bin/env python3
"""Summarize an LLM run (annotation + translation) for manual review.

  ./run.sh compile/smoke_report.py [--dir data/llm/smoke] [--examples 12]

Annotation: JSON validity, label distributions, examples (gist + labels).
Translation: validity, OpenLID-v3 agreement with the target language,
length ratio target/source, examples.
"""
import argparse
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow.parquet as pq  # noqa: E402

import annotate_prompt as A  # noqa: E402
import build_pool as B  # noqa: E402  (LID helpers)
import translate_prompt as T  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(ROOT, "data", "llm", "smoke"))
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()

    reqs = {r["id"]: r for r in pq.read_table(os.path.join(args.dir, "requests.parquet")).to_pylist()}
    meta = {r["id"]: r for r in pq.read_table(os.path.join(args.dir, "requests_meta.parquet")).to_pylist()}
    outs = {}
    for p in sorted(glob.glob(os.path.join(args.dir, "out", "shard*.parquet"))):
        for r in pq.read_table(p).to_pylist():
            outs[r["id"]] = r
    print("requests {}, outputs {}".format(len(reqs), len(outs)))
    fr = collections.Counter(o["finish_reason"] for o in outs.values())
    print("finish reasons:", dict(fr))
    toks = collections.defaultdict(list)
    for i, o in outs.items():
        toks[meta[i]["kind"]].append((o["prompt_tokens"], o["completion_tokens"]))
    for k, v in toks.items():
        print("{}: mean prompt {:.0f} tok, mean completion {:.0f} tok".format(
            k, sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v)))

    # ---------------- annotation
    ann = [(i, o) for i, o in outs.items() if meta[i]["kind"] == "annotate"]
    parsed = [(i, A.parse(o["text"])) for i, o in ann]
    ok = [(i, d) for i, d in parsed if d]
    print("\n== ANNOTATION: {} / {} valid JSON".format(len(ok), len(ann)))
    for field in ("task", "domain", "safety", "locale", "complexity", "response_language"):
        c = collections.Counter(str(d.get(field)) for _, d in ok)
        print("  {:<18} {}".format(field, dict(c.most_common(12))))
    for field in ("constraints", "flags"):
        c = collections.Counter(x for _, d in ok for x in d.get(field) or [])
        print("  {:<18} {}".format(field, dict(c.most_common(12))))
    by_src = collections.defaultdict(list)
    for i, d in ok:
        by_src[meta[i]["source"]].append((i, d))
    print("\n-- examples (prompt head | labels)")
    shown = 0
    for src, items in by_src.items():
        for i, d in items[:2]:
            ctx = json.loads(reqs[i]["messages"])[1]["content"]
            last = ctx.split("[user]: ")[-1][:160].replace("\n", " ")
            print("[{}] {}\n    -> {} | {} / {} / {} | safety={} locale={} c={} lang={} cons={} flags={}".format(
                src.split("/")[-1], last, d.get("gist"), d.get("task"), d.get("secondary_task"),
                d.get("domain"), d.get("safety"), d.get("locale"), d.get("complexity"),
                d.get("response_language"), d.get("constraints"), d.get("flags")))
            shown += 1
        if shown >= args.examples * 2:
            break

    # ---------------- translation
    tr = [(i, o) for i, o in outs.items() if meta[i]["kind"] == "translate"]
    if not tr:
        return
    B.load_model(B.ensure_model(), B.label_map())
    per_lang = collections.defaultdict(lambda: collections.Counter())
    ratios = collections.defaultdict(list)
    examples = collections.defaultdict(list)
    for i, o in tr:
        m = meta[i]
        src_msgs = json.loads(json.loads(reqs[i]["messages"])[1]["content"]
                              .split("Input messages (JSON):\n", 1)[1]
                              .rsplit("\n\nReturn", 1)[0])
        t = T.parse(o["text"], m["n_messages"])
        lang = m["tgt_lang"]
        if t is None:
            per_lang[lang]["invalid"] += 1
            continue
        users = [x for x, s in zip(t, src_msgs) if s["role"] == "user"]
        code, p, _, _ = B.lid(B.clean_for_lid(users[-1], 1000))
        per_lang[lang]["lid_ok" if code == lang else "lid_other:{}".format(code)] += 1
        src_len = sum(len(s["content"]) for s in src_msgs)
        ratios[lang].append(sum(len(x) for x in t) / max(src_len, 1))
        examples[lang].append((src_msgs[-1]["content"][:200], users[-1][:200], code, p))
    print("\n== TRANSLATION")
    for lang in T.LANG_NAMES:
        if lang in per_lang:
            r = ratios[lang]
            print("  {:<4} {}  len-ratio mean {:.2f} (min {:.2f}, max {:.2f})".format(
                lang, dict(per_lang[lang]), sum(r) / max(len(r), 1),
                min(r) if r else 0, max(r) if r else 0))
    print("\n-- examples")
    for lang, ex in examples.items():
        for s, t, code, p in ex[:2]:
            print("[{} lid={} {:.2f}] EN: {}\n      {}: {}".format(
                lang, code, p, s.replace("\n", " "), lang, t.replace("\n", " ")))


if __name__ == "__main__":
    main()
