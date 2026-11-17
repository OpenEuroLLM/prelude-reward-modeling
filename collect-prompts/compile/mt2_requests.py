#!/usr/bin/env python3
"""Stage 5m: text-level translation with TildeOpen-30b (8-shot) and SalamandraTA-7b -- requests.

  ./run.sh compile/mt2_requests.py --validate [--n 8]
  ./run.sh compile/mt2_requests.py --langs lt:tilde,et:tilde,sl:tilde,hu:tilde,is:salta,ga:salta --tag P1 [--round N]

Translators chosen by benchmark 2 (stage 5l) for the languages where Gemma-4 and
TranslateGemma read badly (manual_check/quality_review2/FINDINGS.md):
  tilde::  TildeOpen-30b, base model, 8-shot: 6 FLORES+ devtest sentence pairs plus two
           multi-line pairs (a bullet list, two paragraphs) built from FLORES sentences so
           line structure is kept; SentencePiece prompt ids (see mt_common), token ids saved.
  salta::  SalamandraTA-7b-instruct, chat, the prompt from its model card.
Per message, prose is split into paragraph chunks of <= 1000 characters, code blocks and
letterless pieces kept verbatim (mt_common.segments). SalamandraTA translates sentence/line by
line (given several lines it drops all but the first: 14 of 26 multi-line segments in the first
validation run), so for it each chunk is further split into lines, with newlines and leading
list/heading markers kept verbatim. segments.parquet records each system's layout so outputs
reassemble exactly (mt2_collect.py). Request ids: <system>::<row id>::<msg>::<piece>.

--validate: per language (lv lt et hu sl is mt ga eu) --n single-turn rows of the set that we
translated from English (general/IF purpose, safety none, no IF verifier params, no code-heavy
prose, 80-1200 characters, not read in an earlier review; eu: 1.5 x n, it has one candidate
system), translated by every system whose model card lists the language. Their current
translation (Gemma-4 or TranslateGemma) is the third arm of the blind read.
Outputs: data/llm/mtval/{tilde,salta}/requests.parquet, data/llm/mtval/segments.parquet, sample.json

--langs L:system,... --tag X (production, chosen per language by the validation read,
manual_check/mt_validation/FINDINGS.md): every translated row of the set in these languages,
except rows with IF verifier params or code-heavy prose (they keep their conversation-level
Gemma translation, as in stage 5c), whole context (all messages) translated by the language's
system. --round N: only the slots refilled in round N. Outputs under
data/llm/translate_mt2/run<X>/ (assembly reads every run; a later run replaces an earlier one
per (slot, uid)).
"""
import argparse
import collections
import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import mt_common as M  # noqa: E402

NAME = {"en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
        "pt": "Portuguese", "pl": "Polish", "nl": "Dutch", "lv": "Latvian", "lt": "Lithuanian",
        "et": "Estonian", "fi": "Finnish", "hu": "Hungarian", "cs": "Czech", "sk": "Slovak",
        "sl": "Slovenian", "hr": "Croatian", "bs": "Bosnian", "sr": "Serbian", "mk": "Macedonian",
        "bg": "Bulgarian", "uk": "Ukrainian", "ro": "Romanian", "el": "Greek", "tr": "Turkish",
        "sq": "Albanian", "da": "Danish", "sv": "Swedish", "nb": "Norwegian Bokmål",
        "nn": "Norwegian Nynorsk", "is": "Icelandic", "ga": "Irish", "mt": "Maltese",
        "eu": "Basque", "gl": "Galician", "ca": "Catalan", "ka": "Georgian"}
# languages each model card lists (targets we would use it for)
LANGS = {"tilde": set("lv et lt mt is ga sl hu hr cs sk mk bs sr sq bg uk ro".split()),
         "salta": set("lv et lt mt eu is ga sl hu hr gl cs el sk bg ca da de es fi fr it nl pl pt ro sv uk".split())}
VALIDATE = ["lv", "lt", "et", "hu", "sl", "is", "mt", "ga", "eu"]
MAX_CHARS = 1000
EXEMPLARS = [298, 474, 483, 163, 258, 896, 57, 114]   # benchmark 2 (not in its eval sample)
OUT = os.path.join(M.DATA, "llm", "mtval")
PROD = os.path.join(M.DATA, "llm", "translate_mt2")
LINE_LEVEL = {"salta"}
MARKER = re.compile(r"^(\s*(?:[-*\u2022>]|\d{1,3}[.)]|[A-Za-z][.)]|\([A-Za-z0-9]{1,3}\)|#{1,6})\s+)")


def line_pieces(chunk):
    """Split a prose chunk into lines; newlines and leading list/heading markers verbatim."""
    out = []
    for part in re.split(r"(\n+)", chunk):
        if not part:
            continue
        if part.startswith("\n"):
            out.append(("ws", part))
            continue
        m = MARKER.match(part)
        if m:
            out.append(("keep", m.group(1)))
            part = part[m.end():]
        body = part.strip()
        lead, trail = part[:len(part) - len(part.lstrip())], part[len(part.rstrip()):] if body else ""
        if lead and body:
            out.append(("ws", lead))
        if body:
            out.append(("text" if M.LETTER.search(body) else "keep", body))
        elif part:
            out.append(("ws", part))
        if trail:
            out.append(("ws", trail))
    assert "".join(x for _, x in out) == chunk, (chunk, out)
    return out


def layout(text, system):
    pieces = M.segments(text or "", MAX_CHARS)
    if system not in LINE_LEVEL:
        return pieces
    return [q for k, x in pieces for q in (line_pieces(x) if k == "text" else [(k, x)])]


def tilde_shots(src, tgt):
    """8 exemplar pairs: 6 sentences, one 3-item bullet list, one two-paragraph text."""
    S, T = M.flores("devtest", src), M.flores("devtest", tgt)
    pairs = [(S[i], T[i]) for i in EXEMPLARS[:6]]
    extra = sorted(set(S) - set(M.eval_ids(200, seed=20261110)) - set(EXEMPLARS))
    rng = random.Random(9)
    b, p = rng.sample(extra, 3), rng.sample(extra, 2)
    pairs.append(("\n".join("- " + S[i] for i in b), "\n".join("- " + T[i] for i in b)))
    pairs.append(("\n\n".join(S[i] for i in p), "\n\n".join(T[i] for i in p)))
    return pairs


def tilde_prompt(text, src, tgt, shots):
    s, t = NAME[src], NAME[tgt]
    head = "{} texts with their translations into {}.\n\n".format(s, t)
    ex = "".join("<{0}>\n{2}\n</{0}>\n<{1}>\n{3}\n</{1}>\n\n".format(s, t, a, b) for a, b in shots)
    return head + ex + "<{0}>\n{2}\n</{0}>\n<{1}>\n".format(s, t, text.strip())


def salta_messages(text, src, tgt):
    s, t = NAME[src], NAME[tgt]
    return [{"role": "user", "content": "Translate the following text from {0} into {1}.\n{0}: {2} \n{1}:".format(s, t, text)}]


def build(rows, systems, out_dir):
    """rows: dicts with id, uid, lang, src_lang, sctx (source context). Writes requests per
    system and one segments file (each system's own layout); returns request counts."""
    shots, reqs, segs = {}, collections.defaultdict(list), []
    for r in rows:
        for sysname in systems:
            if r["lang"] not in LANGS[sysname] or (r.get("system") and r["system"] != sysname):
                continue
            for mi, m in enumerate(r["sctx"]):
                for pi, (kind, s) in enumerate(layout(m["content"], sysname)):
                    segs.append({"system": sysname, "id": r["id"], "slot": int(r["id"].split("-")[1]),
                                 "uid": r["uid"], "lang": r["lang"],
                                 "src_lang": r["src_lang"], "msg": mi, "role": m["role"], "piece": pi,
                                 "kind": kind, "content": s})
                    if kind != "text":
                        continue
                    rid = "{}::{}::{}::{}".format(sysname, r["id"], mi, pi)
                    mt = int(min(6144, len(s) + 128))
                    if sysname == "tilde":
                        key = (r["src_lang"], r["lang"])
                        if key not in shots:
                            shots[key] = tilde_shots(*key)
                        p = tilde_prompt(s, r["src_lang"], r["lang"], shots[key])
                        reqs[sysname].append({"id": rid, "messages": None, "prompt": p, "prompt_ids": M.tilde_encode(p),
                                              "stop": json.dumps(["</{}>".format(NAME[r["lang"]])]),
                                              "schema": None, "max_tokens": mt})
                    else:
                        reqs[sysname].append({"id": rid, "messages": json.dumps(salta_messages(s, r["src_lang"], r["lang"]),
                                                                                ensure_ascii=False),
                                              "prompt": None, "prompt_ids": None, "stop": None, "schema": None,
                                              "max_tokens": mt})
    for sysname, rs in reqs.items():
        os.makedirs(os.path.join(out_dir, sysname), exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rs), os.path.join(out_dir, sysname, "requests.parquet"))
    os.makedirs(out_dir, exist_ok=True)
    if segs:    # an empty round (every refill native) writes no segments; assembly skips such runs
        pq.write_table(pa.Table.from_pylist(segs), os.path.join(out_dir, "segments.parquet"))
    return {k: len(v) for k, v in reqs.items()}


def reviewed_ids():
    seen = set()
    for f in glob.glob(os.path.join(M.DATA, "compiled", "manual_check", "**", "*.txt"), recursive=True):
        seen |= set(re.findall(r"mlp128k-\d{6}", open(f, encoding="utf-8").read()))
    for f in glob.glob(os.path.join(M.DATA, "compiled", "manual_check", "**", "*.jsonl"), recursive=True):
        seen |= set(re.findall(r"mlp128k-\d{6}", open(f, encoding="utf-8").read()))
    return seen


def validation_rows(n):
    seen = reviewed_ids()
    rows = [r for r in pq.read_table(os.path.join(M.DATA, "compiled", "prompts.parquet"), columns=[
        "id", "lang", "fill", "source_uid", "source_lang", "purpose", "safety", "if_kwargs", "context",
        "translator"]).to_pylist()
        if r["fill"] == "translated" and r["lang"] in VALIDATE and r["source_lang"] == "en"
        and r["purpose"] in ("general", "if") and r["safety"] == "none" and not r["if_kwargs"]
        and r["id"] not in seen and len(r["context"]) == 1 and r["context"][0]["role"] == "user"]
    need = {r["source_uid"] for r in rows}
    src = {}
    for b in pq.ParquetFile(os.path.join(M.DATA, "pool", "candidates_ctx.parquet")).iter_batches(
            batch_size=20000, columns=["uid", "context"]):
        for u, c in zip(b.column("uid").to_pylist(), b.column("context").to_pylist()):
            if u in need:
                src[u] = json.loads(c)
    ok = collections.defaultdict(list)
    for r in rows:
        s = src.get(r["source_uid"])
        if not s or len(s) != 1 or s[0]["role"] != "user":
            continue
        t = s[0]["content"] or ""
        if not 80 <= len(t) <= 1200 or M.codeish(t) or "```" in t:
            continue
        ok[r["lang"]].append({"id": r["id"], "uid": r["source_uid"], "lang": r["lang"], "src_lang": "en",
                              "sctx": s, "current": r["context"][0]["content"], "translator": r["translator"]})
    rng = random.Random(1114)
    pick = []
    for lang in VALIDATE:
        k = int(n * 1.5) if lang == "eu" else n
        pool = sorted(ok[lang], key=lambda x: x["id"])
        pick += rng.sample(pool, min(k, len(pool)))
        print("{}: {} eligible, {} picked".format(lang, len(pool), min(k, len(pool))))
    return pick


def load_sources(uids):
    src = {}
    for b in pq.ParquetFile(os.path.join(M.DATA, "pool", "candidates_ctx.parquet")).iter_batches(
            batch_size=20000, columns=["uid", "context"]):
        for u, c in zip(b.column("uid").to_pylist(), b.column("context").to_pylist()):
            if u in uids:
                src[u] = json.loads(c)
    return src


def production_rows(lang_sys, round_n):
    rows = [r for r in pq.read_table(os.path.join(M.DATA, "compiled", "prompts.parquet"), columns=[
        "id", "lang", "fill", "source_uid", "source_lang", "if_kwargs", "translator"]).to_pylist()
        if r["fill"] == "translated" and r["lang"] in lang_sys]
    if round_n:
        alloc = pq.read_table(os.path.join(M.DATA, "compiled", "allocation.parquet"),
                              columns=["slot", "uid", "round"]).to_pylist()
        refilled = {(a["slot"], a["uid"]) for a in alloc if a["round"] == round_n}
        rows = [r for r in rows if (int(r["id"].split("-")[1]), r["source_uid"]) in refilled]
    src = load_sources({r["source_uid"] for r in rows})
    out, skipped = [], collections.Counter()
    for r in rows:
        s = src[r["source_uid"]]
        if r["if_kwargs"]:
            skipped["if_params"] += 1
            continue
        if any(M.codeish(m["content"]) for m in s):
            skipped["code_heavy"] += 1
            continue
        out.append({"id": r["id"], "uid": r["source_uid"], "lang": r["lang"], "src_lang": r["source_lang"],
                    "sctx": s, "system": lang_sys[r["lang"]]})
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--langs", default="", help="L:system,... (production)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--round", type=int, default=0)
    args = ap.parse_args()
    if not args.validate:
        lang_sys = dict(x.split(":") for x in args.langs.split(",") if x)
        if not lang_sys or not args.tag:
            sys.exit("production needs --langs and --tag")
        bad = [l for l, s in lang_sys.items() if s not in LANGS or l not in LANGS[s]]
        if bad:
            sys.exit("system does not cover: {}".format(bad))
        rows, skipped = production_rows(lang_sys, args.round)
        missing = sorted({r["src_lang"] for r in rows} - set(NAME))
        if missing:
            sys.exit("no language name for source languages {}".format(missing))
        out_dir = os.path.join(PROD, "run" + args.tag)
        counts = build(rows, sorted(set(lang_sys.values())), out_dir)
        with open(os.path.join(out_dir, "selection.json"), "w") as f:
            json.dump({"langs": lang_sys, "round": args.round, "rows": len(rows), "skipped": dict(skipped),
                       "by_lang": dict(collections.Counter(r["lang"] for r in rows)),
                       "src_langs": dict(collections.Counter(r["src_lang"] for r in rows))}, f, indent=1)
        print("rows:", len(rows), dict(collections.Counter(r["lang"] for r in rows)), "| skipped:", dict(skipped),
              "| source langs:", dict(collections.Counter(r["src_lang"] for r in rows)), "| requests:", counts)
        return
    rows = validation_rows(args.n)
    counts = build(rows, ["tilde", "salta"], OUT)
    with open(os.path.join(OUT, "sample.json"), "w") as f:
        json.dump([{k: r[k] for k in ("id", "uid", "lang", "translator", "current")} | {"source": r["sctx"][0]["content"]}
                   for r in rows], f, ensure_ascii=False, indent=1)
    print("rows:", len(rows), "| requests:", counts)


if __name__ == "__main__":
    main()
