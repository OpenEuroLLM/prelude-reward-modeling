#!/usr/bin/env python3
"""Stage 6: validate translations and assemble the compiled prompt set.

  ./run.sh compile/assemble.py [--out data/compiled]
Inputs : data/compiled/allocation.parquet, data/pool/candidates.parquet,
         data/extracted/** (full records via pointers), data/pool/if_meta.parquet,
         data/llm/translate/{requests_meta.parquet,out/}
Outputs: data/compiled/prompts.parquet      one row per filled, validated slot
         data/compiled/failed_slots.parquet slots whose translation failed checks
         data/compiled/assemble_report.json

Translation checks (validate_context, any translator): per-message sanity (no
"role"/"translations" junk, no run-on strings, per-message length ratio 0.25-4);
no leaked JSON scaffolding (a made-up key such as 'role-user-2-content-translated-
placeholder-...": "' glued to a message by Gemma's JSON mode; Gemma translations
get the prefix stripped when the rest is clean, noted 'leaked_json_key_stripped');
no exotic-script injection (Hangul/Thai/Kana/Indic glyphs absent from the source);
OpenLID-v3 on the translated last user turn (target language, confusable groups
tolerated; Serbian must be Cyrillic); length ratio 0.4-3.0. Gemma translations
also: JSON shape; IF params present in the translated user turn (else the
verifier is dropped, the prompt kept).
Translators: text-level TildeOpen-30b / SalamandraTA-7b translations (data/llm/translate_mt2,
stage 5m, languages chosen by a blind read) replace the slot's translation when complete and
passing the checks plus M.literal_problems on the user turns (inline code, URLs, digits and
<< >> [ ] { } kept, no translator preamble); otherwise TranslateGemma-27B re-translations
(data/llm/translate_tg, stage 5c) replace the Gemma translation of a slot when they pass; Latgalian rows use only
the contexts selected in stage 5e (data/llm/ltg_final; LID cannot separate lv/ltg,
so they are checked there with a Goldfish lv/ltg discriminator) -- a Latgalian
slot without one fails. Slots in data/compiled/force_fail.parquet (wrong-language
rows from trusted labels) fail while they hold the listed uid.
Responses are kept only for native / source-translated items, in the item's
language, and only if reusable (compar:IA model_reuse != false).
"""
import argparse
import bisect
import collections
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import regex  # noqa: E402

import build_pool as B  # noqa: E402
import mt_common as M  # noqa: E402
import translate_prompt as T  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")
EXT = os.path.join(DATA, "extracted")
TRANSLATOR = "google/gemma-4-31B-it"
TG_TRANSLATOR = "google/translategemma-27b-it"
MT2_TRANSLATOR = {"tilde": "TildeAI/TildeOpen-30b", "salta": "BSC-LT/salamandraTA-7b-instruct"}
MT2_NOTE = {"tilde": "retranslated_tildeopen", "salta": "retranslated_salamandrata"}
CYR = regex.compile(r"\p{Cyrillic}")
LAT = regex.compile(r"\p{Latin}")
# Serbian Cyrillic alphabet; any other Cyrillic letter (ы э ъ ё й щ ь ѓ ќ ѕ ...)
# means Russian / Bulgarian / Macedonian / Ukrainian rather than Serbian
SR_ALPHABET = set("абвгдђежзијклљмнњопрстћуфхцчџш")
# OpenLID cannot reliably separate Bosnian / Croatian / Serbian-Latin on short texts
BCMS_LATIN = {"bs", "hr", "sr-Latn"}
GROUPS = [{"bs", "hr", "sr-Latn"}, {"sr", "mk"}, {"nb", "nn", "da", "sv"}, {"lv", "ltg"},
          {"cs", "sk"}]
JUNK = {"", "role", "user", "assistant", "system", "content", "translation", "translations"}


def fetch_records(cands):
    """uid -> full extracted record, via (part, row) pointers."""
    tasks = collections.defaultdict(list)
    for c in cands:
        tasks[c["part"]].append((c["row"], c["uid"]))
    out = {}
    for part, wanted in tasks.items():
        pf = pq.ParquetFile(os.path.join(EXT, part))
        starts, acc = [], 0
        for i in range(pf.num_row_groups):
            starts.append(acc)
            acc += pf.metadata.row_group(i).num_rows
        by_rg = collections.defaultdict(list)
        for row, uid in wanted:
            by_rg[bisect.bisect_right(starts, row) - 1].append((row, uid))
        for rg, ws in by_rg.items():
            sub = pf.read_row_group(rg).take([row - starts[rg] for row, _ in ws]).to_pylist()
            for (row, uid), r in zip(ws, sub):
                assert r["uid"] == uid, (part, row)
                out[uid] = r
    return out


def lang_check(text, tgt):
    clean = B.clean_for_lid(text, 1000)
    code, p, code2, _ = B.lid(clean)
    if tgt == "sr":
        # orthographic test on code-/URL-stripped text: mostly Cyrillic (Latin
        # product names and terms are fine) and only letters of the Serbian alphabet
        cyr, lat = CYR.findall(clean), len(LAT.findall(clean))
        foreign = sum(ch not in SR_ALPHABET for ch in cyr)
        ok = len(cyr) > 0 and len(cyr) >= 0.3 * (len(cyr) + lat) and foreign == 0
        return ok, code if ok else "{}/sr-orth".format(code), p
    if code == tgt:
        return True, code, p
    if tgt in BCMS_LATIN and code in BCMS_LATIN:
        return True, code, p
    if any(tgt in g and code in g for g in GROUPS) and (code2 == tgt or tgt == "ltg"):
        return True, code, p
    if code2 == tgt and p < 0.5:   # e.g. short prompts dominated by English names
        return True, code, p
    return False, code, p


def message_check(src_msgs, tr):
    """Per-message sanity check of a translation -> reason or None. Catches
    structured-output failures (a message rendered as "role" / "translations" or
    as a run-on hyphenated string) that a check of the last user turn misses."""
    for m, t in zip(src_msgs, tr):
        s, t = (m["content"] or "").strip(), (t or "").strip()
        if t.lower() in JUNK and s.lower() not in JUNK:
            return "junk_message"
        if len(s) >= 80 and not 0.25 <= len(t) / len(s) <= 4.0:
            return "message_ratio"
        if len(t) >= 60 and " " not in t and " " in s:
            return "run_on_message"
    return None


WORD = regex.compile(r"\p{L}{2,}")
# Gemma's JSON mode sometimes glues a made-up key to a message, e.g.
# 'role-user-2-content-translated-placeholder-for-json-...": "<translation>"'
LEAK = regex.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+){5,}")
LEAK_WORDS = ("placeholder", "translated", "translation", "json", "role", "content", "message")
LEAK_KEY = regex.compile(r'(?m)^\s*"(?:role|content|translation|translations)"\s*:')


def leaked(text, src):
    """Leaked JSON scaffolding in a translated message (absent from its source)."""
    for m in LEAK.finditer(text or ""):
        tok = m.group(0).lower()
        if m.group(0) not in (src or "") and sum(w in tok for w in LEAK_WORDS) >= 2:
            return True
    return bool(LEAK_KEY.search(text or "")) and not LEAK_KEY.search(src or "")


def strip_leak(text, src):
    """'<junk key>": "<translation>"' -> '<translation>' (text after the last ': "'),
    or None if what remains still contains scaffolding."""
    i = (text or "").rfind(': "')
    if i < 0:
        return None
    out = text[i + 3:].strip()
    if out.endswith('"'):
        out = out[:-1].rstrip()
    return out if out and not leaked(out, src) else None


EP_SOURCE = "lang-uk/every_prompt"
EP_PREFIX = regex.compile(r"^\s*(?:[\p{So}\p{Sk}️‍]+\s*)+")   # leading emoji / symbols
EP_NUMBER = regex.compile(r"^\s*\d{1,3}[.)]\s+")                        # "7. ", "3) "


def clean_every_prompt(context, responses):
    """every_prompt rows are scraped FAQ entries: strip leading emoji / list numbers
    from user turns, and label the scraped answers as web answers, not human replies."""
    out = []
    for m in context:
        c = m["content"] or ""
        if m["role"] == "user":
            s = EP_NUMBER.sub("", EP_PREFIX.sub("", c), count=1)
            c = s if s.strip() else c
        out.append(dict(m, content=c))
    return out, [dict(x, model="web_faq_answer") for x in responses]


def load_translations():
    """(slot, uid) -> request meta and request id -> output, over all rounds:
    data/llm/translate/ (round 0) and data/llm/translate/round*/."""
    tr_meta, tr_out = {}, {}
    tdir = os.path.join(DATA, "llm", "translate")
    for d in [tdir] + sorted(glob.glob(os.path.join(tdir, "round*"))):
        mp = os.path.join(d, "requests_meta.parquet")
        if not os.path.exists(mp):
            continue
        for r in pq.read_table(mp).to_pylist():
            tr_meta[(r["slot"], r["uid"])] = r
        for p in glob.glob(os.path.join(d, "out", "*.parquet")):
            for r in pq.read_table(p).to_pylist():
                tr_out[r["id"]] = r
    return tr_meta, tr_out


def translation_lang_ok(src_ctx, new_ctx, lang):
    """LID of the translated last user turn; for multi-turn conversations whose
    last user turn is too short for LID (< 30 letters, e.g. "komisk?" or a URL),
    all user turns together are checked instead, provided the last turn is not
    an untranslated copy of the source."""
    users = [mm["content"] for mm in new_ctx if mm["role"] == "user"]
    ok, code, p = lang_check(users[-1], lang)
    if ok or len(users) < 2 or len(B.clean_for_lid(users[-1], 1000)) >= 30:
        return ok, code, p
    src_last = [mm["content"] or "" for mm in src_ctx if mm["role"] == "user"][-1]
    if users[-1].strip() == src_last.strip() and len(WORD.findall(src_last)) >= 2:
        return False, code, p
    ok2, code2, p2 = lang_check("\n".join(users), lang)
    return (True, code2, p2) if ok2 else (False, code, p)


def validate_context(context, new_ctx, lang):
    """Checks on a translated conversation, whatever produced it ->
    (reason, why_key), (None, None) when accepted. Latgalian skips LID (it cannot
    separate lv/ltg); its contexts are checked in stage 5e."""
    tr = [mm["content"] for mm in new_ctx]
    if len(tr) != len(context) or any(a["role"] != b["role"] for a, b in zip(context, new_ctx)):
        return "layout", "translation_layout"
    bad = message_check(context, tr)
    if bad:
        return bad, "translation_" + bad
    if any(leaked(t, mm["content"]) for mm, t in zip(context, tr)):
        return "leaked_json_key", "translation_leaked_json_key"
    inj = M.injected("\n".join(mm["content"] or "" for mm in context), "\n".join(tr))
    if inj:
        return "script_injection:" + ",".join(sorted(inj)), "translation_script_injection"
    if lang != "ltg":
        ok, code, _ = translation_lang_ok(context, new_ctx, lang)
        if not ok:
            return "lid:{}".format(code), "translation_wrong_language"
    ratio = sum(len(x) for x in tr) / max(1, sum(len(mm["content"] or "") for mm in context))
    if not 0.4 <= ratio <= 3.0:
        return "length_ratio:{:.2f}".format(ratio), "translation_length_ratio"
    return None, None


def tg_context(msgs, context):
    """Messages by index (from load_tg_overrides) -> full context; messages
    without text pieces (empty in the source) are taken from the source."""
    return [msgs.get(i, {"role": m["role"], "content": m["content"] or ""})
            for i, m in enumerate(context)]


def load_tg_overrides():
    """TranslateGemma re-translations (stage 5c) -> (slot, uid) -> (messages by index,
    complete); messages are reassembled from per-piece outputs (code / whitespace
    verbatim)."""
    base = os.path.join(DATA, "llm", "translate_tg")
    out, rows = {}, {}
    for d in [base] + sorted(glob.glob(os.path.join(base, "round*"))):   # stage 5c, refill rounds, language sets
        sp = os.path.join(d, "segments.parquet")
        if not os.path.exists(sp):
            continue
        for p in glob.glob(os.path.join(d, "out", "*.parquet")):
            for r in pq.read_table(p).to_pylist():
                out[r["id"]] = r
        here = collections.defaultdict(list)
        for s in pq.read_table(sp).to_pylist():
            if s["group"].startswith("tg"):          # tg (5c) / tgN (refill round N) / tgX (language set)
                here[(s["slot"], s["uid"])].append(s)
        rows.update(here)                            # a later run replaces, never mixes with, an earlier one
    res = {}
    for key, ss in rows.items():
        msgs, complete = {}, True
        for s in sorted(ss, key=lambda x: (x["msg"], x["piece"])):
            m = msgs.setdefault(s["msg"], {"role": s["role"], "content": ""})
            if s["kind"] != "text":
                m["content"] += s["content"]
                continue
            o = out.get(s["rid"])
            if not o or o["finish_reason"] != "stop" or not o["text"].strip():
                complete = False
            else:
                m["content"] += o["text"].strip()
        res[key] = (msgs, complete)
    return res


def load_mt2_overrides():
    """TildeOpen / SalamandraTA translations (stage 5m) -> (slot, uid) -> (system, messages by
    index, complete); runs data/llm/translate_mt2/run*/ in name order, a later run replaces an
    earlier one per (slot, uid)."""
    import mt2_collect as C2
    res = {}
    for d in sorted(glob.glob(os.path.join(DATA, "llm", "translate_mt2", "run*"))):
        sp = os.path.join(d, "segments.parquet")
        if not os.path.exists(sp):
            continue
        segs = pq.read_table(sp).to_pylist()
        outs = {sysname: C2.outputs(os.path.join(d, sysname)) for sysname in MT2_TRANSLATOR}
        for (rid, sysname), x in C2.assemble(segs, outs).items():
            msgs = {i: {"role": role, "content": text} for i, (role, text) in x["by_index"].items()}
            res[(int(rid.split("-")[1]), x["uid"])] = (sysname, msgs, x["failed"] == 0)
    return res


def valid_mt2(mt2_ov, key, context, lang, stats):
    """(system, context) of the slot's text-level translation if complete and passing, else None."""
    if key not in mt2_ov:
        return None
    sysname, msgs, complete = mt2_ov[key]
    if not complete:
        stats[sysname + ":incomplete"] += 1
        return None
    if any(M.dangling_fence(m["content"]) for m in context):   # segmented with the old fence rule
        stats[sysname + ":rejected:dangling_fence"] += 1
        return None
    tctx = tg_context(msgs, context)
    reason, why_key = validate_context(context, tctx, lang)
    if reason:
        stats["{}:rejected:{}".format(sysname, why_key)] += 1
        return None
    for a, b in zip(context, tctx):
        if a["role"] == "user" and M.literal_problems(a["content"], b["content"]):
            stats[sysname + ":rejected:literals"] += 1
            return None
    stats[sysname + ":used"] += 1
    return sysname, tctx


MANUAL = os.path.join(DATA, "llm", "translate_manual", "translations.parquet")
MANUAL_TRANSLATOR = "anthropic/claude-opus-5-5"


def load_manual_overrides():
    """Manual translations (stage 5n, manual_collect.py) -> (slot, uid) -> context."""
    if not os.path.exists(MANUAL):
        return {}
    return {(r["slot"], r["uid"]): json.loads(r["context"]) for r in pq.read_table(MANUAL).to_pylist()}


def valid_manual(man_ov, key, context, lang, stats):
    """The slot's manual translation if it matches the source's messages and passes the checks."""
    if key not in man_ov:
        return None
    tctx = man_ov[key]
    if len(tctx) != len(context) or any(a["role"] != b["role"] for a, b in zip(context, tctx)):
        stats["manual:rejected:shape"] += 1
        return None
    reason, why_key = validate_context(context, tctx, lang)
    if reason:
        stats["manual:rejected:" + why_key] += 1
        return None
    for a, b in zip(context, tctx):
        if a["role"] == "user" and M.literal_problems(a["content"], b["content"]):
            stats["manual:rejected:literals"] += 1
            return None
    stats["manual:used"] += 1
    return tctx


def load_ltg_overrides():
    """Latgalian contexts selected in stage 5f -> ((slot, uid) -> (context, system, note), meta)."""
    d = os.path.join(DATA, "llm", "ltg_final")
    p = os.path.join(d, "contexts.parquet")
    if not os.path.exists(p):
        return {}, {}
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    return {(r["slot"], r["uid"]): (json.loads(r["context"]), r["system"], r["note"])
            for r in pq.read_table(p).to_pylist()}, meta


def valid_tg(tg_ov, key, context, lang, stats):
    """The TranslateGemma context of a slot if it exists, is complete and passes the checks."""
    if key not in tg_ov:
        return None
    msgs, complete = tg_ov[key]
    if not complete:
        stats["incomplete"] += 1
        return None
    if any(M.dangling_fence(m["content"]) for m in context):   # segmented with the old fence rule
        stats["rejected:dangling_fence"] += 1
        return None
    tctx = tg_context(msgs, context)
    reason, why_key = validate_context(context, tctx, lang)
    if reason:
        stats["rejected:" + why_key] += 1
        return None
    return tctx


def load_tg_choice():
    """Round-trip QE choice per re-translated row (stage 5h): (slot, uid) -> "old" | "new"."""
    p = os.path.join(DATA, "llm", "translate_tg", "choice.parquet")
    if not os.path.exists(p):
        return {}
    return {(r["slot"], r["uid"]): r["choice"] for r in pq.read_table(p).to_pylist()}


def load_force_fail(out_dir):
    p = os.path.join(out_dir, "force_fail.parquet")
    if not os.path.exists(p):
        return {}
    return {(r["slot"], r["uid"]): r["reason"] for r in pq.read_table(p).to_pylist()}


def check_translation(context, m, o, lang, kwargs, ids):
    """Validate one translation -> (reason, why_key, new_context, kwargs, ids, notes);
    reason is None when the translation is accepted."""
    if not o or o["finish_reason"] != "stop":
        return ("no_translation" if not o else o["finish_reason"]), \
            "translation_missing_or_truncated", None, kwargs, ids, []
    n = m["n_messages"]
    params = json.loads(m["params"])
    if params:
        tr, tp = T.parse_if(o["text"], n, len(params))
    else:
        tr, tp = T.parse(o["text"], n), None
    if tr is None:
        return "bad_json", "translation_bad_json", None, kwargs, ids, []
    notes = []
    tr = list(tr)
    for i, (mm, t) in enumerate(zip(context, tr)):
        if leaked(t, mm["content"]):
            s = strip_leak(t, mm["content"])
            if s is None:
                return "leaked_json_key", "translation_leaked_json_key", None, kwargs, ids, []
            tr[i] = s
            if "leaked_json_key_stripped" not in notes:
                notes.append("leaked_json_key_stripped")
    new_ctx = [{"role": mm["role"], "content": t} for mm, t in zip(context, tr)]
    reason, key = validate_context(context, new_ctx, lang)
    if reason:
        return reason, key, None, kwargs, ids, []
    if params:
        low = [mm["content"] for mm in new_ctx if mm["role"] == "user"][-1].lower()
        if tp is None or not all(x.lower() in low for x in tp):
            notes.append("if_params_not_found_verifier_dropped")
            kwargs, ids = None, []
        else:
            kwargs = T.apply_params(kwargs, dict(zip(params, tp)))
    return None, None, new_ctx, kwargs, ids, notes


def keep_responses(rec, lang):
    """Existing responses worth keeping: reusable, in the row's language, and without
    mixed-script words (generation artifacts) that the prompt does not contain."""
    ctx_mixed = set(M.mixed_script_words("\n".join(m["content"] or "" for m in rec["context"])))
    out = []
    for r in rec["responses"] or []:
        if set(M.mixed_script_words(r["content"])) - ctx_mixed:
            continue
        m = json.loads(r["meta"]) if r["meta"] else {}
        if m.get("model_reuse") is False:
            continue
        code, p, _, _ = B.lid(B.clean_for_lid(r["content"] or "", 600))
        if code != lang and not any(lang in g and code in g for g in GROUPS):
            continue
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "compiled"))
    args = ap.parse_args()
    B.load_model(B.ensure_model(), B.label_map())

    alloc = pq.read_table(os.path.join(args.out, "allocation.parquet")).to_pylist()
    cand = {r["uid"]: r for r in pq.read_table(os.path.join(DATA, "pool", "candidates.parquet"),
                                               columns=["uid", "part", "row", "license_flag",
                                                        "user_key"]).to_pylist()}
    used = [cand[a["uid"]] for a in alloc if a["uid"]]
    recs = fetch_records(used)
    ifm = {r["uid"]: r for r in pq.read_table(os.path.join(DATA, "pool", "if_meta.parquet")).to_pylist()}

    tr_meta, tr_out = load_translations()
    tg_ov = load_tg_overrides()
    mt2_ov = load_mt2_overrides()
    mt2_stats = collections.Counter()
    man_ov = load_manual_overrides()
    tg_choice = load_tg_choice()
    ltg_ov, ltg_meta = load_ltg_overrides()
    force = load_force_fail(args.out)

    rows, failed = [], []
    why = collections.Counter()
    tg_stats = collections.Counter()
    for a in alloc:
        if not a["uid"]:
            why["unfilled"] += 1
            continue
        key2 = (a["slot"], a["uid"])
        if key2 in force:
            failed.append({**a, "reason": "label:" + force[key2]})
            why["wrong_language_label"] += 1
            continue
        rec = recs[a["uid"]]
        c = cand[a["uid"]]
        lang = a["lang"]
        context, responses, preference = rec["context"], [], None
        im = ifm.get(a["uid"])
        kwargs = json.loads(im["kwargs"]) if im and im["kwargs"] else None
        ids = list(im["instruction_ids"]) if im else []
        notes = []
        translator = None
        man = valid_manual(man_ov, key2, context, lang, mt2_stats) \
            if a["fill"] == "translate" and lang != "ltg" else None
        mt2 = valid_mt2(mt2_ov, key2, context, lang, mt2_stats) \
            if a["fill"] == "translate" and lang != "ltg" and man is None else None
        if a["fill"] == "translate" and lang == "ltg":
            new_ctx, system, ltg_note = ltg_ov.get(key2, (None, None, None))
            reason, key = ("ltg_needs_retranslation", "translation_ltg_missing") if new_ctx is None \
                else validate_context(context, new_ctx, lang)
            if reason:
                failed.append({**a, "reason": reason})
                why[key] += 1
                continue
            if ids:
                notes.append("ltg_if_verifier_dropped")
            context, kwargs, ids = new_ctx, None, []
            translator = ltg_meta["translators"][system]
            notes.append(ltg_note)
        elif man is not None:
            context, translator = man, MANUAL_TRANSLATOR
            kwargs, ids = None, []          # rows with IF params were not re-translated
            notes.append("retranslated_manual_claude")
        elif mt2 is not None:
            sysname, context = mt2
            translator = MT2_TRANSLATOR[sysname]
            kwargs, ids = None, []          # rows with IF params were not re-translated
            notes.append(MT2_NOTE[sysname])
        elif a["fill"] == "translate":
            # the round-trip QE (stage 5h) may prefer the Gemma translation of a re-translated row
            prefer_gemma = tg_choice.get(key2) == "old"
            tctx = None if prefer_gemma else valid_tg(tg_ov, key2, context, lang, tg_stats)
            if tctx is None:
                m = tr_meta.get((a["slot"], a["uid"]))
                o = tr_out.get(m["id"]) if m else None
                reason, key, new_ctx, kw2, ids2, n2 = check_translation(context, m, o, lang, kwargs, ids)
                if reason is None:
                    context, translator, kwargs, ids = new_ctx, TRANSLATOR, kw2, ids2
                    notes += n2
                    if prefer_gemma:
                        tg_stats["kept_gemma_roundtrip"] += 1
                    elif key2 in tg_ov:
                        tg_stats["kept_gemma"] += 1
                else:
                    tctx = valid_tg(tg_ov, key2, context, lang, tg_stats) if prefer_gemma else None
                    if tctx is None:
                        failed.append({**a, "reason": reason})
                        why[key] += 1
                        continue
            if tctx is not None:
                context, translator = tctx, TG_TRANSLATOR
                kwargs, ids = None, []      # rows with IF params were not re-translated
                notes.append("retranslated_translategemma")
                tg_stats["used"] += 1
        else:
            responses = keep_responses(rec, lang)
            if responses and len(responses) == len(rec["responses"] or []):
                preference = rec["preference"]
            elif rec["preference"] and responses:
                notes.append("preference_dropped_responses_filtered")
        if rec["source"] == EP_SOURCE:
            context, responses = clean_every_prompt(context, responses)
        rows.append({
            "id": "mlp128k-{:06d}".format(a["slot"]),
            "lang": lang,
            "context": context,
            "responses": responses,
            "preference": preference,
            "fill": "translated" if a["fill"] == "translate" else a["fill"],
            "source": rec["source"], "subset": rec["subset"], "source_uid": a["uid"],
            "source_lang": a["src_lang"],
            "translator": translator,
            "license": rec["license"], "license_flag": c["license_flag"],
            "purpose": a["purpose"], "origin": a["item_origin"], "task": a["item_task"],
            "domain": a["item_domain"], "turns": a["turns"], "safety": a["safety"],
            "locale": a["locale"], "complexity": a["complexity"],
            "constraints": list(a["constraints"]) if a["constraints"] is not None else [],
            "gist": a["gist"], "alloc_level": a["level"],
            "instruction_ids": ids, "if_kwargs": json.dumps(kwargs, ensure_ascii=False) if kwargs else None,
            "notes": notes,
            "source_meta": rec["meta"],
        })
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(args.out, "prompts.parquet"),
                   compression="zstd")
    pq.write_table(pa.Table.from_pylist(failed) if failed else pa.table({"slot": []}),
                   os.path.join(args.out, "failed_slots.parquet"))
    rep = {"rows": len(rows), "failed": len(failed), "why": dict(why),
           "fill": collections.Counter(r["fill"] for r in rows),
           "with_responses": sum(1 for r in rows if r["responses"]),
           "with_preference": sum(1 for r in rows if r["preference"]),
           "if_verifiable": sum(1 for r in rows if r["instruction_ids"]),
           "purpose": collections.Counter(r["purpose"] for r in rows),
           "if_purpose_verifiable_by_fill": collections.Counter(
               "{}:{}".format(r["fill"], bool(r["instruction_ids"])) for r in rows
               if r["purpose"] == "if"),
           "notes": collections.Counter(n for r in rows for n in r["notes"]),
           "translator": collections.Counter(r["translator"] for r in rows if r["translator"]),
           "translategemma": dict(tg_stats),
           "text_level_mt2": dict(mt2_stats),
           "failed_reasons": collections.Counter(f["reason"].split(":")[0] for f in failed)}
    with open(os.path.join(args.out, "assemble_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=dict)
    print(json.dumps(rep, indent=1, default=dict))


if __name__ == "__main__":
    main()
