"""Multilingual (mostly synthetic / translated / web-mined) prompt sources.

  openeurollm/EuroMultiSynth-Instruct   35 languages, natively generated
      single-turn pairs (gemma-4-31b-it). One record per row; subset =
      language config; response = generated reply (model = generator);
      meta keeps domain, persona, judge_score (Qwen3.6-27B, 0-10, only 7-10
      kept by the source).
  akoksal/muri-it                       reverse instructions over native
      documents (MRI) + WikiHow + multilingual NLP tasks. One record per row;
      subset = dataset_name; response = `output` (the human-written document
      for MRI/WikiHow/OpenAssistant -> model="human"; the task's reference
      answer for xP3/SuperNatInst/FLAN -> model=None, meta kind=
      "task_reference"); language = ISO 639-3 code; meta keeps
      subdataset_name (document corpus / task name) and language_name.
      Source splits train/validation/test are kept.
  lang-uk/every_prompt                  schema.org FAQPage/HowTo items crawled
      from the web (every_prompt.jsonlines; scraper/ and bin/ are code).
      FAQPage: one record per Question (subset "FAQPage"); prompt = question
      name (+ question text if it adds something); responses = acceptedAnswer
      (+ suggestedAnswer) texts, model="human", no preference.
      HowTo: one record per page (subset "HowTo"); prompt = the HowTo name;
      response = its steps rendered as a numbered list (none if the page
      has no steps). FAQ pages without any Question are skipped.
      HTML in question/answer text is converted to plain text (tags removed,
      entities unescaped; meta html_stripped=true). language = the page-level
      gcld3 label of the source (meta language_is_reliable). meta keeps url,
      domain, page index (`page`, linking all questions of a page), question
      index, schema_type.
  BSC-LT/ALIA-2606-SFT                  ALIA SFT mixture (es/ca/eu/gl/pt/en and
      translation pairs such as "en-pt"). One record per conversation
      (split_last_turn); subset = dataset_name (per-row origin); meta keeps
      task and id.
  BSC-LT/ALIA-2606-DPO-helpfulness      chosen/rejected pairs judged by
      DeepSeek-V3 (1-5 helpfulness). responses = [chosen, rejected],
      preference chosen_rejected with the scores in "raw"; response model =
      chosen_origin / rejected_origin; meta prompt_id links to ALIA-2606-SFT.
      The same id occurs twice for 268 flores-dev translation prompts with
      different `lang` labels -> source_id gets "#<lang>" for the repeat.
  nvidia/Nemotron-Post-Training-Dataset-v2  only multilingual_{de,fr,it,es}
      and chat were downloaded. One record per row; subset = category (file
      group); language from the category (chat -> None); leading
      <think>...</think> blocks are dropped from the reply (response meta
      reasoning_dropped / reasoning mode); empty system messages are dropped.
      meta keeps uuid, generator, reasoning on/off, per-row license.
      81,876 chat prompts occur twice (reasoning on and off, same context):
      merged into one record with responses [on, off] (no preference;
      meta reasoning="on+off", uuid_reasoning_off).
  ltg/normistral-11b-thinking-training  Norwegian (Bokmål "nb" / Nynorsk
      "nn") RL prompt pool translated from many sources + a small English
      SFT file. subset = file stem (e.g. rl_nb_lmsys, train_sft); meta origin
      (lmsys, oasst, wildguardmix, ...) and variant (nb/nn). RL rows are
      prompt-only: their `target` is a verification reference (often the
      untranslated English gold answer, a math answer, or a harmfulness note
      for wildguardmix) and goes to meta["target"], not to responses. The
      only exception is rl_hardcoded_prompts, whose target is a Norwegian
      assistant message -> response (model="ltg_hardcoded"). wildguardmix
      targets are parsed into meta["harmful"]. original_prompt (English
      source prompt) is kept in meta as a linking key. A few translated
      prompts end with an assistant turn (role misalignment from translation);
      it is cut off (meta trailing_assistant_dropped) rather than used as a
      response. train_sft: separate `reasoning` fields are dropped from the
      final reply (reasoning_dropped).
"""
import hashlib
import html
import json
import os
import re
from urllib.parse import urlparse

import pyarrow.parquet as pq

from common import (iter_jsonl, iter_parquet, msg, norm_messages, register,
                    response, split_last_turn)

THINK_RE = re.compile(r"^\s*<think>(.*?)</think>\s*", re.S)


def strip_think(text):
    """Drop a leading <think>...</think> block. Returns (text, dropped) where
    dropped is True if the block had non-whitespace content. An unterminated
    leading <think> means the whole text is reasoning -> ("", True)."""
    if text is None:
        return None, False
    m = THINK_RE.match(text)
    if m:
        return text[m.end():], bool(m.group(1).strip())
    if text.lstrip().startswith("<think>"):
        return "", True
    return text, False


def _clean_context(context):
    """Strip leading think blocks from earlier assistant turns."""
    out = []
    for m in context:
        if m["role"] == "assistant" and "<think>" in m["content"]:
            m = dict(m, content=strip_think(m["content"])[0])
        out.append(m)
    return out


def _nonempty(s):
    return s is not None and str(s).strip() != ""


def _reply(text, model=None, **meta):
    """Response with reasoning stripped; None if nothing remains."""
    if text is None:
        return None
    text, dropped = strip_think(text)
    if dropped:
        meta["reasoning_dropped"] = True
    if not text.strip():
        return None
    return response(text, model=model, **meta)


def _drop_empty_system(messages):
    return [m for m in messages if not (m["role"] == "system" and not m["content"].strip())]


# ------------------------------------------------------ EuroMultiSynth-Instruct

@register("openeurollm/EuroMultiSynth-Instruct")
def euromultisynth(ctx):
    for path, row in iter_parquet(ctx.files("data/*/train-*.parquet")):
        lang = row.get("language") or os.path.basename(os.path.dirname(path))
        messages = _drop_empty_system(norm_messages(row["messages"] or []))
        context, reply = split_last_turn(messages)
        if context is None or not _nonempty(context[-1]["content"]):
            ctx.skip("no_user_turn")
            continue
        r = _reply(reply, model="google/gemma-4-31b-it")
        yield ctx.record(
            subset=lang, split="train", source_id=row["id"],
            context=_clean_context(context), responses=[r] if r else [],
            language=lang,
            meta={"domain": row.get("domain"), "persona": row.get("persona"),
                  "judge_score": row.get("judge_score"),
                  "judge_model": "Qwen/Qwen3.6-27B"})


# --------------------------------------------------------------------- muri-it

MURI_HUMAN = {"MRI", "WikiHow", "OpenAssistant"}


@register("akoksal/muri-it")
def muri_it(ctx):
    for path in ctx.files("data/*.parquet"):
        name = os.path.basename(path).rsplit(".", 1)[0]  # train-00003-of-00013
        file_split = name.split("-", 1)[0]
        split = {"test": "test", "validation": "validation"}.get(file_split, "train")
        idx = 0
        for _, row in iter_parquet([path]):
            sid = "{}:{}".format(name, idx)
            idx += 1
            if not _nonempty(row["input"]):
                ctx.skip("empty_prompt")
                continue
            ds = row.get("dataset_name")
            if _nonempty(row.get("output")):
                if ds in MURI_HUMAN:
                    resp = [response(row["output"], model="human")]
                else:
                    resp = [response(row["output"], model=None, kind="task_reference")]
            else:
                resp = []
            yield ctx.record(
                subset=ds or "unknown", split=split, source_id=sid,
                context=[msg("user", row["input"])], responses=resp,
                language=row.get("language"),
                meta={"dataset_name": ds,
                      "subdataset_name": row.get("subdataset_name"),
                      "language_name": row.get("language_name"),
                      "original_split": row.get("split")})


# ---------------------------------------------------------------- every_prompt

_TAG_BREAK = re.compile(r"<\s*(br|/p|/li|/div|/h[1-6]|/tr|p|li|div|tr)\b[^<>]*>", re.I)
_TAG = re.compile(r"<\s*/?\s*[a-zA-Z][a-zA-Z0-9:-]*(\s[^<>]*)?/?\s*>")
_WS = re.compile(r"[ \t\r\f\v]+")
_NL = re.compile(r"\n\s*\n\s*\n+")


def _html_text(s):
    """Plain text from schema.org text (which may hold HTML). Returns
    (text, had_html)."""
    if s is None:
        return "", False
    if isinstance(s, list):
        s = "\n".join(str(x) for x in s if isinstance(x, (str, int, float)))
    elif not isinstance(s, str):
        if isinstance(s, (int, float)) and not isinstance(s, bool):
            s = str(s)
        else:
            return "", False
    had = bool(_TAG.search(s))
    if had:
        s = _TAG_BREAK.sub("\n", s)
        s = _TAG.sub("", s)
    if "&" in s:
        s2 = html.unescape(s)
        if s2 != s:
            s = s2
            if _TAG.search(s):  # escaped markup ("&lt;p&gt;")
                had = True
                s = _TAG.sub("", _TAG_BREAK.sub("\n", s))
    s = s.replace("\u00a0", " ")
    lines = [_WS.sub(" ", ln).strip() for ln in s.split("\n")]
    s = _NL.sub("\n\n", "\n".join(lines)).strip()
    return s, had


def _first_str(v):
    if isinstance(v, list):
        v = next((x for x in v if isinstance(x, str) and x.strip()), None)
    return v if isinstance(v, str) else None


def _questions(me):
    """Flatten mainEntity into a list of Question dicts."""
    out = []
    stack = me if isinstance(me, list) else [me]
    for it in stack:
        if isinstance(it, list):
            out.extend(x for x in it if isinstance(x, dict))
        elif isinstance(it, dict):
            if any(k in it for k in ("name", "acceptedAnswer", "suggestedAnswer", "@type", "text")):
                out.append(it)
            else:  # {"1": {...}, "2": {...}}
                out.extend(v for v in it.values() if isinstance(v, dict))
    return out


def _answer_list(v):
    if v is None:
        return []
    return [a for a in (v if isinstance(v, list) else [v]) if isinstance(a, (dict, str))]


def _answer_text(a):
    if isinstance(a, str):
        return _html_text(a)
    t = a.get("text")
    if t is None or (isinstance(t, str) and not t.strip()):
        t = a.get("name")
    return _html_text(t)


def _sub_texts(v):
    """Texts of HowToDirection/HowToTip substeps (recursive)."""
    out = []
    for s in (v if isinstance(v, list) else [v]):
        if isinstance(s, str):
            t, _ = _html_text(s)
        elif isinstance(s, dict):
            t, _ = _html_text(s.get("text"))
            if not t:
                t, _ = _html_text(_first_str(s.get("name")))
            sub = s.get("itemListElement", s.get("itemlistElement"))
            if sub is not None:
                t = "\n".join(x for x in [t] + _sub_texts(sub) if x)
        else:
            t = ""
        if t:
            out.append(t)
    return out


def _step_lines(step, out):
    """Render HowTo steps / sections into (kind, text) lines."""
    if isinstance(step, list):
        for s in step:
            _step_lines(s, out)
        return
    if isinstance(step, str):
        t, _ = _html_text(step)
        if t:
            out.append(("step", t))
        return
    if not isinstance(step, dict):
        return
    sub = step.get("itemListElement", step.get("itemlistElement"))
    name, _ = _html_text(_first_str(step.get("name")))
    if "Section" in str(step.get("@type") or "") and sub is not None:
        if name:
            out.append(("section", name))
        _step_lines(sub, out)
        return
    text, _ = _html_text(step.get("text"))
    body = "\n".join(x for x in [text] + (_sub_texts(sub) if sub is not None else []) if x)
    if name and body and len(name) <= 150 and name.rstrip(".:") not in body:
        out.append(("step", "{}: {}".format(name, body)))
    elif body or name:
        out.append(("step", body or name))


def _render_howto(p):
    lines = []
    _step_lines(p.get("step"), lines)
    out, n = [], 0
    for kind, t in lines:
        if kind == "section":
            out.append("\n" + t if out else t)
            n = 0
        else:
            n += 1
            out.append("{}. {}".format(n, t))
    return "\n".join(out).strip()


@register("lang-uk/every_prompt")
def every_prompt(ctx):
    for page, (_, row) in enumerate(iter_jsonl(ctx.files("every_prompt.jsonlines"))):
        p = row.get("payload")
        if isinstance(p, str):
            try:
                p = json.loads(p)
            except ValueError:
                p = None
        if not isinstance(p, dict):
            ctx.skip("payload_not_object")
            continue
        url = row.get("url")
        try:
            domain = urlparse(url).netloc.lower() if url else None
        except ValueError:
            domain = None
        lang = row.get("language")
        ptype = p.get("@type")
        base = {"url": url, "domain": domain, "page": page,
                "schema_type": row.get("schema_type"),
                "page_type": ptype if isinstance(ptype, str) or ptype is None
                else json.dumps(ptype, ensure_ascii=False),
                "language_is_reliable": row.get("language_is_reliable"),
                "page_text_length": row.get("text_length")}
        me = p.get("mainEntity", p.get("mainentity"))
        if me is None and isinstance(p.get("question"), (list, dict)):
            me = p.get("question")
        if me is None and (p.get("step") is not None or ptype == "HowTo"):
            # ------------------------------------------------------ HowTo
            title, h1 = _html_text(_first_str(p.get("name")) or _first_str(p.get("headline")))
            if not title:
                ctx.skip("howto_no_name")
                continue
            steps = _render_howto(p)
            meta = dict(base)
            if h1:
                meta["html_stripped"] = True
            yield ctx.record(
                subset="HowTo", split="train", source_id=str(page),
                context=[msg("user", title)],
                responses=[response(steps, model="human", kind="howto_steps")] if steps else [],
                language=lang, meta=meta)
            continue
        if me is None:
            ctx.skip("no_questions_on_page")
            continue
        qs = _questions(me)
        if not qs:
            ctx.skip("no_questions_on_page")
            continue
        for qi, q in enumerate(qs):
            name, h1 = _html_text(_first_str(q.get("name")))
            text, h2 = _html_text(_first_str(q.get("text")))
            if not name:
                name, text, h1, h2 = text, "", h2, False
            if not name:
                ctx.skip("empty_question")
                continue
            prompt = name
            if text and text != name and name not in text and text not in name:
                prompt = name + "\n\n" + text
            elif text and len(text) > len(name) and name in text:
                prompt = text
            had_html = h1 or h2
            resps = []
            for kind, key in (("accepted", "acceptedAnswer"), ("suggested", "suggestedAnswer")):
                for a in _answer_list(q.get(key)):
                    t, h = _answer_text(a)
                    if not t:
                        continue
                    had_html = had_html or h
                    rm = {"kind": kind}
                    if isinstance(a, dict):
                        for k in ("upvoteCount", "downvoteCount"):
                            if a.get(k) not in (None, ""):
                                rm[k] = a.get(k)
                    resps.append(response(t, model="human", **rm))
            meta = dict(base, question_index=qi, n_questions=len(qs))
            if had_html:
                meta["html_stripped"] = True
            yield ctx.record(
                subset="FAQPage", split="train",
                source_id="{}-{}".format(page, qi),
                context=[msg("user", prompt)], responses=resps,
                language=lang, meta=meta)


# --------------------------------------------------------------- ALIA-2606-SFT

@register("BSC-LT/ALIA-2606-SFT")
def alia_sft(ctx):
    for _, row in iter_parquet(ctx.files("data/train-*.parquet")):
        messages = _drop_empty_system(norm_messages(row["messages"] or []))
        context, reply = split_last_turn(messages)
        if context is None or not _nonempty(context[-1]["content"]):
            ctx.skip("no_user_turn")
            continue
        r = _reply(reply)
        yield ctx.record(
            subset=row.get("dataset_name") or "unknown", split="train",
            source_id=row["id"], context=_clean_context(context),
            responses=[r] if r else [], language=row.get("lang"),
            meta={"id": row["id"], "task": row.get("task"),
                  "dataset_name": row.get("dataset_name")})


# ------------------------------------------------- ALIA-2606-DPO-helpfulness

@register("BSC-LT/ALIA-2606-DPO-helpfulness")
def alia_dpo(ctx):
    seen = {}
    for _, row in iter_parquet(ctx.files("data/train-*.parquet")):
        sid = row["id"]
        seen[sid] = seen.get(sid, 0) + 1
        if seen[sid] > 1:
            sid = "{}#{}".format(sid, row.get("lang"))
            if sid in seen:
                sid = "{}#{}".format(sid, seen[row["id"]])
            seen[sid] = 1
        messages = _drop_empty_system(norm_messages(row["prompt"] or []))
        context, _ = split_last_turn(messages)
        if context is None or not _nonempty(context[-1]["content"]):
            ctx.skip("no_user_turn")
            continue
        if len(context) != len(messages):
            ctx.skip("prompt_not_ending_with_user")
            continue
        md = row.get("metadata") or {}
        ch = [m for m in (row.get("chosen") or []) if m["role"] == "assistant"]
        rj = [m for m in (row.get("rejected") or []) if m["role"] == "assistant"]
        if not ch or not rj:
            ctx.skip("missing_chosen_or_rejected_reply")
            continue
        rc = _reply(ch[-1]["content"], model=md.get("chosen_origin"),
                    score=md.get("chosen_score")) or response("", model=md.get("chosen_origin"))
        rr = _reply(rj[-1]["content"], model=md.get("rejected_origin"),
                    score=md.get("rejected_score")) or response("", model=md.get("rejected_origin"))
        yield ctx.record(
            subset=md.get("dataset_name") or "default", split="train",
            source_id=sid, context=_clean_context(context),
            responses=[rc, rr],
            preference={"type": "chosen_rejected", "chosen": 0, "rejected": 1,
                        "raw": {"chosen_score": md.get("chosen_score"),
                                "rejected_score": md.get("rejected_score"),
                                "scale": "1-5 helpfulness (DeepSeek-V3-0324 judge)"}},
            language=row.get("lang"),
            meta={"id": row["id"], "prompt_id": md.get("prompt_id"),
                  "prompt_origin": md.get("prompt_origin"),
                  "type": row.get("type"),
                  "preference_origin": row.get("preference_origin")})


# --------------------------------------------- Nemotron-Post-Training-Dataset-v2

NEMOTRON_GROUPS = ("chat", "multilingual_de", "multilingual_fr",
                   "multilingual_it", "multilingual_es")


_NEMO_COLS = ["uuid", "license", "generator", "category", "reasoning", "messages"]


def _ctx_key(context):
    return hashlib.sha1(json.dumps(context, ensure_ascii=False).encode("utf-8")).hexdigest()


def _nemotron_off_rows(paths):
    """reasoning == "off" rows; files without any are skipped after reading
    only their `reasoning` column."""
    for p in paths:
        if "off" not in pq.read_table(p, columns=["reasoning"]).column(0).to_pylist():
            continue
        for _, row in iter_parquet([p], columns=_NEMO_COLS, batch_size=1024):
            if row["reasoning"] == "off":
                yield row


def _nemotron_parse(row):
    messages = _drop_empty_system(norm_messages(row["messages"] or []))
    context, reply = split_last_turn(messages)
    if context is None or not _nonempty(context[-1]["content"]):
        return None, None
    context = _clean_context(context)
    return context, _reply(reply, model=row.get("generator"),
                           reasoning_mode=row.get("reasoning"))


@register("nvidia/Nemotron-Post-Training-Dataset-v2")
def nemotron_v2(ctx):
    """chat contains 81,876 prompts twice (reasoning on and off, identical
    context): merged into one record with responses [on, off]; source_id is
    the reasoning-on uuid, meta uuid_reasoning_off the other."""
    for group in NEMOTRON_GROUPS:
        lang = group.split("_", 1)[1] if group.startswith("multilingual_") else None
        paths = ctx.files("data/{}-*.parquet".format(group))
        off = {}
        if group == "chat":
            for row in _nemotron_off_rows(paths):
                context, r = _nemotron_parse(row)
                if context is not None:
                    slim = {k: row.get(k) for k in _NEMO_COLS if k != "messages"}
                    slim["context"] = context
                    off.setdefault(_ctx_key(context), []).append((slim, r))
        for _, row in iter_parquet(paths, columns=_NEMO_COLS, batch_size=1024):
            if group == "chat" and row["reasoning"] == "off":
                continue  # emitted together with its reasoning-on twin (or at the end)
            context, r = _nemotron_parse(row)
            if context is None:
                ctx.skip("no_user_turn")
                continue
            meta = {"uuid": row["uuid"], "generator": row.get("generator"),
                    "reasoning": row.get("reasoning"),
                    "category": row.get("category"),
                    "license": row.get("license")}
            if r is None and _nonempty(split_last_turn(norm_messages(row["messages"]))[1]):
                meta["reply_empty_after_reasoning_strip"] = True
            resps = [r] if r else []
            twins = off.pop(_ctx_key(context), None) if off else None
            if twins:
                orow, oresp = twins.pop(0)
                if twins:
                    off[_ctx_key(context)] = twins
                if oresp:
                    resps.append(oresp)
                meta["reasoning"] = "on+off"
                meta["uuid_reasoning_off"] = orow["uuid"]
                if orow.get("generator") != row.get("generator"):
                    meta["generator_reasoning_off"] = orow.get("generator")
            yield ctx.record(
                subset=row.get("category") or group, split="train",
                source_id=row["uuid"], context=context,
                responses=resps, language=lang, meta=meta)
        for twins in off.values():  # reasoning-off rows without an "on" twin
            for orow, oresp in twins:
                yield ctx.record(
                    subset=orow.get("category") or group, split="train",
                    source_id=orow["uuid"], context=orow["context"],
                    responses=[oresp] if oresp else [], language=lang,
                    meta={"uuid": orow["uuid"], "generator": orow.get("generator"),
                          "reasoning": orow.get("reasoning"),
                          "category": orow.get("category"),
                          "license": orow.get("license")})


# ------------------------------------------- normistral-11b-thinking-training

_NM_FILE = re.compile(r"^rl_(?:(nb|nn)_)?(.+)$")


@register("ltg/normistral-11b-thinking-training")
def normistral(ctx):
    for path in ctx.files("*.jsonl"):
        stem = os.path.basename(path)[:-len(".jsonl")]
        if stem == "train_sft":
            origin, variant = "sft", None
        else:
            m = _NM_FILE.match(stem)
            variant, origin = (m.group(1), m.group(2)) if m else (None, stem)
        for i, (_, row) in enumerate(iter_jsonl([path])):
            sid = "{}:{}".format(stem, i)
            meta = {"origin": origin, "variant": variant}
            if stem == "train_sft":
                raw = row.get("messages") or []
                messages = _drop_empty_system(norm_messages(raw))
                context, reply = split_last_turn(messages)
                if context is None or not _nonempty(context[-1]["content"]):
                    ctx.skip("no_user_turn")
                    continue
                last_user = max(j for j, m in enumerate(raw) if m.get("role") == "user")
                nxt = next((m for m in raw[last_user + 1:] if m.get("role") == "assistant"), None)
                resps = []
                if reply is not None and _nonempty(reply):
                    rm = {}
                    if nxt is not None and _nonempty(nxt.get("reasoning")):
                        rm["reasoning_dropped"] = True
                    r = _reply(reply, **rm)
                    if r:
                        resps.append(r)
                yield ctx.record(
                    subset=stem, split="train", source_id=sid,
                    context=_clean_context(context), responses=resps,
                    language=row.get("language"), meta=meta)
                continue
            messages = _drop_empty_system(norm_messages(row.get("prompt") or []))
            context, trailing = split_last_turn(messages)
            if context is None or not _nonempty(context[-1]["content"]):
                ctx.skip("no_user_turn")
                continue
            if len(context) != len(messages):
                meta["trailing_assistant_dropped"] = True
            target = row.get("target")
            resps = []
            if isinstance(target, dict) and target.get("role") == "assistant":
                r = _reply(target.get("content"), model="ltg_hardcoded")
                if r:
                    resps.append(r)
            elif target is not None:
                meta["target"] = target
                if origin == "wildguardmix" and isinstance(target, str):
                    if "This prompt is not harmful" in target:
                        meta["harmful"] = False
                    elif "This prompt is harmful" in target:
                        meta["harmful"] = True
            for k in ("category", "subcategory"):
                if row.get(k) is not None:
                    meta[k] = row[k]
            if row.get("original_prompt") is not None:
                meta["original_prompt"] = row["original_prompt"]
            yield ctx.record(
                subset=stem, split="train", source_id=sid,
                context=_clean_context(context), responses=resps,
                language=row.get("language") or variant, meta=meta)
