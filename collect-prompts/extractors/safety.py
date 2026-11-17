"""Safety sources: harmful prompts, adversarial (jailbreak) prompts and
benign-but-sensitive (over-refusal contrast) prompts.

Routing labels. Every record of these sources has these record-meta keys
(null when the source neither gives nor implies them):

  prompt_harm    "harmful" | "unharmful" | null
  adversarial    true | false | null   (jailbreak-style wrapping of the prompt)
  harm_category  source category label: string, list, or null
  prompt_harm_basis  only when prompt_harm is implied by the dataset/subset/
                 category definition rather than annotated per row:
                 "dataset" | "subset" | "category"

Responses that carry labels have these response-meta keys:

  response_harm  "harmful" | "unharmful" | null
  refusal        true | false | null

Original label fields are kept next to them. When rows that are merged into
one record (same prompt) disagree on a prompt label, the normalized key is
null and the distinct raw values are listed under "<key>_conflict".

Representations
  allenai/wildjailbreak   one record per row; context = adversarial prompt if
      present else vanilla, the vanilla prompt in meta.vanilla; completion as
      the response. eval -> test.
  allenai/wildguardmix    one record per distinct prompt (train lists each
      prompt once without a response plus once per refusal/compliance
      response); responses carry refusal / harm labels.
  allenai/coconot         original (noncompliance; train has noncompliant
      responses), contrast (should comply; test only), pref (chosen =
      compliant gpt-4 reply, rejected = noncompliant reply, chosen_rejected).
      meta.expected_behavior = "noncompliance" | "compliance".
  nvidia/Aegis-AI-Content-Safety-Dataset-2.0 and
  nvidia/Nemotron-Safety-Guard-Dataset-v3  one record per (id, prompt text)
      within a language/split; the original response and the synthetic
      refusal_data_augmentation responses with the same id become several
      responses. "REDACTED" prompts (suicide-detection samples that are not
      redistributed) are dropped. Nemotron tag == "jailbreaking" ->
      adversarial = true. Nemotron's adapted_samples.zip is not used (the
      adapted samples are already in the per-language files).
  ToxicityPrompts/PolyGuardMix  one record per distinct (language, source,
      prompt); responses carry refusal / harm labels. subset = metadata.source;
      the wildguardmix_original subset is a copy of allenai/wildguardmix train
      (meta.duplicate_of).
  NASK-PIB/RefusEU        lang_* configs, chosen_rejected (multilingual/ is an
      exact copy of the union of lang_* and is skipped); evaluation config ->
      test, prompts only.
  CohereLabs/aya_redteaming  one record per prompt, all harmful.
  bench-llm/or-bench      80k -> train (unharmful), hard-1k -> test
      (unharmful), toxic -> test (harmful). 80k rows whose prompt is in
      hard-1k have meta.in_or_bench_hard_1k = true (hold-out leakage).
  BSC-LT/ALIA-2606-DPO-safety  chosen_rejected; LLM-judge safety scores in
      the response meta (safety_score) and the preference raw; response_harm
      is left null because many score-0 "chosen" replies still comply.
"""
import collections
import csv
import hashlib
import json
import os
import re

from common import (iter_json, iter_jsonl, iter_parquet, msg, norm_messages,
                    register, response)

csv.field_size_limit(2 ** 31 - 1)

HARMFUL, UNHARMFUL = "harmful", "unharmful"


def _meta(prompt_harm, adversarial, harm_category, **rest):
    m = {"prompt_harm": prompt_harm, "adversarial": adversarial,
         "harm_category": harm_category}
    m.update(rest)
    return m


def _nonempty(s):
    return isinstance(s, str) and s.strip() != ""


def _distinct(values):
    out = []
    for v in values:
        if v is not None and v not in out:
            out.append(v)
    return out


def _agree(values):
    """(value, None) if all non-null values agree, (None, [distinct]) if they
    conflict, (None, None) if there are none."""
    vals = _distinct(values)
    if len(vals) == 1:
        return vals[0], None
    return None, (vals or None)


def _one_or_list(values):
    vals = _distinct(values)
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else vals


def _cats(s):
    """Comma-separated category string -> list ('None'/'' -> [])."""
    if not s or s.strip() in ("None", "none", "null"):
        return []
    return [c.strip() for c in s.split(",") if c.strip()]


def _iter_csv(path, delimiter=","):
    with open(path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f, delimiter=delimiter)):
            yield i, row


def _with_conflict(meta, key, value, conflict):
    meta[key] = value
    if conflict:
        meta[key + "_conflict"] = conflict


# ------------------------------------------------------------ WildJailbreak

@register("allenai/wildjailbreak")
def wildjailbreak(ctx):
    for subset, split, rel in (("train", "train", "train/train.tsv"),
                               ("eval", "test", "eval/eval.tsv")):
        for path in ctx.files(rel):
            for i, row in _iter_csv(path, delimiter="\t"):
                vanilla = row.get("vanilla") or ""
                adv = row.get("adversarial") or ""
                dt = row.get("data_type") or ""
                if _nonempty(adv):
                    prompt, is_adv = adv, True
                elif _nonempty(vanilla):
                    prompt, is_adv = vanilla, False
                else:
                    ctx.skip("empty_prompt")
                    continue
                harm = (HARMFUL if dt.endswith("_harmful") else
                        UNHARMFUL if dt.endswith("_benign") else None)
                meta = _meta(harm, is_adv, None, data_type=dt,
                             vanilla=vanilla if is_adv and vanilla else None)
                if "label" in row:
                    meta["label"] = row["label"]
                if dt.startswith("adversarial") and not is_adv:
                    # 21 train rows: adversarial text missing, vanilla used
                    meta["adversarial_text_missing"] = True
                completion = row.get("completion")
                yield ctx.record(
                    subset=subset, split=split, source_id="{}/{}".format(subset, i),
                    context=[msg("user", prompt)],
                    responses=[response(completion)] if _nonempty(completion) else [],
                    language="en", meta=meta)


# ------------------------------------------------------------- WildGuardMix

_REFUSAL = {"refusal": True, "compliance": False}


@register("allenai/wildguardmix")
def wildguardmix(ctx):
    for split, rel in (("train", "train/*.parquet"), ("test", "test/*.parquet")):
        groups = collections.OrderedDict()   # prompt -> [(row index, row)]
        for i, (_, row) in enumerate(iter_parquet(ctx.files(rel))):
            if not _nonempty(row.get("prompt")):
                ctx.skip("empty_prompt")
                continue
            groups.setdefault(row["prompt"], []).append((i, row))
        for prompt, rows in groups.items():
            responses = []
            for _, r in rows:
                text = r.get("response")
                if not _nonempty(text):
                    if r.get("response_refusal_label") or r.get("response_harm_label"):
                        ctx.skip("labelled_response_without_text")
                    continue
                rmeta = {"response_harm": r.get("response_harm_label"),
                         "refusal": _REFUSAL.get(r.get("response_refusal_label")),
                         "response_refusal_label": r.get("response_refusal_label"),
                         "response_harm_label": r.get("response_harm_label")}
                for k in ("response_refusal_agreement", "response_harm_agreement"):
                    if k in r:
                        rmeta[k] = r[k]
                responses.append(response(text, **rmeta))
            harm, harm_conf = _agree(r["prompt_harm_label"] for _, r in rows)
            adv, adv_conf = _agree(r["adversarial"] for _, r in rows)
            meta = _meta(None, None, _one_or_list(r["subcategory"] for _, r in rows))
            _with_conflict(meta, "prompt_harm", harm, harm_conf)
            _with_conflict(meta, "adversarial", adv, adv_conf)
            meta["subcategory"] = meta["harm_category"]
            if split == "test":
                meta["prompt_harm_agreement"] = rows[0][1].get("prompt_harm_agreement")
            meta["n_source_rows"] = len(rows)
            yield ctx.record(
                subset="default", split=split,
                source_id="{}/{}".format(split, rows[0][0]),
                context=[msg("user", prompt)], responses=responses,
                language="en", meta=meta)


# ------------------------------------------------------------------ CoCoNot

@register("allenai/coconot")
def coconot(ctx):
    for subset, split in (("original", "train"), ("original", "test"),
                          ("contrast", "test"), ("pref", "train")):
        for _, row in iter_parquet(ctx.files("{}/{}-*.parquet".format(subset, split))):
            if not _nonempty(row.get("prompt")):
                ctx.skip("empty_prompt")
                continue
            noncompliance = subset == "original"
            if noncompliance:
                harm = (HARMFUL if row["category"] == "Requests with safety concerns"
                        else UNHARMFUL)
            else:
                harm = UNHARMFUL
            meta = _meta(harm, None, row.get("subcategory"),
                         prompt_harm_basis="category",
                         expected_behavior="noncompliance" if noncompliance else "compliance",
                         category=row.get("category"), subcategory=row.get("subcategory"),
                         id=row.get("id"))
            responses, preference = [], None
            if subset == "pref":
                responses = [
                    response(row["chosen"], model=row.get("chosen_model") or None,
                             refusal=False),
                    response(row["rejected"], model=row.get("rejected_model") or None,
                             refusal=True)]
                preference = {"type": "chosen_rejected", "chosen": 0, "rejected": 1,
                              "raw": "chosen/rejected columns"}
            elif _nonempty(row.get("response")):
                # original/train: noncompliant reference responses
                responses = [response(row["response"], refusal=noncompliance)]
            yield ctx.record(
                subset=subset, split=split,
                source_id="{}/{}/{}".format(subset, split, row["id"]),
                context=[msg("user", row["prompt"])], responses=responses,
                preference=preference, language="en", meta=meta)


# ----------------------------------------------- Aegis 2.0 / Nemotron-v3

_SAFE = {"safe": UNHARMFUL, "unsafe": HARMFUL}
AEGIS_MODEL = "mistralai/Mistral-7B-v0.1"
AEGIS_REFUSAL_MODEL = "google/gemma-2-27b-it"


def _aegis_records(ctx, rows, subset, split, language, sid_prefix, models):
    """Aegis-format rows -> one record per (id, prompt text), in order of
    first appearance. models: (default response model, refusal model);
    language None -> the rows' own `language` label."""
    groups = collections.OrderedDict()
    for r in rows:
        p = r.get("prompt")
        if isinstance(p, str) and p.strip() == "REDACTED":
            ctx.skip("redacted_prompt")
            continue
        if not _nonempty(p):
            ctx.skip("empty_prompt")
            continue
        groups.setdefault((r["id"], p), []).append(r)
    variant = collections.Counter()
    for (rid, prompt), rs in groups.items():
        variant[rid] += 1
        sid = sid_prefix + rid + ("" if variant[rid] == 1 else "#{}".format(variant[rid]))
        responses, seen = [], set()
        for r in rs:
            text = r.get("response")
            if not _nonempty(text):
                if r.get("response_label"):
                    ctx.skip("labelled_response_without_text")
                continue
            if text in seen:
                ctx.skip("duplicate_response_in_group")
                continue
            seen.add(text)
            src = r.get("response_label_source")
            is_ref = src == "refusal_data_augmentation"
            rmeta = {"response_harm": _SAFE.get(r.get("response_label")),
                     "refusal": True if is_ref else None,
                     "response_label": r.get("response_label") or None,
                     "response_label_source": src,
                     "violated_categories": _cats(r.get("violated_categories")) or None}
            if r.get("tag"):
                rmeta["tag"] = r["tag"]
            responses.append(response(text, model=models[1] if is_ref else models[0],
                                      **rmeta))
        label, label_conf = _agree(r.get("prompt_label") for r in rs)
        cats = _distinct(c for r in rs for c in _cats(r.get("violated_categories")))
        tags = _distinct(r.get("tag") for r in rs)
        meta = _meta(_SAFE.get(label), True if tags == ["jailbreaking"] else None,
                     cats or None, id=rid)
        _with_conflict(meta, "prompt_label", label, label_conf)
        if label_conf:
            meta["prompt_harm_conflict"] = [_SAFE.get(v, v) for v in label_conf]
        meta["prompt_label_source"] = _one_or_list(r.get("prompt_label_source") for r in rs)
        if tags:
            meta["tag"] = _one_or_list(tags)
        meta["n_source_rows"] = len(rs)
        yield ctx.record(
            subset=subset, split=split, source_id=sid,
            context=[msg("user", prompt)], responses=responses,
            language=language if language is not None else rs[0].get("language"),
            meta=meta)


@register("nvidia/Aegis-AI-Content-Safety-Dataset-2.0")
def aegis2(ctx):
    for split, names in (("train", ["train.json", "refusals_train.json"]),
                         ("validation", ["validation.json", "refusals_validation.json"]),
                         ("test", ["test.json"])):
        paths = [ctx.files(n)[0] for n in names]
        rows = (row for _, row in iter_json(paths))
        yield from _aegis_records(ctx, rows, "default", split, "en",
                                  split + "/", (AEGIS_MODEL, AEGIS_REFUSAL_MODEL))


_NEMO_SPLIT = {"train": "train", "valid": "validation", "test": "test"}


@register("nvidia/Nemotron-Safety-Guard-Dataset-v3")
def nemotron_safety_guard_v3(ctx):
    for path in ctx.files("*/*.jsonl"):
        lang = os.path.basename(os.path.dirname(path))
        split = _NEMO_SPLIT[os.path.basename(path)[:-len(".jsonl")]]
        # responses are translations / adaptations of Aegis 2.0 (or newly
        # generated jailbreak responses): model unknown
        rows = (row for _, row in iter_jsonl([path]))
        yield from _aegis_records(ctx, rows, lang, split, None,
                                  "{}/{}/".format(lang, split), (None, None))


# ------------------------------------------------------------- PolyGuardMix

_YESNO_HARM = {"yes": HARMFUL, "no": UNHARMFUL}
_YESNO_BOOL = {"yes": True, "no": False}


def _pgm_cats(s):
    """Space-separated S-codes ('S7 S11'), 'None' -> []."""
    return [t for t in re.split(r"[\s,]+", s or "") if t and t != "None"]


def _pgm_group(ctx, rows):
    first, r0 = rows[0]
    md = r0.get("metadata") or {}
    lang, src = md.get("language"), md.get("source")
    if not _nonempty(r0.get("prompt")):
        for _ in rows:
            ctx.skip("empty_prompt")
        return None
    responses, seen = [], set()
    for _, r in rows:
        text = r.get("response")
        if not _nonempty(text):
            continue   # prompt-only rows (labels are placeholders)
        if text in seen:
            ctx.skip("duplicate_response_in_group")
            continue
        seen.add(text)
        responses.append(response(
            text,
            response_harm=_YESNO_HARM.get(r.get("response_harm_label")),
            refusal=_YESNO_BOOL.get(r.get("response_refusal_label")),
            response_harm_label=r.get("response_harm_label"),
            response_refusal_label=r.get("response_refusal_label"),
            response_safety_categories=_pgm_cats(r.get("response_safety_categories")) or None))
    label, label_conf = _agree(r.get("prompt_harm_label") for _, r in rows)
    cats = _distinct(c for _, r in rows for c in _pgm_cats(r.get("prompt_safety_categories")))
    meta = _meta(_YESNO_HARM.get(label), None, cats or None)
    if label_conf:
        meta["prompt_harm_conflict"] = [_YESNO_HARM.get(v, v) for v in label_conf]
    meta["prompt_harm_label"] = label if label_conf is None else label_conf
    meta["pgm_source"] = src
    if src == "wildguardmix_original":
        meta["duplicate_of"] = "allenai/wildguardmix"
    meta["n_source_rows"] = len(rows)
    return ctx.record(
        subset=src or "unknown", split="train", source_id="row{}".format(first),
        context=[msg("user", r0["prompt"])], responses=responses,
        language=lang, meta=meta)


@register("ToxicityPrompts/PolyGuardMix")
def polyguardmix(ctx):
    """Rows of one prompt (prompt-only + refusal + compliance) are scattered
    over all shards: assign a group id per (language, source, prompt),
    stable-sort the row indices by group id and stream the table in that
    order, so groups come out contiguous and in order of first appearance."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.concat_tables([pq.read_table(p) for p in ctx.files("data/train-*.parquet")])
    n = table.num_rows
    gids = np.empty(n, dtype=np.int64)
    key_gid = {}
    i = 0
    for batch in table.select(["prompt", "metadata"]).to_batches(max_chunksize=65536):
        d = batch.to_pydict()
        for p, md in zip(d["prompt"], d["metadata"]):
            md = md or {}
            key = hashlib.blake2b("\x00".join(
                (md.get("language") or "", md.get("source") or "", p or "")
            ).encode("utf-8"), digest_size=16).digest()
            g = key_gid.get(key)
            if g is None:
                g = key_gid[key] = len(key_gid)
            gids[i] = g
            i += 1
    del key_gid
    order = np.argsort(gids, kind="stable")
    cur, rows = -1, []
    chunk = 100_000
    for s in range(0, n, chunk):
        idx = order[s:s + chunk]
        for j, r in zip(idx.tolist(), table.take(pa.array(idx)).to_pylist()):
            g = int(gids[j])
            if g != cur:
                if rows:
                    rec = _pgm_group(ctx, rows)
                    if rec is not None:
                        yield rec
                cur, rows = g, []
            elif r["prompt"] != rows[0][1]["prompt"]:
                raise ValueError("hash collision in PolyGuardMix grouping")
            rows.append((j, r))
    if rows:
        rec = _pgm_group(ctx, rows)
        if rec is not None:
            yield rec


# ------------------------------------------------------------------ RefusEU

@register("NASK-PIB/RefusEU")
def refuseu(ctx):
    # multilingual/ == union of lang_*/ (verified row by row): not read
    for path in ctx.files("lang_*/*.parquet"):
        subset = os.path.basename(os.path.dirname(path))
        split = "test" if os.path.basename(path).startswith("test") else "train"
        seen = collections.Counter()
        for _, row in iter_parquet([path]):
            chosen = norm_messages(row["chosen"])
            rejected = norm_messages(row["rejected"])
            if (len(chosen) < 2 or chosen[:-1] != rejected[:-1]
                    or chosen[-1]["role"] != "assistant"
                    or rejected[-1]["role"] != "assistant"
                    or chosen[-2]["role"] != "user"):
                ctx.skip("chosen_rejected_context_mismatch")
                continue
            if not _nonempty(chosen[-2]["content"]):
                ctx.skip("empty_prompt")
                continue
            # id = "<row_id>_<lang>"; row_id restarts per category
            base = "{}/{}/{}".format(split, row["category"], row["id"])
            seen[base] += 1
            sid = base if seen[base] == 1 else "{}#{}".format(base, seen[base])
            yield ctx.record(
                subset=subset, split=split, source_id=sid, context=chosen[:-1],
                responses=[response(chosen[-1]["content"]),
                           response(rejected[-1]["content"])],
                preference={"type": "chosen_rejected", "chosen": 0, "rejected": 1,
                            "raw": "chosen/rejected columns"},
                language=row.get("lang"),
                meta=_meta(HARMFUL, None, row.get("category"),
                           prompt_harm_basis="dataset", id=row.get("id"),
                           row_id=row.get("row_id"), category=row.get("category")))
    for _, row in iter_parquet(ctx.files("evaluation/*.parquet")):
        if not _nonempty(row.get("prompt")):
            ctx.skip("empty_prompt")
            continue
        yield ctx.record(
            subset="evaluation", split="test",
            source_id="eval/{}/{}".format(row["lang"], row["row_id"]),
            context=[msg("user", row["prompt"])], language=row.get("lang"),
            meta=_meta(HARMFUL, None, None, prompt_harm_basis="dataset",
                       row_id=row.get("row_id")))


# ---------------------------------------------------------- aya_redteaming

@register("CohereLabs/aya_redteaming")
def aya_redteaming(ctx):
    for path in ctx.files("aya_*.jsonl"):
        code = os.path.basename(path)[len("aya_"):-len(".jsonl")]
        for i, (_, row) in enumerate(iter_jsonl([path])):
            if not _nonempty(row.get("prompt")):
                ctx.skip("empty_prompt")
                continue
            cat = row.get("harm_category")
            try:
                cat = json.loads(cat) if isinstance(cat, str) else cat
            except ValueError:
                pass
            yield ctx.record(
                subset=code, split="train", source_id="{}/{}".format(code, i),
                context=[msg("user", row["prompt"])], language=row.get("language"),
                meta=_meta(HARMFUL, None, cat, prompt_harm_basis="dataset",
                           global_or_local=row.get("global_or_local"),
                           literal_translation=row.get("literal_translation"),
                           semantic_translation=row.get("semantic_translation"),
                           explanation=row.get("explanation")))


# ----------------------------------------------------------------- OR-Bench

@register("bench-llm/or-bench")
def or_bench(ctx):
    hard = {row["prompt"] for _, row in _iter_csv(ctx.files("or-bench-hard-1k.csv")[0])}
    for subset, split, harm in (("or-bench-80k", "train", UNHARMFUL),
                                ("or-bench-hard-1k", "test", UNHARMFUL),
                                ("or-bench-toxic", "test", HARMFUL)):
        for i, row in _iter_csv(ctx.files(subset + ".csv")[0]):
            if not _nonempty(row.get("prompt")):
                ctx.skip("empty_prompt")
                continue
            meta = _meta(harm, None, row.get("category"), prompt_harm_basis="subset",
                         category=row.get("category"))
            if subset == "or-bench-80k" and row["prompt"] in hard:
                meta["in_or_bench_hard_1k"] = True
            yield ctx.record(
                subset=subset, split=split, source_id="{}/{}".format(subset, i),
                context=[msg("user", row["prompt"])], language="en", meta=meta)


# ------------------------------------------------------- ALIA DPO safety

@register("BSC-LT/ALIA-2606-DPO-safety")
def alia_dpo_safety(ctx):
    for _, row in iter_parquet(ctx.files("data/*.parquet")):
        context = norm_messages(row["prompt"] or [])
        chosen = norm_messages(row["chosen"] or [])
        rejected = norm_messages(row["rejected"] or [])
        if not context or context[-1]["role"] != "user" or not _nonempty(context[-1]["content"]):
            ctx.skip("no_user_turn")
            continue
        if (len(chosen) != 1 or len(rejected) != 1
                or chosen[0]["role"] != "assistant" or rejected[0]["role"] != "assistant"):
            ctx.skip("unexpected_chosen_rejected_shape")
            continue
        md = row.get("metadata") or {}
        cs, rs = md.get("chosen_score"), md.get("rejected_score")
        yield ctx.record(
            subset="default", split="train", source_id=row["id"], context=context,
            responses=[
                # judge score 0 (safe) .. negative (harmful); not mapped to
                # response_harm: many score-0 "chosen" replies still comply
                # with the harmful request
                response(chosen[0]["content"], model=md.get("chosen_origin"),
                         response_harm=None, safety_score=cs),
                response(rejected[0]["content"], model=md.get("rejected_origin"),
                         response_harm=None, safety_score=rs)],
            preference={"type": "chosen_rejected", "chosen": 0, "rejected": 1,
                        "raw": {"chosen_score": cs, "rejected_score": rs}},
            language=row.get("lang"),
            meta=_meta(HARMFUL, None, row.get("harm_category"),
                       prompt_harm_basis="dataset",
                       harm_category_code=row.get("harm_category_code"),
                       id=row.get("id"), type=row.get("type"),
                       prompt_origin=md.get("prompt_origin"),
                       preference_origin=row.get("preference_origin")))
