#!/usr/bin/env python3
"""Export the compiled prompt set as one shuffled JSONL file with self-describing field names.

  ./run.sh compile/export_jsonl.py [--prompts data/compiled/prompts.parquet] [--out data/compiled/prompts.jsonl] [--seed 20261113]

One row per line, in a random order (fixed seed). The fields, their values and how they were made
are described in data/compiled/prompts.README.md (written by this script, from CARD below).
Mapping from prompts.parquet:
  id -> id, lang -> language, context -> messages,
  fill (native / translated / source_translated) -> translation_status
      (native / machine_translated / translated_in_source_dataset),
  translator -> translation_model, source_lang -> translated_from_language (machine_translated only),
  purpose (general / if / safety_benign / safety_harmful) -> category
      (general / instruction_following / safety_benign / safety_harmful),
  task -> task_type (for safety_harmful rows the parquet holds the allocation cell, "harmful" /
      "jailbreak"; the annotated task is restored from data/llm/annotate),
  domain -> subject_domain, origin -> author_type (real_chatbot_user / human_written / llm_generated /
      web_scraped), safety -> safety_label, locale -> locale_dependence,
  complexity (1-3) -> complexity (simple / moderate / hard), constraints -> explicit_constraint_types,
  gist -> request_summary, instruction_ids + if_kwargs -> verifiable_instructions,
  source + subset + source_uid + license -> source {dataset, subset, item_id, license}.
Left out (all still in prompts.parquet): responses and preference; turns (follows from messages),
license_flag (follows from license), source_lang for non-translated rows (= language), alloc_level
and notes (pipeline internals), source_meta (dataset-specific extras such as WildChat's hashed IPs
and locations, arena model names, EuroMultiSynth personas, every_prompt URLs). Translator labels
with a method suffix (35 Latgalian rows) are reduced to the model id; PLAN.md documents the methods.
The file is written to <out>.tmp and renamed when complete, after checks: row count, unique ids
equal to the parquet's, every conversation ending in a non-empty user turn, no missing labels.
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

import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import annotate_prompt as AP  # noqa: E402
import mt_common as M  # noqa: E402

COLUMNS = ["id", "lang", "context", "fill", "translator", "source_lang", "source", "subset", "source_uid",
           "license", "instruction_ids", "if_kwargs", "purpose", "task", "domain", "origin", "safety",
           "locale", "complexity", "constraints", "gist"]
STATUS = {"native": "native", "translated": "machine_translated", "source_translated": "translated_in_source_dataset"}
CATEGORY = {"general": "general", "if": "instruction_following", "safety_benign": "safety_benign",
            "safety_harmful": "safety_harmful"}
AUTHOR = {"real_user": "real_chatbot_user", "human": "human_written", "synthetic": "llm_generated", "web": "web_scraped"}
COMPLEXITY = {1: "simple", 2: "moderate", 3: "hard"}
LABELS = ("category", "task_type", "subject_domain", "author_type", "safety_label", "locale_dependence",
          "complexity", "explicit_constraint_types", "request_summary")

CARD = """# Multilingual prompt set (128k) for reward modelling

`prompts.jsonl`: {rows:,} prompts in {nlang} languages, one JSON object per line, in random order
(shuffle seed {seed}). Each row is a conversation that ends in a user turn to be answered; the
assistant's answer is not included. Written by `collect-prompts/compile/export_jsonl.py` from
`prompts.parquet`, which also keeps the source datasets' responses and the pipeline's internal
columns.

## Fields

| Field | Meaning |
|---|---|
| `id` | Stable row id (`mlp128k-NNNNNN`); the review notes refer to rows by it. |
| `language` | Language of the prompt: ISO 639-1 code, or 639-3 where there is none (`ltg` Latgalian). |
| `messages` | The conversation, `[{{"role": "system" / "user" / "assistant", "content": ...}}]`, ending in a user turn. Earlier assistant turns are part of the prompt. |
| `translation_status` | `native`: written in this language. `machine_translated`: translated into it by this pipeline. `translated_in_source_dataset`: published as a translation by the source dataset. Counts: {status}. |
| `translation_model` | For `machine_translated` rows, the model that translated it; otherwise null. Icelandic and Irish rows marked `anthropic/claude-opus-5-5` were translated by Claude. |
| `translated_from_language` | For `machine_translated` rows, the language of the original prompt; otherwise null. |
| `category` | `general`, `instruction_following` (from instruction-following datasets), `safety_benign` (sensitive topic, legitimate request), `safety_harmful` (a request to refuse or redirect, including jailbreaks). Counts: {category}. |
| `task_type` | Main task of the final user request ({ntask} types, e.g. `information_seeking`, `coding`, `editing`). |
| `subject_domain` | Subject area ({ndomain} values; `general` if none applies). |
| `author_type` | Who wrote the prompt: `real_chatbot_user` (logs of real chatbot use), `human_written` (annotators, volunteers, experts), `llm_generated` (synthetic), `web_scraped` (questions from web pages). Counts: {author}. |
| `safety_label` | `none`, `sensitive_benign`, `harmful`, `jailbreak` (adversarial framing to obtain harmful content). |
| `locale_dependence` | `culture_specific` (needs knowledge of a country, culture or local context), `language_specific` (about the language itself: grammar, wording, translation, proofreading), or `none`. |
| `complexity` | `simple` (short answer), `moderate`, `hard` (expert-level or long multi-step). |
| `explicit_constraint_types` | Kinds of explicit, checkable requirements the user states (e.g. `length`, `format`, `include_keywords`); empty if none. |
| `request_summary` | One English sentence (about 15 words) saying what the user wants, also for non-English prompts. |
| `verifiable_instructions` | For {nverif:,} rows (mostly `instruction_following`, a few with a safety category): IFEval-style `instruction_ids` with their `kwargs`, so a reply can be checked automatically; otherwise null. `kwargs` is a JSON string, a list with one object or null per instruction id (a string because the same key holds a string in some rows and a list in others, which Arrow-based loaders cannot read as objects). |
| `source` | Where the prompt comes from: `dataset` (Hugging Face id), `subset`, `item_id` within it, and the dataset's `license`. |

## How the labels were made

- **Predicted by an LLM:** `task_type`, `subject_domain`, `safety_label`, `locale_dependence`,
  `complexity`, `explicit_constraint_types` and `request_summary`.
  - Gemma-4-31B-it labelled every candidate prompt once, with a fixed JSON schema
    (`collect-prompts/compile/annotate_prompt.py`).
  - It saw the conversation as written in the source, with long messages shortened.
  - For machine-translated rows, the labels therefore describe the original prompt.
  - These are single-pass model judgements, not human annotations. `complexity` in particular is
    the model's estimate of how hard a good answer is.
- **Derived:**
  - `category` comes from `safety_label` and the kind of source dataset.
  - `author_type` comes from the source dataset (per-source rules, `pool_rules.py`).
- Every label is present for every row.
"""


def restored_tasks(uids):
    """the annotated task of these pool items (data/llm/annotate/out)"""
    out, want = {}, list(uids)
    for p in sorted(glob.glob(os.path.join(M.DATA, "llm", "annotate", "out", "*.parquet"))):
        t = pq.read_table(p, columns=["id", "text", "finish_reason"])
        t = t.filter(pc.is_in(t["id"], value_set=__import__("pyarrow").array(want)))
        for u, text, fr in zip(t["id"].to_pylist(), t["text"].to_pylist(), t["finish_reason"].to_pylist()):
            d = AP.parse(text) if fr == "stop" else None
            if d and d.get("task") in AP.TASKS:
                out[u] = d["task"]
    return out


def row_out(r, task):
    tr = r["fill"] == "translated"
    ds, _, item = r["source_uid"].partition("::")
    assert ds == r["source"], r["id"]
    kw = r["if_kwargs"]
    if isinstance(kw, str):
        kw = json.loads(kw)
    return {
        "id": r["id"],
        "language": r["lang"],
        "messages": [{"role": m["role"], "content": m["content"]} for m in r["context"]],
        "translation_status": STATUS[r["fill"]],
        "translation_model": (r["translator"] or "").split(" (")[0] or None if tr else None,
        "translated_from_language": r["source_lang"] if tr else None,
        "category": CATEGORY[r["purpose"]],
        "task_type": task,
        "subject_domain": r["domain"],
        "author_type": AUTHOR[r["origin"]],
        "safety_label": r["safety"],
        "locale_dependence": r["locale"],
        "complexity": COMPLEXITY[int(r["complexity"])],
        "explicit_constraint_types": list(r["constraints"] or []),
        "request_summary": r["gist"],
        # kwargs stays a JSON string: the same key holds a string in some rows and a list in others
        # (keyword1 / keyword2), which Arrow-based loaders (datasets, the Hub viewer) cannot read as objects
        "verifiable_instructions": ({"instruction_ids": list(r["instruction_ids"]), "kwargs": json.dumps(kw, ensure_ascii=False)}
                                    if r["instruction_ids"] else None),
        "source": {"dataset": ds, "subset": r["subset"], "item_id": item, "license": r["license"]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=os.path.join(M.DATA, "compiled", "prompts.parquet"))
    ap.add_argument("--out", default=os.path.join(M.DATA, "compiled", "prompts.jsonl"))
    ap.add_argument("--seed", type=int, default=20261113)
    args = ap.parse_args()
    rows = pq.read_table(args.prompts, columns=COLUMNS).to_pylist()
    src_ids = [r["id"] for r in rows]
    # safety_harmful rows carry the allocation cell ("harmful" / "jailbreak") as their task
    fix = {r["source_uid"] for r in rows if r["task"] not in AP.TASKS}
    tasks = restored_tasks(fix)
    missing = fix - set(tasks)
    assert not missing, "{} rows without an annotated task, e.g. {}".format(len(missing), sorted(missing)[:3])
    print("task_type restored from the annotations for {} rows".format(len(fix)))
    random.Random(args.seed).shuffle(rows)
    tmp = args.out + ".tmp"
    cnt = collections.defaultdict(collections.Counter)
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            o = row_out(r, tasks.get(r["source_uid"], r["task"]))
            for k in ("translation_status", "category", "author_type", "task_type", "subject_domain", "language"):
                cnt[k][o[k]] += 1
            cnt["verif"][o["verifiable_instructions"] is not None] += 1
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    ids, bad = [], collections.Counter()
    with open(tmp, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            ids.append(o["id"])
            last = o["messages"][-1]
            bad["last_turn"] += not (last["role"] == "user" and (last["content"] or "").strip())
            bad["missing_label"] += any(o[k] in (None, "") for k in LABELS if k != "explicit_constraint_types")
            bad["unknown_task"] += o["task_type"] not in AP.TASKS
    assert len(ids) == len(set(ids)) == len(src_ids) and set(ids) == set(src_ids), "id mismatch"
    assert not sum(bad.values()), dict(bad)
    os.replace(tmp, args.out)
    fmt = lambda c: ", ".join("`{}` {:,}".format(k, v) for k, v in c.most_common())
    card = CARD.format(rows=len(ids), nlang=len(cnt["language"]), seed=args.seed, status=fmt(cnt["translation_status"]),
                       category=fmt(cnt["category"]), author=fmt(cnt["author_type"]), ntask=len(cnt["task_type"]),
                       ndomain=len(cnt["subject_domain"]), nverif=cnt["verif"][True])
    card_path = os.path.join(os.path.dirname(args.out), "prompts.README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card)
    print("{} rows (shuffled, seed {}), {} unique ids -> {} ({:.0f} MB); card -> {}".format(
        len(ids), args.seed, len(set(ids)), args.out, os.path.getsize(args.out) / 1e6, card_path))


if __name__ == "__main__":
    main()
