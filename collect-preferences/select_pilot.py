#!/usr/bin/env python3
"""Pick the pilot subset: about 1,000 pairs, stratified so that the small languages and the safety and
instruction-following slices are well represented, with everything the judges and the analysis need.

  collect-preferences/run.sh select_pilot.py [--n 1000] [--seed 20261114]

Selection: weighted sampling without replacement. Each language gets a target count (English 20% of the
subset; the five large languages 25%, split evenly; the eighteen mid-size languages 30%, split evenly; the
fourteen small languages 25%, split evenly), and within a language the safety slices are weighted 3x and
instruction-following prompts 2x. Pairs with an empty answer are excluded (nothing to judge).

Writes data/preferences/pilot/pairs.parquet, one row per pair: the prompt (messages, JSON), the two final
answers in slot order (x = slot 0, y = slot 1), their models (analysis only; a judge never sees them), modes,
finish reasons, token counts, and the OpenLID language of each answer with its confidence.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "collect-prompts", "compile"))
OUT = os.path.join(ROOT, "data", "preferences", "pilot")

LARGE = ["es", "de", "fr", "it", "pt"]
MID = ["pl", "nl", "tr", "cs", "sv", "ro", "el", "uk", "hu", "da", "fi", "bg", "bs", "nb", "sk", "hr", "ca", "lt"]
SMALL = ["sl", "et", "lv", "sq", "sr", "mk", "ka", "is", "gl", "eu", "nn", "ga", "mt", "ltg"]
TIER_SHARE = {"en": 0.20, "large": 0.25, "mid": 0.30, "small": 0.25}
CAT_WEIGHT = {"safety_harmful": 3.0, "safety_benign": 3.0, "instruction_following": 2.0, "general": 1.0}

LANG_NAMES = {"en": "English", "es": "Spanish", "de": "German", "fr": "French", "it": "Italian", "pt": "Portuguese",
              "pl": "Polish", "nl": "Dutch", "tr": "Turkish", "cs": "Czech", "sv": "Swedish", "ro": "Romanian",
              "el": "Greek", "uk": "Ukrainian", "hu": "Hungarian", "da": "Danish", "fi": "Finnish", "bg": "Bulgarian",
              "bs": "Bosnian", "nb": "Norwegian (Bokmål)", "sk": "Slovak", "hr": "Croatian", "ca": "Catalan",
              "lt": "Lithuanian", "sl": "Slovenian", "et": "Estonian", "lv": "Latvian", "sq": "Albanian",
              "sr": "Serbian", "mk": "Macedonian", "ka": "Georgian", "is": "Icelandic", "gl": "Galician",
              "eu": "Basque", "nn": "Norwegian (Nynorsk)", "ga": "Irish", "mt": "Maltese", "ltg": "Latgalian"}


def tier(lang):
    return "en" if lang == "en" else "large" if lang in LARGE else "mid" if lang in MID else "small"


def pair_row(i, p, x, y, B):
    """one row of a pairs file (also used by build_pairs.py): the prompt and its labels, both answers in slot
    order (x = slot 0, y = slot 1), their models, modes, finish reasons and token counts, and the OpenLID language
    of each answer with its confidence (None for an empty answer). B: build_pool with the OpenLID model loaded."""
    def lid(text):
        return B.lid(B.clean_for_lid(text, 3000))[:2] if text.strip() else (None, 0.0)
    (lx, px), (ly, py) = lid(x["response"]), lid(y["response"])
    return {
        "id": i, "language": p["language"], "language_name": LANG_NAMES[p["language"]], "tier": tier(p["language"]),
        "category": p["category"], "task_type": p["task_type"], "safety_label": p["safety_label"],
        "locale_dependence": p["locale_dependence"], "complexity": p["complexity"],
        "verifiable_instructions": json.dumps(p.get("verifiable_instructions"), ensure_ascii=False)
        if p.get("verifiable_instructions") else None,
        "messages": json.dumps(p["messages"], ensure_ascii=False), "n_turns": len(p["messages"]),
        "response_x": x["response"], "response_y": y["response"],
        "model_x": x["model"], "model_y": y["model"], "group_x": x["model_group"], "group_y": y["model_group"],
        "mode_x": x["mode"], "mode_y": y["mode"], "has_reasoning_x": bool(x["reasoning"].strip()),
        "has_reasoning_y": bool(y["reasoning"].strip()),
        "finish_x": x["finish_reason"], "finish_y": y["finish_reason"],
        "prompt_tokens": x["prompt_tokens"], "tokens_x": x["completion_tokens"], "tokens_y": y["completion_tokens"],
        "chars_x": len(x["response"]), "chars_y": len(y["response"]),
        "lid_x": lx, "lid_conf_x": float(px), "lid_y": ly, "lid_conf_y": float(py),
    }


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261114)
    ap.add_argument("--refresh", action="store_true",
                    help="keep the pairs of the existing pairs.parquet and refresh their answers and derived fields from "
                         "responses.parquet (after continue_truncated.py); the previous file is kept as "
                         "pairs.before_continuation.parquet")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    prompts = {}
    with open(os.path.join(ROOT, "data", "compiled", "prompts.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            prompts[o["id"]] = o
    resp = pq.read_table(os.path.join(ROOT, "data", "responses", "responses.parquet"),
                         columns=["id", "slot", "model", "model_group", "mode", "response", "reasoning", "finish_reason",
                                  "prompt_tokens", "completion_tokens", "think_closed"]).to_pylist()
    pairs = collections.defaultdict(dict)
    for r in resp:
        pairs[r["id"]][r["slot"]] = r
    ids = [i for i, d in pairs.items() if d[0]["response"].strip() and d[1]["response"].strip()]
    ids.sort()
    print("{} pairs, {} with two non-empty answers".format(len(pairs), len(ids)))
    refreshed = None
    if args.refresh:
        import shutil
        old = os.path.join(OUT, "pairs.parquet")
        backup = os.path.join(OUT, "pairs.before_continuation.parquet")
        if not os.path.exists(backup):
            shutil.copy2(old, backup)
        refreshed = pq.read_table(backup, columns=["id"]).column("id").to_pylist()

    # target count per language
    by_lang = collections.defaultdict(list)
    for i in ids:
        by_lang[prompts[i]["language"]].append(i)
    per_tier = collections.defaultdict(list)
    for lang in by_lang:
        per_tier[tier(lang)].append(lang)
    target = {}
    for t, langs in per_tier.items():
        for lang in langs:
            target[lang] = TIER_SHARE[t] * args.n / len(langs)
    chosen = []
    for lang, cand in sorted(by_lang.items()):
        k = int(round(target[lang]))
        k = min(k, len(cand))
        w = np.array([CAT_WEIGHT[prompts[i]["category"]] for i in cand], dtype=float)
        pick = rng.choice(len(cand), size=k, replace=False, p=w / w.sum())
        chosen += [cand[j] for j in pick]
    chosen.sort()
    if refreshed is not None:                     # same pairs as before, current answers
        chosen = refreshed

    import build_pool as B
    B.load_model(os.path.join(ROOT, "data", "models", "openlid-v3", "openlid-v3.bin"), {})
    rows = [pair_row(i, prompts[i], pairs[i][0], pairs[i][1], B) for i in chosen]
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "pairs.parquet")
    pq.write_table(pa.Table.from_pylist(rows), path)
    print("{} pairs -> {}".format(len(rows), path))
    for key in ["tier", "category", "task_type"]:
        print(key, dict(collections.Counter(r[key] for r in rows).most_common()))
    print("languages", dict(sorted(collections.Counter(r["language"] for r in rows).items())))
    print("same model both sides: {}; prompts > 1 turn: {}".format(sum(r["model_x"] == r["model_y"] for r in rows),
                                                                   sum(r["n_turns"] > 1 for r in rows)))


if __name__ == "__main__":
    main()
