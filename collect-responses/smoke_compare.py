#!/usr/bin/env python3
"""Compare sampling settings on the smoke test's general prompts: baseline <model>.jsonl against tagged
runs <model><tag>.jsonl (smoke_test.py run --override ... --tag ...).

  collect-responses/run.sh smoke_compare.py [--tags @p95,@p95norep,@norep]

Per model and mode: stop / cut / open as in the smoke report, lang = OpenLID language matches the prompt
(translation and language_specific prompts excluded), salad = more than 1% of the letters are in a script
the prompt's language does not use (word salad across languages), loop = the text compresses below 0.15
(repetition), tok = mean answer + reasoning tokens (chars / 4).
"""
import argparse
import json
import os
import sys
import unicodedata
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SMOKE = os.path.join(ROOT, "data", "responses", "smoke")
SCRIPTS = {"el": {"GREEK"}, "uk": {"CYRILLIC"}, "bg": {"CYRILLIC"}, "mk": {"CYRILLIC"}, "sr": {"CYRILLIC"},
           "ka": {"GEORGIAN"}}


def script(ch):
    name = unicodedata.name(ch, "")
    return name.split(" ")[0] if name else ""


def foreign_share(text, lang):
    ok = {"LATIN"} | SCRIPTS.get(lang, set())
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(script(c) not in ok for c in letters) / len(letters)


def loop(text):
    b = text.encode("utf-8")
    return len(b) > 1000 and len(zlib.compress(b)) / len(b) < 0.15


def stats(rows, B):
    gen = [r for r in rows if r["smoke_case"] not in ("identity", "overtrigger")]
    if not gen:
        return None
    n = len(gen)
    lang_rows = [r for r in gen if r["locale_dependence"] != "language_specific" and r["task_type"] != "translation"
                 and r["response"]]
    from smoke_test import same_lang
    lang_ok = sum(bool(l) and same_lang(l, r["language"]) for r in lang_rows
                  for l in [B.lid(B.clean_for_lid(r["response"], 3000))[0]])
    text = lambda r: r["response"] or r["reasoning"]  # noqa: E731
    return {"n": n, "stop": sum(r["finish_reason"] == "stop" and r["think_closed"] is not False for r in gen) / n,
            "cut": sum(r["finish_reason"] == "length" for r in gen) / n,
            "lang": lang_ok / max(1, len(lang_rows)),
            "salad": sum(foreign_share(r["response"], r["language"]) > 0.01 for r in gen) / n,
            "loop": sum(loop(text(r)) for r in gen) / n,
            "tok": sum(len(r["response"]) + len(r["reasoning"]) for r in gen) / n / 4}


WEIGHT_TIERS = [(0.25, 0.25), (0.10, 0.5), (0.0, 1.0)]     # salad rate >= threshold -> weight


def weights():
    """word-salad rate per model under the pool's sampling settings (smoke test, general prompts, all modes):
    a response counts if more than 1% of its letters are in a script the prompt's language does not use, or if
    the answer (or the reasoning, when there is no answer) is a loop. The weight for assign.py follows WEIGHT_TIERS."""
    import yaml
    names = [m["name"] for m in yaml.safe_load(open(os.path.join(HERE, "models.yaml")))["models"]]
    lines = ["{:36s} {:>4s} {:>6s} {:>6s} {:>6s} {:>6s}".format("model", "n", "salad", "loop", "either", "weight")]
    out = {}
    for name in names:
        path = os.path.join(SMOKE, name + ".jsonl")
        if not os.path.exists(path):
            lines.append("{:36s} not run".format(name))
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        gen = [r for r in rows if r["smoke_case"] not in ("identity", "overtrigger")]
        sal = [foreign_share(r["response"], r["language"]) > 0.01 for r in gen]
        lp = [loop(r["response"] or r["reasoning"]) for r in gen]
        rate = sum(a or b for a, b in zip(sal, lp)) / len(gen)
        w = next(w for t, w in WEIGHT_TIERS if rate >= t)
        out[name] = (rate, w)
        lines.append("{:36s} {:4d} {:6.0%} {:6.0%} {:6.0%} {:6.2f}".format(name, len(gen), sum(sal) / len(gen),
                                                                          sum(lp) / len(gen), rate, w))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(SMOKE, "WEIGHTS.txt"), "w") as f:
        f.write(text + "\n\nsalad: >1% of the letters in a script the prompt language does not use; loop: compresses "
                "below 0.15; weight: either >= 25% -> 0.25, >= 10% -> 0.5, else 1.0 (assign.py draws models in proportion).\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="@p95,@p95norep,@norep")
    ap.add_argument("--weights", action="store_true", help="word-salad rate and sampling weight per model")
    args = ap.parse_args()
    if args.weights:
        return weights()
    sys.path.insert(0, os.path.join(ROOT, "collect-prompts", "compile"))
    sys.path.insert(0, HERE)
    import build_pool as B
    B.load_model(os.path.join(ROOT, "data", "models", "openlid-v3", "openlid-v3.bin"), {})
    import yaml
    names = [m["name"] for m in yaml.safe_load(open(os.path.join(HERE, "models.yaml")))["models"]]
    lines = ["{:36s} {:6s} {:11s} {:>4s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>6s}".format(
        "model", "mode", "sampling", "n", "stop", "cut", "lang", "salad", "loop", "tok")]
    for name in names:
        variants = [("baseline", os.path.join(SMOKE, name + ".jsonl"))] + \
                   [(t, os.path.join(SMOKE, name + t + ".jsonl")) for t in args.tags.split(",")]
        variants = [(t, p) for t, p in variants if os.path.exists(p)]
        if len(variants) < 2:
            continue
        loaded = [(t, [json.loads(l) for l in open(p, encoding="utf-8")]) for t, p in variants]
        for mode in sorted({r["mode"] for _, rows in loaded for r in rows}):
            for t, rows in loaded:
                s = stats([r for r in rows if r["mode"] == mode], B)
                if s:
                    lines.append("{:36s} {:6s} {:11s} {:4d} {:5.0%} {:5.0%} {:5.0%} {:5.0%} {:5.0%} {:6.0f}".format(
                        name, mode, t, s["n"], s["stop"], s["cut"], s["lang"], s["salad"], s["loop"], s["tok"]))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(SMOKE, "SAMPLING_COMPARE.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
