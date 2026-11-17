#!/usr/bin/env python3
"""Stage 5m: collect TildeOpen / SalamandraTA translations (mt2_requests.py outputs).

  ./run.sh compile/mt2_collect.py --validate

Reassembles every translated message from segments.parquet (code blocks, whitespace and
letterless pieces verbatim) and runs automatic checks per translation: failed segment
(no stop), length ratio vs the source, exotic-script injection, new mixed-script words,
untranslated text (share of the source's longer words copied verbatim).
--validate: writes the blind read (manual_check/mt_validation/<group>.txt): per English
source its 2-3 versions (current translation, TildeOpen, SalamandraTA) shuffled under
opaque ids; key.tsv (hidden from reviewers) maps them back; auto_checks.json.
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

import pyarrow.parquet as pq  # noqa: E402

import mt_common as M  # noqa: E402
from mt2_requests import NAME, OUT  # noqa: E402

VAL = os.path.join(M.DATA, "compiled", "manual_check", "mt_validation")
GROUPS = [("lv", "lt"), ("et", "hu"), ("sl", "is"), ("mt", "ga"), ("eu",)]
WORD = re.compile(r"[^\W\d_]{4,}")


def outputs(d):
    out = {}
    for p in glob.glob(os.path.join(d, "out", "shard*.parquet")):
        for r in pq.read_table(p).to_pylist():
            out[r["id"]] = r
    return out


def seg_text(system, r, tgt):
    """Translated text of one segment, or None if generation did not finish."""
    if r is None or r["finish_reason"] != "stop":
        return None
    t = r["text"]
    if system == "tilde":
        if r.get("token_ids") is not None:
            t = M.tilde_sp().decode([int(x) for x in r["token_ids"]])
        t = t.split("</{}>".format(NAME[tgt]))[0]
        t = re.split(r"\n<(?:{}|English)>".format("|".join(map(re.escape, NAME.values()))), t)[0]
    t = t.strip()
    if system == "salta" and t.startswith(NAME[tgt] + ":"):   # it sometimes repeats the prompt's label
        t = t[len(NAME[tgt]) + 1:].strip()
    return t


def assemble(segs, outs):
    """-> {(row id, system): {"messages": [(role, text)], "failed": n}} (each system's own layout)"""
    by_row = collections.defaultdict(list)
    for s in segs:
        by_row[(s["id"], s["system"])].append(s)
    res = {}
    for (rid, system), ss in by_row.items():
        o = outs.get(system, {})
        ss.sort(key=lambda s: (s["msg"], s["piece"]))
        lang = ss[0]["lang"]
        msgs, failed = collections.OrderedDict(), 0
        for s in ss:
            piece = s["content"]
            if s["kind"] == "text":
                t = seg_text(system, o.get("{}::{}::{}::{}".format(system, rid, s["msg"], s["piece"])), lang)
                if t is None:
                    failed += 1
                    t = ""
                piece = t
            msgs.setdefault(s["msg"], [s["role"], ""])[1] += piece
        res[(rid, system)] = {"messages": [tuple(v) for v in msgs.values()], "failed": failed,
                              "by_index": {i: tuple(v) for i, v in msgs.items()}, "uid": ss[0]["uid"]}
    return res


def checks(src, tr):
    a = len(src.strip()) or 1
    ratio = len(tr.strip()) / a
    sw = {w.lower() for w in WORD.findall(src) if w.isascii()}
    tw = [w.lower() for w in WORD.findall(tr)]
    copied = sum(w in sw for w in tw) / max(len(tw), 1)
    new_mixed = sorted(set(M.mixed_script_words(tr)) - set(M.mixed_script_words(src)))
    return {"ratio": round(ratio, 2), "injected": M.injected(src, tr), "mixed": new_mixed,
            "copied": round(copied, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    if not args.validate:
        sys.exit("only --validate is implemented so far")
    segs = pq.read_table(os.path.join(OUT, "segments.parquet")).to_pylist()
    outs = {s: outputs(os.path.join(OUT, s)) for s in ("tilde", "salta")}
    res = assemble(segs, outs)
    sample = {r["id"]: r for r in json.load(open(os.path.join(OUT, "sample.json")))}
    os.makedirs(VAL, exist_ok=True)
    rng = random.Random(1115)
    key = ["vid\tid\tlang\tsystem\tgroup\tfailed\tratio\tcopied\tinjected\tmixed"]
    auto = collections.defaultdict(collections.Counter)
    for g in GROUPS:
        name = "_".join(g)
        blocks = []
        rows = [r for r in sample.values() if r["lang"] in g]
        rows.sort(key=lambda r: (g.index(r["lang"]), r["id"]))
        for k, r in enumerate(rows, 1):
            versions = [("current:" + r["translator"].split("/")[-1], r["current"], 0)]
            for system in ("tilde", "salta"):
                x = res.get((r["id"], system))
                if x:
                    versions.append((system, "".join(t for role, t in x["messages"] if role == "user"), x["failed"]))
            rng.shuffle(versions)
            blocks.append("### S{:02d} | {} | English source\n{}".format(k, r["lang"], r["source"].strip()))
            for system, text, failed in versions:
                vid = "v{:05x}".format(rng.randrange(16 ** 5))
                c = checks(r["source"], text)
                sysk = system.split(":")[0]
                auto[(r["lang"], sysk)]["n"] += 1
                auto[(r["lang"], sysk)]["failed"] += bool(failed)
                auto[(r["lang"], sysk)]["ratio_off"] += not 0.6 <= c["ratio"] <= 2.0
                auto[(r["lang"], sysk)]["copied>0.3"] += c["copied"] > 0.3
                auto[(r["lang"], sysk)]["script"] += bool(c["injected"] or c["mixed"])
                key.append("\t".join([vid, r["id"], r["lang"], system, name, str(failed), str(c["ratio"]),
                                      str(c["copied"]), json.dumps(c["injected"]), json.dumps(c["mixed"], ensure_ascii=False)]))
                blocks.append("=== {} | {}\n{}".format(vid, r["lang"], text.strip() or "(empty)"))
        with open(os.path.join(VAL, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(blocks) + "\n")
        print(name, len(rows), "sources")
    with open(os.path.join(VAL, "key.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(key) + "\n")
    json.dump({"{}|{}".format(*k): dict(v) for k, v in auto.items()}, open(os.path.join(VAL, "auto_checks.json"), "w"), indent=1)
    print("automatic checks (n / failed / length ratio off / copied>0.3 / script):")
    for (lang, s), c in sorted(auto.items()):
        print("  {:3} {:8} {:2} / {} / {} / {} / {}".format(lang, s, c["n"], c["failed"], c["ratio_off"], c["copied>0.3"], c["script"]))


if __name__ == "__main__":
    main()
