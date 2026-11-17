#!/usr/bin/env python3
"""Stage 5f: final Latgalian translation of each ltg row, from a manual per-row choice.

  ./run.sh compile/ltg_finalize.py [--choices data/compiled/manual_check/ltg_choices.tsv]

choices.tsv: id <tab> system <tab> overrides <tab> reason. `system` names a
candidate system of stage 5d (data/ltg/eval/items.parquet, e.g. T3, T2p, T4p);
overrides "1:FIX,2:T2p" replace single messages by another system's message or
by a targeted re-translation of stage 5e' (FIX, compile/ltg_fix.py). Every ltg
row needs a choice. Each final row is checked (layout, per-message length
ratios, no exotic-script injection, total length ratio) and its user text is
scored with the Goldfish lv/ltg discriminator; a failing row stops the script.
Outputs: data/llm/ltg_final/contexts.parquet (slot, uid, id, system, dbpc, reason,
context), meta.json (translator strings used by assemble.py)
"""
import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import mt_common as M  # noqa: E402
from ltg_fix import fix_messages, source_contexts  # noqa: E402
from ltg_score import DESC, Goldfish  # noqa: E402

EVAL = os.path.join(M.DATA, "ltg", "eval")
OUT = os.path.join(M.DATA, "llm", "ltg_final")
DESC = dict(DESC, FIX="targeted re-translation (manual_check/ltg_fix_inputs.json)")


def message_ok(src_ctx, ctx):
    if len(src_ctx) != len(ctx) or any(a["role"] != b["role"] for a, b in zip(src_ctx, ctx)):
        return "layout"
    for a, b in zip(src_ctx, ctx):
        s, t = (a["content"] or "").strip(), (b["content"] or "").strip()
        if s and not t:
            return "empty_message"
        if len(s) >= 80 and not 0.25 <= len(t) / len(s) <= 4.0:
            return "message_ratio"
    tot = sum(len(m["content"] or "") for m in ctx) / max(1, sum(len(m["content"] or "") for m in src_ctx))
    if not 0.4 <= tot <= 3.0:
        return "length_ratio"
    if M.injected("\n".join(m["content"] or "" for m in src_ctx), "\n".join(m["content"] for m in ctx)):
        return "script_injection"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--choices", default=os.path.join(M.DATA, "compiled", "manual_check", "ltg_choices.tsv"))
    args = ap.parse_args()
    choices = {}
    for ln in open(args.choices, encoding="utf-8"):
        if ln.strip() and not ln.startswith("#"):
            rid, system, overrides, reason = (ln.rstrip("\n").split("\t") + ["", ""])[:4]
            choices[rid] = (system, dict(x.split(":") for x in overrides.split(",") if x), reason)

    segs = [s for s in pq.read_table(os.path.join(M.DATA, "llm", "translate_tg", "segments.parquet")).to_pylist()
            if s["group"] == "lv"]
    uid = {s["id"]: (s["slot"], s["uid"]) for s in segs}
    missing = sorted(set(uid) - set(choices))
    if missing:
        sys.exit("no choice for ltg rows: {}".format(missing))
    items = {(r["id"], r["system"]): r for r in pq.read_table(os.path.join(EVAL, "items.parquet")).to_pylist()}
    fix = fix_messages()
    src = source_contexts(set(uid))

    rows, labels, bad = [], {}, []
    for rid in sorted(uid):
        system, overrides, reason = choices[rid]
        base = items.get((rid, system))
        if base is None or not base["complete"]:
            bad.append((rid, "{} incomplete".format(system)))
            continue
        ctx = json.loads(base["context"])
        sctx = src[rid][0]
        if len(ctx) < len(sctx):    # messages without text pieces (empty in the source)
            ctx += [{"role": m["role"], "content": m["content"] or ""} for m in sctx[len(ctx):]]
        for mi, osys in overrides.items():
            mi = int(mi)
            if osys == "FIX":
                text = fix.get((rid, mi))
            else:
                o = items.get((rid, osys))
                text = json.loads(o["context"])[mi]["content"] if o and o["complete"] else None
            if not text:
                bad.append((rid, "override {}:{} missing".format(mi, osys)))
                break
            ctx[mi] = {"role": ctx[mi]["role"], "content": text}
        else:
            why = message_ok(sctx, ctx)
            if why:
                bad.append((rid, why))
                continue
            label = system if not overrides else "{}+{}".format(system, "+".join(sorted(set(overrides.values()))))
            labels[label] = "TildeAI/TildeOpen-30b ({})".format(
                "; ".join(DESC[s] for s in [system] + sorted(set(overrides.values()) - {system})))
            note = "latgalian_tildeopen_automatic_checks_only" if "not manually verified" in reason \
                else "latgalian_tildeopen_manually_verified"
            rows.append({"slot": uid[rid][0], "uid": uid[rid][1], "id": rid, "system": label, "reason": reason,
                         "note": note, "context": json.dumps(ctx, ensure_ascii=False)})
    if bad:
        sys.exit("rows failing: {}".format(bad))
    gf = Goldfish()
    users = ["\n".join(m["content"] for m in json.loads(r["context"]) if m["role"] == "user") for r in rows]
    for r, d in zip(rows, gf.deltas(users)):
        r["dbpc"] = d
    os.makedirs(OUT, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(OUT, "contexts.parquet"))
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump({"translators": labels}, f, indent=1)
    ds = [r["dbpc"] for r in rows if r["dbpc"] is not None]
    print("rows:", len(rows), "| systems:", dict(collections.Counter(r["system"] for r in rows)))
    print("Goldfish Δbpc of the user text: mean {:.2f}, min {:.2f}, Latgalian-like {}/{}".format(
        sum(ds) / len(ds), min(ds), sum(d > 0 for d in ds), len(ds)))
    for r in rows:
        if r["dbpc"] is not None and r["dbpc"] < 0.5:
            print("  low Δbpc: {} {} {:.2f}".format(r["id"], r["system"], r["dbpc"]))


if __name__ == "__main__":
    main()
