#!/usr/bin/env python3
"""Stage 5k: Latgalian for ltg slots refilled with a new donor (no stage-5f context).

  ./run.sh compile/ltg_new.py requests          # TildeOpen T3 (24-shot) + T1 (8-shot), direct
  (GPU: retranslate_ltg.sbatch PHASES=tilde TILDE_REQ=.../data/ltg/new/requests.parquet TILDE_OUT=.../out)
  ./run.sh compile/ltg_new.py show              # reassembled candidates + Goldfish score, for reading
  ./run.sh compile/ltg_new.py add SLOT:SYS ...  # append the chosen ones to data/llm/ltg_final/

The candidates are read against the source before choosing (as in stage 5f);
`add` runs the same structure checks as ltg_finalize.py.
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import mt_common as M  # noqa: E402
from ltg_requests import tilde_prompt  # noqa: E402
from ltg_score import DESC  # noqa: E402

D = os.path.join(M.DATA, "ltg", "new")
FINAL = os.path.join(M.DATA, "llm", "ltg_final")
SYSTEMS = {"T3": 24, "T1": 8}


def pending():
    """ltg slots of the allocation without a Latgalian context (slot, uid, src_lang)."""
    done = {(r["slot"], r["uid"]) for r in pq.read_table(os.path.join(FINAL, "contexts.parquet")).to_pylist()}
    al = pq.read_table(os.path.join(M.DATA, "compiled", "allocation.parquet"),
                       columns=["slot", "uid", "lang", "fill", "src_lang"]).to_pylist()
    return [a for a in al if a["lang"] == "ltg" and a["fill"] == "translate" and (a["slot"], a["uid"]) not in done]


def contexts(uids):
    out = {}
    for b in pq.ParquetFile(os.path.join(M.DATA, "pool", "candidates_ctx.parquet")).iter_batches(
            batch_size=20000, columns=["uid", "context"]):
        for u, c in zip(b.column("uid").to_pylist(), b.column("context").to_pylist()):
            if u in uids:
                out[u] = json.loads(c)
    return out


def requests():
    todo = pending()
    src = contexts({a["uid"] for a in todo})
    reqs, layout = [], []
    for a in todo:
        for mi, m in enumerate(src[a["uid"]]):
            for pi, (kind, s) in enumerate(M.segments(m["content"] or "", 500)):
                layout.append({"slot": a["slot"], "uid": a["uid"], "msg": mi, "piece": pi, "role": m["role"],
                               "kind": kind, "content": s})
                if kind != "text":
                    continue
                for sysn, k in SYSTEMS.items():
                    prompt = tilde_prompt(a["src_lang"], s, k)
                    reqs.append({"id": "{}::{}::{}::{}".format(sysn, a["slot"], mi, pi), "messages": None,
                                 "prompt": prompt, "prompt_ids": M.tilde_encode(prompt),
                                 "stop": json.dumps(["</Latgalian>"]), "schema": None,
                                 "max_tokens": int(min(4096, len(s) * 0.8 + 128))})
    os.makedirs(D, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(reqs), os.path.join(D, "requests.parquet"))
    pq.write_table(pa.Table.from_pylist(layout), os.path.join(D, "layout.parquet"))
    print("pending ltg slots:", [(a["slot"], a["src_lang"]) for a in todo], "| requests:", len(reqs))


def candidates():
    out = {}
    for p in sorted(os.listdir(os.path.join(D, "out"))):
        for r in pq.read_table(os.path.join(D, "out", p)).to_pylist():
            if r["finish_reason"] == "stop" and r.get("token_ids") is not None:
                t = M.tilde_sp().decode([int(x) for x in r["token_ids"]]).split("</Latgalian>")[0]
                out[r["id"]] = t.split("\n<")[0].strip()
    rows = collections.defaultdict(list)
    for x in pq.read_table(os.path.join(D, "layout.parquet")).to_pylist():
        rows[(x["slot"], x["uid"])].append(x)
    res = {}
    for key, xs in rows.items():
        for sysn in SYSTEMS:
            msgs, ok = collections.OrderedDict(), True
            for x in sorted(xs, key=lambda x: (x["msg"], x["piece"])):
                m = msgs.setdefault(x["msg"], {"role": x["role"], "content": ""})
                if x["kind"] == "text":
                    t = out.get("{}::{}::{}::{}".format(sysn, x["slot"], x["msg"], x["piece"]))
                    ok = ok and bool(t)
                    m["content"] += t or ""
                else:
                    m["content"] += x["content"]
            res[(key, sysn)] = (list(msgs.values()), ok)
    return res


def main():
    cmd = sys.argv[1]
    if cmd == "requests":
        return requests()
    from ltg_finalize import message_ok
    from ltg_score import Goldfish
    cands = candidates()
    src = contexts({k[0][1] for k in cands})
    if cmd == "show":
        gf = Goldfish()
        for (key, sysn), (ctx, ok) in sorted(cands.items()):
            user = "\n".join(m["content"] for m in ctx if m["role"] == "user")
            print("=== slot {} {} complete={} check={} Δbpc={}".format(
                key[0], sysn, ok, message_ok(src[key[1]], ctx) if ok else "-",
                round(gf.deltas([user])[0] or 0, 2)))
            for a, b in zip(src[key[1]], ctx):
                print("  SRC [{}] {}".format(a["role"], (a["content"] or "")[:600].replace("\n", " ⏎ ")))
                print("  LTG [{}] {}".format(b["role"], b["content"][:600].replace("\n", " ⏎ ")))
        return
    if cmd == "add":
        rows = pq.read_table(os.path.join(FINAL, "contexts.parquet")).to_pylist()
        meta = json.load(open(os.path.join(FINAL, "meta.json")))
        gf = Goldfish()
        keys = {k[0][0]: k[0] for k in cands}
        from ltg_fix import fix_messages
        fixes = fix_messages()
        for arg in sys.argv[2:]:      # SLOT:SYS or SLOT:SYS:MSG=FIX[,MSG=FIX]
            parts = arg.split(":")
            slot, sysn = parts[0], parts[1]
            over = [int(x.split("=")[0]) for x in parts[2].split(",")] if len(parts) > 2 else []
            key = keys[int(slot)]
            ctx, ok = cands[(key, sysn)]
            ctx = [dict(m) for m in ctx]
            rid = "mlp128k-{:06d}".format(key[0])
            for mi in over:
                t = fixes.get((rid, mi))
                if not t:
                    sys.exit("slot {}: no FIX output for message {}".format(slot, mi))
                ctx[mi]["content"] = t
            label = sysn + ("+FIX" if over else "")
            bad = None if ok else "incomplete"
            bad = bad or message_ok(src[key[1]], ctx)
            if bad:
                sys.exit("slot {} {}: {}".format(slot, label, bad))
            user = "\n".join(m["content"] for m in ctx if m["role"] == "user")
            rows = [r for r in rows if (r["slot"], r["uid"]) != key]     # replace an earlier choice
            rows.append({"slot": key[0], "uid": key[1], "id": rid, "system": label,
                         "reason": "refilled slot (round 7): read against the source, {} chosen".format(label),
                         "note": "latgalian_tildeopen_manually_verified", "dbpc": gf.deltas([user])[0],
                         "context": json.dumps(ctx, ensure_ascii=False)})
            meta["translators"].setdefault(label, "TildeAI/TildeOpen-30b ({})".format(
                "; ".join([DESC[sysn]] + (["targeted re-translation (manual_check/ltg_fix_inputs.json)"] if over else []))))
        pq.write_table(pa.Table.from_pylist(rows), os.path.join(FINAL, "contexts.parquet"))
        json.dump(meta, open(os.path.join(FINAL, "meta.json"), "w"), indent=1)
        print("ltg_final rows:", len(rows))


if __name__ == "__main__":
    main()
