#!/usr/bin/env python3
"""Stage 7: statistics of the compiled prompt set.

  ./run.sh compile/dataset_stats.py [--out data/compiled]
Inputs : <out>/prompts.parquet, compile/quotas.tsv
Outputs: <out>/stats.json   all tables below, machine-readable
         <out>/STATS.md     the same as Markdown tables

Tables: per-language counts vs. quota and fill mix; purpose / origin / task /
domain / turns / safety / complexity / locale distributions; language-content
independence (Jensen-Shannon divergence of each language's distribution from
the global one, per dimension); sources and licence flags; responses and
preferences; IF verifiers; prompt length per language; notes.
"""
import argparse
import collections
import json
import os
import statistics

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")
DIMS = ["purpose", "origin", "task", "domain", "turns", "complexity", "locale"]


def quotas():
    q = {}
    with open(os.path.join(HERE, "quotas.tsv")) as f:
        for line in f:
            if line.startswith("#") or line.startswith("code\t"):
                continue
            p = line.rstrip("\n").split("\t")
            q[p[0]] = int(p[-1])
    return q


def js(p, q):
    keys = sorted(set(p) | set(q))
    pv = np.array([p.get(k, 0.0) for k in keys]) + 1e-12
    qv = np.array([q.get(k, 0.0) for k in keys]) + 1e-12
    pv, qv = pv / pv.sum(), qv / qv.sum()
    m = (pv + qv) / 2
    return float(0.5 * (pv * np.log2(pv / m)).sum() + 0.5 * (qv * np.log2(qv / m)).sum())


def dist(rows, key):
    c = collections.Counter(r[key] for r in rows)
    n = sum(c.values())
    return {k: v / n for k, v in c.most_common()}


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(DATA, "compiled"))
    args = ap.parse_args()
    cols = ["id", "lang", "context", "responses", "preference", "fill", "source", "license_flag",
            "purpose", "origin", "task", "domain", "turns", "safety", "locale", "complexity",
            "instruction_ids", "notes", "translator"]
    rows = []
    pf = pq.ParquetFile(os.path.join(args.out, "prompts.parquet"))
    for b in pf.iter_batches(batch_size=20000, columns=cols):
        for r in b.to_pylist():
            users = [m["content"] or "" for m in r["context"] if m["role"] == "user"]
            r["user_chars"] = sum(len(u) for u in users)
            r["n_msgs"] = len(r["context"])
            r["n_resp"] = len(r["responses"] or [])
            r["has_pref"] = r["preference"] is not None
            r["has_if"] = bool(r["instruction_ids"])
            del r["context"], r["responses"], r["preference"]
            rows.append(r)
    q = quotas()
    st = {"rows": len(rows), "unique_ids": len({r["id"] for r in rows})}
    by_lang = collections.defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)

    # languages
    lang_rows = []
    for l, n in sorted(q.items(), key=lambda x: -x[1]):
        g = by_lang.get(l, [])
        fc = collections.Counter(r["fill"] for r in g)
        lang_rows.append({
            "lang": l, "quota": n, "rows": len(g), "missing": n - len(g),
            "native": fc["native"], "source_translated": fc["source_translated"],
            "translated": fc["translated"],
            "native_share": round((fc["native"] + fc["source_translated"]) / max(1, len(g)), 3),
            "with_responses": sum(r["n_resp"] > 0 for r in g),
            "median_user_chars": int(statistics.median([r["user_chars"] for r in g])) if g else 0,
        })
    st["languages"] = lang_rows
    st["fill"] = dict(collections.Counter(r["fill"] for r in rows))

    # content distributions and language independence
    st["distributions"] = {d: dist(rows, d) for d in DIMS}
    st["distributions"]["safety"] = dist([r for r in rows if r["purpose"].startswith("safety")], "safety")
    st["independence_js"] = {}
    for d in ["purpose", "task", "domain", "turns", "complexity"]:
        glob = st["distributions"][d]
        per = {l: round(js(dist(g, d), glob), 4) for l, g in by_lang.items()}
        vals = list(per.values())
        st["independence_js"][d] = {"mean": round(float(np.mean(vals)), 4),
                                    "max": max(per.items(), key=lambda x: x[1]), "per_lang": per}

    # sources, licences, responses, IF
    st["sources"] = collections.Counter(r["source"] for r in rows).most_common()
    st["license_flag"] = dict(collections.Counter(r["license_flag"] for r in rows))
    st["responses"] = {"rows_with_responses": sum(r["n_resp"] > 0 for r in rows),
                       "responses_total": sum(r["n_resp"] for r in rows),
                       "rows_with_preference": sum(r["has_pref"] for r in rows)}
    st["if"] = {"if_purpose_rows": sum(r["purpose"] == "if" for r in rows),
                "rows_with_verifier": sum(r["has_if"] for r in rows),
                "verifier_by_fill": dict(collections.Counter(r["fill"] for r in rows if r["has_if"]))}
    st["turns_messages"] = dict(sorted(collections.Counter(min(r["n_msgs"], 10) for r in rows).items()))
    st["notes"] = dict(collections.Counter(n for r in rows for n in r["notes"]))

    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump(st, f, indent=1, default=list, ensure_ascii=False)

    # markdown
    md = ["# Compiled prompt set: statistics", "",
          "{} rows ({} unique ids). Fill: {}.".format(
              st["rows"], st["unique_ids"],
              ", ".join("{} {}".format(k, v) for k, v in st["fill"].items())), "",
          "## Languages", "",
          md_table(["lang", "quota", "rows", "native", "src-transl.", "translated", "native share",
                    "with responses", "median user chars"],
                   [[x["lang"], x["quota"], x["rows"], x["native"], x["source_translated"],
                     x["translated"], x["native_share"], x["with_responses"], x["median_user_chars"]]
                    for x in lang_rows]), ""]
    for d in DIMS + ["safety"]:
        items = list(st["distributions"][d].items())[:25]
        md += ["## {}".format(d), "", md_table([d, "share"], [[k, "{:.3f}".format(v)] for k, v in items]), ""]
    md += ["## Language-content independence (JS divergence from the global distribution)", "",
           md_table(["dimension", "mean over languages", "max (language)"],
                    [[d, v["mean"], "{} ({})".format(v["max"][1], v["max"][0])]
                     for d, v in st["independence_js"].items()]), "",
           "## Sources", "", md_table(["source", "rows"], st["sources"]), "",
           "## Licence flags", "", md_table(["flag", "rows"], list(st["license_flag"].items())), "",
           "## Responses / preferences / IF verifiers", "",
           md_table(["metric", "value"], list(st["responses"].items()) + list(st["if"].items())), "",
           "## Notes", "", md_table(["note", "rows"], list(st["notes"].items())), ""]
    with open(os.path.join(args.out, "STATS.md"), "w") as f:
        f.write("\n".join(md))
    print(json.dumps({k: st[k] for k in ("rows", "unique_ids", "fill", "responses", "if")}, indent=1))
    print("independence (mean JS):", {d: v["mean"] for d, v in st["independence_js"].items()})
    print("languages short of quota:", [(x["lang"], x["missing"]) for x in lang_rows if x["missing"]])


if __name__ == "__main__":
    main()
