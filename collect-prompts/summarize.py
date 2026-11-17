#!/usr/bin/env python3
"""Validate extracted outputs and summarize them.

Checks that every part file has exactly the unified schema and that the row
counts match stats.json, then writes data/extracted/SUMMARY.md.

  ./run.sh summarize.py
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow.parquet as pq  # noqa: E402

import common  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "extracted")


def main():
    sources = []
    with open(os.path.join(HERE, "datasets.txt")) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                sources.append(line.split()[:2])

    rows, problems = [], []
    tot = {"records": 0, "resp1": 0, "resp2": 0, "pref": 0, "heldout": 0}
    for hf_id, cat in sources:
        d = os.path.join(OUT, hf_id.replace("/", "__"))
        stats_path = os.path.join(d, "stats.json")
        if not os.path.exists(stats_path):
            problems.append("{}: not extracted".format(hf_id))
            continue
        st = json.load(open(stats_path))
        n_parquet = 0
        size = 0
        for p in sorted(glob.glob(os.path.join(d, "part-*.parquet"))):
            pf = pq.ParquetFile(p)
            if not pf.schema_arrow.equals(common.SCHEMA):
                problems.append("{}: schema mismatch in {}".format(hf_id, p))
            n_parquet += pf.metadata.num_rows
            size += os.path.getsize(p)
        if n_parquet != st["records"]:
            problems.append("{}: {} rows on disk vs {} in stats".format(
                hf_id, n_parquet, st["records"]))
        if st["duplicate_uids"]:
            problems.append("{}: {} duplicate uids".format(hf_id, st["duplicate_uids"]))
        n = st["records"]
        nr = st["n_responses(5=5+)"]
        r1 = n - nr.get("0", 0)
        r2 = n - nr.get("0", 0) - nr.get("1", 0)
        pref = n - st["preference_type"].get("None", 0)
        held = n - st["split"].get("train", 0)
        langs = [k for k in st["language_top50"] if k != "None"][:5]
        rows.append((hf_id, cat, n, r1, r2, pref, held, st["n_languages"],
                     ", ".join(langs), size))
        tot["records"] += n
        tot["resp1"] += r1
        tot["resp2"] += r2
        tot["pref"] += pref
        tot["heldout"] += held

    pct = lambda a, b: "{:.0f}%".format(100.0 * a / b) if b else "-"
    lines = [
        "# Extracted prompt sources",
        "",
        "| Source | Category | Records | >=1 resp | >=2 resp | Pref labels | Val/test | #lang | Top languages (source labels) | Size |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for (hf_id, cat, n, r1, r2, pref, held, nl, langs, size) in rows:
        lines.append("| {} | {} | {:,} | {} | {} | {:,} | {:,} | {} | {} | {:.1f} GB |".format(
            hf_id, cat, n, pct(r1, n), pct(r2, n), pref, held, nl, langs,
            size / 1e9))
    lines += [
        "",
        "**Total:** {:,} records; {:,} with >=1 response, {:,} with >=2; "
        "{:,} with preference labels; {:,} in validation/test splits.".format(
            tot["records"], tot["resp1"], tot["resp2"], tot["pref"], tot["heldout"]),
        "",
        "Counts are before deduplication, language ID and filtering. "
        "Language labels are as given by each source.",
    ]
    if problems:
        lines += ["", "## Problems", ""] + ["- " + p for p in problems]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT, "SUMMARY.md"), "w") as f:
        f.write(text)
    print(text)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
