#!/usr/bin/env python3
"""Build a small annotation + translation request set for the GPU smoke test.

  ./run.sh compile/make_smoke_requests.py
Writes data/llm/smoke/requests.parquet (+ requests_meta.parquet).
Reads one random row group per source (memory-light).
"""
import glob
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import annotate_prompt as A  # noqa: E402
import translate_prompt as T  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
EXT = os.path.join(ROOT, "data", "extracted")
OUT = os.path.join(ROOT, "data", "llm", "smoke")

ANNOTATE = {
    "allenai__WildChat-4.8M": 40, "ministere-culture__comparia-conversations": 20,
    "openeurollm__EuroMultiSynth-Instruct": 30, "allenai__wildjailbreak": 15,
    "allenai__IF_multi_constraints_upto5": 15, "OpenAssistant__oasst2": 15,
    "CohereLabs__aya_redteaming": 10, "ToxicityPrompts__PolyGuardMix": 10,
    "nvidia__HelpSteer3": 15, "bench-llm__or-bench": 10,
    "lmarena-ai__arena-human-preference-140k": 20,
}
TARGETS = ["ltg", "mt", "ga", "is", "ka", "sr", "eu", "lv", "de"]


def sample_rows(src, n, rng, pred=None, groups=3):
    parts = sorted(glob.glob(os.path.join(EXT, src, "part-*.parquet")))
    rows = []
    for _ in range(groups):
        pf = pq.ParquetFile(rng.choice(parts))
        g = rng.randrange(pf.num_row_groups)
        rows += [r for r in pf.read_row_group(g).to_pylist() if pred is None or pred(r)]
    rng.shuffle(rows)
    return rows[:n]


def main():
    rng = random.Random(0)
    reqs, meta = [], []
    for src, n in ANNOTATE.items():
        for r in sample_rows(src, n, rng, pred=lambda r: r["split"] == "train"):
            rid = "ann::" + r["uid"]
            reqs.append({"id": rid, "messages": json.dumps(A.messages(r["context"]), ensure_ascii=False),
                         "schema": json.dumps(A.SCHEMA), "max_tokens": 300})
            meta.append({"id": rid, "kind": "annotate", "uid": r["uid"], "source": r["source"],
                         "tgt_lang": None, "n_messages": len(r["context"])})

    # translation probes: English WildChat prompts (mostly single-turn, some multi-turn)
    def english_ok(r):
        return (r["language"] == "English" and 30 <= len(r["context"][-1]["content"]) <= 1200
                and len(r["context"]) <= 3 and not json.loads(r["meta"]).get("toxic"))
    probes = sample_rows("allenai__WildChat-4.8M", 8, rng, pred=english_ok, groups=4)
    probes += sample_rows("allenai__IF_multi_constraints_upto5", 2, rng,
                          pred=lambda r: len(r["context"][-1]["content"]) < 1200)
    for r in probes:
        for tgt in TARGETS:
            rid = "tr::{}::{}".format(tgt, r["uid"])
            reqs.append({"id": rid,
                         "messages": json.dumps(T.messages(r["context"], "en", tgt), ensure_ascii=False),
                         "schema": json.dumps(T.schema(len(r["context"]))), "max_tokens": 3000})
            meta.append({"id": rid, "kind": "translate", "uid": r["uid"], "source": r["source"],
                         "tgt_lang": tgt, "n_messages": len(r["context"])})

    os.makedirs(OUT, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(reqs), os.path.join(OUT, "requests.parquet"))
    pq.write_table(pa.Table.from_pylist(meta), os.path.join(OUT, "requests_meta.parquet"))
    kinds = {}
    for m in meta:
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    print(kinds, "->", OUT)


if __name__ == "__main__":
    main()
