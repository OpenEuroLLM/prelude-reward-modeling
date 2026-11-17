#!/usr/bin/env python3
"""Convert downloaded raw datasets into the unified prompt schema
(see common.py) using the per-source extractors in extractors/*.py.

Output: <out>/<org>__<name>/part-#####.parquet + stats.json

  ./run.sh extract.py --only allenai/WildChat-1M            # full run
  ./run.sh extract.py --only allenai/WildChat-1M --limit 2000
        # quick test: writes to <out>/_sample/<org>__<name>/ and prints
        # a few truncated example records
  ./run.sh extract.py --list                               # registry status
"""
import argparse
import collections
import importlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # run.sh uses `python -I`; add our own code dir only

import common  # noqa: E402

ROOT = os.path.dirname(HERE)
DEFAULT_RAW = os.path.join(ROOT, "data", "raw")
DEFAULT_OUT = os.path.join(ROOT, "data", "extracted")
SOURCES = os.path.join(HERE, "datasets.txt")

# Licenses missing from (or wrong in) the HF metadata, taken from dataset cards.
LICENSE_OVERRIDES = {
    "lmsys/lmsys-chat-1m":
        "lmsys-chat-1m-agreement (commercial use ok, no redistribution)",
    "lmarena-ai/arena-human-preference-100k":
        "cc-by-4.0 (prompts); outputs under each provider's terms",
    "allenai/tulu-3-wildchat-unused": "odc-by",
    "allenai/coconot": "ai2-impact-lr (card) / odc-by (Tulu 3 card): unresolved",
}


def load_extractors():
    ext_dir = os.path.join(HERE, "extractors")
    for name in sorted(os.listdir(ext_dir)):
        if name.endswith(".py") and not name.startswith("_"):
            importlib.import_module("extractors." + name[:-3])


def read_sources():
    rows = []
    with open(SOURCES) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                rows.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return rows


def trunc(v, n=300):
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + "...[{} chars]".format(len(v))
    if isinstance(v, list):
        return [trunc(x, n) for x in v[:4]] + (
            ["...[{} items]".format(len(v))] if len(v) > 4 else [])
    if isinstance(v, dict):
        return {k: trunc(x, n) for k, x in v.items()}
    return v


class Stats:
    def __init__(self):
        self.n = 0
        self.split = collections.Counter()
        self.subset = collections.Counter()
        self.lang = collections.Counter()
        self.turns = collections.Counter()
        self.n_resp = collections.Counter()
        self.pref = collections.Counter()
        self.prompt_chars = 0
        self.uids = set()
        self.dup_uids = 0

    def add(self, r):
        self.n += 1
        self.split[r["split"]] += 1
        self.subset[r["subset"]] += 1
        self.lang[r["language"]] += 1
        self.turns[min(r["n_user_turns"], 10)] += 1
        self.n_resp[min(len(r["responses"]), 5)] += 1
        self.pref[json.loads(r["preference"])["type"]
                  if r["preference"] else None] += 1
        self.prompt_chars += len(r["context"][-1]["content"])
        if r["uid"] in self.uids:
            self.dup_uids += 1
        else:
            self.uids.add(r["uid"])

    def as_dict(self, skipped, seconds):
        return {
            "records": self.n,
            "duplicate_uids": self.dup_uids,
            "skipped": skipped,
            "split": dict(self.split),
            "subset": dict(self.subset.most_common(50)),
            "language_top50": {str(k): v for k, v in self.lang.most_common(50)},
            "n_languages": len(self.lang),
            "n_user_turns(10=10+)": {str(k): v for k, v in sorted(self.turns.items())},
            "n_responses(5=5+)": {str(k): v for k, v in sorted(self.n_resp.items())},
            "preference_type": {str(k): v for k, v in self.pref.items()},
            "mean_last_prompt_chars": round(self.prompt_chars / max(self.n, 1), 1),
            "seconds": round(seconds, 1),
        }


def run_one(hf_id, category, args):
    raw_dir = os.path.join(args.raw, hf_id.replace("/", "__"))
    manifest = os.path.join(raw_dir, "_download_info.json")
    if not os.path.exists(manifest):
        print("  not downloaded yet (no _download_info.json): " + raw_dir)
        return False
    with open(manifest) as f:
        lic = json.load(f).get("license")
    if isinstance(lic, list):
        lic = ",".join(lic)
    lic = LICENSE_OVERRIDES.get(hf_id, lic)
    ctx = common.Context(hf_id, category, lic, raw_dir)
    fn = common.REGISTRY[hf_id]
    sample = args.limit is not None
    out_dir = os.path.join(args.out, "_sample" if sample else "",
                           hf_id.replace("/", "__"))
    writer = common.ShardWriter(out_dir)
    stats = Stats()
    examples = []
    t0 = time.time()
    for rec in fn(ctx):
        stats.add(rec)
        writer.write(rec)
        if sample and len(examples) < 3 and stats.n % max(args.limit // 3, 1) == 1:
            examples.append(rec)
        if sample and stats.n >= args.limit:
            break
        if stats.n % 200_000 == 0:
            print("  ... {:,} records ({:.0f}s)".format(stats.n, time.time() - t0))
    writer.close()
    st = stats.as_dict(ctx.skipped, time.time() - t0)
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(st, f, indent=2, ensure_ascii=False)
    print(json.dumps(st, ensure_ascii=False, indent=1))
    if sample:
        for ex in examples:
            ex = dict(ex)
            for k in ("preference", "meta"):
                ex[k] = json.loads(ex[k]) if ex[k] else None
            print("EXAMPLE:", json.dumps(trunc(ex), ensure_ascii=False))
    print("  -> " + out_dir)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    load_extractors()
    sources = read_sources()
    if args.list:
        for hf_id, cat in sources:
            done = os.path.exists(os.path.join(
                args.out, hf_id.replace("/", "__"), "stats.json"))
            print("{:<52} extractor={:<5} extracted={}".format(
                hf_id, "yes" if hf_id in common.REGISTRY else "NO", done))
        return
    if args.only:
        known = {s for s, _ in sources}
        bad = set(args.only) - known
        if bad:
            sys.exit("not in datasets.txt: {}".format(sorted(bad)))
        sources = [s for s in sources if s[0] in set(args.only)]
    ok = True
    for hf_id, cat in sources:
        print("== " + hf_id)
        if hf_id not in common.REGISTRY:
            print("  no extractor registered")
            ok = False
            continue
        ok &= run_one(hf_id, cat, args)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
