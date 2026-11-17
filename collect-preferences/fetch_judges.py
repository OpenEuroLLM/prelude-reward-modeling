#!/usr/bin/env python3
"""Download the judge candidates and instruments of judges.yaml into data/models/<name>/.

  collect-preferences/run.sh fetch_judges.py [--only name1,name2] [--tools] [--check]

Same conventions as collect-responses/fetch_pool.py: Hugging Face safetensors only (Mistral consolidated.*
duplicates, PyTorch .bin copies, GGUF and original/ folders are skipped), resumable, and every shard listed
in model.safetensors.index.json must exist with its full size afterwards (`--check` runs only that test).
`revision: main` is resolved to the commit sha at download time; the sha is recorded in
<dir>/_download_info.json and printed, so that it can be pinned in judges.yaml.
Needs HF_TOKEN in the environment and network access (login node; CPU container).
"""
import argparse
import json
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "data", "models")
IGNORE = ["*.gguf", "original/*", "consolidated*", "*.pth", "*.pt", "pytorch_model*.bin", "*.msgpack",
          "*.h5", "*.ot", "*.onnx", "onnx/*", "*.tflite"]


def entries(tools):
    with open(os.path.join(HERE, "judges.yaml"), encoding="utf-8") as f:
        pool = yaml.safe_load(f)
    out = list(pool["judges"]) + (list(pool.get("tools") or []) if tools else [])
    names = [m["name"] for m in out]
    assert len(names) == len(set(names)), "duplicate names"
    return out


def check(m, fresh=False):
    """problems with the local copy (empty list = complete). fresh: the files themselves, right after
    snapshot_download (which moves each file into place only once it is complete) and before the info file
    is written; otherwise the info file must also record a completed download."""
    d = os.path.join(MODELS, m["name"])
    info = os.path.join(d, "_download_info.json")
    if not fresh and (not os.path.isfile(info) or not json.load(open(info)).get("complete")):
        return ["download not finished"]
    if m.get("allow"):
        return [p for p in m["allow"] if not os.path.isfile(os.path.join(d, p))]
    if not os.path.isfile(os.path.join(d, "config.json")):
        return ["no config.json"]
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(idx):
        shards = sorted(set(json.load(open(idx))["weight_map"].values()))
    else:
        shards = [p for p in os.listdir(d) if p.endswith(".safetensors")] or ["model.safetensors"]
    bad = [s for s in shards if not os.path.isfile(os.path.join(d, s)) or os.path.getsize(os.path.join(d, s)) == 0]
    return ["missing or empty shard " + s for s in bad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated names")
    ap.add_argument("--tools", action="store_true", help="include the instruments (tools:) as well")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    todo = entries(args.tools)
    if args.only:
        want = set(args.only.split(","))
        todo = [m for m in todo if m["name"] in want]
        assert len(todo) == len(want), "unknown names: {}".format(want - {m["name"] for m in todo})
    converted = [m["name"] for m in todo if m.get("convert")]
    if converted and not args.check:
        print("skipped (converted by convert_judges.py, see `convert` in judges.yaml):", ", ".join(converted))
        todo = [m for m in todo if not m.get("convert")]
    if args.check:
        rc = 0
        for m in todo:
            problems = check(m)
            print("{:28s} {}".format(m["name"], "ok" if not problems else "; ".join(problems[:3])))
            rc |= bool(problems)
        sys.exit(rc)
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    for m in todo:
        d = os.path.join(MODELS, m["name"])
        t0 = time.time()
        info = api.model_info(m["hf"], revision=m.get("revision") or "main")
        sha = info.sha
        print("{}: {} @ {} -> {}".format(m["name"], m["hf"], sha, d), flush=True)
        snapshot_download(repo_id=m["hf"], revision=sha, local_dir=d, max_workers=16,
                          allow_patterns=m.get("allow"), ignore_patterns=IGNORE)
        problems = check(m, fresh=True)
        with open(os.path.join(d, "_download_info.json"), "w") as f:
            json.dump({"repo": m["hf"], "revision": sha, "family": m.get("family"), "licence": m.get("licence"),
                       "downloaded": time.strftime("%Y-%m-%dT%H:%M:%S"), "complete": not problems}, f, indent=1)
        size = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(d) for x in fs) / 1e9
        print("{}: {} GB in {:.0f} min; {}".format(m["name"], round(size), (time.time() - t0) / 60,
                                                   "complete" if not problems else "PROBLEMS: " + "; ".join(problems[:3])), flush=True)


if __name__ == "__main__":
    main()
