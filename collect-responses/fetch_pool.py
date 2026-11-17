#!/usr/bin/env python3
"""Download the response-sampling model pool (models.yaml) into data/models/<name>/.

  collect-responses/run.sh fetch_pool.py [--only name1,name2] [--check]

Each model is fetched at its pinned `revision`, in Hugging Face safetensors format only (Mistral
`consolidated.*` duplicates, PyTorch .bin copies, GGUF and `original/` folders are skipped).
Resumable: files already present are not fetched again. Afterwards every shard listed in
model.safetensors.index.json must exist with its full size; `--check` only runs that test.
Writes <dir>/_download_info.json (repo, revision, group, licence) next to the weights.
"""
import argparse
import json
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "data", "models")
IGNORE = ["*.gguf", "original/*", "consolidated*", "*.pth", "*.pt", "pytorch_model*.bin", "*.msgpack",
          "*.h5", "*.ot", "*.onnx", "onnx/*", "*.tflite"]


def load_pool():
    with open(os.path.join(HERE, "models.yaml"), encoding="utf-8") as f:
        pool = yaml.safe_load(f)
    names = [m["name"] for m in pool["models"]]
    assert len(names) == len(set(names)), "duplicate model names"
    return pool


def check(m):
    """problems with the local copy of model m (empty list = complete)"""
    d = os.path.join(MODELS, m["name"])
    if not os.path.isfile(os.path.join(d, "config.json")):
        return ["no config.json"]
    info = os.path.join(d, "_download_info.json")
    if not os.path.isfile(info) or json.load(open(info))["revision"] != m["revision"]:
        return ["missing or different _download_info.json revision"]
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(idx):
        shards = sorted(set(json.load(open(idx))["weight_map"].values()))
    else:
        shards = [f for f in os.listdir(d) if f.endswith(".safetensors")]
    if not shards:
        return ["no safetensors"]
    sizes = json.load(open(info)).get("sizes", {})
    bad = []
    for s in shards:
        p = os.path.join(d, s)
        if not os.path.isfile(p):
            bad.append("missing " + s)
        elif s in sizes and os.path.getsize(p) != sizes[s]:
            bad.append("size mismatch " + s)
    return bad


def fetch(m):
    from huggingface_hub import HfApi, snapshot_download
    d = os.path.join(MODELS, m["name"])
    snapshot_download(m["hf"], revision=m["revision"], local_dir=d, ignore_patterns=IGNORE, max_workers=8)
    info = HfApi().model_info(m["hf"], revision=m["revision"], files_metadata=True)
    sizes = {s.rfilename: s.size for s in info.siblings if s.rfilename.endswith(".safetensors")}
    with open(os.path.join(d, "_download_info.json"), "w") as f:
        json.dump({"id": m["hf"], "revision": m["revision"], "name": m["name"], "group": m["group"],
                   "license": m.get("license"), "sizes": sizes}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated model names")
    ap.add_argument("--check", action="store_true", help="only verify the local copies")
    args = ap.parse_args()
    models = load_pool()["models"]
    if args.only:
        want = set(args.only.split(","))
        unknown = want - {m["name"] for m in models}
        assert not unknown, "unknown names: {}".format(sorted(unknown))
        models = [m for m in models if m["name"] in want]
    failed = []
    for m in models:
        if not args.check and check(m):
            print("fetch {:38s} {} @ {}".format(m["name"], m["hf"], m["revision"][:12]), flush=True)
            try:
                fetch(m)
            except Exception as e:      # keep going; the summary lists what is missing
                print("  ERROR {}: {}".format(type(e).__name__, e), flush=True)
        problems = check(m)
        gb = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(MODELS, m["name"]))
                 for f in fs) / 1e9 if os.path.isdir(os.path.join(MODELS, m["name"])) else 0
        print("{:4s} {:38s} {:6.1f} GB {}".format("ok" if not problems else "BAD", m["name"], gb,
                                                 "; ".join(problems[:3])), flush=True)
        if problems:
            failed.append(m["name"])
    print("{} of {} complete{}".format(len(models) - len(failed), len(models),
                                       "; incomplete: " + ",".join(failed) if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
