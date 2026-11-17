#!/usr/bin/env python3
"""Download a model snapshot into data/models/<name>/ (pinned revision recorded).

  ./run.sh fetch_model.py google/gemma-4-31B-it
"""
import json
import os
import sys

from huggingface_hub import HfApi, snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    repo = sys.argv[1]
    info = HfApi().model_info(repo)
    target = os.path.join(ROOT, "data", "models", repo.split("/")[-1])
    snapshot_download(repo, revision=info.sha, local_dir=target, max_workers=8)
    with open(os.path.join(target, "_download_info.json"), "w") as f:
        json.dump({"id": repo, "revision": info.sha,
                   "license": (info.card_data or {}).get("license")
                   if info.card_data else None}, f, indent=2)
    print(target)


if __name__ == "__main__":
    main()
