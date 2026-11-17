#!/usr/bin/env python3
"""Download the raw prompt-source datasets listed in datasets.txt.

Each dataset goes to <out>/<org>__<name>/ (exact repo layout, pinned to the
commit that was current at download time) plus a _download_info.json
manifest with the revision, license, gating and file list.

datasets.txt columns:  <hf_id>  <category>  [include-globs, comma-separated]
If include-globs are given, only matching files (plus README.md) are fetched.

Run inside the container:
  ./run.sh download.py --dry-run            # sizes only, nothing downloaded
  ./run.sh download.py                      # download everything
  ./run.sh download.py --only allenai/WildChat-1M nvidia/HelpSteer3
"""
import argparse
import collections
import datetime
import fnmatch
import json
import os
import sys

from huggingface_hub import HfApi, snapshot_download

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SOURCES = os.path.join(HERE, "datasets.txt")
DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "data", "raw")


def read_sources(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            include = parts[2].split(",") if len(parts) > 2 else None
            rows.append({"id": parts[0],
                         "category": parts[1] if len(parts) > 1 else "",
                         "include": include})
    return rows


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return "{:.1f} {}".format(n, unit)
        n /= 1024.0


def select(files, include):
    if not include:
        return files
    return [(name, size) for name, size in files
            if name == "README.md"
            or any(fnmatch.fnmatch(name, pat) for pat in include)]


def dry_run(api, src):
    info = api.dataset_info(src["id"], files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    chosen = select(files, src["include"])
    total = sum(s for _, s in chosen)
    print("{:<52} {:>6} files {:>10}   (repo total {})".format(
        src["id"], len(chosen), human(total),
        human(sum(s for _, s in files))))
    # Per top-level directory breakdown, to help choose include-globs.
    by_dir = collections.Counter()
    for name, size in chosen:
        top = name.split("/", 1)[0] if "/" in name else "."
        by_dir[top] += size
    if len(by_dir) > 1:
        for top, size in by_dir.most_common(12):
            print("    {:<48} {:>10}".format(top + "/", human(size)))
    return total


def download(api, src, out_root, workers):
    info = api.dataset_info(src["id"], files_metadata=True)
    target = os.path.join(out_root, src["id"].replace("/", "__"))
    allow = (src["include"] + ["README.md"]) if src["include"] else None
    snapshot_download(repo_id=src["id"], repo_type="dataset",
                      revision=info.sha, local_dir=target,
                      allow_patterns=allow, max_workers=workers)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    manifest = {
        "id": src["id"],
        "category": src["category"],
        "revision": info.sha,
        "last_modified": str(info.last_modified),
        "downloaded_at": datetime.datetime.utcnow().isoformat() + "Z",
        "gated": info.gated,
        "license": (info.card_data or {}).get("license")
                   if info.card_data else None,
        "include": src["include"],
        "files": [{"name": n, "size": s} for n, s in select(files, src["include"])],
    }
    with open(os.path.join(target, "_download_info.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=DEFAULT_SOURCES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="*", help="subset of dataset IDs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    sources = read_sources(args.sources)
    if args.only:
        wanted = set(args.only)
        sources = [s for s in sources if s["id"] in wanted]
        missing = wanted - {s["id"] for s in sources}
        if missing:
            sys.exit("not in {}: {}".format(args.sources, sorted(missing)))

    api = HfApi()
    if args.dry_run:
        grand = 0
        for src in sources:
            try:
                grand += dry_run(api, src)
            except Exception as e:
                print("{:<52} ERROR {}".format(src["id"], e))
        print("\nTOTAL selected: {}".format(human(grand)))
        return

    os.makedirs(args.out, exist_ok=True)
    failed = []
    for i, src in enumerate(sources, 1):
        print("[{}/{}] {}".format(i, len(sources), src["id"]), flush=True)
        try:
            path = download(api, src, args.out, args.workers)
            print("    -> {}".format(path), flush=True)
        except Exception as e:
            print("    FAILED: {}".format(e), flush=True)
            failed.append(src["id"])
    if failed:
        print("\nFailed: {}".format(" ".join(failed)))
        sys.exit(1)


if __name__ == "__main__":
    main()
