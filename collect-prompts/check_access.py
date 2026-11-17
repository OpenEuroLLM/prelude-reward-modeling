#!/usr/bin/env python3
"""Check gating status and your access for each HF dataset in datasets.txt.

Stdlib only (runs on any Python >= 3.6, no container needed). Reads HF_TOKEN
from the environment.

For every dataset it reports:
  gated  : no / auto (instant on click) / manual (owner must approve)
  access : yes / NO (request it at the printed URL) / ? (could not tell)

Usage:
  python3 check_access.py [datasets.txt]
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://huggingface.co/api/datasets/"
RESOLVE = "https://huggingface.co/datasets/{id}/resolve/main/{path}"
DATA_EXT = (".parquet", ".jsonl", ".json", ".jsonl.gz", ".json.gz", ".csv",
            ".tsv", ".arrow", ".zst", ".gz", ".zip", ".tar")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # surface 30x as HTTPError so we can read the status


def request(url, token, method="GET", follow=True):
    req = urllib.request.Request(url, method=method)
    if token:
        req.add_header("Authorization", "Bearer " + token)
    opener = (urllib.request.build_opener() if follow
              else urllib.request.build_opener(NoRedirect))
    try:
        with opener.open(req, timeout=30) as r:
            return r.status, r.read() if method == "GET" else b""
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:  # network errors
        return None, str(e).encode()


def pick_data_file(siblings):
    names = [s.get("rfilename", "") for s in siblings or []]
    for n in names:
        if n.lower().endswith(DATA_EXT):
            return n
    for n in names:
        if n not in (".gitattributes", "README.md"):
            return n
    return None


def check(ds_id, token):
    status, body = request(API + ds_id, token)
    if status != 200:
        return {"gated": "?", "access": "?",
                "note": "API HTTP {}".format(status)}
    info = json.loads(body.decode())
    gated = info.get("gated") or "no"
    lic = (info.get("cardData") or {}).get("license", "")
    if isinstance(lic, list):
        lic = ",".join(lic)
    if gated == "no":
        return {"gated": "no", "access": "yes", "license": lic}
    path = pick_data_file(info.get("siblings"))
    if path is None:
        return {"gated": gated, "access": "?", "license": lic,
                "note": "no files listed"}
    url = RESOLVE.format(id=ds_id, path=urllib.parse.quote(path))
    code, _ = request(url, token, method="HEAD", follow=False)
    if code in (200, 301, 302, 303, 307, 308):
        access = "yes"
    elif code in (401, 403):
        access = "NO"
    else:
        access = "?"
    return {"gated": gated, "access": access, "license": lic,
            "note": "" if access != "?" else "HEAD HTTP {}".format(code)}


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "datasets.txt")
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            rows.append((parts[0], parts[1] if len(parts) > 1 else ""))

    fmt = "{:<52} {:<24} {:<7} {:<7} {:<16} {}"
    print(fmt.format("dataset", "category", "gated", "access", "license", "note"))
    todo = []
    for ds_id, cat in rows:
        r = check(ds_id, token)
        print(fmt.format(ds_id, cat, r["gated"], r["access"],
                         str(r.get("license", ""))[:16], r.get("note", "")),
              flush=True)
        if r["access"] == "NO":
            todo.append((ds_id, r["gated"]))
    if todo:
        print("\nRequest access (log in as the token's user, accept terms):")
        for ds_id, gated in sorted(todo, key=lambda t: t[1] != "manual"):
            print("  [{}] https://huggingface.co/datasets/{}".format(gated, ds_id))


if __name__ == "__main__":
    main()
