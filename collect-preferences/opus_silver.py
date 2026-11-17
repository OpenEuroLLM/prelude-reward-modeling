#!/usr/bin/env python3
"""Silver annotations from Claude Opus for the pilot pairs: the same judge prompt, both orders, through the
Message Batches API (asynchronous, half price). Evaluation use only: these labels pick and weight the open
judges and measure their biases; they are never training labels.

  collect-preferences/run.sh opus_silver.py submit [--pairs ...] [--out DIR] [--limit N]   # creates the batch
  collect-preferences/run.sh opus_silver.py status [--out DIR]
  collect-preferences/run.sh opus_silver.py fetch [--out DIR]                               # -> votes parquet

The API key is read from ~/.anthropic_api_key and never printed. Plain HTTPS with `requests` (the SDK is not
in the LUMI containers). Opus 5.5 always thinks (adaptive); effort is its only control. Output:
<out>/votes.parquet with the same columns as judge.py; the default <out> is data/preferences/pilot/v3/opus
(round 3, prompt v3); the v1 round is in data/preferences/pilot/opus.
"""
import argparse
import json
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from judge_prompt import PROMPT_VERSION, SCHEMA, build, extract  # noqa: E402

API = "https://api.anthropic.com/v1"
MODEL = "claude-opus-5-5"
OUT = os.path.join(ROOT, "data", "preferences", "pilot", "v3", "opus")
MAX_TOKENS = 16000           # thinking included; the v1 round used 8,000


def headers():
    key = open(os.path.expanduser("~/.anthropic_api_key")).read().strip()
    return {"x-api-key": key, "anthropic-version": "2026-10-21", "content-type": "application/json"}


def request_for(pair, order, effort, version):
    system, user = build(pair, order, version)
    return {"custom_id": "{}_o{}".format(pair["id"], order),        # allowed: [a-zA-Z0-9_-]
            "params": {"model": MODEL, "max_tokens": MAX_TOKENS, "system": system,
                       "messages": [{"role": "user", "content": user}],
                       "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}}}}


def submit(args):
    import pyarrow.parquet as pq
    pairs = pq.read_table(args.pairs).to_pylist()
    if args.limit:
        pairs = pairs[:args.limit]
    reqs = [request_for(p, order, args.effort, args.prompt) for p in pairs for order in (0, 1)]
    chars = sum(len(r["params"]["system"]) + len(r["params"]["messages"][0]["content"]) for r in reqs)
    print("{} requests, about {:.1f}M input tokens (chars/3.5)".format(len(reqs), chars / 3.5 / 1e6))
    if args.dry_run:
        print(json.dumps(reqs[0]["params"], ensure_ascii=False)[:3000])
        return
    r = requests.post(API + "/messages/batches", headers=headers(), json={"requests": reqs}, timeout=300)
    if r.status_code >= 400:
        raise SystemExit("batch creation failed: HTTP {} {}".format(r.status_code, r.text[:2000]))
    batch = r.json()
    os.makedirs(args.out, exist_ok=True)
    if os.path.exists(os.path.join(args.out, "batch.json")):
        raise SystemExit("{} already has a batch; use another --out".format(args.out))
    with open(os.path.join(args.out, "batch.json"), "w") as f:
        json.dump({"id": batch["id"], "created": batch.get("created_at"), "n": len(reqs), "pairs": args.pairs,
                   "prompt_version": args.prompt, "effort": args.effort, "model": MODEL,
                   "max_tokens": MAX_TOKENS}, f, indent=1)
    print("batch {} submitted: {}".format(batch["id"], batch.get("processing_status")))


def status(args):
    meta = json.load(open(os.path.join(args.out, "batch.json")))
    r = requests.get(API + "/messages/batches/" + meta["id"], headers=headers(), timeout=60)
    r.raise_for_status()
    b = r.json()
    print(b.get("processing_status"), b.get("request_counts"))
    return b


def fetch(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    meta = json.load(open(os.path.join(args.out, "batch.json")))
    b = status(args)
    if b.get("processing_status") != "ended":
        print("not finished yet")
        return
    r = requests.get(b["results_url"], headers=headers(), timeout=600)
    r.raise_for_status()
    pairs = {p["id"]: p for p in pq.read_table(meta["pairs"]).to_pylist()}
    rows, usage = [], {"input": 0, "output": 0}
    for line in r.text.splitlines():
        if not line.strip():
            continue
        res = json.loads(line)
        pid, order = res["custom_id"].rsplit("_o", 1)
        order = int(order)
        base = {"pair_id": pid, "order": order, "judge": MODEL, "judge_revision": None,
                "prompt_version": meta.get("prompt_version", "v1"), "thinking": "adaptive-" + meta["effort"],
                "language": pairs[pid]["language"]}
        if res["result"]["type"] != "succeeded":
            rows.append({**base, "status": res["result"]["type"], "raw": json.dumps(res["result"])[:4000]})
            continue
        msg = res["result"]["message"]
        text = "".join(c.get("text", "") for c in msg["content"] if c.get("type") == "text")
        obj = extract(text)
        verdict = obj["verdict"] if obj else None
        prefers = None if verdict is None else ("x" if (verdict == "A") == (order == 0) else "y")
        u = msg.get("usage") or {}
        usage["input"] += u.get("input_tokens", 0)
        usage["output"] += u.get("output_tokens", 0)
        rows.append({**base, "status": "ok" if obj else ("refusal" if msg.get("stop_reason") == "refusal" else "parse_error"),
                     "verdict": verdict, "prefers": prefers, "request": (obj or {}).get("request"),
                     "response_a": (obj or {}).get("response_a"), "response_b": (obj or {}).get("response_b"),
                     "comparison": (obj or {}).get("comparison"), "raw": text, "finish_reason": msg.get("stop_reason"),
                     "prompt_tokens": u.get("input_tokens"), "completion_tokens": u.get("output_tokens")})
    schema = pa.schema([("pair_id", pa.string()), ("order", pa.int8()), ("judge", pa.string()), ("judge_revision", pa.string()),
                        ("prompt_version", pa.string()), ("thinking", pa.string()), ("language", pa.string()),
                        ("status", pa.string()), ("verdict", pa.string()), ("prefers", pa.string()),
                        ("request", pa.string()), ("response_a", pa.string()), ("response_b", pa.string()),
                        ("comparison", pa.string()), ("raw", pa.string()), ("finish_reason", pa.string()),
                        ("prompt_tokens", pa.int32()), ("completion_tokens", pa.int32())])
    path = os.path.join(args.out, "votes.parquet")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    ok = sum(r["status"] == "ok" for r in rows)
    print("{} results ({} ok) -> {}; tokens in {:.2f}M out {:.2f}M (batch price: about ${:.0f})".format(
        len(rows), ok, path, usage["input"] / 1e6, usage["output"] / 1e6,
        usage["input"] / 1e6 * 2.0 + usage["output"] / 1e6 * 10.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["submit", "status", "fetch"])
    ap.add_argument("--pairs", default=os.path.join(ROOT, "data", "preferences", "pilot", "pairs.parquet"))
    ap.add_argument("--effort", default="high")
    ap.add_argument("--prompt", default=PROMPT_VERSION, help="judge prompt version (judge_prompt.PROMPTS)")
    ap.add_argument("--out", default=OUT, help="directory for batch.json and votes.parquet")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.pairs, args.out = os.path.join(ROOT, args.pairs), os.path.join(ROOT, args.out)   # relative to the repo root
    {"submit": submit, "status": status, "fetch": fetch}[args.cmd](args)


if __name__ == "__main__":
    main()
