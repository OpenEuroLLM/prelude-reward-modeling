#!/usr/bin/env python3
"""Completeness of the pilot runs: for every (judge, prompt, thinking) run in the votes directory, the calls
done against the 2 x pairs expected, statuses, and the finished/unfinished verdict. Unfinished runs are
resumed by rerunning the same judge.py / pilot job arguments.

  collect-preferences/run.sh pilot_status.py [--votes data/preferences/pilot/votes] [--pairs ...]
"""
import argparse
import collections
import glob
import os

import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PILOT = os.path.join(ROOT, "data", "preferences", "pilot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(PILOT, "pairs.parquet"))
    ap.add_argument("--votes", default=os.path.join(PILOT, "votes"))
    args = ap.parse_args()
    args.pairs, args.votes = os.path.join(ROOT, args.pairs), os.path.join(ROOT, args.votes)
    expected = 2 * pq.read_table(args.pairs, columns=["id"]).num_rows
    runs = collections.defaultdict(lambda: {"files": 0, "status": collections.Counter(), "tokens": 0})
    for f in sorted(glob.glob(os.path.join(args.votes, "*.parquet"))):
        t = pq.read_table(f, columns=["judge", "prompt_version", "thinking", "status", "completion_tokens"]).to_pylist()
        if not t:
            continue
        r = runs["{}|{}|{}".format(t[0]["judge"], t[0]["prompt_version"], t[0]["thinking"])]
        r["files"] += 1
        r["status"].update(x["status"] for x in t)
        r["tokens"] += sum(x["completion_tokens"] or 0 for x in t)
    print("{:36s} {:>5s} {:>11s} {:>7s} {:>7s} {:>7s} {:>7s}  {}".format("run", "files", "calls", "ok", "regex", "parse", "tokens", "state"))
    for run, r in sorted(runs.items()):
        n = sum(r["status"].values())
        print("{:36s} {:5d} {:5d}/{:5d} {:7d} {:7d} {:7d} {:7.0f}  {}".format(
            run, r["files"], n, expected, r["status"]["ok"], r["status"]["ok_regex"], r["status"]["parse_error"],
            r["tokens"] / max(1, n), "finished" if n >= expected else "UNFINISHED ({} calls missing)".format(expected - n)))


if __name__ == "__main__":
    main()
