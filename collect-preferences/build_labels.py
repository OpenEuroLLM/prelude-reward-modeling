#!/usr/bin/env python3
"""Preference labels of the full run: one row per pair of data/preferences/pairs.parquet.

  collect-preferences/run.sh build_labels.py                       # after the tie-break runs
  collect-preferences/run.sh build_labels.py --allow-incomplete    # a preview while votes are still missing

p_x is the share of the pair's valid votes that prefer answer x, over every vote of the pair: its three judges
(assign_judges.py) in both orders, and the tie-breaker if it had one. A judge that changes its verdict with the
order therefore contributes 0.5. The label is p_x after two rules (README, "Label (post-processing)"):
  think_tag   an answer that contains <think> or </think> is strongly dispreferred (label 0 or 1) whatever the
              votes, unless both answers have them; then the votes stand.
  too_long    a pair that no judge could read (every call too_long) has no label, except where one answer ran
              to the length limit and is over four times longer than the other, the conversation is under
              60,000 characters, and the other answer stopped normally without looping (under half of its word
              10-grams repeated): then the long answer is strongly dispreferred (all such long answers were
              read on 2026-11-16: every one is degenerate).
Pairs without a valid vote for another reason (parse errors only) have no label.

Writes data/preferences/labels.parquet: id, language, group, judges (the voting judges, tie-breaker last), votes
(judge, order, prefers of every valid vote), n_votes, votes_x, p_x, label, rule, and data/preferences/LABELS.txt
(the summary printed here, with agreement and Brier score against the Opus silver votes on the pilot pairs).
"""
import argparse
import collections
import glob
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PREF = os.path.join(ROOT, "data", "preferences")
THINK_TAGS = ("<think>", "</think>")


def repeated_share(text, n=10):
    """share of the word n-grams of text that already occurred earlier in it"""
    w = text.split()
    seen, rep = set(), 0
    for i in range(max(0, len(w) - n + 1)):
        g = tuple(w[i:i + n])
        rep += g in seen
        seen.add(g)
    return rep / max(1, len(w) - n + 1)


def too_long_rule(p):
    """'x' / 'y': the answer to disprefer in a pair no judge could read; None: no label"""
    conv = sum(len(m.get("content") or "") for m in json.loads(p["messages"]))
    lx, ly = len(p["response_x"]), len(p["response_y"])
    big, small = ("x", "y") if lx >= ly else ("y", "x")
    if conv > 60_000 or not max(lx, ly) > 4 * min(lx, ly) + 1 or p["finish_" + big] != "length":
        return None
    if p["finish_" + small] != "stop" or repeated_share(p["response_" + small]) >= 0.5:
        return None
    return big


def load_votes(dirs):
    v = collections.defaultdict(dict)                   # pair -> (judge, order) -> (status, prefers)
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(PREF, "votes", d, "*.parquet"))):
            for r in pq.read_table(path, columns=["pair_id", "order", "judge", "status", "prefers"]).to_pylist():
                v[r["pair_id"]][(r["judge"], r["order"])] = (r["status"], r["prefers"])
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-incomplete", action="store_true", help="label from the votes there are (a preview; writes nothing)")
    args = ap.parse_args()

    pairs = pq.read_table(os.path.join(PREF, "pairs.parquet"), columns=[
        "id", "language", "messages", "response_x", "response_y", "finish_x", "finish_y"]).to_pylist()
    assigned = {r["id"]: (r["group"], list(r["judges"])) for r in pq.read_table(os.path.join(PREF, "assigned", "judges.parquet")).to_pylist()}
    breaker = {}                                        # pair -> its tie-break judge
    for path in glob.glob(os.path.join(PREF, "assigned", "tiebreak_*.parquet")):
        j = os.path.basename(path)[len("tiebreak_"):-len(".parquet")]
        for i in pq.read_table(path, columns=["id"]).column("id").to_pylist():
            breaker[i] = j
    votes = load_votes(["pass1", "substitutes", "tiebreak"])

    missing = collections.Counter()
    for i, (g, js) in assigned.items():
        for j in js + ([breaker[i]] if i in breaker else []):
            for o in (0, 1):
                if (j, o) not in votes.get(i, {}):
                    missing[j] += 1
    if missing and not args.allow_incomplete:
        raise SystemExit("{} calls are missing ({}): finish the runs first, or --allow-incomplete for a preview".format(
            sum(missing.values()), dict(missing)))

    rows, why = [], collections.Counter()
    for p in pairs:
        i = p["id"]
        g, js = assigned[i]
        js = js + ([breaker[i]] if i in breaker else [])
        calls = {(j, o): votes.get(i, {}).get((j, o)) for j in js for o in (0, 1)}
        valid = [(j, o, c[1]) for (j, o), c in calls.items() if c and c[1] in ("x", "y")]
        kx = sum(v == "x" for _, _, v in valid)
        p_x = kx / len(valid) if valid else None
        tag = {s: any(t in p["response_" + s] for t in THINK_TAGS) for s in ("x", "y")}
        label, rule = p_x, None
        if not valid:
            label = None
            all_too_long = calls and all(c and c[0] == "too_long" for c in calls.values())
            if not all_too_long:
                why["no label: no valid vote (parse errors or missing calls)"] += 1
            else:
                bad = too_long_rule(p)
                if bad is None:
                    why["no label: too long for every judge"] += 1
                elif tag["x"] != tag["y"] and tag["y" if bad == "x" else "x"]:
                    why["no label: too long, the rules conflict (think tags in the other answer)"] += 1
                else:
                    label, rule = (0.0 if bad == "x" else 1.0), "too_long"
                    why["rule too_long: degenerate long answer dispreferred"] += 1
        elif tag["x"] != tag["y"]:
            label, rule = (0.0 if tag["x"] else 1.0), "think_tag"
            why["rule think_tag: tagged answer dispreferred"] += 1
        else:
            why["votes" + (" (both answers with think tags)" if tag["x"] else "")] += 1
        rows.append({"id": i, "language": p["language"], "group": g, "judges": js,
                     "votes": [{"judge": j, "order": o, "prefers": v} for j, o, v in valid],
                     "n_votes": len(valid), "votes_x": kx, "p_x": p_x, "label": label, "rule": rule})

    L = ["LABELS: {} pairs, {} labelled{}".format(len(rows), sum(r["label"] is not None for r in rows),
                                                  "; PREVIEW, {} calls still missing".format(sum(missing.values())) if missing else "")]
    for k, n in sorted(why.items()):
        L.append("  {:70s} {:7d}".format(k, n))
    n_tb = sum(r["id"] in breaker for r in rows)
    voted = [r for r in rows if r["rule"] is None and r["label"] is not None]
    L.append("votes per pair: " + ", ".join("{}: {}".format(k, n) for k, n in sorted(collections.Counter(r["n_votes"] for r in voted).items())))
    L.append("tie-breaker on {} pairs; even split after all votes: {} ({:.2%} of the pairs labelled by votes)".format(
        n_tb, sum(r["p_x"] == 0.5 for r in voted), sum(r["p_x"] == 0.5 for r in voted) / max(1, len(voted))))
    bins = collections.Counter(min(4, int(abs(r["label"] - 0.5) * 10)) for r in voted)
    L.append("|label - 0.5| (votes only): " + ", ".join("{:.1f}-{:.1f}: {:.1%}".format(b / 10, (b + 1) / 10 if b < 4 else 0.5, bins[b] / max(1, len(voted))) for b in range(5)))
    L.append("unanimous (label 0 or 1): {:.1%} of the pairs labelled by votes".format(sum(r["label"] in (0.0, 1.0) for r in voted) / max(1, len(voted))))

    # agreement with the Opus silver votes on the pilot pairs (both orders, consistent only)
    opus = collections.defaultdict(dict)
    for r in pq.read_table(os.path.join(PREF, "pilot", "opus", "votes.parquet"), columns=["pair_id", "order", "prefers"]).to_pylist():
        opus[r["pair_id"]][r["order"]] = r["prefers"]
    ref = {i: d[0] for i, d in opus.items() if d.get(0) in ("x", "y") and d.get(0) == d.get(1)}
    by = {r["id"]: r for r in rows}
    for title, keep in (("all labels", lambda r: True), ("labels by votes", lambda r: r["rule"] is None)):
        ids = [i for i in ref if i in by and by[i]["label"] is not None and keep(by[i])]
        y = {i: 1.0 if ref[i] == "x" else 0.0 for i in ids}
        agree = sum(1.0 if (by[i]["label"] > 0.5) == (y[i] == 1.0) and by[i]["label"] != 0.5 else 0.5 if by[i]["label"] == 0.5 else 0.0 for i in ids)
        brier = sum((by[i]["label"] - y[i]) ** 2 for i in ids)
        L.append("against Opus on the pilot pairs where Opus is consistent ({}): agreement {:.2%} (an even split counts half), Brier {:.4f}, {} pairs".format(
            title, agree / max(1, len(ids)), brier / max(1, len(ids)), len(ids)))
    print("\n".join(L))
    if missing:
        print("preview only: nothing written")
        return
    out = os.path.join(PREF, "labels.parquet")
    pq.write_table(pa.Table.from_pylist(rows), out)
    open(os.path.join(PREF, "LABELS.txt"), "w").write("\n".join(L) + "\n")
    print("-> {}, {}".format(out, os.path.join(PREF, "LABELS.txt")))


if __name__ == "__main__":
    main()
