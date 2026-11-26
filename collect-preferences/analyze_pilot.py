#!/usr/bin/env python3
"""Pilot analysis: how reliable is each judge, and how do the votes of a judge pool behave?

  collect-preferences/run.sh analyze_pilot.py [--votes data/preferences/pilot/votes] [--opus data/preferences/pilot/opus/votes.parquet]

For every (judge, thinking) run found in the votes directory, and for the Opus silver run, a pair gets a
preference from its two presentation orders: x or y when both orders agree, "incons" otherwise (counted as
0.5). Reported per judge: parse errors, position consistency and the direction of the position bias,
agreement with Opus overall and by language tier, category and task type, behaviour on pairs where exactly
one answer was flagged by language ID, length bias (how often the longer answer wins, compared with Opus),
and self-family preference (win rate of a judge's own family's responders under that judge vs under Opus).
Then the judges' pairwise agreement, and a simulation of the pool: vote fractions over all open judges in
both orders, their distribution, and the agreement of the majority with Opus. Writes
data/preferences/pilot/REPORT.txt.
"""
import argparse
import collections
import glob
import itertools
import math
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PILOT = os.path.join(ROOT, "data", "preferences", "pilot")
sys.path.insert(0, HERE)
from judge_prompt import same_lang  # noqa: E402

from families import RESPONDER_FAMILY  # noqa: E402  (re-exported for the analysis scripts)


def load_votes(path_glob):
    import pyarrow.parquet as pq
    runs = collections.defaultdict(dict)        # run -> pair_id -> {order: row}
    for p in sorted(glob.glob(path_glob)):
        for r in pq.read_table(p).to_pylist():
            run = "{}|{}|{}".format(r["judge"], r["prompt_version"], r["thinking"])
            runs[run].setdefault(r["pair_id"], {})[r["order"]] = r
    return runs


def preference(rows):
    """-> ('x'|'y'|'incons'|None, p_x) from the two orders of one pair"""
    v = [rows[o]["prefers"] for o in (0, 1) if o in rows and rows[o].get("prefers")]
    if len(v) < 2:
        return (None, None) if not v else (v[0], 1.0 if v[0] == "x" else 0.0)
    if v[0] == v[1]:
        return v[0], 1.0 if v[0] == "x" else 0.0
    return "incons", 0.5


def pct(a, b):
    return "{:5.1f}%".format(100.0 * a / b) if b else "    - "


def soft_p(rows):
    """share of a pair's two calls (orders) preferring x; a call without a verdict counts as half"""
    return sum({"x": 1.0, "y": 0.0}.get((rows.get(o) or {}).get("prefers"), 0.5) for o in (0, 1)) / 2


def agreement(p_a, p_b, ids, ref=False):
    """expected agreement of one call of A with one call of B over the pairs both judged: an inconsistent pair or a
    call without a verdict scores half, so inconsistency costs agreement. ref=True: B is the reference (Opus), and
    only the pairs where it is consistent count. -> (agreement sum, pairs)"""
    s = n = 0
    for i in ids:
        if i not in p_a or i not in p_b or (ref and p_b[i] not in (0.0, 1.0)):
            continue
        s += p_a[i] * p_b[i] + (1 - p_a[i]) * (1 - p_b[i])
        n += 1
    return s, n


def main():
    import pyarrow.parquet as pq
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(PILOT, "pairs.parquet"))
    ap.add_argument("--votes", default=os.path.join(PILOT, "votes"))
    ap.add_argument("--opus", default=os.path.join(PILOT, "opus", "votes.parquet"))
    ap.add_argument("--out", default=os.path.join(PILOT, "REPORT.txt"))
    ap.add_argument("--pool-prompt", default="v1", help="prompt version whose thinking-off runs form the simulated pool")
    ap.add_argument("--pool-judges", help="comma-separated judge names for the simulated pool (default: all)")
    ap.add_argument("--pool-runs", help="comma-separated runs (judge|prompt|thinking) forming the simulated pool; "
                                        "overrides --pool-prompt and --pool-judges")
    args = ap.parse_args()
    for k in ("pairs", "votes", "opus", "out"):                       # relative paths are relative to the repo root
        setattr(args, k, os.path.join(ROOT, getattr(args, k)))
    pairs = {p["id"]: p for p in pq.read_table(args.pairs).to_pylist()}
    judges = {j["name"]: j for j in yaml.safe_load(open(os.path.join(HERE, "judges.yaml")))["judges"]}
    runs = load_votes(os.path.join(args.votes, "*.parquet"))
    opus = load_votes(args.opus) if os.path.exists(args.opus) else {}
    opus_run = next(iter(opus), None)
    prefs = {run: {i: preference(rows) for i, rows in d.items()} for run, d in list(runs.items()) + list(opus.items())}
    ref = prefs.get(opus_run, {})
    softp = {run: {i: soft_p(rows) for i, rows in d.items()} for run, d in list(runs.items()) + list(opus.items())}
    refp = softp.get(opus_run, {})
    L = []

    def flagged_side(p):
        """'x' or 'y' if exactly that answer is flagged by LID as another language; else None"""
        if p["task_type"] == "translation" or p["locale_dependence"] == "language_specific":
            return None
        f = [s for s in ("x", "y") if p["lid_" + s] and p["lid_conf_" + s] >= 0.5 and len(p["response_" + s].strip()) >= 20
             and not same_lang(p["lid_" + s], p["language"])]
        return f[0] if len(f) == 1 else None

    L.append("PILOT: {} pairs; runs: {}; reference: {}".format(len(pairs), ", ".join(sorted(runs)), opus_run or "none"))
    L.append("")
    hdr = "{:34s} {:>6s} {:>6s} {:>6s} {:>7s} {:>7s} {:>7s} {:>7s} | {:>7s} {:>7s} | {:>7s} {:>7s} {:>7s}".format(
        "run", "pairs", "calls", "parse", "tokens", "consis", "biasA", "biasB", "agrOpus", "cover", "lang-ok", "longWin", "OpusLW")
    L.append(hdr)
    for run in sorted(list(runs) + ([opus_run] if opus_run else [])):
        d = runs.get(run) or opus.get(run)
        calls = [r for rows in d.values() for r in rows.values()]
        n_calls = len(calls)
        parse = sum(not r["status"].startswith("ok") for r in calls)      # ok / ok_regex both carry a verdict
        toks = sum(r.get("completion_tokens") or 0 for r in calls) / max(1, n_calls)
        pr = prefs[run]
        decided = [i for i, (v, _) in pr.items() if v is not None]
        consistent = sum(pr[i][0] in ("x", "y") for i in decided)
        # position bias: inconsistent pairs where both orders chose A (first) or both chose B (second)
        bias_a = bias_b = 0
        for i in decided:
            if pr[i][0] == "incons":
                v0 = d[i][0]["verdict"] if 0 in d[i] else None
                if v0 == "A":
                    bias_a += 1
                elif v0 == "B":
                    bias_b += 1
        same, both = agreement(softp[run], refp, list(pairs), ref=True) if run != opus_run and ref else (0, 0)
        # language fidelity: pairs with exactly one flagged side -> share preferring the other side
        lf = [(i, flagged_side(pairs[i])) for i in decided if flagged_side(pairs[i])]
        lf_ok = sum(pr[i][0] not in (None, "incons") and pr[i][0] != f for i, f in lf)
        lf_n = sum(pr[i][0] in ("x", "y") for i, _ in lf)
        # length bias: share of consistent pairs where the longer answer wins (ratio > 1.5)
        lw = lw_n = ow = ow_n = 0
        for i in decided:
            p = pairs[i]
            if pr[i][0] not in ("x", "y") or min(p["chars_x"], p["chars_y"]) == 0:
                continue
            ratio = max(p["chars_x"], p["chars_y"]) / min(p["chars_x"], p["chars_y"])
            if ratio < 1.5:
                continue
            longer = "x" if p["chars_x"] > p["chars_y"] else "y"
            lw += pr[i][0] == longer
            lw_n += 1
            if ref.get(i, (None,))[0] in ("x", "y"):
                ow += ref[i][0] == longer
                ow_n += 1
        L.append("{:34s} {:6d} {:6d} {:>6s} {:7.0f} {:>7s} {:>7s} {:>7s} | {:>7s} {:7d} | {:>7s} {:>7s} {:>7s}".format(
            run, len(decided), n_calls, pct(parse, n_calls), toks, pct(consistent, len(decided)), pct(bias_a, len(decided)),
            pct(bias_b, len(decided)), pct(same, both), both, pct(lf_ok, lf_n), pct(lw, lw_n), pct(ow, ow_n)))
    L.append("consis: both orders agree; biasA/biasB: inconsistent pairs where the first/second position won both times; "
             "agrOpus: agreement with Opus on the pairs where Opus is consistent (cover = how many), each of the judge's "
             "two calls scoring 1 if it matches, 0 if not and 1/2 without a verdict (an inconsistent pair scores 1/2); lang-ok: pairs with exactly "
             "one LID-flagged answer where the judge preferred the other one; longWin: share of consistent pairs (length "
             "ratio > 1.5) won by the longer answer, OpusLW: the same for Opus on the same pairs.")
    L.append("")

    def clean(p):
        """both answers substantial, complete and in the expected language: the genuinely close comparisons"""
        return (min(p["chars_x"], p["chars_y"]) >= 200 and p["finish_x"] != "length" and p["finish_y"] != "length"
                and flagged_side(p) is None)

    # agreement with Opus by tier / category / task type
    if ref:
        L.append("AGREEMENT WITH OPUS by slice (pairs where Opus is consistent; an inconsistent judge pair scores 1/2):")
        slices = [("tier", lambda p: p["tier"]), ("tier, clean pairs only", lambda p: p["tier"] if clean(p) else None),
                  ("category", lambda p: p["category"]), ("task_type", lambda p: p["task_type"])]
        for key, fn in slices:
            vals = sorted({fn(p) for p in pairs.values()} - {None})
            L.append("  " + key)
            for run in sorted(runs):
                cells = []
                for v in vals:
                    ids = [i for i, p in pairs.items() if fn(p) == v]
                    s, b = agreement(softp[run], refp, ids, ref=True)
                    cells.append("{}={}({})".format(v, pct(s, b).strip(), b))
                L.append("    {:34s} {}".format(run, "  ".join(cells)))
        L.append("")

        # length bias by tier: share of consistent pairs with length ratio > 1.5 won by the longer answer, judge vs Opus
        L.append("LENGTH BIAS by tier (longer answer wins, length ratio > 1.5): judge vs Opus on the same pairs")
        for run in sorted(runs):
            cells = []
            for t in ("en", "large", "mid", "small"):
                lw = ow = n = 0
                for i, p in pairs.items():
                    if p["tier"] != t or min(p["chars_x"], p["chars_y"]) == 0:
                        continue
                    if prefs[run].get(i, (None,))[0] not in ("x", "y") or ref.get(i, (None,))[0] not in ("x", "y"):
                        continue
                    if max(p["chars_x"], p["chars_y"]) / min(p["chars_x"], p["chars_y"]) < 1.5:
                        continue
                    longer = "x" if p["chars_x"] > p["chars_y"] else "y"
                    lw += prefs[run][i][0] == longer
                    ow += ref[i][0] == longer
                    n += 1
                cells.append("{}={} vs {} ({})".format(t, pct(lw, n).strip(), pct(ow, n).strip(), n))
            L.append("  {:34s} {}".format(run, "  ".join(cells)))
        L.append("")

    # position consistency by language tier, every run
    L.append("POSITION CONSISTENCY by tier (pairs where both orders agree):")
    for run in sorted(list(runs) + ([opus_run] if opus_run else [])):
        pr = prefs[run]
        cells = []
        for t in ("en", "large", "mid", "small"):
            ids = [i for i in pairs if pairs[i]["tier"] == t and pr.get(i, (None,))[0]]
            cells.append("{}={}({})".format(t, pct(sum(pr[i][0] in ("x", "y") for i in ids), len(ids)).strip(), len(ids)))
        L.append("  {:34s} {}".format(run, "  ".join(cells)))
    L.append("")

    # self-family preference
    L.append("SELF-FAMILY CHECK: win rate of responders from the judge's own family under that judge vs under Opus")
    for run in sorted(runs):
        fam = judges.get(run.split("|")[0], {}).get("family")
        ids = [i for i, p in pairs.items() if fam and (RESPONDER_FAMILY.get(p["model_x"]) == fam) != (RESPONDER_FAMILY.get(p["model_y"]) == fam)]
        w = n = ow = on = 0
        for i in ids:
            own = "x" if RESPONDER_FAMILY.get(pairs[i]["model_x"]) == fam else "y"
            if prefs[run].get(i, (None,))[0] in ("x", "y"):
                w += prefs[run][i][0] == own
                n += 1
            if ref.get(i, (None,))[0] in ("x", "y"):
                ow += ref[i][0] == own
                on += 1
        L.append("  {:34s} family {:8s} pairs with one own-family answer: {:4d}  own wins: {} (judge) vs {} (Opus)".format(
            run, str(fam), len(ids), pct(w, n), pct(ow, on)))
    L.append("")

    # inter-judge agreement
    names = sorted(runs)
    if len(names) > 1:
        L.append("INTER-JUDGE AGREEMENT (one call of each judge, both orders, all pairs both judged; inconsistency costs):")
        for a, b in itertools.combinations(names, 2):
            s, n = agreement(softp[a], softp[b], list(pairs))
            L.append("  {:34s} {:34s} {} ({})".format(a, b, pct(s, n), n))
        L.append("")

    # pool simulation: thinking-off runs of distinct judges with one prompt version, both orders -> vote fraction for x
    if args.pool_runs:
        pool = [r for r in args.pool_runs.split(",") if r in runs]
    else:
        pool = [r for r in names if r.split("|")[1] == args.pool_prompt and r.endswith("|off")
                and (not args.pool_judges or r.split("|")[0] in args.pool_judges.split(","))]
    if len(pool) >= 2:
        L.append("POOL SIMULATION over {} judges x 2 orders ({} votes per pair): {}".format(len(pool), 2 * len(pool), ", ".join(pool)))
        frac = {}
        for i in pairs:
            votes = [runs[r][i][o]["prefers"] for r in pool for o in (0, 1) if i in runs[r] and o in runs[r][i] and runs[r][i][o].get("prefers")]
            if len(votes) == 2 * len(pool):
                frac[i] = sum(v == "x" for v in votes) / len(votes)
        hist = collections.Counter(round(f * 2 * len(pool)) for f in frac.values())
        L.append("  pairs with all votes: {}; votes for x: {}".format(len(frac), dict(sorted(hist.items()))))
        mid = sum(1 for f in frac.values() if 0.2 < f < 0.8)
        L.append("  middle band (0.2 < fraction < 0.8, would get a second pass): {} ({})".format(mid, pct(mid, len(frac))))
        if ref:
            maj = {i: ("x" if f > 0.5 else "y" if f < 0.5 else "tie") for i, f in frac.items()}
            both = [i for i in frac if refp.get(i) in (0.0, 1.0)]          # Opus consistent in both orders
            rw = {i: "x" if refp[i] == 1.0 else "y" for i in both}
            hit = {i: 0.5 if maj[i] == "tie" else float(maj[i] == rw[i]) for i in both}
            same = sum(hit.values())
            L.append("  majority vs Opus (a tied vote counts half): {} ({} pairs); unanimous pairs vs Opus: {}".format(
                pct(same, len(both)), len(both),
                pct(sum(maj[i] == rw[i] for i in both if frac[i] in (0.0, 1.0)), sum(1 for i in both if frac[i] in (0.0, 1.0)))))
            # agreement of Opus with the fraction, by band
            bands = collections.defaultdict(lambda: [0, 0])
            for i in both:
                b = "unanimous" if frac[i] in (0.0, 1.0) else "strong" if frac[i] <= 0.2 or frac[i] >= 0.8 else "middle"
                bands[b][0] += hit[i]
                bands[b][1] += 1
            L.append("  majority agreement with Opus by band: " + "  ".join("{}={} ({})".format(k, pct(*v), v[1]) for k, v in bands.items()))
        L.append("")

    text = "\n".join(L)
    print(text)
    with open(args.out, "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
