#!/usr/bin/env python3
"""Metrics of trained reward models, from the rewards that train_rm.py wrote.

  train-reward-model/run.sh eval_report.py RUN [RUN ...] [--out data/reward_model/REPORT.md] [--detail]

Validation (the 1,008 pilot pairs, data/reward_model/runs/<run>/val_<last step>.parquet):
  acc     accuracy against the majority of the judge votes, pairs with an even split left out
  agree   expected agreement with one random vote: p_a if the RM prefers a, 1 - p_a if b, 0.5 on a tie
  opus    accuracy against Claude Opus 5.5 (prompt v3) on the pairs where Opus gives the same verdict in both
          orders (evaluation only; Opus is never a training label)
  ll      log loss of sigmoid(ra - rb) against p_a
  long    share of the pairs with a length ratio >= 1.5 (tokens) where the RM prefers the longer answer,
          next to the label mass of the longer answer ("lab")
Benchmarks (runs/<run>/scores_<bench>.parquet, official scoring rules):
  rb2     RewardBench 2: per subset the share of prompts whose correct answer outscores all 3 wrong ones
          (an exact tie at the top gets 1/k), Ties by its weighted formula; the score is the mean of the 6 subsets
  rmb     RM-Bench: chosen style i vs rejected style j over the 3 x 3 styles; easy (j < i), normal (i = j),
          hard (j > i: a concise correct answer against a more detailed wrong one); the mean over 4 domains
  mrb     M-RewardBench, 11 European languages: accuracy per category, the mean of the 4 categories per
          language, the mean over languages ("mrb_small": 300 pairs per language)
  probes  share of the length probes (prepare_probes.py) where the original answer beats its padded copy and
          its truncated copy
"""
import argparse
import collections
import glob
import json
import math
import os
import re
import sys

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "reward_model")
OPUS = os.path.join(ROOT, "data", "preferences", "pilot", "v3", "opus", "votes.parquet")
PILOT = os.path.join(ROOT, "data", "preferences", "pilot", "pairs.parquet")


def opus_ref():
    d = collections.defaultdict(dict)
    for r in pq.read_table(OPUS, columns=["pair_id", "order", "prefers"]).to_pylist():
        d[r["pair_id"]][r["order"]] = r["prefers"]
    return {i: ("a" if v[0] == "x" else "b") for i, v in d.items() if v.get(0) in ("x", "y") and v.get(0) == v.get(1)}


def last_val(run_dir):
    fs = sorted(glob.glob(os.path.join(run_dir, "val_*.parquet")))
    return fs[-1] if fs else None


def val_metrics(run_dir, model, detail=False):
    f = last_val(run_dir)
    if not f:
        return {}
    v = pq.read_table(f).to_pandas()
    meta = pq.read_table(os.path.join(DATA, "tok", model, "val.parquet"),
                         columns=["id", "language", "category", "a_ids", "b_ids", "prompt_ids"]).to_pandas()
    tier = dict(zip(*pq.read_table(PILOT, columns=["id", "tier"]).to_pydict().values()))
    v = v.merge(meta, on="id")
    d = (v.r_a - v.r_b).to_numpy()
    p = v.p_a.to_numpy()
    nt = p != 0.5
    out = {"step": int(re.search(r"val_(\d+)", f).group(1))}
    out["acc"] = np.mean((d[nt] > 0) == (p[nt] > 0.5))
    out["agree"] = np.mean(np.where(d > 0, p, np.where(d < 0, 1 - p, 0.5)))
    out["ll"] = np.mean(np.logaddexp(0, -d) * p + np.logaddexp(0, d) * (1 - p))
    ref = opus_ref()
    o = np.array([i in ref for i in v.id])
    w = np.array([ref.get(i) == "a" for i in v.id])
    out["opus"] = np.mean((d[o] > 0) == w[o])
    out["opus_n"] = int(o.sum())
    la = v.a_ids.map(len).to_numpy() + v.prompt_ids.map(len).to_numpy()
    lb = v.b_ids.map(len).to_numpy() + v.prompt_ids.map(len).to_numpy()
    lr = np.maximum(v.a_ids.map(len), v.b_ids.map(len)) / np.maximum(1, np.minimum(v.a_ids.map(len), v.b_ids.map(len)))
    m = lr.to_numpy() >= 1.5
    a_longer = (v.a_ids.map(len) > v.b_ids.map(len)).to_numpy()
    out["long"] = np.mean(np.where(a_longer[m], d[m] > 0, d[m] < 0))
    out["long_lab"] = np.mean(np.where(a_longer[m], p[m], 1 - p[m]))
    out["long_n"] = int(m.sum())
    if detail:
        v["tier"] = v.id.map(tier)
        rows = {}
        for col in ("tier", "category"):
            for g, sub in v.groupby(col):
                dd = (sub.r_a - sub.r_b).to_numpy()
                pp = sub.p_a.to_numpy()
                k = pp != 0.5
                oo = np.array([i in ref for i in sub.id])
                ww = np.array([ref.get(i) == "a" for i in sub.id])
                rows["{}={}".format(col, g)] = {"n": len(sub), "acc": np.mean((dd[k] > 0) == (pp[k] > 0.5)),
                                                 "opus": np.mean((dd[oo] > 0) == ww[oo]) if oo.any() else float("nan")}
        bands = [(1.5, 2), (2, 4), (4, 1e9)]
        for a, b in bands:
            mm = (lr.to_numpy() >= a) & (lr.to_numpy() < b)
            rows["ratio {}-{}".format(a, b if b < 1e9 else "inf")] = {
                "n": int(mm.sum()), "rm_longer": np.mean(np.where(a_longer[mm], d[mm] > 0, d[mm] < 0)),
                "label_longer": np.mean(np.where(a_longer[mm], p[mm], 1 - p[mm]))}
        lang_mean = {}
        for g, sub in v.groupby("language"):
            lang_mean[g] = float(np.mean(np.concatenate([sub.r_a, sub.r_b])))
        out["detail"] = rows
        out["lang_reward_mean"] = lang_mean
    return out


def bench_scores(run_dir, bench):
    f = os.path.join(run_dir, "scores_{}.parquet".format(bench))
    if not os.path.exists(f):
        return None
    s = dict(zip(*pq.read_table(f).to_pydict().values()))
    meta = pq.read_table(os.path.join(DATA, "bench", bench + "_meta.parquet"),
                         columns=["id", "item", "role", "k", "group", "language"]).to_pandas()
    meta["r"] = meta.id.map(s)
    return meta


def rb2(meta):
    res = {}
    for g, sub in meta.groupby("group"):
        if g == "Ties":
            continue
        acc = []
        for item, it in sub.groupby("item"):
            c = it[it.role == "chosen"].r.to_numpy()
            r = it[it.role == "rejected"].r.to_numpy()
            scores = np.concatenate([c[:1], r])
            mx = scores.max()
            acc.append(1.0 / np.sum(scores == mx) if scores[0] == mx else 0.0)
        res[g] = float(np.mean(acc))
    ties = meta[meta.group == "Ties"]
    if len(ties):
        stats = {}
        for item, it in ties.groupby("item"):
            typ, pid = str(item).split(":")
            c = it[it.role == "chosen"].r.to_numpy()
            r = it[it.role == "rejected"].r.to_numpy()
            dm = c.max() - c.min() if len(c) > 1 else None
            stats[(typ, int(pid))] = (c.min() - r.max() > 0, dm, c.min() - r.max())
        ref = {p: v for (t, p), v in stats.items() if t == "ref"}
        tied = {p: v for (t, p), v in stats.items() if t != "ref"}
        ref_acc = np.mean([v[0] for v in ref.values()]) if ref else 0.0
        tied_acc = np.mean([v[0] for v in tied.values()]) if tied else 0.0
        both = sorted(set(ref) & set(tied))
        dcm = np.array([tied[p][1] for p in both], dtype=float)
        cit = np.array([tied[p][2] for p in both], dtype=float)
        cir = np.array([ref[p][2] for p in both], dtype=float)
        cp = np.mean(cit > dcm)
        cph = np.mean(np.minimum(cir, cit) > dcm)
        with np.errstate(divide="ignore", invalid="ignore"):
            ms = np.nan_to_num(np.tanh(np.minimum(cir, cit) / dcm - 1), nan=0.0)
        res["Ties"] = float(0.3 * tied_acc + 0.3 * ref_acc + 0.2 * cp + 0.2 * cph + 0.01 * np.mean(ms))
    res["score"] = float(np.mean([res[k] for k in res]))
    return res


def rmbench(meta):
    res = {}
    per_domain = {}
    for g, sub in meta.groupby("group"):
        dom = "safety" if g.startswith("safety") else g
        mats = []
        for item, it in sub.groupby("item"):
            c = it[it.role == "chosen"].sort_values("k").r.to_numpy()
            r = it[it.role == "rejected"].sort_values("k").r.to_numpy()
            if len(c) == 3 and len(r) == 3:
                mats.append((c[:, None] > r[None, :]).astype(float))
        per_domain.setdefault(dom, []).extend(mats)
    for dom, mats in per_domain.items():
        m = np.mean(mats, axis=0)
        res[dom + "_easy"] = float(np.tril(m, -1).sum() / 3)
        res[dom + "_normal"] = float(np.mean(np.diag(m)))
        res[dom + "_hard"] = float(np.triu(m, 1).sum() / 3)
        res[dom] = float(np.mean([res[dom + "_easy"], res[dom + "_normal"], res[dom + "_hard"]]))
    doms = [d for d in ("chat", "math", "code", "safety") if d in res]
    for lvl in ("easy", "normal", "hard"):
        res[lvl] = float(np.mean([res[d + "_" + lvl] for d in doms]))
    res["score"] = float(np.mean([res[d] for d in doms]))
    return res


def mrb(meta):
    res = {}
    langs = {}
    for (lang, cat), sub in meta.groupby(["language", "group"]):
        piv = sub.pivot_table(index="item", columns="role", values="r")
        langs.setdefault(lang, {})[cat] = float(np.mean(piv["chosen"] > piv["rejected"]))
    for lang, cats in langs.items():
        res[lang] = float(np.mean(list(cats.values())))
    res["score"] = float(np.mean([res[l] for l in langs]))
    return res


def probes(meta):
    res = {}
    for g, sub in meta.groupby("group"):
        piv = sub.pivot_table(index="item", columns="role", values="r")
        res[g] = float(np.mean(piv["chosen"] > piv["rejected"]))
    return res


def run_model(run_dir):
    a = json.load(open(os.path.join(run_dir, "args.json")))
    return a["model"], a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    rows = []
    for run in args.runs:
        rd = run if os.path.isdir(run) else os.path.join(DATA, "runs", run)
        model, a = run_model(rd)
        r = {"run": os.path.basename(rd.rstrip("/")), "model": model}
        r.update(val_metrics(rd, model, detail=args.detail))
        for bench, fn in (("rb2", rb2), ("rmbench", rmbench), ("mrb", mrb), ("mrb_small", mrb), ("probes", probes)):
            meta = bench_scores(rd, bench)
            if meta is not None:
                r[bench] = fn(meta)
        rows.append(r)
    hdr = "| run | step | acc | agree | opus (n) | ll | longer wins rm / labels | RB2 | RM-Bench (hard) | M-RB |"
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for r in rows:
        if "acc" not in r:
            print("| {} | no validation yet |".format(r["run"]))
            continue
        b2 = "{:.1f}".format(100 * r["rb2"]["score"]) if "rb2" in r else "-"
        rmb = "{:.1f} ({:.1f})".format(100 * r["rmbench"]["score"], 100 * r["rmbench"]["hard"]) if "rmbench" in r else "-"
        m = r.get("mrb") or r.get("mrb_small")
        mm = "{:.1f}{}".format(100 * m["score"], "" if "mrb" in r else "*") if m else "-"
        print("| {} | {} | {:.1f} | {:.1f} | {:.1f} ({}) | {:.3f} | {:.1f} / {:.1f} | {} | {} | {} |".format(
            r["run"], r["step"], 100 * r["acc"], 100 * r["agree"], 100 * r["opus"], r["opus_n"], r["ll"],
            100 * r["long"], 100 * r["long_lab"], b2, rmb, mm))
    if args.detail:
        for r in rows:
            print("\n##", r["run"])
            for k, v in r.get("detail", {}).items():
                print("  {:32} ".format(k) + "  ".join("{} {:.3f}".format(a, b) if isinstance(b, float) else "{} {}".format(a, b)
                                                  for a, b in v.items()))
            for bench in ("rb2", "rmbench", "mrb", "mrb_small", "probes"):
                if bench in r:
                    print("  {}: ".format(bench) + "  ".join("{} {:.1f}".format(k, 100 * v) for k, v in r[bench].items()))
            if "lang_reward_mean" in r:
                lm = r["lang_reward_mean"]
                print("  mean reward by language: " + "  ".join("{} {:+.2f}".format(k, lm[k]) for k in sorted(lm, key=lm.get)))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1, default=float)


if __name__ == "__main__":
    main()
