#!/usr/bin/env python3
"""Plots of the preference dataset (preferences.jsonl), in the style of collect-prompts/compile/plot_stats.py.

  collect-preferences/run.sh plot_dataset.py [--jsonl data/preferences/preferences.jsonl] [--out-dir data/preferences/plots]

Writes <name>.png (200 dpi) and <name>.pdf for each figure, plus tables/<name>.csv with the numbers behind it:
  label_distribution   prob_a_better over the rows, in bins of one sixth (the resolution of six votes)
  agreement            rows labelled by votes, per language and per task_type: unanimous (0 or 1), majority, tie
  model_win_rates      expected win rate of each responder model's answers (the label mass they receive), with
                       the thinking / direct split for the models that have both modes
  length               answer length per model, and the label mass of the longer answer by length-ratio band,
                       for all pairs and for clean pairs (both answers >= 200 characters, neither cut at the
                       token budget, neither flagged in another language than the prompt's: length_bias.py)
The prompt-distribution figures of the same rows come from plot_stats.py --jsonl data/preferences/preferences.jsonl.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "collect-prompts", "compile"))
import plot_stats as PS  # noqa: E402  (palette, rcParams, helpers)
from judge_prompt import same_lang  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

DATA = os.path.join(ROOT, "data")
GROUPS = ["prelude", "strong", "mid", "weak"]
GROUP_COLOR = dict(zip(GROUPS, PS.SLOTS[:4]))
AGREE = ["unanimous", "majority", "tie"]
AGREE_COLOR = [PS.RAMP[650], PS.RAMP[450], PS.RAMP[250]]
BANDS = [(1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 4.0), (4.0, float("inf"))]
BAND_NAMES = ["1.0–1.2", "1.2–1.5", "1.5–2", "2–4", "> 4"]


def models_yaml():
    with open(os.path.join(ROOT, "collect-responses", "models.yaml"), encoding="utf-8") as f:
        ms = yaml.safe_load(f)["models"]
    return {m["hf"]: (m["name"], m["group"]) for m in ms}


def load(path, names):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            p = o["preference"]
            r = {"language": o["language"], "task_type": o["task_type"], "locale_dependence": o["locale_dependence"],
                 "prob": p["prob_a_better"], "rule": p["rule"], "n_votes": len(p["votes"])}
            for s in ("a", "b"):
                x = o["response_" + s]
                r["model_" + s], r["group_" + s] = names[x["model"]]
                r["mode_" + s] = x["mode"]
                r["chars_" + s] = len(x["content"])
                r["cut_" + s] = x["finish_reason"] == "length"
                r["flag_" + s] = flagged(o, x)
            rows.append(r)
    return rows


def flagged(o, x):
    """the answer is in another language than the prompt (OpenLID, as in length_bias.py)"""
    if o["task_type"] == "translation" or o["locale_dependence"] == "language_specific":
        return False
    lid, conf = x["detected_language"], x["detected_language_confidence"]
    return bool(lid and conf >= 0.5 and len(x["content"].strip()) >= 20 and not same_lang(lid, o["language"]))


def agreement(prob):
    return "unanimous" if prob in (0.0, 1.0) else "tie" if prob == 0.5 else "majority"


# ---------------------------------------------------------------- figures

def fig_label_distribution(rows, out_dir):
    probs = np.array([r["prob"] for r in rows])
    edges = np.array([1, 3, 5, 7, 9, 11]) / 12
    bins = np.digitize(probs, edges)
    counts = np.bincount(bins, minlength=7)
    share = 100 * counts / len(rows)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(range(7), share, width=0.72, color=PS.SLOTS[0])
    for i, (c, s) in enumerate(zip(counts, share)):
        ax.text(i, s + share.max() * 0.015, PS.pct(s), ha="center", va="bottom", fontsize=8.5, color=PS.INK2)
    ax.set_xticks(range(7))
    ax.set_xticklabels(["0", "1/6", "1/3", "1/2", "2/3", "5/6", "1"])
    ax.set_xlabel("prob_a_better", family=PS.MONO)
    ax.set_ylim(0, share.max() * 1.12)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    PS.save(fig, out_dir, "label_distribution")
    PS.write_table(out_dir, "label_distribution", ["bin_centre", "bin_lo", "bin_hi", "rows", "share"],
                   [(round(k / 6, 4), round(edges[k - 1], 4) if k else 0.0, round(edges[k], 4) if k < 6 else 1.0,
                     int(counts[k]), round(counts[k] / len(rows), 6)) for k in range(7)])


def stacked_shares(ax, keys, shares, labels, colors, legend_field, legend_ncol):
    left = np.zeros(len(keys))
    for j in range(shares.shape[1]):
        ax.barh(range(len(keys)), shares[:, j], left=left, height=0.68, color=colors[j],
                edgecolor=PS.SURFACE, linewidth=0.8)
        left += shares[:, j]
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    PS.hbar_axis(ax, labels)
    ax.spines["bottom"].set_visible(True)
    PS.field_legend(ax, legend_field, AGREE, colors, ncol=legend_ncol)


def fig_agreement(rows, names, out_dir):
    voted = [r for r in rows if r["rule"] is None]
    table = []
    fig = plt.figure(figsize=(15, 10.5))
    panels = (("language", fig.add_axes([0.10, 0.05, 0.36, 0.86]), lambda l: "{} ({})".format(names[l], l)),
              ("task_type", None, lambda t: t))
    for field, ax, label in panels:
        by = collections.defaultdict(list)
        for r in voted:
            by[r[field]].append(agreement(r["prob"]))
        keys = sorted(by, key=lambda k: -len(by[k]))
        if ax is None:   # the shorter panel, aligned at the top with the same bar pitch
            h = 0.86 * len(keys) / len(panels[0][1].get_yticks())
            ax = fig.add_axes([0.63, 0.05 + 0.86 - h, 0.33, h])
        shares = np.array([[collections.Counter(by[k])[v] / len(by[k]) for v in AGREE] for k in keys])
        stacked_shares(ax, keys, shares, [label(k) for k in keys], AGREE_COLOR, "prob_a_better", 3)
        if field == "task_type":
            for t in ax.get_yticklabels():
                t.set_family(PS.MONO)
        table += [(field, k, len(by[k])) + tuple(round(x, 4) for x in shares[i]) for i, k in enumerate(keys)]
    PS.save(fig, out_dir, "agreement")
    PS.write_table(out_dir, "agreement", ["field", "value", "rows_labelled_by_votes"] + AGREE, table)


def fig_model_win_rates(rows, out_dir):
    wins = collections.defaultdict(list)          # model -> win probabilities of its answers
    by_mode = collections.defaultdict(list)       # (model, mode) -> the same
    group = {}
    for r in rows:
        for s, w in (("a", r["prob"]), ("b", 1 - r["prob"])):
            m = r["model_" + s]
            wins[m].append(w)
            by_mode[(m, r["mode_" + s])].append(w)
            group[m] = r["group_" + s]
    models = sorted(wins, key=lambda m: -np.mean(wins[m]))
    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    y = np.arange(len(models))
    ax.axvline(0.5, color=PS.GRID, lw=0.8, zorder=0)
    table = []
    for i, m in enumerate(models):
        w = np.array(wins[m])
        mean, ci = w.mean(), 1.96 * w.std() / np.sqrt(len(w))
        c = GROUP_COLOR[group[m]]
        modes = {md: np.mean(by_mode[(m, md)]) for md in ("direct", "thinking") if (m, md) in by_mode}
        if len(modes) == 2:
            ax.plot([modes["direct"], modes["thinking"]], [i, i], color=PS.BASE, lw=1.2, zorder=1)
            ax.scatter([modes["direct"]], [i], s=30, marker="o", facecolor=PS.SURFACE, edgecolor=c, linewidth=1.1, zorder=2)
            ax.scatter([modes["thinking"]], [i], s=36, marker="^", facecolor=PS.SURFACE, edgecolor=c, linewidth=1.1, zorder=2)
        ax.plot([mean - ci, mean + ci], [i, i], color=c, lw=2.2, solid_capstyle="butt", zorder=3)
        ax.scatter([mean], [i], s=46, color=c, edgecolor=PS.SURFACE, linewidth=1.0, zorder=4)
        ax.annotate(PS.pct(100 * mean), (mean, i), xytext=(0, 7), textcoords="offset points", ha="center",
                    va="bottom", fontsize=7.5, color=PS.INK2)
        table.append([m, group[m], len(w), round(mean, 4), round(ci, 4)]
                     + [round(modes.get(md, float("nan")), 4) if md in modes else "" for md in ("direct", "thinking")]
                     + [len(by_mode.get((m, md), [])) for md in ("direct", "thinking")])
    PS.hbar_axis(ax, models)
    for t in ax.get_yticklabels():
        t.set_family(PS.MONO)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=PS.GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(True)
    ax.set_xlabel("win rate")
    handles = [Patch(facecolor=GROUP_COLOR[g], edgecolor="none") for g in GROUPS]
    handles += [Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor=PS.SURFACE, markeredgecolor=PS.INK2),
                Line2D([], [], marker="^", ls="", markersize=6.5, markerfacecolor=PS.SURFACE, markeredgecolor=PS.INK2)]
    ax.legend(handles, GROUPS + ["direct", "thinking"], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=6,
              fontsize=8.5, columnspacing=1.2, handletextpad=0.5, borderaxespad=0.5)
    PS.save(fig, out_dir, "model_win_rates")
    PS.write_table(out_dir, "model_win_rates", ["model", "group", "answers", "win_rate", "ci95_halfwidth",
                                                  "win_rate_direct", "win_rate_thinking", "answers_direct", "answers_thinking"],
                   table)


def clean(r):
    return (min(r["chars_a"], r["chars_b"]) >= 200 and not r["cut_a"] and not r["cut_b"]
            and not r["flag_a"] and not r["flag_b"])


def fig_length(rows, out_dir):
    # answer length per model
    chars = collections.defaultdict(list)
    group = {}
    for r in rows:
        for s in ("a", "b"):
            chars[r["model_" + s]].append(r["chars_" + s])
            group[r["model_" + s]] = r["group_" + s]
    models = sorted(chars, key=lambda m: -np.median(chars[m]))
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_axes([0.17, 0.08, 0.42, 0.88])
    stats, table = [], []
    for m in models:
        c = np.array(chars[m])
        q = np.percentile(c, [5, 25, 50, 75, 95])
        stats.append(dict(whislo=q[0], q1=q[1], med=q[2], q3=q[3], whishi=q[4], fliers=[]))
        table.append(["length", m, len(c)] + [int(x) for x in q] + [round(np.mean(c == 0), 4)])
    bp = ax.bxp(stats, positions=range(len(models)), vert=False, widths=0.6, patch_artist=True,
                medianprops=dict(color=PS.INK, linewidth=1.4), whiskerprops=dict(color=PS.MUTED, linewidth=0.8),
                capprops=dict(color=PS.MUTED, linewidth=0.8), showcaps=True)
    for box, m in zip(bp["boxes"], models):
        box.set(facecolor=PS.tint(GROUP_COLOR[group[m]], 0.55), edgecolor=GROUP_COLOR[group[m]], linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(8, 100000)
    ax.xaxis.set_major_locator(mticker.FixedLocator([10, 100, 1000, 10000, 100000]))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(["10", "100", "1,000", "10K", "100K"]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(axis="x", color=PS.GRID, lw=0.6)
    ax.set_axisbelow(True)
    PS.hbar_axis(ax, models)
    for t in ax.get_yticklabels():
        t.set_family(PS.MONO)
    ax.spines["bottom"].set_visible(True)
    ax.set_xlabel("characters")
    ax.legend([Patch(facecolor=PS.tint(GROUP_COLOR[g], 0.55), edgecolor=GROUP_COLOR[g]) for g in GROUPS], GROUPS,
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8.5, columnspacing=1.2, handletextpad=0.5,
              borderaxespad=0.5)

    # label mass of the longer answer, by length-ratio band
    ax2 = fig.add_axes([0.70, 0.40, 0.28, 0.50])
    series = (("all pairs", lambda r: True), ("clean pairs", clean))
    width = 0.36
    for j, (name, keep) in enumerate(series):
        vals, ns = [], []
        for a, b in BANDS:
            sel = [r for r in rows if keep(r) and a <= max(r["chars_a"], r["chars_b"]) / max(1, min(r["chars_a"], r["chars_b"])) < b]
            longer = [r["prob"] if r["chars_a"] > r["chars_b"] else 1 - r["prob"] for r in sel if r["chars_a"] != r["chars_b"]]
            vals.append(100 * np.mean(longer) if longer else float("nan"))
            ns.append(len(longer))
        x = np.arange(len(BANDS)) + (j - 0.5) * width
        ax2.bar(x, vals, width=width - 0.03, color=PS.SLOTS[j])
        for xi, v in zip(x, vals):
            ax2.text(xi, v + 0.6, "{:.0f}".format(v), ha="center", va="bottom", fontsize=7.5, color=PS.INK2)
        table += [("longer_wins", name, BAND_NAMES[k], ns[k], round(vals[k] / 100, 4)) for k in range(len(BANDS))]
    ax2.axhline(50, color=PS.MUTED, lw=0.8, ls=(0, (3, 3)))
    ax2.set_xticks(range(len(BANDS)))
    ax2.set_xticklabels(BAND_NAMES)
    ax2.set_xlabel("length ratio")
    ax2.set_ylim(0, 60)
    ax2.set_yticks([0, 25, 50])
    ax2.set_yticklabels(["0", "25%", "50%"])
    ax2.set_ylabel("label mass of the longer answer")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.legend([Patch(facecolor=PS.SLOTS[j], edgecolor="none") for j in range(2)], [s[0] for s in series],
               loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5, handletextpad=0.5, borderaxespad=0.5)
    PS.save(fig, out_dir, "length")
    PS.write_table(out_dir, "length", ["table", "model_or_series", "answers_or_band", "n_or_p5", "p25_or_share",
                                        "median", "p75", "p95", "empty_share"], table)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=os.path.join(DATA, "preferences", "preferences.jsonl"))
    ap.add_argument("--out-dir", default=os.path.join(DATA, "preferences", "plots"))
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)
    rows = load(args.jsonl, models_yaml())
    names = PS.language_names()
    print("{} rows".format(len(rows)), flush=True)
    fig_label_distribution(rows, args.out_dir)
    fig_agreement(rows, names, args.out_dir)
    fig_model_win_rates(rows, args.out_dir)
    fig_length(rows, args.out_dir)
    print("figures ->", args.out_dir)


if __name__ == "__main__":
    main()
