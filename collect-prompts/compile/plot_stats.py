#!/usr/bin/env python3
"""Plots of the released prompt set, from the JSONL so that they show exactly what is published.

  ./run.sh compile/plot_stats.py [--jsonl data/compiled/prompts.jsonl] [--out-dir data/compiled/plots]

Writes <name>.png (200 dpi) and <name>.pdf for each figure, plus tables/<name>.csv with the
numbers behind it (the table view of each chart):
  languages_donut       prompts per language (outer ring), grouped by language family (inner ring)
  language_composition  per language: translation_status, author_type, locale_dependence (shares)
  label_donuts          the low-cardinality fields: translation_status, category, author_type,
                        complexity, safety_label, locale_dependence, user turns, and translation_model /
                        translated_from_language of the machine-translated rows
  task_domain           task_type and subject_domain, ranked
  prompt_length         prompt length in characters (all messages) per language
  sources               rows per source dataset (top 25 and the rest), split by author_type
  allocation            prompts per language vs a split proportional to HPLT 3.0 documents
category and the single/multi-turn mix are not shown per language: the allocation fixes them
(cells are purpose x origin x task x domain x turns), so they are nearly the same everywhere;
the per-language numbers are in tables/language_composition.csv and tables/prompt_length.csv.

Colours: the categorical slots of the dataviz reference palette in fixed order (validated for
colour-vision deficiency on adjacent pairs, including the donut's wrap-around pair); ordered
values (complexity, user turns) use steps of one blue ramp.
"""
import argparse
import collections
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data")

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RAMP = {250: "#86b6ef", 350: "#5598e7", 450: "#2a78d6", 550: "#1c5cab", 650: "#104281"}  # blue
INK, INK2, MUTED, GRID, BASE, OTHER = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#c3c2b7"
SURFACE = "#ffffff"
MONO = "DejaVu Sans Mono"

# Inner ring of the donut, in ring order (clockwise from the top; colour = slot of the same index),
# with the direction of each family's languages (by size; the largest language always comes first).
# Order, directions and DONUT_TURN were chosen by a search over all arrangements for the least
# label displacement: the runs of small languages land at the sides of the ring.
FAMILIES = [
    ("Germanic", ["en", "de", "nl", "sv", "da", "nb", "nn", "is"], "asc"),
    ("Slavic", ["pl", "cs", "uk", "bg", "bs", "sk", "hr", "sl", "sr", "mk"], "asc"),
    ("Romance", ["es", "fr", "it", "pt", "ro", "ca", "gl"], "desc"),
    ("Uralic", ["hu", "fi", "et"], "desc"),
    ("Baltic", ["lt", "lv", "ltg"], "desc"),
    ("Other families", ["tr", "el", "sq", "ka", "eu", "ga", "mt"], "desc"),
]
DONUT_TURN = 12   # degrees counter-clockwise, applied after centring the largest language at the top
OTHER_NOTE = "Turkic, Greek, Albanian, Kartvelian, Basque, Celtic, Semitic"
ORDERS = {  # value order (= colour order) of the categorical fields
    "translation_status": ["native", "machine_translated", "translated_in_source_dataset"],
    "category": ["general", "instruction_following", "safety_benign", "safety_harmful"],
    "author_type": ["real_chatbot_user", "human_written", "llm_generated", "web_scraped"],
    "complexity": ["simple", "moderate", "hard"],
    "safety_label": ["none", "sensitive_benign", "harmful", "jailbreak"],
    "locale_dependence": ["none", "culture_specific", "language_specific"],
    "user_turns": ["1", "2", "3", "4–5", "6+"],
}
COLORS = {f: SLOTS[:len(v)] for f, v in ORDERS.items()}
COLORS["complexity"] = [RAMP[250], RAMP[450], RAMP[650]]
COLORS["user_turns"] = [RAMP[k] for k in (250, 350, 450, 550, 650)]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.titlepad": 10,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
    "xtick.major.pad": 4, "ytick.major.pad": 6,
    "legend.frameon": False, "legend.fontsize": 8.5, "legend.handlelength": 1.0, "legend.handleheight": 1.0,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "pdf.fonttype": 42,
})


def tint(c, f):
    r, g, b = mcolors.to_rgb(c)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def pct(x):
    return "{:.1f}%".format(x) if x >= 1 or x == 0 else "{:.2f}%".format(x)


def turns_bucket(n):
    return str(n) if n <= 3 else "4–5" if n <= 5 else "6+"


def load(path):
    keys = ("language", "translation_status", "translation_model", "translated_from_language", "category",
            "task_type", "subject_domain", "author_type", "safety_label", "locale_dependence", "complexity")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            r = {k: o[k] for k in keys}
            r["dataset"] = o["source"]["dataset"]
            r["chars"] = sum(len(m["content"] or "") for m in o["messages"])
            r["n_user_turns"] = sum(m["role"] == "user" for m in o["messages"])
            r["user_turns"] = turns_bucket(r["n_user_turns"])
            rows.append(r)
    return rows


def language_names():
    with open(os.path.join(HERE, "languages.tsv"), encoding="utf-8") as f:
        lines = [l.rstrip("\n").split("\t") for l in f if l.strip() and not l.startswith("#")]
    names = {r[0]: r[3] for r in lines[1:]}
    names["sr"] = "Serbian"   # mixed Latin / Cyrillic script in the set
    return names


def save(fig, out_dir, name):
    for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
        fig.savefig(os.path.join(out_dir, "{}.{}".format(name, ext)), bbox_inches="tight", pad_inches=0.15, **kw)
    plt.close(fig)


def write_table(out_dir, name, head, rows):
    with open(os.path.join(out_dir, "tables", name + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(head)
        w.writerows(rows)


def spread(targets, gap, lo, hi):
    """Label positions for targets sorted top-down: at least `gap` apart, within [lo, hi], and as close
    to the targets as possible (least squares, via isotonic regression on targets + k * gap)."""
    blocks = []                                   # pool-adjacent-violators, non-increasing
    for k, t in enumerate(targets):
        blocks.append([t + k * gap, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    z = [s / c for s, c in blocks for _ in range(c)]
    ys = [v - k * gap for k, v in enumerate(z)]
    if ys and ys[0] > hi:
        ys[0] = hi
        for k in range(1, len(ys)):
            ys[k] = min(ys[k], ys[k - 1] - gap)
    if ys and ys[-1] < lo:
        ys[-1] = lo
        for k in range(len(ys) - 2, -1, -1):
            ys[k] = max(ys[k], ys[k + 1] + gap)
    return ys


def hbar_axis(ax, labels):
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.4, -0.6)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)


def field_legend(ax, field, order, colors, ncol=2):
    leg = ax.legend([Patch(facecolor=c, edgecolor="none") for c in colors], order, title=field,
                    loc="lower left", bbox_to_anchor=(0, 1.0), ncol=ncol, fontsize=8.5, columnspacing=1.2,
                    handletextpad=0.4, borderaxespad=0.5, alignment="left")
    leg.get_title().set_fontproperties({"family": MONO, "weight": "semibold", "size": 10})
    return leg


# ---------------------------------------------------------------- figures

def fig_languages_donut(rows, names, out_dir):
    cnt = collections.Counter(r["language"] for r in rows)
    fam_of = {l: f for f, ls, _ in FAMILIES for l in ls}
    assert set(cnt) == set(fam_of), set(cnt) ^ set(fam_of)
    total = sum(cnt.values())
    fams = []
    for i, (f, ls, d) in enumerate(FAMILIES):
        ls = sorted(ls, key=lambda l: cnt[l] * (-1 if d == "desc" else 1))
        if i == 0:   # the largest language first, at the top of the ring
            ls = [max(ls, key=cnt.get)] + [l for l in ls if l != max(ls, key=cnt.get)]
        fams.append((f, ls))
    langs = [l for _, ls in fams for l in ls]
    vals = [cnt[l] for l in langs]
    fam_vals = [sum(cnt[l] for l in ls) for _, ls in fams]
    colors = [SLOTS[i] for i, (_, ls) in enumerate(fams) for _ in ls]

    fig = plt.figure(figsize=(12, 8.3))
    ax = fig.add_axes([0, 0, 1, 1])
    start = 90 + 180 * vals[0] / total + DONUT_TURN     # the largest language at the top
    wp = dict(edgecolor=SURFACE, linewidth=1.2)
    wedges, _ = ax.pie(vals, radius=1.0, colors=colors, startangle=start, counterclock=False,
                       wedgeprops=dict(width=0.30, **wp))
    ax.pie(fam_vals, radius=0.665, colors=[tint(SLOTS[i], 0.5) for i in range(len(fams))],
           startangle=start, counterclock=False, wedgeprops=dict(width=0.11, **wp))
    ax.text(0, 0, "{:,}".format(total), ha="center", va="center", fontsize=24, weight="semibold", color=INK)

    def label(x, y, name, figure, ha):
        nt = ax.text(x, y, name, ha=ha, va="center", fontsize=9, color=INK)
        # count and share in secondary ink, on the outer side of the name (anchored to its extent)
        side = 1 if ha == "left" else -1
        ax.annotate(figure, xy=(1 if side > 0 else 0, 0.5), xycoords=nt, xytext=(6 * side, 0),
                    textcoords="offset points", ha=ha, va="center", fontsize=8.5, color=INK2)

    items = []
    for w, l, v in zip(wedges, langs, vals):
        a = np.deg2rad((w.theta1 + w.theta2) / 2)
        items.append((l, v, np.cos(a), np.sin(a)))
    # the top slice is labelled above the ring; the rest in two columns with leader lines,
    # spread so that labels never overlap
    l0, v0 = items[0][0], items[0][1]
    ax.text(0, 1.13, names[l0], ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.text(0, 1.05, "{:,} · {}".format(v0, pct(100 * v0 / total)), ha="center", va="bottom", fontsize=8.5, color=INK2)
    for side in (1, -1):
        its = sorted([i for i in items[1:] if (i[2] >= 0) == (side > 0)], key=lambda i: -i[3])
        ys = spread([1.15 * i[3] for i in its], gap=0.07, lo=-1.45, hi=1.25)
        for (l, v, x, y), ty in zip(its, ys):
            tx = side * 1.24
            ax.plot([1.005 * x, 1.05 * x, tx - side * 0.03], [1.005 * y, 1.05 * y, ty], color=MUTED, lw=0.6,
                    solid_capstyle="round", clip_on=False)
            label(tx, ty, names[l], "{:,} · {}".format(v, pct(100 * v / total)), "left" if side > 0 else "right")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.32, 1.32)
    ax.set_aspect("equal")
    order = sorted(range(len(fams)), key=lambda i: -fam_vals[i])
    handles = [Patch(facecolor=SLOTS[i], edgecolor="none") for i in order]
    labels = ["{} {}".format(fams[i][0], pct(100 * fam_vals[i] / total)) for i in order]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.01), ncol=len(order), fontsize=9,
              columnspacing=1.6, handletextpad=0.5)
    save(fig, out_dir, "languages_donut")
    write_table(out_dir, "languages", ["language", "name", "family", "prompts", "share"],
                [(l, names[l], fam_of[l], cnt[l], round(cnt[l] / total, 6))
                 for l in sorted(langs, key=lambda l: -cnt[l])])


def fig_language_composition(rows, names, out_dir):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["language"]].append(r)
    langs = sorted(by, key=lambda l: -len(by[l]))
    fields = ("translation_status", "author_type", "locale_dependence")
    fig, axes = plt.subplots(1, 3, figsize=(15, 10.5), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, field in zip(axes, fields):
        order, colors = ORDERS[field], COLORS[field]
        shares = np.array([[collections.Counter(r[field] for r in by[l])[v] / len(by[l]) for v in order]
                           for l in langs])
        left = np.zeros(len(langs))
        for j in range(len(order)):
            ax.barh(range(len(langs)), shares[:, j], left=left, height=0.68, color=colors[j],
                    edgecolor=SURFACE, linewidth=0.8)
            left += shares[:, j]
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
        hbar_axis(ax, ["{} ({})".format(names[l], l) for l in langs])
        ax.spines["bottom"].set_visible(True)
        field_legend(ax, field, order, colors)
    for ax in axes[1:]:
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    save(fig, out_dir, "language_composition")
    head = ["language", "prompts"] + ["{}={}".format(f, v) for f in fields + ("category", "user_turns")
                                      for v in ORDERS[f]]
    write_table(out_dir, "language_composition", head,
                [[l, len(by[l])] + [round(collections.Counter(r[f] for r in by[l])[v] / len(by[l]), 4)
                                    for f in fields + ("category", "user_turns") for v in ORDERS[f]]
                 for l in langs])


def donut(ax, counter, order, colors, title, center, shown=None):
    vals = [counter.get(v, 0) for v in order]
    total = sum(vals)
    ax.pie(vals, radius=1.0, colors=colors, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.34, edgecolor=SURFACE, linewidth=1.2))
    if center:
        ax.text(0, 0, center, ha="center", va="center", fontsize=13, weight="semibold", color=INK)
    ax.set_title(title, family=MONO, fontsize=10, loc="center", pad=6)
    ax.set_aspect("equal")
    ax.legend([Patch(facecolor=c, edgecolor="none") for c in colors],
              ["{}  {}".format(s, pct(100 * x / total)) for s, x in zip(shown or order, vals)],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, fontsize=8.5, handletextpad=0.5)
    return list(zip(order, vals))


def fig_label_donuts(rows, out_dir):
    n = len(rows)
    mt = [r for r in rows if r["translation_status"] == "machine_translated"]
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 14), gridspec_kw={"hspace": 0.55, "wspace": 0.35})
    table = []
    for ax, field in zip(axes.flat, ("translation_status", "category", "author_type", "complexity",
                                     "safety_label", "locale_dependence", "user_turns")):
        title = "user turns" if field == "user_turns" else field
        res = donut(ax, collections.Counter(r[field] for r in rows), ORDERS[field], COLORS[field], title, None)
        table += [(field, v, x) for v, x in res]
    mc = collections.Counter(r["translation_model"] for r in mt)
    order = [m for m, _ in mc.most_common()]
    res = donut(axes.flat[7], mc, order, SLOTS[:len(order)], "translation_model",
                "{:,}".format(len(mt)), shown=[m.split("/")[-1] for m in order])
    table += [("translation_model", v, x) for v, x in res]
    sc = collections.Counter(r["translated_from_language"] for r in mt)
    top = [l for l, _ in sc.most_common(4)]
    rest = "other ({} languages)".format(len(sc) - len(top))
    folded = collections.Counter({l: sc[l] for l in top})
    folded[rest] = sum(v for l, v in sc.items() if l not in top)
    donut(axes.flat[8], folded, top + [rest], SLOTS[:len(top)] + [OTHER], "translated_from_language",
          "{:,}".format(len(mt)))
    table += [("translated_from_language", l, v) for l, v in sc.most_common()]
    save(fig, out_dir, "label_donuts")
    write_table(out_dir, "label_donuts", ["field", "value", "prompts"], table)


def ranked_panel(ax, counter, title, n):
    items = counter.most_common()
    vals = np.array([v for _, v in items]) / n * 100
    ax.barh(range(len(items)), vals, height=0.62, color=SLOTS[0])
    hbar_axis(ax, [k for k, _ in items])
    for i, v in enumerate(vals):
        ax.text(v + vals.max() * 0.012, i, pct(v), va="center", ha="left", fontsize=8, color=INK2)
    ax.set_xlim(0, vals.max() * 1.13)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=8.5)
    for t in ax.get_yticklabels():
        t.set_family(MONO)
    ax.set_title(title, family=MONO, fontsize=10)
    return items


def fig_task_domain(rows, out_dir):
    n = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8), gridspec_kw={"wspace": 0.5})
    t = ranked_panel(axes[0], collections.Counter(r["task_type"] for r in rows), "task_type", n)
    d = ranked_panel(axes[1], collections.Counter(r["subject_domain"] for r in rows), "subject_domain", n)
    save(fig, out_dir, "task_domain")
    write_table(out_dir, "task_domain", ["field", "value", "prompts"],
                [("task_type", k, v) for k, v in t] + [("subject_domain", k, v) for k, v in d])


def fig_prompt_length(rows, names, out_dir):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["language"]].append(r)
    langs = sorted(by, key=lambda l: -len(by[l]))
    stats, table = [], []
    for l in langs:
        c = np.array([r["chars"] for r in by[l]])
        q = np.percentile(c, [5, 25, 50, 75, 95])
        stats.append(dict(whislo=q[0], q1=q[1], med=q[2], q3=q[3], whishi=q[4], fliers=[]))
        table.append([l, len(c)] + [int(x) for x in q] + [round(np.mean([r["n_user_turns"] > 1 for r in by[l]]), 4)])
    fig, ax = plt.subplots(figsize=(10, 10.5))
    ax.bxp(stats, positions=range(len(langs)), vert=False, widths=0.6, patch_artist=True,
           boxprops=dict(facecolor=tint(SLOTS[0], 0.55), edgecolor=SLOTS[0], linewidth=0.8),
           medianprops=dict(color=INK, linewidth=1.4), whiskerprops=dict(color=MUTED, linewidth=0.8),
           capprops=dict(color=MUTED, linewidth=0.8), showcaps=True)
    ax.set_xscale("log")
    ax.set_xlim(8, 30000)
    ax.xaxis.set_major_locator(mticker.FixedLocator([10, 30, 100, 300, 1000, 3000, 10000, 30000]))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(["10", "30", "100", "300", "1,000", "3,000", "10K", "30K"]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    hbar_axis(ax, ["{} ({})".format(names[l], l) for l in langs])
    ax.set_xlabel("characters")
    save(fig, out_dir, "prompt_length")
    write_table(out_dir, "prompt_length", ["language", "prompts", "chars_p5", "chars_p25", "chars_median",
                                           "chars_p75", "chars_p95", "multi_turn_share"], table)


def fig_sources(rows, out_dir, top=25):
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["dataset"]][r["author_type"]] += 1
    ds = sorted(by, key=lambda d: -sum(by[d].values()))
    shown, rest = ds[:top], ds[top:]
    order = ORDERS["author_type"]
    fig, ax = plt.subplots(figsize=(10, 8.6))
    left = np.zeros(len(shown))
    for j, a in enumerate(order):
        v = np.array([by[d][a] for d in shown])
        ax.barh(range(len(shown)), v, left=left, height=0.64, color=SLOTS[j], edgecolor=SURFACE, linewidth=0.8)
        left += v
    restv = sum(sum(by[d].values()) for d in rest)
    ax.barh([len(shown)], [restv], height=0.64, color=OTHER)
    tot = list(left) + [restv]
    for i, v in enumerate(tot):
        ax.text(v + max(tot) * 0.01, i, "{:,}".format(int(v)), va="center", fontsize=8, color=INK2)
    hbar_axis(ax, shown + ["{} other datasets".format(len(rest))])
    ax.set_xlim(0, max(tot) * 1.1)
    ax.set_xticks([])
    field_legend(ax, "author_type", order, SLOTS[:len(order)], ncol=4)
    save(fig, out_dir, "sources")
    write_table(out_dir, "sources", ["dataset", "prompts"] + order,
                [[d, sum(by[d].values())] + [by[d][a] for a in order] for d in ds])


def fig_allocation(rows, names, out_dir):
    cnt = collections.Counter(r["language"] for r in rows)
    with open(os.path.join(HERE, "quotas.tsv"), encoding="utf-8") as f:
        lines = [l.rstrip("\n").split("\t") for l in f if l.strip() and not l.startswith("#")]
    freq = {d["code"]: float(d["frequency"]) for d in (dict(zip(lines[0], r)) for r in lines[1:])}
    assert set(freq) == set(cnt)
    n, z = sum(cnt.values()), sum(freq.values())
    prop = {l: n * freq[l] / z for l in freq}
    langs = sorted(freq, key=lambda l: -freq[l])
    fig, ax = plt.subplots(figsize=(9.5, 10))
    y = np.arange(len(langs))
    for i, l in enumerate(langs):
        ax.plot([prop[l], cnt[l]], [i, i], color=BASE, lw=1.4, solid_capstyle="round", zorder=1)
    ax.scatter([prop[l] for l in langs], y, s=34, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.2, zorder=2)
    ax.scatter([cnt[l] for l in langs], y, s=40, color=SLOTS[0], edgecolor=SURFACE, linewidth=1.0, zorder=3)
    for l in ("en", "ltg"):     # the two ends, labelled on the outer side of each marker
        i, a, p = langs.index(l), cnt[l], prop[l]
        s = 1 if a > p else -1
        ax.annotate("{:,}".format(a), (a, i), xytext=(7 * s, 0), textcoords="offset points",
                    ha="left" if s > 0 else "right", va="center", fontsize=8, color=INK2)
        ax.annotate("{:,.0f}".format(p) if p >= 1 else "{:.2f}".format(p), (p, i), xytext=(-7 * s, 0),
                    textcoords="offset points", ha="right" if s > 0 else "left", va="center", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 3e5)
    ax.xaxis.set_major_locator(mticker.FixedLocator([0.1, 1, 10, 100, 1000, 10000, 100000]))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(["0.1", "1", "10", "100", "1,000", "10K", "100K"]))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    hbar_axis(ax, ["{} ({})".format(names[l], l) for l in langs])
    ax.set_xlabel("prompts")
    ax.legend([Line2D([], [], marker="o", ls="", markersize=6.5, markerfacecolor=SLOTS[0], markeredgecolor=SURFACE),
               Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor=SURFACE, markeredgecolor=MUTED)],
              ["∝ √HPLT (this set)", "∝ HPLT"],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5, borderaxespad=0.5)
    save(fig, out_dir, "allocation")
    write_table(out_dir, "allocation", ["language", "hplt_documents", "prompts", "prompts_if_proportional"],
                [(l, int(freq[l]), cnt[l], round(prop[l], 2)) for l in langs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default=os.path.join(DATA, "compiled", "prompts.jsonl"))
    ap.add_argument("--out-dir", default=os.path.join(DATA, "compiled", "plots"))
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)
    rows = load(args.jsonl)
    names = language_names()
    print("{} rows".format(len(rows)))
    fig_languages_donut(rows, names, args.out_dir)
    fig_language_composition(rows, names, args.out_dir)
    fig_label_donuts(rows, args.out_dir)
    fig_task_domain(rows, args.out_dir)
    fig_prompt_length(rows, names, args.out_dir)
    fig_sources(rows, args.out_dir)
    fig_allocation(rows, names, args.out_dir)
    print("figures ->", args.out_dir)


if __name__ == "__main__":
    main()
