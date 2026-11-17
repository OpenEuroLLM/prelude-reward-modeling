#!/usr/bin/env python3
"""Elo-scale ratings of the responder models from the preference labels (Bradley–Terry, as LMArena computes them).

  collect-preferences/run.sh elo.py [--labels data/preferences/labels.parquet] [--pairs data/preferences/pairs.parquet]
                                    [--out data/preferences/ELO.txt] [--boot 500]

Every labelled pair is a game between model_x and model_y with a soft outcome: label = P(x is better) counts as
`label` wins for x and 1 - label for y, so a pair the judges split counts half a win each. The Bradley–Terry
strengths theta maximise the binomial log-likelihood sum label*log s(d) + (1-label)*log(1-s(d)) over the pairs,
d = theta_x - theta_y (Newton's method; a tiny ridge fixes the free constant). As in LMArena, the rating is
400/ln(10) * theta, shifted so that the mean over the models is 1000: 100 points more means winning about 64% of the
games against that opponent, 200 about 76%, 400 about 91%. Pairs with the same model on both sides carry no
information about the models and are left out of the model fit (they do count in the model x mode fit when the
modes differ). Intervals: 95% percentile bootstrap over the pairs (--boot resamples).
Writes --out with: A. the models; B. model x mode; C. the models per language tier (en / large / mid / small);
D. per prompt category; and the figures data/preferences/plots/elo.{png,pdf} (the models by their Hugging Face id,
rating and interval over all pairs) and elo_tiers.{png,pdf} (the same, one panel per language tier beside it).
"""
import argparse
import collections
import math
import os
import time

import numpy as np
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PREF = os.path.join(ROOT, "data", "preferences")
SCALE = 400 / math.log(10)
TIERS = ("en", "large", "mid", "small")
CATEGORIES = ("general", "instruction_following", "safety_benign", "safety_harmful")
GROUPS = ("prelude", "strong", "mid", "weak")


def cells(a, b, y):
    """aggregate games into cells (a, b, outcome) with counts: the likelihood only depends on these"""
    yq = np.round(y, 4)
    keys = np.stack([a, b, yq], axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    n = np.bincount(inv.ravel(), minlength=len(uniq)).astype(float)
    return uniq[:, 0].astype(int), uniq[:, 1].astype(int), uniq[:, 2], n


def fit(a, b, y, n, players, lam=1e-3):
    """Bradley–Terry strengths (mean 0) by Newton's method on cells with weights n"""
    theta = np.zeros(players)
    idx = np.arange(players)
    for _ in range(200):
        p = 1 / (1 + np.exp(-(theta[a] - theta[b])))
        r = n * (y - p)
        g = np.bincount(a, r, players) - np.bincount(b, r, players) - lam * theta
        h = n * p * (1 - p)
        H = np.zeros((players, players))
        np.add.at(H, (a, b), h)
        np.add.at(H, (b, a), h)
        H[idx, idx] = -(np.bincount(a, h, players) + np.bincount(b, h, players)) - lam
        step = np.linalg.solve(H, g)
        theta -= step
        if np.abs(step).max() < 1e-9:
            break
    return theta - theta.mean()


def rate(a, b, y, players, boot, rng):
    """-> ratings, (low, high) of the 95% interval, games per player; players without a game get nan"""
    ca, cb, cy, n = cells(a, b, y)
    games = np.bincount(a, minlength=players) + np.bincount(b, minlength=players)
    theta = fit(ca, cb, cy, n, players)
    draws = np.empty((boot, players))
    N = n.sum()
    for i in range(boot):
        draws[i] = fit(ca, cb, cy, rng.multinomial(int(N), n / N).astype(float), players)
    r = SCALE * theta + 1000
    lo, hi = SCALE * np.percentile(draws, 2.5, axis=0) + 1000, SCALE * np.percentile(draws, 97.5, axis=0) + 1000
    r[games == 0] = lo[games == 0] = hi[games == 0] = np.nan
    return r, lo, hi, games


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=os.path.join(PREF, "labels.parquet"))
    ap.add_argument("--pairs", default=os.path.join(PREF, "pairs.parquet"))
    ap.add_argument("--out", default=os.path.join(PREF, "ELO.txt"))
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    lab = {r["id"]: r["label"] for r in pq.read_table(args.labels, columns=["id", "label"]).to_pylist() if r["label"] is not None}
    P = pq.read_table(args.pairs, columns=["id", "model_x", "model_y", "mode_x", "mode_y", "group_x", "group_y", "tier",
                                           "category"]).to_pylist()
    P = [p for p in P if p["id"] in lab]
    for p in P:
        p["y"] = lab[p["id"]]
    group = {}
    for p in P:
        group[p["model_x"]], group[p["model_y"]] = p["group_x"], p["group_y"]
    R = pq.read_table(os.path.join(ROOT, "data", "responses", "responses.parquet"), columns=["model", "model_hf"])
    hf_name = dict(set(zip(R["model"].to_pylist(), R["model_hf"].to_pylist())))
    del R
    models = sorted(group, key=lambda m: (GROUPS.index(group[m]), m))
    mi = {m: i for i, m in enumerate(models)}
    lines = []
    out = lines.append

    def table(rows, names, r, lo, hi, games, share=None, extra=None):
        order = sorted(range(len(names)), key=lambda i: -np.nan_to_num(r[i], nan=-1e9))
        w = max(len(x) for x in names)
        out("  rank  {:<{w}}  {:<8}  rating   95% interval     games{}".format("player", "group", "   label share" if share is not None else "", w=w))
        for k, i in enumerate(order, 1):
            if games[i] == 0:
                continue
            s = "  {:>4}  {:<{w}}  {:<8}  {:>6.0f}   [{:>4.0f}, {:>4.0f}]  {:>8,}".format(
                k, names[i], extra[i] if extra else "", r[i], lo[i], hi[i], int(games[i]), w=w)
            if share is not None:
                s += "   {:>6.3f}".format(share[i])
            out(s)

    # A. models
    G = [p for p in P if p["model_x"] != p["model_y"]]
    a = np.array([mi[p["model_x"]] for p in G])
    b = np.array([mi[p["model_y"]] for p in G])
    y = np.array([p["y"] for p in G])
    r, lo, hi, games = rate(a, b, y, len(models), args.boot, rng)
    won = np.bincount(a, y, len(models)) + np.bincount(b, 1 - y, len(models))
    share = won / np.maximum(games, 1)
    clear = y != 0.5
    pred = (r[a] > r[b]) == (y > 0.5)
    out("ELO RATINGS: Bradley-Terry on the labels, 400/ln(10) per unit of log-odds, mean over the models 1000; "
        "{:,} labelled pairs, {:,} between different models; 95% percentile bootstrap over the pairs, {} resamples (seed {})".format(
            len(P), len(G), args.boot, args.seed))
    out("100 points more = about 64% of the games won against that opponent, 200 = 76%, 400 = 91%. The fitted order "
        "has the majority side of {:.1f}% of the {:,} pairs with a majority.".format(100 * pred[clear].mean(), int(clear.sum())))
    out("")
    out("A. MODELS (label share: the share of the label mass the model won in its games)")
    table(None, models, r, lo, hi, games, share=share, extra=[group[m] for m in models])
    overall = {m: r[mi[m]] for m in models}
    out("")

    # B. model x mode
    pm = sorted({(p["model_" + s], p["mode_" + s]) for p in P for s in "xy"}, key=lambda t: (GROUPS.index(group[t[0]]), t))
    pmi = {t: i for i, t in enumerate(pm)}
    names = ["{} {}".format(m, "thinking" if md == "think" else "direct") for m, md in pm]
    G2 = [p for p in P if (p["model_x"], p["mode_x"]) != (p["model_y"], p["mode_y"])]
    a2 = np.array([pmi[(p["model_x"], p["mode_x"])] for p in G2])
    b2 = np.array([pmi[(p["model_y"], p["mode_y"])] for p in G2])
    y2 = np.array([p["y"] for p in G2])
    r2, lo2, hi2, g2 = rate(a2, b2, y2, len(pm), args.boot, rng)
    out("B. MODEL x MODE ({:,} pairs between different players)".format(len(G2)))
    table(None, names, r2, lo2, hi2, g2, extra=[group[m] for m, _ in pm])
    out("")

    # C, D. breakdowns
    panels = collections.OrderedDict([("all", (r, lo, hi, games))])
    for title, field, values in (("C. BY LANGUAGE TIER", "tier", TIERS), ("D. BY CATEGORY", "category", CATEGORIES)):
        res = {}
        for v in values:
            S = [p for p in G if p[field] == v]
            av = np.array([mi[p["model_x"]] for p in S]); bv = np.array([mi[p["model_y"]] for p in S]); yv = np.array([p["y"] for p in S])
            res[v] = rate(av, bv, yv, len(models), args.boot, rng)
            if field == "tier":
                panels[v] = res[v]
        out("{} (rating and half-width of the 95% interval; pairs: {})".format(
            title, ", ".join("{} {:,}".format(v, int(res[v][3].sum() // 2)) for v in values)))
        w = max(len(m) for m in models)
        out("  {:<{w}}  {:<8}  {}".format("model", "group", "  ".join("{:>14}".format(v) for v in values), w=w))
        for m in sorted(models, key=lambda m: -overall[m]):
            cols = []
            for v in values:
                rv, lv, hv, gv = res[v]
                i = mi[m]
                cols.append("{:>14}".format("-") if gv[i] == 0 else "{:>7.0f} +-{:>4.0f}".format(rv[i], (hv[i] - lv[i]) / 2))
            out("  {:<{w}}  {:<8}  {}".format(m, group[m], "  ".join(cols), w=w))
        out("")

    out("{:.0f} s".format(time.time() - t0))
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    figure(models, hf_name, group, overall, panels, os.path.join(PREF, "plots"))


def figure(models, names, group, overall, panels, out_dir):
    """elo: rating with its interval per model (Hugging Face id), all pairs; elo_tiers: one panel per key of panels"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    INK, INK2, GRID = "#1f1f1f", "#5a5a5a", "#e4e4e4"
    PRELUDE, OTHER = "#53bef7", "#e0a970"      # the Prelude checkpoints; the LAB complement (L, -a, -b) for the others
    colour = lambda m: PRELUDE if group[m] == "prelude" else OTHER
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK, "pdf.fonttype": 42})
    order = sorted(models, key=lambda m: overall[m])     # best at the top
    ypos = {m: i for i, m in enumerate(order)}
    os.makedirs(out_dir, exist_ok=True)

    def title(fig, ax, y):
        """'Elo scores on prelude-reward-modeling', the name in monospace, above the first panel"""
        t1 = ax.text(0.0, y, "Elo scores on ", transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, color=INK)
        fig.canvas.draw()
        x1 = ax.transAxes.inverted().transform(t1.get_window_extent())[1, 0]
        ax.text(x1, y, "prelude-reward-modeling", transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5,
                color=INK, family="DejaVu Sans Mono")

    def draw(keys, fname):
        fig, axes = plt.subplots(1, len(keys), figsize=(3.0 * len(keys) + 4.0, 0.33 * len(models) + 0.9), sharey=True,
                                 squeeze=False, gridspec_kw={"wspace": 0.12})
        title(fig, axes[0][0], 1.01 if len(keys) == 1 else 1.065)
        top = max(np.nanmax(panels[k][0]) for k in keys)
        for ax, key in zip(axes[0], keys):
            r, lo, hi, games = panels[key]
            for m in models:
                i = models.index(m)
                if games[i] == 0:
                    continue
                ax.barh(ypos[m], r[i], height=0.68, color=colour(m), edgecolor="none")
                ax.errorbar(r[i], ypos[m], xerr=[[r[i] - lo[i]], [hi[i] - r[i]]], fmt="none", ecolor=INK,
                            elinewidth=0.9, capsize=2.5, capthick=0.9)
                ax.text(hi[i] + 0.012 * top, ypos[m], "{:.0f} ±{:.0f}".format(r[i], (hi[i] - lo[i]) / 2),
                        ha="left", va="center", fontsize=8.5, color=INK)
            ax.set_xlim(0, top * 1.16)
            ax.set_xticks([0, 500, 1000, 1500])
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            ax.tick_params(axis="y", length=0)
            if len(keys) > 1:
                ax.text(0.0, 1.01, key, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5, weight="semibold", color=INK)
            ax.set_xlabel("rating")
        axes[0][0].set_yticks(range(len(order)))
        axes[0][0].set_yticklabels([names[m] for m in order])
        axes[0][-1].legend([Patch(facecolor=PRELUDE), Patch(facecolor=OTHER)], ["Prelude", "other"], loc="lower right",
                           frameon=False, handlelength=1.0, handleheight=1.0, fontsize=8.5)
        for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
            fig.savefig(os.path.join(out_dir, "{}.{}".format(fname, ext)), bbox_inches="tight", pad_inches=0.1, **kw)
        plt.close(fig)

    draw(["all"], "elo")
    draw(list(panels), "elo_tiers")
    print("figures -> {}/elo.{{png,pdf}}, elo_tiers.{{png,pdf}}".format(out_dir))


if __name__ == "__main__":
    main()
