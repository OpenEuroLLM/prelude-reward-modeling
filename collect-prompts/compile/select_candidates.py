#!/usr/bin/env python3
"""Stage 2: choose the candidates that will be annotated (and later allocated).

Reads data/pool/index/**, writes data/pool/candidates.parquet (index rows +
`role`: native | source_translated | donor) and candidates_report.json.

Steps
  1. eligibility: not excluded, train split, licence allowed, target language,
     length bounds, policy-excluded subsets dropped; LID-only language decisions
     need en: >= 4 chars & p >= 0.5; large langs: (>= 20 chars & p >= 0.7) or
     (>= 8 chars & p >= 0.85); others: >= 20 chars & p >= 0.9; and are
     rejected when the response is confidently English; rejected records are
     rescued when the source label and a confident response LID agree
     (code/markup/short strings are otherwise mislabelled e.g. nn, ltg, mt)
  2. exact dedup on key_all, keeping the richest copy
     (preference > >=2 responses > 1 response > none; real_user > human >
     synthetic > web > template; not translated; source order in pool_rules)
  3. per-user cap (user_key)
  4. per language l and purpose p (need = quota_l * share_p): sample up to
     NATIVE_FACTOR * need native candidates and up to SRC_TR_FACTOR * need
     source-translated ones; sources share the budget ~ sqrt(availability)
  5. donors for translation: per purpose (and origin, for general prompts),
     DONOR_FACTOR * expected shortfall, 75% English, 25% other rich languages;
     never translated-by-source, never already a native candidate

Memory-heavy (whole index): run via slurm/select_candidates.sbatch.
"""
import argparse
import collections
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import pool_rules as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
INDEX = os.path.join(ROOT, "data", "pool", "index")
OUT = os.path.join(ROOT, "data", "pool")

SEED = 1234
PURPOSE_SHARE = {"general": 0.80, "if": 0.10, "safety": 0.10}
GENERAL_ORIGIN_SHARE = {"real_user": 0.60, "human": 0.15, "synthetic": 0.25}
NATIVE_FACTOR = 5
SRC_TR_FACTOR = 2
DONOR_FACTOR = 3
USABLE_RATE = 0.6            # expected share of candidates surviving annotation filters
PER_USER_CAP = 50
DONOR_LANGS_SECONDARY = ["fr", "es", "de", "it", "pt", "pl", "nl"]
LICENSE_EXCLUDED = {"no_redistribution"}
EXCLUDE_SUBSETS = {("ToxicityPrompts/PolyGuardMix", "wildguardmix_translated_nllb")}
ORIGIN_RANK = {"real_user": 0, "human": 1, "synthetic": 2, "web": 3, "template": 4}
LARGE_LANGS = {"en", "es", "de", "fr", "it", "pt", "pl", "nl", "tr", "cs", "sv", "ro",
               "el", "uk", "hu", "da", "fi", "bg", "ca"}
COLS = ["uid", "source", "subset", "split", "part", "row", "origin", "translated",
        "purpose", "license_flag", "exclude", "lang", "lang_rule", "lid_last_p",
        "lid_all_p", "lid_chars_last", "resp_lang_p", "lang_src",
        "n_user_turns", "last_user_chars", "user_chars", "n_resp", "n_resp_reusable",
        "resp_lang", "pref_type", "prompt_harm", "adversarial", "if_verifiable",
        "user_key", "key_all"]


def quotas():
    q = {}
    with open(os.path.join(HERE, "quotas.tsv"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("code\t") or not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            q[p[0]] = int(p[5])
    return q


TARGETS = set()


def load(targets):
    d = ds.dataset(INDEX, format="parquet")
    f = (ds.field("exclude").is_null() & (ds.field("split") == "train")
         & ds.field("lang").isin(sorted(targets))
         & (ds.field("last_user_chars") >= 10) & (ds.field("last_user_chars") <= 8000)
         & (ds.field("user_chars") <= 24000))
    t = d.to_table(columns=COLS, filter=f)
    df = t.to_pandas()
    for c in ("source", "subset", "split", "origin", "purpose", "license_flag",
              "resp_lang", "pref_type", "prompt_harm", "part"):
        df[c] = df[c].astype("category")
    return df


def refine_lang(df):
    """Stricter acceptance of LID-only language decisions (see module doc)."""
    rule = df["lang_rule"].astype(str)
    lang = df["lang"].astype(str)
    trusted = rule.str.contains(r"\+source|source_over_lid|source_only", regex=True)
    p = np.where(rule.str.startswith("lid_last"), df["lid_last_p"], df["lid_all_p"])
    n = df["lid_chars_last"]
    large = lang.isin(LARGE_LANGS)
    ok_en = (lang == "en") & (p >= 0.5) & (n >= 4)
    ok_large = large & (lang != "en") & (((p >= 0.7) & (n >= 20)) | ((p >= 0.85) & (n >= 8)))
    ok_other = ~large & (p >= 0.9) & (n >= 20)
    ok = (ok_en | ok_large | ok_other) & ~rule.str.contains("weak")
    resp_en = ((df["resp_lang"].astype(str) == "en") & (df["resp_lang_p"].fillna(0) >= 0.9)
               & (lang != "en") & (p < 0.98))
    keep = trusted | (ok & ~resp_en)
    # rescue: source label and a confident response LID agree on a target language
    # (recovers e.g. unfenced code dumps with an English request, LID-labelled 'nn')
    src = df["lang_src"].astype(str)
    rescue = (~keep & (src == df["resp_lang"].astype(str))
              & (df["resp_lang_p"].fillna(0) >= 0.9) & src.isin(list(TARGETS)))
    df.loc[rescue, "lang"] = df.loc[rescue, "lang_src"]
    df.loc[rescue, "lang_rule"] = "rescued_src+resp"
    return keep | rescue


def sqrt_allocate(avail, budget):
    """Split `budget` over groups ~ sqrt(avail), capped at avail (water-filling)."""
    alloc = {k: 0 for k in avail}
    left = budget
    active = {k for k, v in avail.items() if v > 0}
    while left > 0 and active:
        w = {k: math.sqrt(avail[k]) for k in active}
        z = sum(w.values())
        given = 0
        for k in list(active):
            share = int(math.floor(left * w[k] / z)) or (1 if left - given > 0 else 0)
            take = min(share, avail[k] - alloc[k], left - given)
            alloc[k] += take
            given += take
            if alloc[k] >= avail[k]:
                active.discard(k)
        if given == 0:
            break
        left -= given
    return alloc


def sample_by_source(sub, budget, rng):
    if budget <= 0 or sub.empty:
        return sub.iloc[0:0]
    avail = sub.groupby("source", observed=True).size().to_dict()
    alloc = sqrt_allocate(avail, min(budget, len(sub)))
    parts = []
    for src, n in alloc.items():
        if n > 0:
            g = sub[sub["source"] == src]
            parts.append(g.sample(n=n, random_state=rng.integers(1 << 31)))
    return pd.concat(parts) if parts else sub.iloc[0:0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    q = quotas()
    rep = {"quotas": q}

    TARGETS.update(q)
    df = load(set(q))
    rep["eligible_before_policy"] = len(df)
    lic_bad = df["license_flag"].isin(list(LICENSE_EXCLUDED))
    sub_bad = pd.Series([(s, b) in EXCLUDE_SUBSETS for s, b in zip(df["source"], df["subset"])],
                        index=df.index)
    df = df[~lic_bad & ~sub_bad].copy()
    keep = refine_lang(df)
    rep["lang_rejected"] = df[~keep].groupby("lang", observed=True).size().to_dict()
    df = df[keep].copy()
    rep["rescued"] = df[df["lang_rule"] == "rescued_src+resp"].groupby("lang").size().to_dict()
    df["lang"] = df["lang"].astype("category")
    rep["eligible"] = len(df)
    print("eligible rows:", len(df), flush=True)

    # ---- 2. exact dedup, richest copy first
    src_rank = {s: i for i, s in enumerate(R.SOURCES)}
    df["_rich"] = (df["pref_type"].notna().astype(int) * 4
                   + (df["n_resp"] >= 2).astype(int) * 2 + (df["n_resp"] >= 1).astype(int))
    df["_orank"] = df["origin"].astype(str).map(ORIGIN_RANK).astype(int)
    df["_srank"] = df["source"].astype(str).map(src_rank).astype(int)
    df = df.sort_values(["key_all", "_rich", "_orank", "translated", "_srank"],
                        ascending=[True, False, True, True, True], kind="mergesort")
    dup_counts = df.groupby("key_all", observed=True).size()
    df = df.drop_duplicates("key_all", keep="first")
    df["n_copies"] = df["key_all"].map(dup_counts).astype(int)
    rep["after_dedup"] = len(df)
    print("after exact dedup:", len(df), flush=True)

    # ---- 3. per-user cap
    has_user = df["user_key"].notna()
    shuffled = df[has_user].sample(frac=1.0, random_state=SEED)
    capped = shuffled.groupby("user_key", observed=True).head(PER_USER_CAP)
    df = pd.concat([df[~has_user], capped])
    rep["after_user_cap"] = len(df)
    print("after per-user cap:", len(df), flush=True)

    # ---- 4. native / source-translated candidates per (lang, purpose)
    chosen, avail_rep = [], {}
    native_avail = collections.Counter()
    for lang, ql in q.items():
        for p, share in PURPOSE_SHARE.items():
            need = ql * share
            g = df[(df["lang"] == lang) & (df["purpose"] == p)]
            nat = g[~g["translated"]]
            tr = g[g["translated"]]
            native_avail[(lang, p)] = len(nat)
            if p == "general":
                for o in GENERAL_ORIGIN_SHARE:
                    native_avail[(lang, p, o)] = int((nat["origin"] == o).sum())
            sn = sample_by_source(nat, int(math.ceil(NATIVE_FACTOR * need)), rng)
            st = sample_by_source(tr, int(math.ceil(SRC_TR_FACTOR * need)), rng)
            sn = sn.assign(role="native")
            st = st.assign(role="source_translated")
            chosen += [sn, st]
            avail_rep["{}/{}".format(lang, p)] = {
                "need": round(need, 1), "native_avail": len(nat), "native_sampled": len(sn),
                "src_translated_avail": len(tr), "src_translated_sampled": len(st)}
    cand = pd.concat(chosen)
    rep["per_lang_purpose"] = avail_rep

    # ---- 5. donors
    demand = collections.Counter()
    for lang, ql in q.items():
        if lang == "en":
            continue
        for p, share in PURPOSE_SHARE.items():
            if p == "general":
                for o, os_ in GENERAL_ORIGIN_SHARE.items():
                    need = ql * share * os_
                    demand[(p, o)] += max(0.0, need - USABLE_RATE * native_avail[(lang, p, o)])
            else:
                need = ql * share
                demand[(p, None)] += max(0.0, need - USABLE_RATE * native_avail[(lang, p)])
    taken = set(cand["uid"])
    pool = df[~df["translated"] & ~df["uid"].isin(taken)]
    donors = []
    for (p, o), dem in demand.items():
        n = int(math.ceil(DONOR_FACTOR * dem / USABLE_RATE))
        g = pool[pool["purpose"] == p]
        if o is not None:
            g = g[g["origin"] == o]
        en = g[g["lang"] == "en"]
        sec = g[g["lang"].isin(DONOR_LANGS_SECONDARY)]
        d_en = sample_by_source(en, int(0.75 * n), rng)
        d_sec = sample_by_source(sec, n - len(d_en), rng)
        donors += [d_en.assign(role="donor"), d_sec.assign(role="donor")]
        rep.setdefault("donors", {})["{}/{}".format(p, o)] = {
            "demand": round(dem, 1), "target": n, "en": len(d_en), "secondary": len(d_sec)}
    cand = pd.concat([cand] + donors)
    cand = cand.drop(columns=["_rich", "_orank", "_srank"])
    rep["candidates"] = int(len(cand))
    rep["by_role"] = cand["role"].value_counts().to_dict()
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "candidates.parquet")
    cand.to_parquet(out, index=False)
    with open(os.path.join(args.out, "candidates_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("eligible", "after_dedup", "after_user_cap",
                                          "candidates", "by_role")}, default=str))
    print("donors:", json.dumps(rep.get("donors"), default=str))
    print("->", out)


if __name__ == "__main__":
    main()
