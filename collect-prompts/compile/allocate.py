#!/usr/bin/env python3
"""Stage 4: allocate 128k slots (language x content cell) and fill them.

Inputs
  data/pool/candidates.parquet            candidate index rows (+ role)
  data/llm/annotate/out/*.parquet          LLM annotations (id = uid)
  data/pool/neighbors/*.parquet            kNN (any language) + hold-out max-sim
  data/pool/clusters.parquet               k-means cluster per candidate
  data/holdout/prompts.parquet             hold-out exact-match keys
  data/pool/if_meta.parquet                IF constraint ids / transferability
  compile/quotas.tsv
Outputs (data/compiled/)
  allocation.parquet   one row per slot: lang, cell, chosen uid, fill type
  allocation_report.json

Design (see PLAN.md):
  * content cell = (purpose, origin, task, domain, turns); purpose in
    general | if | safety_harmful | safety_benign; origin only for general
  * global plan: purpose/origin shares fixed, cell shares within them
    proportional to N(cell)^ALPHA over all eligible candidates
  * languages are assigned to plan slots by a random permutation of the quota
    multiset, so language is independent of content by construction
  * filling passes (--order native-first, default):
      native L0 -> native L1 -> source_translated L0 -> donor L0 ->
      source_translated L1 -> donor L1 -> {native, source_tr, donor} L2 -> ... L3
    (--order exact-first puts source_translated/donor L0 before native L1)
  * --refill failed_slots.parquet: reopen failed slots of an existing allocation and
    fill them with fresh (shorter) donors; refilled slots get round = previous + 1
    L0 exact cell, L1 domain relaxed, L2 task relaxed, L3 purpose(+origin) only
  * within a pass: least-used clusters first, bonus for reusable responses in
    the right language and for preference labels; near-duplicates (any
    language) of chosen items are blocked; at most USER_CAP items per user and
    CLUSTER_CAP per embedding cluster
"""
import argparse
import collections
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import annotate_prompt as A  # noqa: E402
import pool_rules as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "data")

TOTAL = 128_000
SEED = 20261107
PURPOSE_TARGET = {"general": 0.80, "if": 0.10, "safety_harmful": 0.05, "safety_benign": 0.05}
ORIGIN_TARGET = {"real_user": 0.60, "human": 0.12, "synthetic": 0.25, "web": 0.03}
EXCLUDED_SOURCES = {"akoksal/muri-it"}   # document-anchored reverse instructions, MT'd
ALPHA = 0.5
MIN_CELL = 3
DROP_FLAGS = {"nonsense", "greeting_or_test", "not_a_request", "model_identity",
              "contains_pii", "minor_sexual", "incomplete", "sexual_explicit"}
USER_CAP = 5
CLUSTER_CAP = 60          # ~4x the mean of 128k / 8192 clusters; caps templates
DONOR_LANGS = {"en", "fr", "es", "de", "it", "pt", "pl", "nl"}
NO_CASE_LANGS = {"ka"}
LANG_GROUPS = [{"nb", "nn", "no"}, {"bs", "hr", "sr"}]


def quotas():
    q = {}
    with open(os.path.join(HERE, "quotas.tsv"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("code\t") or not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            q[p[0]] = int(p[5])
    return q


def read_dir(pattern, columns=None):
    parts = [pq.read_table(p, columns=columns).to_pandas() for p in sorted(glob.glob(pattern))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns or [])


def lang_ok(lang, annot_lang, task, constraints):
    a = R.norm_lang(annot_lang) or str(annot_lang or "").lower()
    if a == lang or task == "translation" or "response_language" in constraints:
        return True
    return any(lang in g and a in g for g in LANG_GROUPS)


def ctx_chars(uids):
    """uid -> total characters of the conversation (user + assistant turns)."""
    out = {}
    for b in pq.ParquetFile(os.path.join(DATA, "pool", "candidates_ctx.parquet")).iter_batches(
            batch_size=20000, columns=["uid", "context"]):
        for u, ctx in zip(b.column("uid").to_pylist(), b.column("context").to_pylist()):
            if u in uids:
                out[u] = sum(len(m["content"] or "") for m in json.loads(ctx))
    return out


def load(args):
    cand = pq.read_table(os.path.join(DATA, "pool", "candidates.parquet")).to_pandas()
    ann = read_dir(os.path.join(DATA, "llm", "annotate", "out", "*.parquet"),
                   ["id", "text", "finish_reason"])
    parsed = [A.parse(t) if fr == "stop" else None for t, fr in zip(ann["text"], ann["finish_reason"])]
    ann = pd.DataFrame({"uid": ann["id"], "a": parsed})
    cand = cand.merge(ann, on="uid", how="left")
    nb = read_dir(os.path.join(DATA, "pool", "neighbors", "*.parquet"))
    cand = cand.merge(nb.rename(columns={"id": "uid"}), on="uid", how="left")
    cl = pq.read_table(os.path.join(DATA, "pool", "clusters.parquet")).to_pandas()
    cand = cand.merge(cl.rename(columns={"id": "uid"}), on="uid", how="left")
    ifm = pq.read_table(os.path.join(DATA, "pool", "if_meta.parquet"),
                        columns=["uid", "transferable", "has_case_constraint"]).to_pandas()
    cand = cand.merge(ifm, on="uid", how="left")
    cand["transferable"] = np.where(cand["transferable"].isna(), True, cand["transferable"]).astype(bool)
    cand["has_case_constraint"] = np.where(cand["has_case_constraint"].isna(), False,
                                           cand["has_case_constraint"]).astype(bool)
    ho_keys = set(pq.read_table(os.path.join(DATA, "holdout", "prompts.parquet"),
                                columns=["key"]).column("key").to_pylist())
    cand["ho_exact"] = cand["key_all"].isin(ho_keys)
    return cand


def derive(c):
    a = c["a"]
    ok = a.notna()
    get = lambda k, d=None: a.map(lambda x: x.get(k, d) if isinstance(x, dict) else d)
    c["task"] = get("task")
    c["domain"] = get("domain")
    c["safety"] = get("safety")
    c["locale"] = get("locale")
    c["complexity"] = get("complexity")
    c["annot_lang"] = get("response_language")
    c["constraints"] = get("constraints", []).map(lambda v: v or [])
    c["flags"] = get("flags", []).map(lambda v: v or [])
    c["gist"] = get("gist")
    c["turns"] = np.where(c["n_user_turns"] > 1, "multi", "single")
    orig = c["origin"].astype(str).map({"real_user": "real_user", "human": "human",
                                        "web": "web", "synthetic": "synthetic"})
    c["origin_g"] = orig  # template -> NaN (not used)
    prior = c["purpose"].astype(str)
    saf = c["safety"].astype(str)
    p = np.where(saf.isin(["harmful", "jailbreak"]), "safety_harmful",
        np.where((saf == "sensitive_benign") & ((prior == "safety") | (orig == "real_user")),
                 "safety_benign",
        np.where(prior == "if", "if", "general")))
    c["purpose_f"] = p
    c["task_f"] = np.where(c["purpose_f"] == "safety_harmful", saf, c["task"])
    if ARGS.origin_free:   # origin not part of the content cell (only excluded origins dropped)
        c["origin_f"] = np.where(c["origin_g"].isna(), None, "any")
    else:
        c["origin_f"] = np.where(c["purpose_f"] == "general", c["origin_g"], "any")
    reasons = collections.Counter()
    drop = pd.Series(False, index=c.index)

    def mark(mask, why):
        nonlocal drop
        reasons[why] += int((mask & ~drop).sum())
        drop = drop | mask

    mark(c["source"].astype(str).isin(EXCLUDED_SOURCES), "excluded_source")
    mark(~ok, "no_annotation")
    mark(c["flags"].map(lambda f: bool(DROP_FLAGS & set(f))), "quality_flag")
    mark(c["ho_exact"] | (c["ho_sim"].fillna(0) >= ARGS.holdout_thr), "holdout_contamination")
    mark(c["origin_f"].isna(), "template_origin")
    lok = [lang_ok(l, al, t, cs) for l, al, t, cs in
           zip(c["lang"].astype(str), c["annot_lang"], c["task"], c["constraints"])]
    mark(~pd.Series(lok, index=c.index) & ok, "annotated_language_mismatch")
    donor_bad = (c["role"] == "donor") & ((c["locale"] == "language_specific")
                                          | c["task"].isin(["translation", "language_learning"]))
    mark(donor_bad, "donor_language_specific")
    c = c[~drop].copy()
    return c, dict(reasons)


def cell_key(df, level):
    cols = {0: ["purpose_f", "origin_f", "task_f", "domain", "turns"],
            1: ["purpose_f", "origin_f", "task_f", "turns"],
            2: ["purpose_f", "origin_f", "turns"],
            3: ["purpose_f", "origin_f"]}[level]
    return df[cols].astype(str).agg("|".join, axis=1)


def plan(c, rng):
    """Global slot plan: list of L0 cells with exact counts (largest remainder)."""
    c0 = cell_key(c, 0)
    counts = c0.value_counts()
    rows = []
    for cell, n in counts.items():
        if n < MIN_CELL:
            continue
        purpose, origin = cell.split("|")[:2]
        rows.append((cell, purpose, origin, n))
    df = pd.DataFrame(rows, columns=["cell", "purpose", "origin", "n"])
    df["w"] = df["n"] ** ALPHA
    df["group_share"] = [PURPOSE_TARGET[p] * (ORIGIN_TARGET.get(o, 1.0) if p == "general" else 1.0)
                         for p, o in zip(df["purpose"], df["origin"])]
    df["share"] = df["group_share"] * df["w"] / df.groupby(["purpose", "origin"])["w"].transform("sum")
    exact = df["share"] / df["share"].sum() * TOTAL
    base = np.floor(exact).astype(int)
    rem = TOTAL - base.sum()
    order = np.argsort(-(exact - base).values + rng.random(len(df)) * 1e-9)
    base.iloc[order[:rem]] += 1
    df["slots"] = base
    return df


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-thr", type=float, default=0.90)
    ap.add_argument("--dup-thr", type=float, default=0.92)
    ap.add_argument("--out", default=os.path.join(DATA, "compiled"))
    ap.add_argument("--order", choices=["native-first", "exact-first"], default="native-first")
    ap.add_argument("--refill", help="failed_slots.parquet: reopen these slots of the "
                                     "existing <out>/allocation.parquet and refill with donors")
    ap.add_argument("--refill-max-chars", type=int,
                    help="refill only: donors' whole conversation (user + assistant) at most "
                         "this many characters (translation failure grows with length)")
    ap.add_argument("--alternates", type=int, default=1,
                    help="refill only: K candidates per reopened slot (1 primary in the "
                         "allocation + K-1 in <out>/alternates_roundN.parquet)")
    ap.add_argument("--origin-free", action="store_true",
                    help="do not keep the real_user/human/synthetic/web mix equal across languages")
    ap.add_argument("--refill-native", action="store_true",
                    help="refill only: also try natives / source-translated at levels L0-L1 first "
                         "(for reopened native slots, e.g. dropped low-quality rows)")
    ap.add_argument("--refill-exclude-sources", default="",
                    help="refill only: comma-separated sources not to pick from (their rows already in "
                         "the allocation keep their metadata)")
    ap.add_argument("--refill-exclude-uids", default="",
                    help="refill only: comma-separated files with one uid per line not to pick (broken "
                         "template families: template_drop.py; template families: families.py)")
    ARGS = args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    q = quotas()
    rep = {"params": vars(args)}

    c = load(args)
    rep["candidates"] = len(c)
    c, rep["dropped"] = derive(c)
    rep["eligible"] = len(c)
    print("eligible after filters:", len(c), rep["dropped"], flush=True)

    SLOT_COLS = ["slot", "lang", "cell0", "purpose", "origin", "task", "domain", "turns",
                 "cell1", "cell2", "cell3", "uid", "fill", "level", "round"]
    if args.refill:
        # reopen only the failed slots of an existing allocation; everything already
        # selected (including the failed donors) stays used / blocked
        prev = pq.read_table(os.path.join(args.out, "allocation.parquet")).to_pandas()
        failed = set(pq.read_table(args.refill, columns=["slot"]).column("slot").to_pylist())
        if "round" not in prev:
            prev["round"] = 0
        S = prev[SLOT_COLS].copy()
        prev_used = set(prev["uid"].dropna())
        for fn in glob.glob(os.path.join(args.out, "alternates_round*.parquet")):
            prev_used |= set(pq.read_table(fn, columns=["uid"]).column("uid").to_pylist())
        reopen = S["slot"].isin(failed)
        S.loc[reopen, "uid"] = None
        S.loc[reopen, "fill"] = None
        S.loc[reopen, "level"] = -1
        S.loc[reopen, "round"] = int(prev["round"].max()) + 1
        rep["refill"] = {"reopened": int(reopen.sum())}
        print("refill: reopened {} slots".format(int(reopen.sum())), flush=True)
    else:
        P = plan(c, rng)
        rep["plan_cells"] = int((P["slots"] > 0).sum())
        slots = np.repeat(P["cell"].values, P["slots"].values)
        langs = np.concatenate([[l] * n for l, n in q.items()])
        rng.shuffle(slots)
        rng.shuffle(langs)
        S = pd.DataFrame({"slot": np.arange(TOTAL), "lang": langs, "cell0": slots})
        parts = S["cell0"].str.split("|", expand=True)
        S["purpose"], S["origin"], S["task"], S["domain"], S["turns"] = (parts[i] for i in range(5))
        S["cell1"] = S[["purpose", "origin", "task", "turns"]].agg("|".join, axis=1)
        S["cell2"] = S[["purpose", "origin", "turns"]].agg("|".join, axis=1)
        S["cell3"] = S[["purpose", "origin"]].agg("|".join, axis=1)
        S["uid"], S["fill"], S["level"], S["round"] = None, None, -1, 0

    for lvl in range(4):
        c["cell%d" % lvl] = cell_key(c, lvl)
    resp_ok = (c["n_resp_reusable"] > 0) & (c["resp_lang"].astype(str) == c["lang"].astype(str))
    c["bonus"] = resp_ok.astype(float) * 1.0 + c["pref_type"].notna().astype(float) * 0.5
    c["rand"] = rng.random(len(c))
    c = c.set_index("uid", drop=False)
    c["_pos"] = np.arange(len(c))
    pos_of = dict(zip(c.index, range(len(c))))
    nn_block = {u: [x for x, s in zip(ids, sims) if s >= args.dup_thr]
                for u, ids, sims in zip(c["uid"], c["nn_ids"], c["nn_sims"])
                if isinstance(ids, (list, np.ndarray))}
    used, blocked = set(), set()
    # used | blocked by position in c: pick's vectorised filter (pool.index.isin(set) converted
    # the whole set on every call, which made a refill of a few thousand slots take an hour)
    taken = np.zeros(len(c), dtype=bool)

    def mark(us):
        for u in us:
            i = pos_of.get(u)
            if i is not None:
                taken[i] = True
    user_n = collections.Counter()
    cluster_n = collections.Counter()
    if args.refill:
        used |= prev_used
        for u in prev_used:
            blocked.update(nn_block.get(u, ()))
        excl = [s for s in args.refill_exclude_sources.split(",") if s]
        if excl:
            blocked.update(c.index[c["source"].astype(str).isin(excl)])
            print("refill: not picking from", excl, flush=True)
        if args.refill_exclude_uids:
            xu = set()
            for fn in args.refill_exclude_uids.split(","):
                with open(fn) as f:
                    xu |= {ln.strip() for ln in f if ln.strip()}
            blocked.update(c.index[c.index.isin(xu)])
            print("refill: not picking {} listed uids ({} in the pool)".format(len(xu), int(c.index.isin(xu).sum())), flush=True)
        mark(used)
        mark(blocked)
        kept = S["uid"].dropna()
        for u in kept:
            if u in c.index:
                cluster_n[c.at[u, "cluster"]] += 1
                uk = c.at[u, "user_key"]
                if isinstance(uk, str):
                    user_n[uk] += 1

    def pick(pool, k, lang_for_user=None):
        """Pick up to k uids from candidate frame `pool` (diversity order)."""
        if pool.empty or k <= 0:
            return []
        pool = pool[~taken[pool["_pos"].to_numpy()]]
        if pool.empty:
            return []
        order = pool.assign(_cl=pool["cluster"].map(lambda x: cluster_n[x]))
        order = order.sort_values(["_cl", "bonus", "rand"], ascending=[True, False, True])
        out = []
        for u, cl, uk in zip(order.index, order["cluster"], order["user_key"]):
            if u in used or u in blocked:
                continue
            if isinstance(uk, str) and user_n[uk] >= USER_CAP:
                continue
            if cluster_n[cl] >= CLUSTER_CAP:
                continue
            out.append(u)
            used.add(u)
            taken[pos_of[u]] = True
            nb = nn_block.get(u, ())
            blocked.update(nb)
            mark(nb)
            cluster_n[cl] += 1
            if isinstance(uk, str):
                user_n[uk] += 1
            if len(out) >= k:
                break
        return out

    native = c[c["role"] == "native"]
    srctr = c[c["role"] == "source_translated"]
    # donors: sampled donors plus unused natives of the rich donor languages
    # (prompts about a specific language, translation and language-learning tasks
    # can't be translated without breaking them -- applies to natives used as donors too)
    donor_ok = ~((c["locale"] == "language_specific")
                 | c["task"].isin(["translation", "language_learning"]))
    donors = c[c["role"].isin(["donor", "native"]) & c["lang"].astype(str).isin(DONOR_LANGS)
               & donor_ok]
    if args.refill:   # shorter conversations only (long ones failed translation)
        donors = donors[(donors["user_chars"] <= 6000) & (donors["n_user_turns"] <= 6)]
        if args.refill_max_chars:
            tot = ctx_chars(set(donors.index))
            donors = donors[[tot.get(u, 10 ** 9) <= args.refill_max_chars for u in donors.index]]
            print("refill donors within {} chars: {}".format(args.refill_max_chars, len(donors)),
                  flush=True)
    groups = {}
    for name, df in (("native", native), ("source_translated", srctr), ("donor", donors)):
        for lvl in range(4):
            # note: grouping by a one-element list yields 1-tuples as keys, so the
            # donor groups (keyed by cell only) are grouped by the Series itself
            key = [df["lang"].astype(str), df["cell%d" % lvl]] if name != "donor" else df["cell%d" % lvl]
            groups[(name, lvl)] = {k: g for k, g in df.groupby(key, observed=True)}
    if args.order == "native-first":   # any native fit before any translation (per level pair)
        passes = [("native", 0), ("native", 1), ("source_translated", 0), ("donor", 0),
                  ("source_translated", 1), ("donor", 1)]
    else:                               # exact cell (incl. translation) before relaxed natives
        passes = [("native", 0), ("source_translated", 0), ("donor", 0), ("native", 1),
                  ("source_translated", 1), ("donor", 1)]
    passes += [("native", 2), ("source_translated", 2), ("donor", 2),
               ("native", 3), ("source_translated", 3), ("donor", 3)]
    if args.refill:   # natives/source-translated at L0/L1 were exhausted before any
        # donor was used; keep the original preference order for the rest
        if not args.refill_native:
            passes = [p for p in passes if p[0] == "donor" or p[1] >= 2]
    def run_passes(S):
        """Fill the open slots of S in place, pass by pass (preference order)."""
        for name, lvl in passes:
            open_ = S[S["uid"].isna()]
            if open_.empty:
                break
            filled = 0
            for (lang, cell), idx in open_.groupby(["lang", "cell%d" % lvl]).groups.items():
                if name == "donor":
                    g = groups[(name, lvl)].get(cell)
                    if g is not None:
                        keep = (g["lang"].astype(str) != lang) & g["transferable"]
                        if lang in NO_CASE_LANGS:
                            keep &= ~g["has_case_constraint"]
                        g = g[keep]
                else:
                    g = groups[(name, lvl)].get((lang, cell))
                if g is None or g.empty:
                    continue
                chosen = pick(g, len(idx))
                for slot, u in zip(idx, chosen):
                    S.at[slot, "uid"] = u
                    S.at[slot, "fill"] = "translate" if name == "donor" else name
                    S.at[slot, "level"] = lvl
                filled += len(chosen)
            print("pass {:<18} L{}: filled {:>6}, open {:>6}".format(
                name, lvl, filled, int(S["uid"].isna().sum())), flush=True)

    def with_meta(df):
        return df.merge(c.reset_index(drop=True)[["uid", "lang", "source", "subset", "origin", "role", "task_f", "domain",
                       "safety", "locale", "complexity", "constraints", "gist", "n_resp",
                       "n_resp_reusable", "resp_lang", "pref_type", "if_verifiable", "cluster",
                       "ho_sim"]].rename(columns={"lang": "src_lang", "task_f": "item_task",
                                                  "domain": "item_domain", "origin": "item_origin"}),
                        on="uid", how="left")

    run_passes(S)
    alts = []
    if args.refill and args.alternates > 1:
        # extra candidates for the reopened slots, translated alongside the primary;
        # pick_alternates.py keeps the first one that passes validation
        for k in range(1, args.alternates):
            print("alternate", k, flush=True)
            A = S.loc[reopen.values, SLOT_COLS].copy()
            A["uid"], A["fill"], A["level"] = None, None, -1
            run_passes(A)
            A["rank"] = k
            alts.append(A[A["uid"].notna()])
    S = with_meta(S[SLOT_COLS])
    os.makedirs(args.out, exist_ok=True)
    S.to_parquet(os.path.join(args.out, "allocation.parquet"), index=False)
    if alts:
        rnd = int(S.loc[S["slot"].isin(failed), "round"].max())
        A = with_meta(pd.concat(alts)[SLOT_COLS + ["rank"]])
        A.to_parquet(os.path.join(args.out, "alternates_round{}.parquet".format(rnd)), index=False)
        rep["alternates"] = {"round": rnd, "rows": len(A), "fill": A["fill"].value_counts().to_dict()}
        print("alternates for round {}: {} rows".format(rnd, len(A)), flush=True)

    # ---- report
    rep["unfilled"] = int(S["uid"].isna().sum())
    rep["fill"] = S["fill"].value_counts().to_dict()
    rep["level"] = S["level"].value_counts().sort_index().to_dict()
    per = {}
    for lang, g in S.groupby("lang"):
        per[lang] = {"n": len(g), **{k: round(v, 3) for k, v in
                     g["fill"].value_counts(normalize=True).items()}}
    rep["per_lang_fill"] = per

    def js(p, qd):
        keys = set(p) | set(qd)
        pv = np.array([p.get(k, 0) for k in keys]) + 1e-12
        qv = np.array([qd.get(k, 0) for k in keys]) + 1e-12
        m = (pv + qv) / 2
        return float(0.5 * (pv * np.log2(pv / m)).sum() + 0.5 * (qv * np.log2(qv / m)).sum())

    filled = S[S["uid"].notna()]
    rep["independence_js"] = {}
    for dim in ("purpose", "item_task", "item_domain", "turns"):
        glob_d = filled[dim].value_counts(normalize=True).to_dict()
        rep["independence_js"][dim] = {l: round(js(g[dim].value_counts(normalize=True).to_dict(), glob_d), 4)
                                       for l, g in filled.groupby("lang")}
    with open(os.path.join(args.out, "allocation_report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    print(json.dumps({k: rep.get(k) for k in ("candidates", "eligible", "plan_cells", "unfilled",
                                          "fill", "level")}, default=str))


if __name__ == "__main__":
    main()
