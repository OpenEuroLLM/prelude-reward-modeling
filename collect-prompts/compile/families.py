#!/usr/bin/env python3
"""Stage 6g: template families in the candidate pool, and a per-family cap on the compiled set.

  ./run.sh compile/families.py                     # detect -> data/pool/families.parquet; report on the set
  ./run.sh compile/families.py --cap 2 --apply     # fail the set's rows over the cap (force_fail.parquet,
                                                   # reason family_cap); write data/compiled/family_uids.txt

A family is a group of prompts that share most of their first user turn, e.g. one WildChat user
running the same instructions over many inputs ("As a prompt generator for a generative AI called
Midjourney ..."), a scraped FAQ site's question template, a synthetic task generator's frame.
Detection, on the first user turn (lowercased, digit runs -> 0), three MinHash (64 hashes) / LSH
signals whose joins are merged into connected components:
  full   first 200 words, bigrams under 15 words else trigrams, estimated Jaccard >= 0.5
         (near-duplicates; templates with a small variable part); prompts under 8 words not grouped
         (short common forms such as "write a poem about X" are not templates)
  frame  first + last 30 words of prompts of >= 80 words (15 + 15 for 40-79 words), trigrams, >= 0.5 (the same
         instructions around a long variable payload; added after refill round 14 brought in e.g.
         35 rows of "Create a long detailed text about <X> with the following summary ...")
  faq    every_prompt questions of 6-40 words with the same first 5 and last 2 words (scraped FAQ
         slot templates such as "How does KAYAK find such great <hotel> hotel deals?"; an exact
         key, because a loose MinHash chained different questions; this source only)
Exact signals (added 2026-11-12: the MinHash ones missed scripts with a long payload and a
generator's repeated scenarios, e.g. 64 EuroMultiSynth prompts opening "Estou a desenvolver um
script em Python para automatizar a recolha de dados de concursos públicos em Portugal e preciso"):
  prefix the same first LONG_PREFIX (15) words, same source; not in PREAMBLE_SOURCES, whose shared
         openings are constraint or format preambles before different tasks ("When answering,
         please adhere to these constraints: ...", "Solve the following math problem ... boxed";
         requiring the prefix to be half the prompt still merged short problems)
  ratio  same source, first 4 and last 2 words; common prefix + common suffix >= 60% of the shorter
         prompt (>= 6 words): slot templates and scripts with a short payload ("Welche Airlines, die
         von <X> nach <Y> fliegen, bieten aufgrund von COVID-19 flexible Umbuchungen an?"). No
         suffix-only rule: IF datasets append the same constraint sentence to unrelated prompts.
         In PREAMBLE_SOURCES the common suffix alone must be half the prompt (a Natural-Instructions
         preamble + the same appended constraint chained 292 rows of different tasks; a math
         preamble + 2 shared final words merged different short problems).
  exact  identical user turns (all of them, lowercased words): duplicates; the cap keeps one per
         language and family. A guard: the pool is deduplicated on whole contexts, so this finds
         none now (conversations sharing a first turn, then diverging, are ratio families).
Opening signal (added 2026-11-12 after round 19: 4,797 set rows over a cap of 2 still shared their
first 10 words, 3,883 of them EuroMultiSynth scenario openings such as "Ik ben bezig met het
plannen van een weekendje weg ..."; also every_prompt FAQ frames, "How far in advance should I book
a flight from <X> to <Y>?", and WildChat scripts copied into other datasets, "Write a concise
summary of the following. Be sure to preserve important details: <text>"):
  opening  the same first OPEN_WORDS (10) words (boilerplate stripped) in prompts of >= OPEN_MIN
         (12) words, across sources. An exact key, so its groups cannot chain; a partition of its
         own. Not grouped: format / constraint preambles before different tasks (the user: "but
         not the math ones"): prompts with a \\boxed answer format, IF constraint preambles
         (OPEN_EXEMPT: "Make sure to follow these requirements when responding: ...", "When
         answering, please adhere to these constraints: ...", "Please include the following
         question in your response without changing anything ..."), and openings that are mostly
         numbers (data: phone lists, lap times).
The exact signals form their own families (within one source), capped separately from the
MinHash ones, so that a link of one kind cannot chain through the other: a WildChat user's script
translating Natural-Instructions prompts into Tunisian Arabic contains those prompts, so MinHash
joined each script prompt to the NI prompt it carried, and a shared NI opening then chained 59
different NI tasks into the script's family. They also skip known NI / FLAN boilerplate openings
(BOILERPLATE: "Given the task definition and input, reply with output.", ...), so NI prompts
group by task, not by boilerplate. A row is failed when it is over the cap in any partition.
The allocation's own guards do not catch these: the
near-duplicate block covers 10 nearest neighbours per pick and the cluster cap is 60 per cluster,
while a family can have thousands of pool items spread over many clusters.
Cap: keep at most C rows per family in the set (Latgalian rows first, then natives, then
translated; by row id within each), fail the rest. family_uids.txt lists every pool item of a family of >= MIN_FAMILY pool
items: refill with --refill-exclude-uids so refills only pick prompts outside any family.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import mt_common as M  # noqa: E402

NH, BANDS = 64, 16
ROWS = NH // BANDS
PRIME = (1 << 31) - 1
JACCARD = 0.5
MIN_WORDS = 8
MIN_FAMILY = 3
LONG_PREFIX = 15        # prefix signal: shared first words
FRAME_RATIO, FRAME_MIN, FRAME_BLOCK = 0.6, 6, 400   # ratio signal: share of the shorter prompt, min words, max block
HT = 40                 # head / tail words kept per prompt for the exact signals
BOILERPLATE = tuple(tuple(b.split()) for b in (     # NI / FLAN openings, toks_of form
    "given the task definition and input reply with output",
    "given the task definition example input output solve the new input case",
    "you will be given a definition of a task first then some input of the task",
    "you will be given a definition of a task first then an example follow the example to solve a new instance of the task",
    "teacher now understand the problem solve this instance",
    "part 0 definition", "detailed instructions", "definition", "instructions", "q", "task"))
PREAMBLE_SOURCES = ("allenai/IF_multi_constraints", "nvidia/Nemotron-RL-instruction", "nvidia/Nemotron-Post-Training",
                    "allenai/tulu-3-sft-personas-instruction")
OPEN_WORDS, OPEN_MIN = 10, 12   # opening signal: shared first words, min prompt words
OPEN_EXEMPT = re.compile(r"\b(follows? these|adhere to these|comply with these|satisfy these|"
                         r"include the following question in your response)\b|"
                         # IFEval-style constraints placed before the task
                         r"^(provide two (distinct|different)|give two (distinct|different)|(your|the) (entire )?(response|answer|reply)|"
                         r"in your (entire )?(response|answer|reply)|answer with|first repeat|repeat the request|wrap your|"
                         r"do not include|finish your response|end your response|highlight at least|response must be|"
                         r"there should be|use at least)\b")
NUMERIC = re.compile(r"[0_]+")  # a digit-run token after toks_of
WORD = re.compile(r"\w+", re.U)
DIG = re.compile(r"\d+")
FAM = os.path.join(M.DATA, "pool", "families.parquet")


def user_key(ctx):
    """exact-duplicate key: all user turns, lowercased words (digits kept)"""
    t = "\x1f".join(" ".join(WORD.findall((m.get("content") or "").lower())) for m in ctx if m.get("role") == "user")
    return hashlib.blake2b(t.encode(), digest_size=12).hexdigest()


def strip_boiler(toks):
    """drop leading NI / FLAN boilerplate (repeatedly: "Detailed Instructions: Definition: ...")"""
    i, moved = 0, True
    while moved:
        moved = False
        for b in BOILERPLATE:
            if tuple(toks[i:i + len(b)]) == b:
                i += len(b)
                moved = True
                break
    return toks[i:]


def lcp(a, b):
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k


def first_user(ctx):
    return next((m.get("content") or "" for m in ctx if m.get("role") == "user"), "")


def toks_of(text):
    return [DIG.sub("0", t.lower()) for t in WORD.findall(text)]


def grams(toks, n, tag=""):
    return {zlib.crc32((tag + " ".join(toks[i:i + n])).encode()) & 0x7FFFFFFF for i in range(len(toks) - n + 1)}


def sh_full(toks, uid):
    """the first 200 words: near-duplicates and templates with a small variable part"""
    t = toks[:200]
    if len(t) < MIN_WORDS:
        return None
    return grams(t, 2 if len(t) < 15 else 3)


def sh_frame(toks, uid):
    """first and last 30 words of long prompts: the same instructions around a long variable
    payload (a summary, a pasted text, data)"""
    if len(toks) >= 80:
        return grams(toks[:30], 3, "h:") | grams(toks[-30:], 3, "t:")
    if len(toks) >= 40:     # medium prompts: first and last 15 words
        return grams(toks[:15], 3, "h:") | grams(toks[-15:], 3, "t:")
    return None


def faq_key(toks, uid):
    """every_prompt's scraped FAQ questions: exact first-5 + last-2 words ("How does KAYAK find such
    great <hotel> hotel deals?", "How much is a hotel in <city> for tonight?"); an exact key, since a
    loose MinHash on bigrams chained different questions ("What are the signs of ...", "What are the
    advantages of ...")"""
    if not uid.startswith("lang-uk/every_prompt::") or not 6 <= len(toks) <= 40:
        return None
    return " ".join(toks[:5]) + " | " + " ".join(toks[-2:])


# name, shingles, bands, rows per band, estimated-Jaccard threshold
SIGNALS = (("full", sh_full, 16, 4, JACCARD), ("frame", sh_frame, 16, 4, 0.5))


def detect(keep_text):
    rng = np.random.default_rng(20261111)
    A = rng.integers(1, PRIME, NH, dtype=np.int64)[:, None]
    B = rng.integers(0, PRIME, NH, dtype=np.int64)[:, None]
    uids, text = [], {}
    sig = {name: ([], []) for name, *_ in SIGNALS}       # name -> (doc indices, signatures)
    faq = collections.defaultdict(list)                  # exact FAQ key -> doc indices
    exact = collections.defaultdict(list)                # user-turns key -> doc indices
    opening = collections.defaultdict(list)              # first OPEN_WORDS words (hashed) -> doc indices
    head, tail, ln, srcs = {}, {}, {}, []                # doc index -> first / reversed last HT words, length
    for b in pq.ParquetFile(os.path.join(M.DATA, "pool", "candidates_ctx.parquet")).iter_batches(
            batch_size=20000, columns=["uid", "context"]):
        for u, c in zip(b.column("uid").to_pylist(), b.column("context").to_pylist()):
            ctx = json.loads(c)
            t = first_user(ctx)
            if u in keep_text:
                text[u] = t[:160].replace("\n", " ")
            toks = toks_of(t)
            i = len(uids)
            uids.append(u)
            srcs.append(u.split("::")[0])
            for name, fn, *_ in SIGNALS:
                sh = fn(toks, u)
                if not sh:
                    continue
                h = np.fromiter(sh, dtype=np.int64, count=len(sh))[None, :]
                sig[name][0].append(i)
                sig[name][1].append(((A * h + B) % PRIME).min(axis=1).astype(np.int32))
            fk = faq_key(toks, u)
            if fk:
                faq[fk].append(i)
            xt = strip_boiler(toks)
            if len(xt) >= FRAME_MIN:
                head[i], tail[i], ln[i] = tuple(xt[:HT]), tuple(xt[-HT:][::-1]), len(xt)
            if len(xt) >= OPEN_MIN and "boxed" not in toks:
                ok = " ".join(xt[:OPEN_WORDS])
                if not OPEN_EXEMPT.search(ok) and sum(bool(NUMERIC.fullmatch(w)) for w in xt[:OPEN_WORDS]) < OPEN_WORDS // 2:
                    opening[hashlib.blake2b(ok.encode(), digest_size=8).digest()].append(i)
            exact[user_key(ctx)].append(i)
    n = len(uids)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for name, _, bands, rows, thr in SIGNALS:
        idx, ss = np.array(sig[name][0]), np.stack(sig[name][1])
        joined = 0
        for band in range(bands):
            key = np.ascontiguousarray(ss[:, band * rows:(band + 1) * rows]).view(np.dtype((np.void, 4 * rows))).ravel()
            _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            multi = np.flatnonzero(cnt[inv] > 1)
            order = multi[np.argsort(inv[multi], kind="stable")]
            for g in np.split(order, np.flatnonzero(np.diff(inv[order])) + 1):
                if len(g) < 2:
                    continue
                agree = (ss[g[1:]] == ss[g[0]]).mean(axis=1) >= thr
                for x in g[1:][agree]:
                    ra, rb = find(idx[g[0]]), find(idx[x])
                    if ra != rb:
                        parent[rb] = ra
                        joined += 1
        print("signal {}: {} prompts, {} joins".format(name, len(idx), joined), flush=True)
    joined = 0
    for members in faq.values():
        for x in members[1:]:
            ra, rb = find(members[0]), find(x)
            if ra != rb:
                parent[rb] = ra
                joined += 1
    print("signal faq (exact key): {} keys shared by >= 2, {} joins".format(sum(len(m) > 1 for m in faq.values()), joined), flush=True)

    roots_m = np.array([find(i) for i in range(n)])
    parent = np.arange(n)   # the exact signals: a partition of their own (find / union read this one)

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            return 1
        return 0

    # prefix: within (source, first 4 words) sorted by head, adjacent items sharing >= LONG_PREFIX words
    # (exact: everything between two such items shares the same prefix)
    blocks = collections.defaultdict(list)
    for i, h in head.items():
        blocks[(srcs[i], h[:4])].append(i)
    joined = 0
    for g in blocks.values():
        g.sort(key=head.get)
        for a, b2 in zip(g, g[1:]):
            k = lcp(head[a], head[b2])
            if k >= LONG_PREFIX and not srcs[a].startswith(PREAMBLE_SOURCES):
                joined += union(a, b2)
    print("signal prefix (>= {} words): {} joins".format(LONG_PREFIX, joined), flush=True)
    blocks = collections.defaultdict(list)
    for i, h in head.items():
        blocks[(srcs[i], h[:4], tail[i][:2])].append(i)
    joined = 0
    for g in blocks.values():
        if not 2 <= len(g) <= FRAME_BLOCK:
            continue
        for x in range(len(g)):
            a = g[x]
            for b2 in g[x + 1:]:
                m = min(ln[a], ln[b2])
                suf = lcp(tail[a], tail[b2])
                if srcs[a].startswith(PREAMBLE_SOURCES) and suf < 0.5 * m:
                    continue    # a preamble + an appended constraint around different tasks
                if min(m, lcp(head[a], head[b2]) + suf) >= max(FRAME_MIN, FRAME_RATIO * m):
                    joined += union(a, b2)
    print("signal ratio (prefix + suffix >= {:.0%}): {} joins".format(FRAME_RATIO, joined), flush=True)
    dup, joined = {}, 0
    for k, g in exact.items():
        if len(g) > 1:
            for x in g:
                dup[uids[x]] = k
            for x in g[1:]:
                joined += union(g[0], x)
    print("signal exact (user turns): {} keys shared by >= 2, {} joins".format(sum(len(g) > 1 for g in exact.values()), joined), flush=True)
    roots_x = np.array([find(i) for i in range(n)])
    oroot, osize = {}, {}       # the opening signal: exact key groups, a third partition
    for g in opening.values():
        if len(g) >= 2:
            for x in g:
                oroot[x], osize[x] = g[0], len(g)
    print("signal opening (first {} words): {} keys shared by >= 2, {} prompts".format(
        OPEN_WORDS, sum(len(g) > 1 for g in opening.values()), len(oroot)), flush=True)
    size_m, size_x = collections.Counter(roots_m.tolist()), collections.Counter(roots_x.tolist())
    fam = {}
    for i, (u, rm, rx) in enumerate(zip(uids, roots_m.tolist(), roots_x.tolist())):
        if size_m[rm] >= 2 or size_x[rx] >= 2 or i in oroot:
            fam[u] = {"uid": u, "family": rm if size_m[rm] >= 2 else None, "size": size_m[rm],
                      "xfamily": rx if size_x[rx] >= 2 else None, "xsize": size_x[rx], "dup": dup.get(u),
                      "ofamily": oroot.get(i), "osize": osize.get(i, 1)}
    print("uids in families of >= 2: MinHash {} | exact {} | opening {} | any {}; >= {} pool items: {}".format(
        sum(r["family"] is not None for r in fam.values()), sum(r["xfamily"] is not None for r in fam.values()),
        sum(r["ofamily"] is not None for r in fam.values()), len(fam),
        MIN_FAMILY, sum(max(r["size"], r["xsize"], r["osize"]) >= MIN_FAMILY for r in fam.values())), flush=True)
    pq.write_table(pa.Table.from_pylist(list(fam.values())), FAM)
    return fam, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=2)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="read data/pool/families.parquet instead of detecting")
    args = ap.parse_args()
    D = os.path.join(M.DATA, "compiled")
    rows = pq.read_table(os.path.join(D, "prompts.parquet"),
                         columns=["id", "lang", "fill", "source", "source_uid", "purpose"]).to_pylist()
    if args.reuse and os.path.exists(FAM):
        fam = {r["uid"]: r for r in pq.read_table(FAM).to_pylist()}
        text = {}
    else:
        fam, text = detect({r["source_uid"] for r in rows})
    by_fam = collections.defaultdict(list)      # ("m", MinHash) / ("x", exact-signal) / ("o", opening) family -> set rows
    pool_size = {}
    for r in rows:
        f = fam.get(r["source_uid"])
        if not f:
            continue
        for kind, key, sk in (("m", "family", "size"), ("x", "xfamily", "xsize"), ("o", "ofamily", "osize")):
            if f.get(key) is not None:
                by_fam[(kind, f[key])].append(r)
                pool_size[(kind, f[key])] = f[sk]
    sizes = collections.Counter(len(v) for v in by_fam.values())
    print("set rows in a family (>= 2 in the set): {} in {} families".format(
        sum(len(v) for v in by_fam.values() if len(v) >= 2), sum(len(v) >= 2 for v in by_fam.values())))
    print("family size in the set -> families:", dict(sorted(sizes.items())))
    # Latgalian rows first (hand-verified TildeOpen translations, costly to redo), then natives
    for v in by_fam.values():
        v.sort(key=lambda r: (r["lang"] != "ltg", r["fill"] != "native", r["id"]))
    for c in (1, 2, 3, 5):
        print("cap {}: {} rows over the cap (MinHash families {}, exact-signal families {}, opening groups {})".format(
            c, len({r["id"] for v in by_fam.values() for r in v[c:]}),
            *(sum(max(0, len(v) - c) for k, v in by_fam.items() if k[0] == kind) for kind in "mxo")))
    over, ndup = [], 0
    for f, v in by_fam.items():
        seen, kept = set(), []
        for r in v:             # exact duplicates: one per language
            d = fam[r["source_uid"]].get("dup") if f[0] == "x" else None
            if d and (d, r["lang"]) in seen:
                over.append((f, r))
                ndup += 1
                continue
            if d:
                seen.add((d, r["lang"]))
            kept.append(r)
        over += [(f, r) for r in kept[args.cap:]]
    print("exact duplicates (same user turns and language) to fail:", ndup)
    print("over the cap in more than one partition:", len(over) - len({r["id"] for _, r in over}))
    over = list({r["id"]: (f, r) for f, r in over}.values())    # once per row
    print("cap {}: rows to fail by language".format(args.cap), dict(collections.Counter(r["lang"] for _, r in over).most_common()))
    print("  by fill", dict(collections.Counter(r["fill"] for _, r in over)), "| by purpose",
          dict(collections.Counter(r["purpose"] for _, r in over)))
    top = sorted(by_fam.items(), key=lambda kv: -len(kv[1]))
    for f, v in top[:30]:
        ex = text.get(v[0]["source_uid"], "")
        print("{:4d} set / {:5d} pool {} | {:34} | {}".format(len(v), pool_size[f], f[0], v[0]["source"][:34], ex[:120]))
    if args.apply:
        p = os.path.join(D, "force_fail.parquet")
        old = pq.read_table(p).to_pylist() if os.path.exists(p) else []
        have = {(r["slot"], r["uid"]) for r in old}
        new = [{"slot": int(r["id"].split("-")[1]), "id": r["id"], "uid": r["source_uid"], "reason": "family_cap"}
               for _, r in over if (int(r["id"].split("-")[1]), r["source_uid"]) not in have]
        pq.write_table(pa.Table.from_pylist(old + new), p)
        print("force_fail: {} existing + {} new".format(len(old), len(new)))
        allu = sorted(u for u, r in fam.items() if max(r["size"], r.get("xsize") or 0, r.get("osize") or 0) >= MIN_FAMILY)
        with open(os.path.join(D, "family_uids.txt"), "w") as fh:
            fh.write("\n".join(allu) + "\n")
        print("family_uids.txt:", len(allu), "uids")


if __name__ == "__main__":
    main()
