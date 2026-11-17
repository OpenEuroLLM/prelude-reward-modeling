#!/usr/bin/env python3
"""Stage 1 of compiling: index every extracted record into the candidate pool.

For each record of data/extracted/<src>/part-*.parquet this writes one index
row (no full text) to data/pool/index/<src>/<part>.parquet with:
  - pointer back to the record (part file + row number)
  - provenance from pool_rules.py (origin, translated, purpose, license, exclude)
  - language: OpenLID-v3 on the last user turn and on all user turns, the
    source's own label, and the final decision (+ the rule that decided it)
  - response language (first response), response/preference availability
  - safety (prompt_harm, adversarial) and instruction-following labels
  - user/session key (for per-user caps) and exact-dedup keys

  ./run.sh compile/build_pool.py [--only <hf_id> ...] [--workers 30] [--force]
  ./run.sh compile/build_pool.py --only allenai/coconot --max-rows 2000 --test

Memory: the 1.2 GB OpenLID model is loaded once in the parent and shared
copy-on-write with forked workers; each worker streams small batches and
writes its output incrementally (peak RSS per part is logged). Run the full
build as a Slurm batch job (compile/slurm/build_pool.sbatch), not inside an
interactive session.
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import resource
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import regex  # noqa: E402

import pool_rules as R  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
EXTRACTED = os.path.join(ROOT, "data", "extracted")
OUT = os.path.join(ROOT, "data", "pool", "index")
MODEL_DIR = os.path.join(ROOT, "data", "models", "openlid-v3")

INDEX_SCHEMA = pa.schema([
    ("uid", pa.string()), ("source", pa.string()), ("subset", pa.string()),
    ("split", pa.string()), ("part", pa.string()), ("row", pa.int64()),
    ("origin", pa.string()), ("translated", pa.bool_()), ("purpose", pa.string()),
    ("license_flag", pa.string()), ("exclude", pa.string()), ("trust_lang", pa.bool_()),
    ("lang", pa.string()), ("lang_rule", pa.string()),
    ("lid_last", pa.string()), ("lid_last_p", pa.float32()),
    ("lid_last2", pa.string()), ("lid_last2_p", pa.float32()),
    ("lid_all", pa.string()), ("lid_all_p", pa.float32()),
    ("lid_chars_last", pa.int32()),
    ("lang_src_raw", pa.string()), ("lang_src", pa.string()),
    ("n_user_turns", pa.int16()), ("n_msgs", pa.int16()),
    ("last_user_chars", pa.int32()), ("user_chars", pa.int32()),
    ("has_system", pa.bool_()), ("has_code", pa.bool_()),
    ("n_resp", pa.int16()), ("n_resp_reusable", pa.int16()),
    ("resp_lang", pa.string()), ("resp_lang_p", pa.float32()),
    ("resp_model0", pa.string()), ("pref_type", pa.string()),
    ("prompt_harm", pa.string()), ("adversarial", pa.bool_()),
    ("if_verifiable", pa.bool_()), ("if_n_constraints", pa.int16()),
    ("user_key", pa.string()), ("key_all", pa.string()), ("key_last", pa.string()),
])

# ---------------------------------------------------------------- text utils
CODE_RE = regex.compile(r"```.*?(?:```|$)", regex.S)
INLINE_CODE_RE = regex.compile(r"`[^`\n]{1,200}`")
URL_RE = regex.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\w+")
TAG_RE = regex.compile(r"<[^>\n]{1,100}>")
NONWORD_RE = regex.compile(r"[^\p{Word}\p{Zs}]|\d")   # OpenLID-v3 preprocessing
SPACES_RE = regex.compile(r"\s\s+")
LETTER_RE = regex.compile(r"\p{L}")
KEY_RE = regex.compile(r"[^\p{L}\p{N}]+")


def clean_for_lid(text, limit):
    t = text[: limit * 4]
    t = CODE_RE.sub(" ", t)
    t = INLINE_CODE_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    t = TAG_RE.sub(" ", t)
    t = t.strip().replace("\n", " ").lower()
    t = SPACES_RE.sub(" ", t)
    t = NONWORD_RE.sub("", t)
    t = SPACES_RE.sub(" ", t).strip()
    return t[:limit]


def text_key(texts):
    s = " ".join(" ".join(KEY_RE.sub(" ", t.lower()).split()) for t in texts)
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest()


# ----------------------------------------------------------------------- LID
_MODEL = None
_LABEL2CODE = {}


def label_to_code(label):
    lab = label.replace("__label__", "")
    if lab in _LABEL2CODE:
        return _LABEL2CODE[lab]
    base, _, script = lab.partition("_")
    if base == "srp" and script == "Latn":
        return "sr-Latn"
    return R.ISO3.get(base, lab)


def load_model(model_path, label2code):
    """Load once in the parent; forked workers share it copy-on-write."""
    global _MODEL, _LABEL2CODE
    import fasttext
    fasttext.FastText.eprint = lambda *a, **k: None
    _MODEL = fasttext.load_model(model_path)
    _LABEL2CODE = label2code


def lid(clean):
    """-> (code1, p1, code2, p2) or Nones. Uses the C++ binding directly
    (fasttext's numpy wrapper is incompatible with numpy 2)."""
    if not clean or not LETTER_RE.search(clean):
        return None, 0.0, None, 0.0
    preds = _MODEL.f.predict(clean, 2, 0.0, "replace")
    out = [(label_to_code(lab), float(p)) for p, lab in preds]
    while len(out) < 2:
        out.append((None, 0.0))
    return out[0][0], out[0][1], out[1][0], out[1][1]


def decide_lang(l1, p1, l2, la, pa_, n_last, lang_src, trust):
    cand, rule = None, None
    if l1 and p1 >= 0.5 and n_last >= 20:
        cand, rule = l1, "lid_last"
    elif la and pa_ >= 0.5:
        cand, rule = la, "lid_all"
    elif l1 and p1 >= 0.3:
        cand, rule = l1, "lid_last_weak"
    if trust and lang_src:
        if cand is None:
            return lang_src, "source_only"
        if cand == lang_src:
            return cand, rule + "+source"
        if R.confusable(cand, lang_src) or l2 == lang_src:
            return lang_src, "source_over_lid"
        return cand, rule + "!source_mismatch"
    return cand or "und", rule or "und"


# ------------------------------------------------------------------ records
USER_KEYS = ("hashed_ip", "visitor_id", "evaluation_session_id", "session_hash",
             "user_id", "annotator_id")


def if_info(meta):
    ids = meta.get("instruction_id_list")
    if isinstance(ids, list) and ids:
        return True, len(ids)
    cons = meta.get("constraints")
    if isinstance(cons, list) and cons:
        ver = any(isinstance(c, dict) and c.get("eval_funcs") for c in cons)
        return ver, len(cons)
    return False, 0


def index_record(r, part_rel, row_no):
    meta = json.loads(r["meta"]) if r["meta"] else {}
    rule = R.rules_for(r["source"], r["subset"], meta)
    ctx = r["context"]
    users = [m["content"] or "" for m in ctx if m["role"] == "user"]
    last = users[-1] if users else ""

    c_last = clean_for_lid(last, 1000)
    c_all = clean_for_lid("\n".join(u[:600] for u in users[-4:]), 2000) \
        if len(users) > 1 else c_last
    l1, p1, l2, p2 = lid(c_last)
    la, pa_ = (l1, p1) if c_all is c_last else lid(c_all)[:2]
    lang_src = R.norm_lang(r["language"])
    lang, lang_rule = decide_lang(l1, p1, l2, la, pa_, len(c_last), lang_src,
                                  rule["trust_lang"])

    resps = r["responses"] or []
    rl, rp = (None, 0.0)
    if resps:
        rl, rp, _, _ = lid(clean_for_lid(resps[0]["content"] or "", 600))
    reusable = 0
    for x in resps:
        m = json.loads(x["meta"]) if x["meta"] else {}
        if m.get("model_reuse") is not False:
            reusable += 1
    pref = json.loads(r["preference"]) if r["preference"] else None
    user_key = None
    for k in USER_KEYS:
        if meta.get(k):
            user_key = "{}:{}".format(k, meta[k])
            break
    harm = meta.get("prompt_harm")
    adv = meta.get("adversarial")
    if_ver, if_n = if_info(meta)
    text_all = "\n".join(users)
    return {
        "uid": r["uid"], "source": r["source"], "subset": r["subset"],
        "split": r["split"], "part": part_rel, "row": row_no,
        "origin": rule["origin"],
        "translated": R.resolve_translated(rule, lang),
        "purpose": rule["purpose"], "license_flag": rule["license_flag"],
        "exclude": rule["exclude"], "trust_lang": rule["trust_lang"],
        "lang": lang, "lang_rule": lang_rule,
        "lid_last": l1, "lid_last_p": p1, "lid_last2": l2, "lid_last2_p": p2,
        "lid_all": la, "lid_all_p": pa_, "lid_chars_last": len(c_last),
        "lang_src_raw": None if r["language"] is None else str(r["language"]),
        "lang_src": lang_src,
        "n_user_turns": len(users), "n_msgs": len(ctx),
        "last_user_chars": len(last), "user_chars": len(text_all),
        "has_system": any(m["role"] == "system" for m in ctx),
        "has_code": "```" in text_all,
        "n_resp": len(resps), "n_resp_reusable": reusable,
        "resp_lang": rl, "resp_lang_p": rp,
        "resp_model0": resps[0]["model"] if resps else None,
        "pref_type": pref.get("type") if pref else None,
        "prompt_harm": None if harm is None else str(harm),
        "adversarial": adv if isinstance(adv, bool) else None,
        "if_verifiable": if_ver, "if_n_constraints": if_n,
        "user_key": user_key,
        "key_all": text_key(users), "key_last": text_key([last]),
    }


COLUMNS = ["uid", "source", "subset", "split", "language", "context",
           "responses", "preference", "meta"]


def process_part(task):
    src_dir, part_path, out_path, max_rows = task
    t0 = time.time()
    part_rel = os.path.relpath(part_path, EXTRACTED)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp"
    writer = pq.ParquetWriter(tmp, INDEX_SCHEMA, compression="zstd")
    rows, n = [], 0
    pf = pq.ParquetFile(part_path)
    done = False
    for batch in pf.iter_batches(batch_size=256, columns=COLUMNS):
        for r in batch.to_pylist():
            rows.append(index_record(r, part_rel, n))
            n += 1
            if max_rows and n >= max_rows:
                done = True
                break
        if len(rows) >= 20_000 or done:
            writer.write_table(pa.Table.from_pylist(rows, schema=INDEX_SCHEMA))
            rows = []
        if done:
            break
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=INDEX_SCHEMA))
    writer.close()
    os.replace(tmp, out_path)
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    return src_dir, os.path.basename(part_path), n, time.time() - t0, rss_gb


def ensure_model():
    path = os.path.join(MODEL_DIR, "openlid-v3.bin")
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        hf_hub_download("HPLT/OpenLID-v3", "openlid-v3.bin", local_dir=MODEL_DIR)
    return path


def label_map():
    m = {}
    with open(os.path.join(HERE, "languages.tsv"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or line.startswith("code\t") or not line.strip():
                continue
            code, _aliases, hplt, _name = line.rstrip("\n").split("\t")
            m[hplt] = code
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-rows", type=int, help="per part file (testing)")
    ap.add_argument("--test", action="store_true",
                    help="write to data/pool/_test_index and print rows")
    args = ap.parse_args()

    out_root = os.path.join(ROOT, "data", "pool", "_test_index") if args.test else OUT
    sources = args.only or sorted(R.SOURCES)
    unknown = set(sources) - set(R.SOURCES)
    if unknown:
        sys.exit("no rules for {}".format(sorted(unknown)))
    tasks = []
    for src in sources:
        d = src.replace("/", "__")
        for p in sorted(glob.glob(os.path.join(EXTRACTED, d, "part-*.parquet"))):
            out = os.path.join(out_root, d, os.path.basename(p))
            if os.path.exists(out) and not args.force and not args.test:
                continue
            tasks.append((d, p, out, args.max_rows))
            if args.test:
                break
    print("{} part files to index".format(len(tasks)), flush=True)
    if not tasks:
        return
    load_model(ensure_model(), label_map())
    # largest files first so the tail of the run is short
    tasks.sort(key=lambda t: -os.path.getsize(t[1]))
    t0 = time.time()
    total = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(min(args.workers, len(tasks)), maxtasksperchild=4) as pool:
        for i, (d, part, n, sec, rss) in enumerate(
                pool.imap_unordered(process_part, tasks), 1):
            total += n
            print("[{}/{}] {} {} rows={:,} {:.0f}s peakRSS={:.1f}GB (total {:,}, {:.0f}s)"
                  .format(i, len(tasks), d, part, n, sec, rss, total,
                          time.time() - t0), flush=True)
    if args.test:
        for d, _, out, _ in tasks:
            t = pq.read_table(out).slice(0, 3).to_pylist()
            for row in t:
                print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
