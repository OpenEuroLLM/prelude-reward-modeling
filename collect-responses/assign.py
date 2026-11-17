#!/usr/bin/env python3
"""Which models answer which prompt: two responses per prompt, each from a (model, mode) drawn at random.

  collect-responses/run_gpu.sh assign.py [--seed 20261113] [--workers 8]    # CPU only (GPU container for
                                                                          # its transformers)

Every prompt of data/compiled/prompts.jsonl gets two slots. For each slot independently (the same model may
be drawn twice) a (model, mode) pair is drawn with probability proportional to the model's `weight` in
models.yaml (default 1; models that often produce word salad are downweighted), split evenly over the modes
the model was trained for (direct / think: a hybrid model answers in thinking mode half of the time).
A pair is eligible only if the prompt, rendered with that model's chat template, leaves room for the answer
within the model's max_model_len: at least MIN_NEW[mode] tokens (or the whole budget if it is smaller).
A pair is also ineligible if the model's chat template rejects the conversation (28 comparia conversations
have two user turns in a row, which e.g. Mistral-style templates refuse); the conversation is never altered.
Ineligible pairs are dropped and the draw is repeated among the rest. Draws use a per-prompt RNG, so the
result does not depend on how prompts are split over workers.

Writes data/responses/assignment.parquet: id, slot, model, mode, prompt_tokens, max_tokens, seed
(the request's sampling seed), and prints counts per model and mode.
"""
import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from smoke_test import pool, prepare_messages  # noqa: E402

OUT = os.path.join(ROOT, "data", "responses")
MIN_NEW = {"direct": 2048, "think": 8192}
SLOTS = 2
_TOK = {}


def tokenizer(name):
    if name not in _TOK:
        from transformers import AutoTokenizer
        _TOK[name] = AutoTokenizer.from_pretrained(os.path.join(ROOT, "data", "models", name))
    return _TOK[name]


def prompt_tokens(prompt, m, mode):
    tok = tokenizer(m["name"])
    msgs = prepare_messages(prompt, m)
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                   **(m["modes"][mode].get("template_kwargs") or {}))
    return len(tok.encode(text, add_special_tokens=False))


def seed_of(*parts):
    return int.from_bytes(hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=4).digest(), "big") & 0x7fffffff


def assign_chunk(job):
    prompts, seed = job
    p, models = pool()
    budgets = p["defaults"]["max_new_tokens"]
    pairs = [(name, mode, m.get("weight", 1.0) / len(m["modes"])) for name, m in models.items() for mode in m["modes"]]
    from jinja2.exceptions import TemplateError
    rows, failed, rejected = [], [], []
    for pr in prompts:
        rng = random.Random(seed_of(seed, pr["id"]))
        cache = {}
        for slot in range(SLOTS):
            cand = [x for x in pairs if x[2] > 0]
            while cand:
                name, mode, _ = rng.choices(cand, weights=[w for _, _, w in cand])[0]
                m = models[name]
                key = (name, mode)
                if key not in cache:
                    try:
                        n = prompt_tokens(pr, m, mode)
                    except TemplateError:                     # the template refuses this conversation
                        rejected.append((pr["id"], name, mode))
                        n = None
                    if n is None:
                        cache[key] = (None, None)
                    else:
                        budget = (m.get("max_new_tokens") or {}).get(mode, budgets[mode])
                        room = m["max_model_len"] - n
                        cache[key] = (n, min(budget, room)) if room >= min(budget, MIN_NEW[mode]) else (n, None)
                n, max_tokens = cache[key]
                if max_tokens is None:
                    cand = [x for x in cand if (x[0], x[1]) != key]
                    continue
                rows.append({"id": pr["id"], "slot": slot, "model": name, "mode": mode, "prompt_tokens": n,
                             "max_tokens": max_tokens, "seed": seed_of(seed, pr["id"], slot, "sample")})
                break
            else:
                failed.append((pr["id"], slot))
    return rows, failed, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261113)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="first N prompts only (testing)")
    ap.add_argument("--out", default=os.path.join(OUT, "assignment.parquet"))
    args = ap.parse_args()
    prompts = []
    with open(os.path.join(ROOT, "data", "compiled", "prompts.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            prompts.append({"id": o["id"], "messages": o["messages"]})
            if args.limit and len(prompts) >= args.limit:
                break
    prompts.sort(key=lambda o: o["id"])
    size = (len(prompts) + args.workers * 4 - 1) // (args.workers * 4)
    jobs = [(prompts[i:i + size], args.seed) for i in range(0, len(prompts), size)]
    rows, failed, rejected = [], [], []
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as workers:
        for k, (r, f, x) in enumerate(workers.imap(assign_chunk, jobs)):
            rows += r
            failed += f
            rejected += x
            print("chunk {}/{} done, {:.0f}s".format(k + 1, len(jobs), time.time() - t0), flush=True)
    assert not failed, "{} slots without an eligible model, e.g. {}".format(len(failed), failed[:5])
    import pyarrow as pa
    import pyarrow.parquet as pq
    path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    p, models = pool()
    by = collections.Counter((r["model"], r["mode"]) for r in rows)
    total = len(rows)
    print("{} prompts, {} responses -> {}".format(len(prompts), total, path))
    for name, m in models.items():
        for mode in m["modes"]:
            print("  {:36s} {:6s} weight {:4.2f}  {:6d} ({:4.1f}%)".format(name, mode, m.get("weight", 1.0),
                                                                          by[(name, mode)], 100 * by[(name, mode)] / total))
    same = sum(1 for i in range(0, total, 2) if rows[i]["model"] == rows[i + 1]["model"])
    print("prompts answered twice by the same model: {} ({:.1%})".format(same, same / (total / 2)))
    if rejected:
        print("drawn pairs whose chat template refused the conversation (redrawn): {} on {} prompts, models {}".format(
            len(rejected), len({i for i, _, _ in rejected}), dict(collections.Counter(n for _, n, _ in rejected))))


if __name__ == "__main__":
    main()
