#!/usr/bin/env python3
"""Training data for the reward models: the released preference pairs, split into train and validation and
tokenized per backbone.

  train-reward-model/run.sh prepare_data.py split
      data/preferences/preferences.jsonl -> data/reward_model/pairs.parquet: one row per labelled pair, without
      the judges' rationales. split = "val" for the 1,008 pilot pairs (data/preferences/pilot/pairs.parquet,
      the pairs with Claude Opus reference labels), "train" for all others.
  train-reward-model/run.sh prepare_data.py tokenize --model qwen3.8-27b
      -> data/reward_model/tok/<model>/{train,val}.parquet and meta.json. Per pair: the prompt (the
      conversation rendered with the backbone's chat template as its generation prompt, thinking disabled:
      what the model sees when it starts the answer), the ids of the two answers, and the label. The end-of-turn ids that close an answer are in
      meta.json. Answers are tokenized as plain text (special-token strings inside an answer stay text).
      Truncation happens at training time.
  train-reward-model/run.sh prepare_data.py stats --model qwen3.8-27b
      token-length distribution of prompt + answer, and how many pairs a maximum length would cut.

The reward model sees what the judges saw: the conversation and one final answer, verbatim. Never the
answer's hidden reasoning, never the model name, and no added system prompt. The Qwen3.8 chat template adds a
"Reasoning effort" system prompt unless thinking is disabled, which is why every template is rendered with
enable_thinking=False.
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PREF = os.path.join(ROOT, "data", "preferences")
OUT = os.path.join(ROOT, "data", "reward_model")
MODELS = os.path.join(ROOT, "data", "models")
SENTINEL = "ANSWER"   # private-use characters, absent from the data

META_COLS = ["id", "split", "language", "category", "task_type", "subject_domain", "safety_label", "complexity",
             "translation_status", "author_type", "model_a", "model_b", "mode_a", "mode_b", "finish_a", "finish_b",
             "lid_a", "lid_b", "p_a", "rule", "n_votes"]


def split(args):
    pilot = set(pq.read_table(os.path.join(PREF, "pilot", "pairs.parquet"), columns=["id"]).column("id").to_pylist())
    cols = collections.defaultdict(list)
    n = 0
    with open(os.path.join(PREF, "preferences.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            a, b, pr = r["response_a"], r["response_b"], r["preference"]
            row = {"id": r["id"], "split": "val" if r["id"] in pilot else "train", "language": r["language"],
                   "category": r["category"], "task_type": r["task_type"], "subject_domain": r["subject_domain"],
                   "safety_label": r["safety_label"], "complexity": r["complexity"],
                   "translation_status": r["translation_status"], "author_type": r["author_type"],
                   "model_a": a["model"], "model_b": b["model"], "mode_a": a["mode"], "mode_b": b["mode"],
                   "finish_a": a["finish_reason"], "finish_b": b["finish_reason"],
                   "lid_a": a["detected_language"], "lid_b": b["detected_language"],
                   "p_a": float(pr["prob_a_better"]), "rule": pr["rule"], "n_votes": len(pr["votes"]),
                   "messages": json.dumps(r["messages"], ensure_ascii=False),
                   "response_a": a["content"], "response_b": b["content"]}
            for k, v in row.items():
                cols[k].append(v)
            n += 1
    t = pa.table(cols)
    os.makedirs(OUT, exist_ok=True)
    pq.write_table(t, os.path.join(OUT, "pairs.parquet"), compression="zstd")
    sp = collections.Counter(cols["split"])
    missing = pilot - set(i for i, s in zip(cols["id"], cols["split"]) if s == "val")
    print("{} pairs: {}; pilot pairs without a label: {}".format(n, dict(sp), sorted(missing)))
    langs = {s: collections.Counter(l for l, x in zip(cols["language"], cols["split"]) if x == s) for s in sp}
    print("languages: train {}, val {}; in train but not val: {}".format(
        len(langs["train"]), len(langs["val"]), sorted(set(langs["train"]) - set(langs["val"]))))


def load_tokenizer(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(os.path.join(MODELS, model))


def render(tok, messages):
    """-> (prefix, suffix): the conversation exactly as the model sees it when it writes the answer (the
    template's generation prompt, thinking disabled), and the end of the assistant turn.

    The prefix is not the template's rendering of a finished turn: Gemma 4's template strips the (empty)
    thinking block from finished turns, but the model always opens its answer with it
    ("<|turn>model\\n<|channel>thought\\n<channel|>"), and without it every answer starts where the model
    expects that block. For Qwen3.8 both renderings are the same."""
    prefix = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    full = tok.apply_chat_template(messages + [{"role": "assistant", "content": SENTINEL}], tokenize=False,
                                   enable_thinking=False)
    i = full.index(SENTINEL)
    return prefix, full[i + len(SENTINEL):].rstrip("\n")


def tokenize(args):
    tok = load_tokenizer(args.model)
    t = pq.read_table(os.path.join(OUT, "pairs.parquet"))
    rows = t.select(["messages", "response_a", "response_b"]).to_pydict()
    n = t.num_rows
    prefixes, suffixes = [], collections.Counter()
    for m in rows["messages"]:
        p, s = render(tok, json.loads(m))
        prefixes.append(p)
        suffixes[s] += 1
    if len(suffixes) != 1:
        raise SystemExit("the end of the assistant turn differs between rows: {}".format(suffixes.most_common(5)))
    suffix = next(iter(suffixes))
    suffix_ids = tok(suffix, add_special_tokens=False)["input_ids"]

    def enc(texts, plain):
        out = []
        for i in range(0, len(texts), 2048):
            out += tok(texts[i:i + 2048], add_special_tokens=False, split_special_tokens=plain)["input_ids"]
        return out

    prompt_ids = enc(prefixes, plain=False)
    a_ids = enc(rows["response_a"], plain=True)
    b_ids = enc(rows["response_b"], plain=True)
    meta = t.select([c for c in META_COLS]).to_pydict()
    d = os.path.join(OUT, "tok", args.model)
    os.makedirs(d, exist_ok=True)
    i32 = pa.list_(pa.int32())
    for s in ("train", "val"):
        idx = [i for i in range(n) if meta["split"][i] == s]
        cols = {c: [meta[c][i] for i in idx] for c in META_COLS}
        cols["prompt_ids"] = pa.array([prompt_ids[i] for i in idx], type=i32)
        cols["a_ids"] = pa.array([a_ids[i] for i in idx], type=i32)
        cols["b_ids"] = pa.array([b_ids[i] for i in idx], type=i32)
        pq.write_table(pa.table(cols), os.path.join(d, s + ".parquet"), compression="zstd")
        print(s, len(idx))
    info = {"model": args.model, "suffix": suffix, "suffix_ids": suffix_ids,
            "example": tok.decode(prompt_ids[0][-40:] + a_ids[0][:20]) + " ... " + tok.decode(a_ids[0][-10:] + suffix_ids),
            "pad_token_id": tok.pad_token_id}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(info, f, ensure_ascii=False, indent=1)
    print(json.dumps(info, ensure_ascii=False, indent=1))


def stats(args):
    d = os.path.join(OUT, "tok", args.model)
    suffix = len(json.load(open(os.path.join(d, "meta.json")))["suffix_ids"])
    for s in ("train", "val"):
        t = pq.read_table(os.path.join(d, s + ".parquet"), columns=["prompt_ids", "a_ids", "b_ids"])
        lp = np.array([len(x) for x in t.column("prompt_ids").to_pylist()])
        la = np.array([len(x) for x in t.column("a_ids").to_pylist()])
        lb = np.array([len(x) for x in t.column("b_ids").to_pylist()])
        seq = np.concatenate([lp + la, lp + lb]) + suffix
        pair = lp * 2 + la + lb + 2 * suffix
        q = [50, 90, 95, 99, 99.9, 100]
        print("== {} ({} pairs)".format(s, len(lp)))
        print("  prompt tokens  mean {:.0f}  pct {}".format(lp.mean(), np.percentile(lp, q).astype(int).tolist()))
        print("  sequence       mean {:.0f}  pct {}".format(seq.mean(), np.percentile(seq, q).astype(int).tolist()))
        print("  pair (2 seqs)  mean {:.0f}  sum {:.1f}M tokens".format(pair.mean(), pair.sum() / 1e6))
        for L in (4096, 8192, 12288, 16384, 32768):
            cut = np.minimum(seq, L)
            print("  max {:>6}: sequences cut {:5.2f}%, prompts over half {:5.2f}%, tokens kept {:5.1f}%, total {:.1f}M".format(
                L, 100 * (seq > L).mean(), 100 * (lp > L // 2).mean(), 100 * cut.sum() / seq.sum(), cut.sum() / 1e6))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["split", "tokenize", "stats"])
    ap.add_argument("--model", default="qwen3.8-27b")
    args = ap.parse_args()
    {"split": split, "tokenize": tokenize, "stats": stats}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
