#!/usr/bin/env python3
"""Offline batch inference with vLLM over a requests parquet (stages 3 and 5).

Requests parquet columns:
  id          str   unique request id
  messages    str   JSON list of {role, content}
  schema      str   JSON schema for structured output, or null
  max_tokens  int   (optional; falls back to --max-tokens)
  prompt      str   (optional) raw completion prompt; used instead of `messages`
                    (base models without a chat template, e.g. few-shot MT)
  prompt_ids  list  (optional) token ids of `prompt`, sent as they are (models
                    whose HF tokenizer is wrong in this container, e.g. TildeOpen:
                    transformers 5 converts its SentencePiece model badly)
  stop        str   (optional) JSON list of stop strings

Prompt construction: a non-null `prompt` is tokenized as is (with BOS);
otherwise `messages` go through the chat template -- by default exactly as
LLM.chat renders them; with --chat-tokens the model's own template is applied
by the tokenizer and the token ids are sent (for templates with custom
message fields that vLLM's chat parser rejects, e.g. TranslateGemma).

Outputs: <out>/shard<i>-<k>.parquet with id, text, finish_reason,
prompt_tokens, completion_tokens. Resumable: ids already in <out> are skipped.
Requests whose prompt + max_tokens exceed --max-model-len are written with
finish_reason="too_long" and empty text (never sent to the engine).

Requests are streamed through the engine (LLMEngine.add_request / step) with
at most --window queued at a time, longest max_tokens first, and finished
outputs are flushed every --flush-every rows or --flush-secs seconds. A job
killed at its time limit therefore loses only the requests still in flight,
and a few very long generations no longer hold back a whole chunk.

Runs inside the LAIFS container on a GPU node (see slurm/llm_multi.sbatch):
  python compile/llm_runner.py --requests R.parquet --out DIR --model PATH \\
      --tp 2 --shard 0 --num-shards 4
"""
import argparse
import glob
import hashlib
import json
import os
import time

import pyarrow as pa
import pyarrow.parquet as pq

OUT_SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string()),
                        ("finish_reason", pa.string()),
                        ("prompt_tokens", pa.int32()),
                        ("completion_tokens", pa.int32())])
OUT_SCHEMA_IDS = OUT_SCHEMA.append(pa.field("token_ids", pa.list_(pa.int32())))


def shard_of(id_, n, salt=""):
    return int(hashlib.md5((salt + id_).encode("utf-8")).hexdigest(), 16) % n


def write(rows, out, shard, schema=OUT_SCHEMA):
    k = len(glob.glob(os.path.join(out, "shard{}-*.parquet".format(shard))))
    path = os.path.join(out, "shard{}-{:05d}.parquet".format(shard, k))
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-salt", default="",
                    help="re-deal ids over shards (e.g. on a resubmission where the unfinished "
                         "ids all sit in a few shards); skipping finished ids is shard-independent")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--window", type=int, default=2048,
                    help="max requests queued in the engine at once")
    ap.add_argument("--chunk", type=int, help=argparse.SUPPRESS)  # old name of --window
    ap.add_argument("--flush-every", type=int, default=500)
    ap.add_argument("--flush-secs", type=float, default=120.0)
    ap.add_argument("--limit", type=int, help="process at most N requests (testing)")
    ap.add_argument("--chat-tokens", action="store_true",
                    help="apply the model's chat template with the tokenizer and send token ids "
                         "(templates with custom message fields, e.g. TranslateGemma)")
    ap.add_argument("--tokenizer-mode", default="auto", help="vLLM tokenizer mode")
    ap.add_argument("--save-token-ids", action="store_true",
                    help="also store the generated token ids (decode them outside vLLM when "
                         "the model's HF tokenizer is wrong)")
    args = ap.parse_args()
    schema = OUT_SCHEMA_IDS if args.save_token_ids else OUT_SCHEMA
    if args.chunk:
        args.window = args.chunk

    os.makedirs(args.out, exist_ok=True)
    done = set()
    for p in glob.glob(os.path.join(args.out, "shard*.parquet")):
        done.update(pq.read_table(p, columns=["id"]).column("id").to_pylist())
    table = pq.read_table(args.requests)
    ids = table.column("id").to_pylist()
    mask = pa.array([shard_of(i, args.num_shards, args.shard_salt) == args.shard and i not in done
                     for i in ids])
    todo = table.filter(mask).to_pylist()
    del table, ids
    if args.limit:
        todo = todo[:args.limit]
    # longest first: the long tail starts early instead of trailing at the end
    todo.sort(key=lambda r: -(r.get("max_tokens") or args.max_tokens))
    print("shard {}/{}: {} requests ({} already done)".format(
        args.shard, args.num_shards, len(todo), len(done)), flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams

    kw = dict(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              enable_prefix_caching=True, max_num_seqs=args.max_num_seqs, seed=0,
              tokenizer_mode=args.tokenizer_mode)
    llm = None
    for mm in ({"image": 0, "audio": 0}, {"image": 0}, None):   # multimodal / text-only models
        try:
            llm = LLM(**kw) if mm is None else LLM(limit_mm_per_prompt=mm, **kw)
            break
        except Exception as e:
            if mm is None:
                raise
            print("retrying with a smaller multimodal limit:", repr(e)[:200], flush=True)
    tok = llm.get_tokenizer()
    eng = llm.llm_engine
    tmpl_kw = {"enable_thinking": False}
    if args.chat_tokens:   # older transformers ignore a separate chat_template.jinja
        tf = os.path.join(args.model, "chat_template.jinja")
        if os.path.exists(tf):
            with open(tf, encoding="utf-8") as f:
                tmpl_kw = {"chat_template": f.read()}
        else:
            tmpl_kw = {}

    pending = iter(todo)
    buf, waiting = [], set()        # waiting: our ids currently in the engine
    alias = {}                      # engine-internal request id -> our id
    n_done, toks_in, toks_out = 0, 0, 0
    t_start = last_flush = time.time()

    def feed():
        while len(waiting) < args.window:
            r = next(pending, None)
            if r is None:
                return
            mt = r.get("max_tokens") or args.max_tokens
            raw = r.get("prompt")
            ei = None
            if r.get("prompt_ids") is not None:   # pre-tokenized outside vLLM
                ids = [int(x) for x in r["prompt_ids"]]
                n_prompt = len(ids)
                ei = TokensPrompt(prompt_token_ids=ids)
            elif raw is not None:            # raw completion prompt (base models)
                ids = tok(raw, add_special_tokens=True)["input_ids"]
                n_prompt = len(ids)
                ei = TokensPrompt(prompt_token_ids=ids)
            else:
                conv = json.loads(r["messages"])
                try:
                    ids = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=True,
                                                  **({} if args.chat_tokens else tmpl_kw))
                except Exception as e:     # e.g. unsupported language code in the template
                    print("template failed for {}: {!r}".format(r["id"], e)[:300], flush=True)
                    buf.append({"id": r["id"], "text": "", "finish_reason": "template_error",
                                "prompt_tokens": 0, "completion_tokens": 0})
                    continue
                if hasattr(ids, "keys"):   # newer tokenizers return a BatchEncoding/dict
                    ids = ids["input_ids"]
                if args.chat_tokens:
                    n_prompt = len(ids)
                    ei = TokensPrompt(prompt_token_ids=list(ids))
                else:
                    # vLLM's own rendering (as in LLM.chat) is one token longer than the
                    # tokenizer's template ids for Gemma 4; keep a token of slack
                    n_prompt = len(ids) + 1
            if n_prompt + mt > args.max_model_len:
                buf.append({"id": r["id"], "text": "", "finish_reason": "too_long",
                            "prompt_tokens": n_prompt, "completion_tokens": 0})
                continue
            if ei is None:
                try:   # identical prompt construction to LLM.chat (pinned vLLM 0.26)
                    ei = llm._preprocess_chat_one(conv, chat_template_kwargs=tmpl_kw)
                except Exception as e:
                    print("render failed for {}: {!r}".format(r["id"], e)[:300], flush=True)
                    buf.append({"id": r["id"], "text": "", "finish_reason": "too_long",
                                "prompt_tokens": n_prompt, "completion_tokens": 0})
                    continue
            so = StructuredOutputsParams(json=json.loads(r["schema"])) \
                if r.get("schema") else None
            stop = json.loads(r["stop"]) if r.get("stop") else None
            sp = SamplingParams(temperature=args.temperature, max_tokens=mt,
                                structured_outputs=so, seed=0, stop=stop,
                                output_kind=RequestOutputKind.FINAL_ONLY)
            rid = eng.add_request(r["id"], ei, sp)
            if rid != r["id"]:
                alias[rid] = r["id"]
            waiting.add(r["id"])

    while True:
        feed()
        if not waiting and not eng.has_unfinished_requests():
            break
        for o in eng.step():
            if not o.finished:
                continue
            # outputs carry the external id (= ours); fall back to the alias map
            our = o.request_id if o.request_id in waiting else alias.pop(o.request_id, None)
            if our not in waiting:
                raise RuntimeError("unexpected output id {!r}".format(o.request_id))
            waiting.discard(our)
            c = o.outputs[0]
            row = {"id": our, "text": c.text, "finish_reason": c.finish_reason,
                   "prompt_tokens": len(o.prompt_token_ids),
                   "completion_tokens": len(c.token_ids)}
            if args.save_token_ids:
                row["token_ids"] = list(c.token_ids)
            buf.append(row)
            toks_in += len(o.prompt_token_ids)
            toks_out += len(c.token_ids)
        if buf and (len(buf) >= args.flush_every or time.time() - last_flush >= args.flush_secs):
            write(buf, args.out, args.shard, schema)
            n_done += len(buf)
            buf, last_flush = [], time.time()
            el = time.time() - t_start
            print("shard {}: {}/{} done, total {:.0f}s, {:.2f} req/s, in {:.0f} tok/s, "
                  "out {:.0f} tok/s".format(args.shard, n_done, len(todo), el, n_done / el,
                                            toks_in / el, toks_out / el), flush=True)
    if buf:
        write(buf, args.out, args.shard, schema)
        n_done += len(buf)
    el = time.time() - t_start
    print("shard {}: {}/{} done, total {:.0f}s, {:.2f} req/s, in {:.0f} tok/s, "
          "out {:.0f} tok/s".format(args.shard, n_done, len(todo), el, n_done / el,
                                    toks_in / el, toks_out / el), flush=True)


if __name__ == "__main__":
    main()
