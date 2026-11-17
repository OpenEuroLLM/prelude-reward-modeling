#!/usr/bin/env python3
"""Continue the responses that stopped at their token budget, up to larger budgets.

  collect-responses/run_gpu.sh continue_truncated.py check            # CPU: rebuild and verify every output prefix
  collect-responses/run_gpu.sh continue_truncated.py plan [--gcds 32] # CPU: requests and work units
  sbatch -N 4 -t 03:00:00 collect-responses/slurm/continue.sbatch   # GPU: one `continue_truncated.py work` per GCD
  collect-responses/run_gpu.sh continue_truncated.py status
  collect-responses/run_gpu.sh continue_truncated.py assemble         # CPU: merge into data/responses/responses.parquet

The first sampling pass gave every answer 4,096 new tokens (2,048 for the four weak models, although their
contexts allow 4,096 to 8,192) and every thinking run 16,384. 9,834 of the 256,000 responses stopped there.
Each of them is continued from the point where it stopped, with the same model, prompt and sampling
(temperature 1.0, top-p 1.0), up to CAP new tokens in total (direct 16,384, think 32,768), capped by the
model's context. Continuing rather than regenerating gives exactly the answer the larger budget would have
produced: a fresh sample would replace a long answer by one that mostly finishes early and bias the data
towards short answers. Every truncated response is continued, also loops and word salad, so that no
heuristic decides which ones get the larger budget; those that run into the new limit stay truncated.

The stored texts are decoded with special tokens removed and stripped, so the generated prefix is rebuilt
(rebuild): direct mode: the answer. Thinking that had closed: the think-start marker (when the prompt does
not end with it; the token counts show that these models generate it), the reasoning, the end marker and the
answer. Thinking that had not closed: the stored text already holds the marker's text part where the model
wrote one (Qwen-style "<think>", Gemma's "thought"), so only a special-token part is added back (Gemma's
"<|channel>"). `check` verifies for every request that the rendered prompt has the stored number of tokens
and that the rebuilt prefix splits (smoke_test.split_output) into the stored reasoning and answer (equal
after NFKC: OLMo's tokenizer normalises a few characters). Outputs whose text lost more than 10% of their
tokens (loops of special tokens that decoding removed) cannot be rebuilt and are not continued; they are
listed in skipped.parquet. The stored text is never changed: the new text is decoded relative to the prefix
and appended, and a thinking that closes in the continuation is split at its end marker as split_output
does. Prompts are rendered as in sample.py; the default system prompt's {date} is the date of the sampling
job (ORIGINAL_DATE). Outputs go to data/responses/continue/samples/<model>/ (flushed regularly, resumable
like sample.py); `assemble` keeps the previous file as responses.before_continuation.parquet and marks
continued rows. A request enters the engine only when its largest possible size fits into the KV cache
beside the running ones (kv_budget), so vLLM never preempts and recomputes these long sequences; the first
job (23757877), without this, ran at about 60 tokens/s per GCD and hit its time limit with 487 left.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import sys
import time
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import sample as S  # noqa: E402
from smoke_test import find_sub, pool, prepare_messages, render, split_output, stop_ids  # noqa: E402

RESP = os.path.join(ROOT, "data", "responses")
CONT = os.path.join(RESP, "continue")
CAP = {"direct": 16384, "think": 32768}
ORIGINAL_DATE = "2026-11-14"      # sampling job 23715022 ran 2026-11-14 03:27-07:19 (sacct)
THINK_START = {"</think>": "<think>\n", "<channel|>": "<|channel>thought\n"}
THINK_CLOSE = {"</think>": "\n</think>\n\n", "<channel|>": "<channel|>"}


def paths(base):
    return {"requests": os.path.join(base, "requests.parquet"), "plan": os.path.join(base, "plan.json"),
            "samples": os.path.join(base, "samples")}


def cont_seed(seed):
    return int(hashlib.md5("{}#continue".format(seed).encode()).hexdigest()[:8], 16)


def prefix_text(r, m, tok, prompt_text):
    """the generated text so far, with the markers that decoding and split_output removed put back"""
    if r["mode"] == "direct":
        return r["response"] or ""
    end = m["modes"]["think"]["think_end"]
    lead = THINK_START[end]
    in_prompt = prompt_text.endswith(lead) or prompt_text.endswith(lead.rstrip("\n"))
    if r["think_closed"]:
        return ("" if in_prompt else lead) + (r["reasoning"] or "") + THINK_CLOSE[end] + (r["response"] or "")
    reasoning = r["reasoning"] or ""
    if in_prompt:
        return reasoning
    kept = tok.decode(tok.encode(lead, add_special_tokens=False), skip_special_tokens=True).strip()
    if not kept:
        return lead + reasoning                       # the marker is all special tokens
    if reasoning.startswith(kept):
        return lead[:lead.index(kept)] + reasoning    # add back only the special-token part
    return reasoning                                  # the model wrote no marker


def rebuild(r, m, tok, prompt):
    """-> prompt ids, generated prefix ids, whether the prefix lost tokens (cannot be continued)"""
    kw = m["modes"][r["mode"]].get("template_kwargs") or {}
    p_ids, p_text = render(tok, prepare_messages(prompt, m, ORIGINAL_DATE), kw)
    gen, _ = prefix_ids(tok, p_ids, p_text, prefix_text(r, m, tok, p_text))
    lossy = r["completion_tokens"] - len(gen) > max(100, 0.1 * r["completion_tokens"])
    return p_ids, gen, lossy


def nfkc(x):
    return unicodedata.normalize("NFKC", x or "")


def extend_text(tok, prefix, new):
    """text of `new` as it continues `prefix` (decoding the suffix alone can drop a leading space)"""
    a = tok.decode(list(prefix), skip_special_tokens=True)
    b = tok.decode(list(prefix) + list(new), skip_special_tokens=True)
    return b[len(a):] if b.startswith(a) else tok.decode(list(new), skip_special_tokens=True)


def assemble_output(r, m, tok, stops, gen, new, finish):
    """-> reasoning, answer, think_closed, leaks: the stored text with the continuation appended"""
    special = set(getattr(tok, "all_special_ids", []) or []) | stops
    new = list(new)
    if finish == "stop":
        while new and new[-1] in special:
            new.pop()
    end = m["modes"]["think"].get("think_end") if r["mode"] == "think" else None
    _, _, _, leaks = split_output(tok, list(gen) + list(new), finish, end, stops)
    if r["mode"] == "direct":
        return "", ((r["response"] or "") + extend_text(tok, gen, new)).strip(), None, leaks
    if r["think_closed"]:
        return r["reasoning"] or "", ((r["response"] or "") + extend_text(tok, gen, new)).strip(), True, leaks
    end_ids = tok.encode(end, add_special_tokens=False)
    full = list(gen) + new
    k = find_sub(full, end_ids)
    if k < 0:
        return ((r["reasoning"] or "") + extend_text(tok, gen, new)).strip(), "", False, leaks
    reasoning = ((r["reasoning"] or "") + extend_text(tok, gen, full[len(gen):k])).strip()
    reasoning = re.sub(r"^\s*(<think>|thought\s*\n)", "", reasoning).strip()
    answer = tok.decode(full[k + len(end_ids):], skip_special_tokens=True).strip()
    return reasoning, answer, True, leaks


def prefix_ids(tok, p_ids, p_text, text):
    """token ids of the generated prefix: the prompt and prefix encoded together where that keeps the prompt's
    ids unchanged (no merge across the boundary), else the prefix on its own"""
    ids = tok.encode(p_text + text, add_special_tokens=False)
    if ids[:len(p_ids)] == p_ids:
        return ids[len(p_ids):], False
    return tok.encode(text, add_special_tokens=False), True


def load_truncated():
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    return ds.dataset(os.path.join(RESP, "responses.parquet")).to_table(filter=pc.field("finish_reason") == "length").to_pylist()


def load_prompts(ids):
    out = {}
    with open(os.path.join(ROOT, "data", "compiled", "prompts.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o["id"] in ids:
                out[o["id"]] = {"id": o["id"], "messages": o["messages"]}
    return out


# ------------------------------------------------------------------ check / plan (CPU)
def check(args):
    from transformers import AutoTokenizer
    p, models = pool()
    rows = load_truncated()
    prompts = load_prompts({r["id"] for r in rows})
    by = collections.defaultdict(list)
    for r in rows:
        by[r["model"]].append(r)
    bad, dtok = collections.Counter(), []
    shown = 0
    for name, rs in sorted(by.items()):
        m = models[name]
        path = os.path.join(ROOT, "data", "models", name)
        tok = AutoTokenizer.from_pretrained(path)
        stops, _ = stop_ids(tok, path)
        c = collections.Counter()
        for r in rs:
            p_ids, gen, lossy = rebuild(r, m, tok, prompts[r["id"]])
            if len(p_ids) != r["prompt_tokens"]:
                c["prompt length"] += 1
                continue
            if lossy:
                c["lossy (not continued)"] += 1
                continue
            end = m["modes"]["think"].get("think_end") if r["mode"] == "think" else None
            reasoning, answer, closed, _ = split_output(tok, list(gen), "length", end, stops)
            if not (nfkc(reasoning) == nfkc(r["reasoning"]) and nfkc(answer) == nfkc(r["response"]) and closed == r["think_closed"]):
                c["split"] += 1
                if shown < 6:
                    shown += 1
                    print("MISMATCH {} {} {}: closed {} vs {}; reasoning {} chars vs {}; answer {} vs {}".format(
                        name, r["mode"], r["request_id"], closed, r["think_closed"], len(reasoning), len(r["reasoning"] or ""),
                        len(answer), len(r["response"] or "")))
                continue
            dtok.append(len(gen) - r["completion_tokens"])
        bad.update({k: v for k, v in c.items() if k != "lossy (not continued)"})
        print("{:38s} {:5d} truncated: {}".format(name, len(rs), dict(c) or "all rebuilt"), flush=True)
    dtok.sort()
    q = lambda f: dtok[min(len(dtok) - 1, int(f * len(dtok)))] if dtok else 0
    print("{} requests, problems: {}; rebuilt minus stored generated tokens: median {}, 1% {}, 99% {}".format(
        len(rows), dict(bad) or "none", q(0.5), q(0.01), q(0.99)))
    return 0 if not bad else 1


def plan(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    P = paths(args.base)
    p, models = pool()
    rows = load_truncated()
    prompts = load_prompts({r["id"] for r in rows})
    reqs, skipped, toks = [], [], {}
    for r in rows:
        m = models[r["model"]]
        if r["model"] not in toks:
            toks[r["model"]] = AutoTokenizer.from_pretrained(os.path.join(ROOT, "data", "models", r["model"]))
        _, gen, lossy = rebuild(r, m, toks[r["model"]], prompts[r["id"]])
        new_max = min(CAP[r["mode"]], m["max_model_len"] - r["prompt_tokens"])
        r = dict(r)
        r["new_max_tokens"] = new_max
        r["cont_seed"] = cont_seed(r["seed"])
        r["rebuilt_tokens"] = len(gen)
        if lossy or new_max <= r["completion_tokens"]:
            r["skip_reason"] = "text lost tokens" if lossy else "no room in the context"
            skipped.append(r)
            continue
        reqs.append(r)
    if args.only:
        keep = set(args.only.split(","))
        reqs = [r for r in reqs if r["model"] in keep]
    if args.limit:
        per = collections.Counter()
        out = []
        for r in sorted(reqs, key=lambda r: (r["model"], r["mode"], not r["think_closed"], r["request_id"])):
            k = (r["model"], r["mode"], r["think_closed"])
            if per[k] < args.limit:
                per[k] += 1
                out.append(r)
        reqs = out
    os.makedirs(args.base, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(reqs), P["requests"], compression="zstd")
    if skipped:
        pq.write_table(pa.Table.from_pylist(skipped), os.path.join(args.base, "skipped.parquet"), compression="zstd")
    print("not continued: {} ({})".format(len(skipped), dict(collections.Counter(r["skip_reason"] for r in skipped))))
    work = collections.Counter()
    for r in reqs:
        work[r["model"]] += (r["new_max_tokens"] - r["completion_tokens"]) + (r["prompt_tokens"] + r["completion_tokens"]) / 20
    total = sum(work.values())
    target = total / (args.gcds * args.units_per_gcd)
    units = []
    for model, w in work.items():
        n = max(1, round(w / target))
        for k in range(n):
            units.append({"unit": "{}__{:03d}of{:03d}".format(model, k, n), "model": model, "shard": k, "num_shards": n, "est": w / n})
    units.sort(key=lambda u: -u["est"])
    json.dump({"units": units, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "gcds": args.gcds, "max_new": args.max_new},
              open(P["plan"], "w"), indent=1)
    up = sum(r["new_max_tokens"] - r["completion_tokens"] for r in reqs)
    print("{} requests (of {} truncated), at most {:.1f}M new tokens, {} units -> {}".format(
        len(reqs), len(rows), up / 1e6, len(units), P["plan"]))
    for model, w in sorted(work.items(), key=lambda x: -x[1]):
        print("  {:36s} {:5.1f}% of the work, {} units".format(model, 100 * w / total, sum(u["model"] == model for u in units)))


# ------------------------------------------------------------------ work (GPU)
def out_schema():
    import pyarrow as pa
    s = S.out_schema()
    for f, t in [("cont_seed", pa.int64()), ("cont_tokens", pa.int32()), ("prefix_tokens", pa.int32()),
                 ("orig_completion_tokens", pa.int32()), ("orig_max_tokens", pa.int32())]:
        s = s.append(pa.field(f, t))
    return s


def done_ids(samples, model, unit):
    import pyarrow.parquet as pq
    ids = set()
    for f in glob.glob(os.path.join(samples, model, unit + "-*.parquet")):
        ids.update(pq.read_table(f, columns=["request_id"]).column("request_id").to_pylist())
    return ids


def write(rows, samples, model, unit):
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = os.path.join(samples, model)
    os.makedirs(d, exist_ok=True)
    prefix = "{}-{}".format(unit, S.worker_tag())
    k = len(glob.glob(os.path.join(d, prefix + "-*.parquet")))
    path = os.path.join(d, "{}-{:05d}.parquet".format(prefix, k))
    pq.write_table(pa.Table.from_pylist(rows, schema=out_schema()), path + ".tmp")
    os.replace(path + ".tmp", path)


def kv_budget(engine):
    """Tokens the KV cache holds (the engine core reports it to the front end at start-up), 5% kept free;
    None if the engine does not say."""
    cc = engine.vllm_config.cache_config
    n = getattr(cc, "kv_cache_size_tokens", None) or (getattr(cc, "num_gpu_blocks", None) or 0) * (getattr(cc, "block_size", None) or 0)
    return int(0.95 * n) if n else None


def run_unit(u, eng, p, models, prompts, reqs, samples, args, max_new):
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    name, m = u["model"], models[u["model"]]
    mine = [r for r in reqs[name] if S.shard_of(r["request_id"], u["num_shards"]) == u["shard"]]
    done = done_ids(samples, name, u["unit"])
    todo = [r for r in mine if r["request_id"] not in done]
    print("{}: {} requests, {} done before".format(u["unit"], len(mine), len(mine) - len(todo)), flush=True)
    if not todo:
        return True
    llm, tok, stops, turn = eng.get(name, m)
    engine = llm.llm_engine
    # Admission control: a request enters the engine only when its largest possible size (prompt, prefix and
    # all new tokens) fits into the KV cache beside the requests already running. Without it, vLLM admits
    # far more of these long continuations than the cache can grow into, then preempts and recomputes them
    # over and over (the first run reached about 60 tokens/s per GCD).
    budget = kv_budget(engine)
    print("{}: KV cache budget {} tokens".format(u["unit"], budget), flush=True)
    sp_base = dict(p["defaults"]["sampling"])
    sp_base.update({k: v for k, v in (m.get("sampling") or {}).items() if k != "source"})
    todo.sort(key=lambda r: -(r["new_max_tokens"] - r["completion_tokens"]))
    pending = collections.deque(todo)
    waiting, info, buf, alias = set(), {}, [], {}
    head, load = [], {}     # head: the next request, rebuilt, waiting for room; load: request id -> tokens reserved
    n_done, toks_out, t0, last = 0, 0, time.time(), time.time()

    def feed():
        while len(waiting) < args.window:
            if not head:
                if not pending:
                    return
                r = pending.popleft()
                p_ids, gen, _ = rebuild(r, m, tok, prompts[r["id"]])
                left = min(r["new_max_tokens"] - r["completion_tokens"], m["max_model_len"] - len(p_ids) - len(gen))
                if max_new:
                    left = min(left, max_new)
                if left < 1:                               # no room left in the context: kept as it was
                    buf.append({k: r[k] for k in ("request_id", "id", "slot", "mode", "reasoning", "response", "finish_reason",
                                                  "think_closed", "leaks", "prompt_tokens", "completion_tokens", "max_tokens",
                                                  "seed", "cont_seed")} | {"model": name, "cont_tokens": 0, "prefix_tokens": len(gen),
                                                                             "orig_completion_tokens": r["completion_tokens"],
                                                                             "orig_max_tokens": r["max_tokens"]})
                    continue
                head.append((r, p_ids, gen, left))
            r, p_ids, gen, left = head[0]
            need = len(p_ids) + len(gen) + left
            if budget and waiting and sum(load.values()) + need > budget:
                return
            head.clear()
            sp = SamplingParams(max_tokens=left, seed=r["cont_seed"], stop_token_ids=sorted(turn),
                                output_kind=RequestOutputKind.FINAL_ONLY, **sp_base)
            x = engine.add_request(r["request_id"], TokensPrompt(prompt_token_ids=p_ids + gen), sp)
            if x is not None and x != r["request_id"]:
                alias[x] = r["request_id"]
            waiting.add(r["request_id"])
            info[r["request_id"]] = (r, gen, len(p_ids))
            load[r["request_id"]] = need

    try:
        while True:
            feed()
            if not waiting and not engine.has_unfinished_requests():
                break
            for o in engine.step():
                if not o.finished:
                    continue
                req = o.request_id if o.request_id in waiting else alias.pop(o.request_id, None)
                if req not in waiting:
                    raise RuntimeError("unexpected output id {!r}".format(o.request_id))
                waiting.discard(req)
                load.pop(req, None)
                r, gen, n_prompt = info.pop(req)
                c = o.outputs[0]
                reasoning, answer, closed, leaks = assemble_output(r, m, tok, stops, gen, c.token_ids, c.finish_reason)
                buf.append({"request_id": req, "id": r["id"], "slot": r["slot"], "model": name, "mode": r["mode"],
                            "reasoning": reasoning, "response": answer, "finish_reason": c.finish_reason,
                            "think_closed": closed, "leaks": leaks, "prompt_tokens": r["prompt_tokens"],
                            "completion_tokens": r["completion_tokens"] + len(c.token_ids), "max_tokens": r["new_max_tokens"],
                            "seed": r["seed"], "cont_seed": r["cont_seed"], "cont_tokens": len(c.token_ids),
                            "prefix_tokens": len(gen), "orig_completion_tokens": r["completion_tokens"],
                            "orig_max_tokens": r["max_tokens"]})
                toks_out += len(c.token_ids)
            if buf and (len(buf) >= args.flush_every or time.time() - last >= args.flush_secs):
                write(buf, samples, name, u["unit"])
                n_done += len(buf)
                buf, last = [], time.time()
                el = time.time() - t0
                print("{}: {}/{} done, {:.0f}s, out {:.0f} tok/s".format(u["unit"], n_done, len(todo), el, toks_out / el), flush=True)
    finally:
        if buf:
            write(buf, samples, name, u["unit"])
            n_done += len(buf)
    print("{}: finished {} in {:.0f}s, out {:.0f} tok/s".format(u["unit"], n_done, time.time() - t0,
                                                              toks_out / max(1, time.time() - t0)), flush=True)
    return len(done_ids(samples, name, u["unit"])) >= len(mine)


def work(args):
    import pyarrow.parquet as pq
    P = paths(args.base)
    p, models = pool()
    meta = json.load(open(P["plan"]))
    units = meta["units"]
    job = os.environ.get("SLURM_JOB_ID", "local")
    claims = os.path.join(P["samples"], "_claims", job)
    done_dir = os.path.join(P["samples"], "_done")
    os.makedirs(claims, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)
    reqs = collections.defaultdict(list)
    for r in pq.read_table(P["requests"]).to_pylist():
        reqs[r["model"]].append(r)
    prompts = load_prompts({r["id"] for rs in reqs.values() for r in rs})
    eng = S.Engine()
    me = "{}:{}".format(socket.gethostname(), os.environ.get("HIP_VISIBLE_DEVICES", "?"))
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    signal.signal(signal.SIGTERM, S.on_term)
    while True:
        if S.STOPPING:
            print(me, "SIGTERM: exiting", flush=True)
            os._exit(0)
        if deadline and time.time() > deadline:
            print(me, "stopping: time budget used", flush=True)
            break
        order = sorted(units, key=lambda u: (u["model"] != eng.name, -u["est"]))
        unit = None
        for u in order:
            if os.path.exists(os.path.join(done_dir, u["unit"])):
                continue
            try:
                os.mkdir(os.path.join(claims, u["unit"]))
            except FileExistsError:
                continue
            unit = u
            break
        if unit is None:
            print(me, "no units left", flush=True)
            break
        print(me, "claimed", unit["unit"], flush=True)
        try:
            complete = run_unit(unit, eng, p, models, prompts, reqs, P["samples"], args, meta.get("max_new"))
        except S.Stop:
            print(me, "SIGTERM: outputs flushed, exiting", flush=True)
            os._exit(0)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(me, "FAILED", unit["unit"], repr(e)[:300], flush=True)
            eng.close()
            continue
        if complete:
            open(os.path.join(done_dir, unit["unit"]), "w").write(me + "\n")
    eng.close()


# ------------------------------------------------------------------ status / assemble (CPU)
def load_outputs(samples):
    import pyarrow.parquet as pq
    rows = {}
    for f in sorted(glob.glob(os.path.join(samples, "*", "*.parquet"))):
        for r in pq.read_table(f).to_pylist():
            rows.setdefault(r["request_id"], r)
    return rows


def status(args):
    import pyarrow.parquet as pq
    P = paths(args.base)
    reqs = pq.read_table(P["requests"], columns=["request_id", "model", "mode"]).to_pylist()
    out = load_outputs(P["samples"])
    by = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for r in reqs:
        x = by[(r["model"], r["mode"])]
        x[0] += 1
        o = out.get(r["request_id"])
        if o:
            x[1] += 1
            x[2][o["finish_reason"]] += 1
    print("{}/{} continued".format(sum(1 for r in reqs if r["request_id"] in out), len(reqs)))
    for (model, mode), (n, d, fin) in sorted(by.items()):
        print("  {:36s} {:6s} {:5d}/{:5d}  {}".format(model, mode, d, n, dict(fin)))


def assemble(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    P = paths(args.base)
    reqs = pq.read_table(P["requests"], columns=["request_id"]).column("request_id").to_pylist()
    out = load_outputs(P["samples"])
    missing = [x for x in reqs if x not in out]
    assert not missing, "{} continuations missing, e.g. {}".format(len(missing), missing[:5])
    path = os.path.join(RESP, "responses.parquet")
    backup = os.path.join(RESP, "responses.before_continuation.parquet")
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    t = pq.read_table(backup)
    rows = t.to_pylist()
    n = 0
    for r in rows:
        o = out.get(r["request_id"])
        r["continued"], r["cont_seed"], r["cont_tokens"] = o is not None, None, None
        if o is None:
            continue
        for k in ("reasoning", "response", "finish_reason", "think_closed", "leaks", "completion_tokens", "max_tokens"):
            r[k] = o[k]
        r["cont_seed"], r["cont_tokens"] = o["cont_seed"], o["cont_tokens"]
        n += 1
    schema = t.schema
    for f, ty in [("continued", pa.bool_()), ("cont_seed", pa.int64()), ("cont_tokens", pa.int32())]:
        if f not in schema.names:
            schema = schema.append(pa.field(f, ty))
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path + ".tmp", compression="zstd")
    os.replace(path + ".tmp", path)
    fin = collections.Counter(r["finish_reason"] for r in rows)
    print("{} of {} responses continued -> {} (previous file: {}); finish reasons now {}".format(
        n, len(rows), path, backup, dict(fin)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "plan", "work", "status", "assemble"])
    ap.add_argument("--base", default=CONT, help="output directory (relative to the repository root), e.g. for a test")
    ap.add_argument("--gcds", type=int, default=32)
    ap.add_argument("--units-per-gcd", type=int, default=3)
    ap.add_argument("--only", help="plan: only these models (comma-separated), for a test")
    ap.add_argument("--limit", type=int, help="plan: at most N requests per model, mode and thinking state, for a test")
    ap.add_argument("--max-new", type=int, help="plan: cap the new tokens per request, for a test")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--flush-secs", type=float, default=300.0)
    ap.add_argument("--max-hours", type=float)
    args = ap.parse_args()
    args.base = os.path.join(ROOT, args.base)
    sys.exit({"check": check, "plan": plan, "work": work, "status": status, "assemble": assemble}[args.cmd](args) or 0)


if __name__ == "__main__":
    main()
