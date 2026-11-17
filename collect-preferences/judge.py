#!/usr/bin/env python3
"""Run one judge model over a pairs file with vLLM: every pair in both presentation orders, forced binary
verdict, JSON output.

  collect-preferences/run_gpu.sh judge.py --judge qwen3.5-27b --pairs data/preferences/pilot/pairs.parquet \
      --out data/preferences/pilot/v3/votes [--thinking on|off] [--shard 0/4] [--tp 2] [--limit N]

One process per engine replica; `--shard i/n` takes every n-th pair so that several replicas (e.g. four TP=2
replicas on a node) share the work. Writes <out>/<judge>__<prompt_version>__<thinking>__<shard>.parquet with
one row per call: the raw output, the parsed analysis fields, the verdict (A/B) mapped back to x/y, the
probabilities of the two verdict tokens where available, token counts, the decoding settings and seed, and
status (ok: JSON parsed; ok_regex: only the verdict could be recovered from the answer text; parse_error;
too_long). `--reparse` re-extracts everything from the stored raw outputs after a parser fix, without
rerunning the model. `--skip-own-family` leaves out, before the sharding, every pair with an answer from the
judge's family (families.py), as in the full run; `--dry-run` prints the counts without loading the model.

Thinking is on by default for every judge that has a thinking mode; gpt-oss always reasons, and its `on` is
reasoning effort high, `off` medium. Decoding is the vendor's recommendation for the judge and mode
(judges.yaml `sampling`), never greedy for a model that reasons, with a seed per call derived from (pair,
order, attempt). A call that gives no verdict, almost always a reasoning that ran out of budget, is retried
once with a new seed and twice the budget (capped by the context); the row records the attempt and the first
attempt's status.

Requests stream through vLLM's engine (continuous batching, so one long call does not hold up a batch).
Finished calls are written every 200 calls or 5 minutes and on SIGTERM; a rerun with the same arguments skips
the calls already in the file. With thinking off the JSON schema is enforced by vLLM's structured outputs
(not for gpt-oss, whose analysis channel precedes the final answer). With thinking on no grammar is applied (from
the first token it would suppress the thinking), so the JSON is free-form: 0.25% of the full run's answers were
not valid JSON (their verdict was still unambiguous, status ok_regex; export_jsonl.py re-parses them). A future
run should constrain only the final answer (vLLM structured outputs together with a reasoning parser). With
thinking on the answer is read after the end of the reasoning; an output without the end-of-reasoning marker counts only if it stopped right after
its JSON object (GLM-4.5-Air often answers inside an unclosed think block), never when it hit the budget.
Prompts are rendered with the tokenizer (chat template plus the judge's thinking switch) and sent as token
ids; a prompt that leaves less than the budget within the judge's context is recorded as too_long.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import re
import signal
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from judge_prompt import PROMPT_VERSION, PROMPTS, SCHEMA, build, extract_span  # noqa: E402
from families import own_family  # noqa: E402

MAX_CHARS = 120_000       # conversation + both answers; longer pairs are recorded as too_long (very few)
DEFAULT_BUDGET = 16384    # new tokens per call (judges.yaml max_tokens overrides); the retry gets twice this
COLUMNS = pa.schema([("pair_id", pa.string()), ("order", pa.int8()), ("judge", pa.string()), ("judge_revision", pa.string()),
                     ("prompt_version", pa.string()), ("thinking", pa.string()), ("language", pa.string()),
                     ("status", pa.string()), ("verdict", pa.string()), ("prefers", pa.string()),
                     ("p_a", pa.float64()), ("p_b", pa.float64()), ("request", pa.string()), ("response_a", pa.string()),
                     ("response_b", pa.string()), ("comparison", pa.string()), ("raw", pa.string()),
                     ("finish_reason", pa.string()), ("prompt_tokens", pa.int32()), ("completion_tokens", pa.int32()),
                     ("attempt", pa.int8()), ("first_status", pa.string()), ("first_completion_tokens", pa.int32()),
                     ("seed", pa.int64()), ("max_tokens", pa.int32()), ("sampling", pa.string())])
THINK_END = {"qwen": "</think>", "glm": "</think>", "gemma": "<channel|>", "cohere": "<|END_THINKING|>", "mistral": "[/THINK]",
             "apertus": "<|inner_suffix|>"}   # gpt-oss: its final channel header
FENCE = re.compile(r"^[\s`]*$")
STOPPING = False


def on_term(signum, frame):
    global STOPPING
    STOPPING = True


def load_judge(name):
    pool = yaml.safe_load(open(os.path.join(HERE, "judges.yaml"), encoding="utf-8"))
    for j in pool["judges"]:
        if j["name"] == name:
            return j
    raise SystemExit("unknown judge " + name)


def keyed(d):
    """judges.yaml thinking/sampling maps with their keys as "off"/"on" (YAML 1.1 reads bare off/on as booleans)"""
    return {("on" if k is True else "off" if k is False else str(k)): v for k, v in (d or {}).items()}


def call_seed(pair_id, order, attempt):
    return int(hashlib.md5("{}#{}#{}".format(pair_id, order, attempt).encode()).hexdigest()[:8], 16)


FINAL_RE = re.compile(r"<\|channel\|>final\s*(?:<\|constrain\|>\s*\w+\s*)?<\|message\|>")   # gpt-oss final channel header
VERDICT_RE = re.compile(r'"verdict"\s*:\s*"([AB])"')


def final_text(text, judge):
    """the answer part of a raw output: after the gpt-oss final channel header (which may carry a
    <|constrain|>json tag), or after a closing think tag"""
    if judge["family"] == "openai":
        ms = list(FINAL_RE.finditer(text))
        if ms:
            text = text[ms[-1].end():]
        return re.sub(r"<\|(return|end|call)\|>.*$", "", text, flags=re.S)
    for tag in ("</think>", "<channel|>", "<|END_THINKING|>", "[/THINK]", "<|inner_suffix|>"):
        k = text.rfind(tag)
        if k >= 0:
            return text[k + len(tag):]
    return text


def parse(raw, judge, finish=None, thinking="off"):
    """-> (status, obj, verdict). ok: the answer parsed as JSON; ok_regex: no valid JSON but a single verdict
    letter in the answer part; parse_error: nothing usable (e.g. cut off inside the reasoning)"""
    if judge["family"] == "openai" and not FINAL_RE.search(raw):
        return "parse_error", None, None                  # never reached its final channel
    end = THINK_END.get(judge["family"]) if thinking == "on" else None
    open_reasoning = (end is not None and end not in raw) or ("<think>" in raw and "</think>" not in raw)
    if open_reasoning and finish == "length":
        return "parse_error", None, None                  # cut off inside the reasoning: any JSON there is a draft
    final = final_text(raw, judge)
    obj, k = extract_span(final)
    if obj and (not open_reasoning or FENCE.match(final[k:])):
        return "ok", obj, obj["verdict"]
    if open_reasoning:
        return "parse_error", None, None                  # a draft inside reasoning that went on, or nothing
    found = set(VERDICT_RE.findall(final))
    if len(found) == 1:
        return "ok_regex", None, found.pop()
    return "parse_error", None, None


def verdict_probs(out, tok):
    """P(A), P(B) from the top log-probabilities at the verdict token, or (None, None)"""
    try:
        if not out.logprobs:
            return None, None
        pieces = []
        for i, tid in enumerate(out.token_ids):
            lp = out.logprobs[i].get(tid)
            pieces.append(lp.decoded_token if lp is not None and lp.decoded_token is not None else tok.decode([tid]))
        text = "".join(pieces)
        m = list(re.finditer(r'"verdict"\s*:\s*"', text))
        if not m:
            return None, None
        pos = m[-1].end()                     # index of the verdict letter in the text
        acc = 0
        for i, piece in enumerate(pieces):
            if acc + len(piece) > pos:
                probs = {"A": 0.0, "B": 0.0}
                for tid, lp in out.logprobs[i].items():
                    s = (lp.decoded_token or "").strip().strip('"')
                    if s in probs:
                        probs[s] += math.exp(lp.logprob)
                return probs["A"], probs["B"]
            acc += len(piece)
    except Exception:
        pass
    return None, None


def field(obj, key):
    """an analysis field as text (without schema enforcement a judge may nest objects or lists in it)"""
    v = (obj or {}).get(key)
    return v if v is None or isinstance(v, str) else json.dumps(v, ensure_ascii=False)


def prefers(verdict, order):
    return None if verdict is None else ("x" if (verdict == "A") == (order == 0) else "y")


def write(rows, path):
    pq.write_table(pa.Table.from_pylist(rows, schema=COLUMNS), path + ".tmp")
    os.replace(path + ".tmp", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--pairs", default=os.path.join(ROOT, "data", "preferences", "pilot", "pairs.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "preferences", "pilot", "v3", "votes"))
    ap.add_argument("--prompt", default=PROMPT_VERSION, choices=sorted(PROMPTS), help="judge prompt version (judge_prompt.PROMPTS)")
    ap.add_argument("--thinking", choices=["off", "on"], help="default: on if the judge has a thinking mode")
    ap.add_argument("--shard", default="0/1", help="i/n: this process takes pairs i, i+n, i+2n, ...")
    ap.add_argument("--tp", type=int, help="tensor parallel size (default: judges.yaml)")
    ap.add_argument("--max-tokens", type=int, help="budget of the first attempt (default: judges.yaml max_tokens or 16384)")
    ap.add_argument("--no-retry", action="store_true", help="do not retry calls without a verdict")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--window", type=int, default=256, help="calls in flight in the engine")
    ap.add_argument("--flush-every", type=int, default=200)
    ap.add_argument("--flush-secs", type=float, default=300.0)
    ap.add_argument("--skip-own-family", action="store_true",
                    help="leave out the pairs with an answer from the judge's family (families.py), before sharding")
    ap.add_argument("--dry-run", action="store_true", help="print what would be judged, then exit without loading the model")
    ap.add_argument("--reparse", action="store_true",
                    help="re-extract the verdicts from the stored raw outputs of the existing output file, then exit")
    args = ap.parse_args()
    args.pairs, args.out = os.path.join(ROOT, args.pairs), os.path.join(ROOT, args.out)   # relative to the repo root
    judge = load_judge(args.judge)
    switches = keyed(judge.get("thinking"))
    thinking = args.thinking or ("on" if "on" in switches else "off")
    sampling = keyed(judge.get("sampling")).get(thinking)
    if sampling is None:
        raise SystemExit("judges.yaml gives no sampling for {} with thinking {}".format(judge["name"], thinking))
    i_shard, n_shard = (int(x) for x in args.shard.split("/"))

    pairs = pq.read_table(args.pairs).to_pylist()
    n_all = len(pairs)
    if args.skip_own_family:
        pairs = [p for p in pairs if not own_family(judge["family"], p)]
    skipped = n_all - len(pairs)
    pairs = [p for k, p in enumerate(pairs) if k % n_shard == i_shard]
    if args.limit:
        pairs = pairs[:args.limit]
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "{}__{}__{}__{}of{}.parquet".format(args.judge, args.prompt, thinking, i_shard, n_shard))
    path = os.path.join(ROOT, "data", "models", judge["name"])
    info = os.path.join(path, "_download_info.json")
    revision = json.load(open(info))["revision"] if os.path.exists(info) else None

    rows = pq.read_table(out_path).to_pylist() if os.path.exists(out_path) else []       # resume
    if args.reparse:
        before = collections.Counter(r["status"] for r in rows)
        for r in rows:
            if r["status"] == "too_long" or not r.get("raw"):
                continue
            status, obj, verdict = parse(r["raw"], judge, r.get("finish_reason"), r["thinking"])
            r.update(status=status, verdict=verdict, request=field(obj, "request"), response_a=field(obj, "response_a"),
                     response_b=field(obj, "response_b"), comparison=field(obj, "comparison"), prefers=prefers(verdict, r["order"]))
        write(rows, out_path)
        print("{}: reparsed {}: {} -> {}".format(args.judge, out_path, dict(before), dict(collections.Counter(r["status"] for r in rows))))
        return
    done = {(r["pair_id"], r["order"]) for r in rows}
    jobs = []
    for p in pairs:
        n_chars = sum(len(m.get("content") or "") for m in json.loads(p["messages"])) + len(p["response_x"]) + len(p["response_y"])
        for order in (0, 1):
            if (p["id"], order) in done:
                continue
            base = {"pair_id": p["id"], "order": order, "judge": judge["name"], "judge_revision": revision,
                    "prompt_version": args.prompt, "thinking": thinking, "language": p["language"]}
            if n_chars > MAX_CHARS:
                rows.append({**base, "status": "too_long", "verdict": None, "prefers": None})
                continue
            system, user = build(p, order, args.prompt)
            jobs.append((base, [{"role": "system", "content": system}, {"role": "user", "content": user}]))
    budget = args.max_tokens or judge.get("max_tokens") or DEFAULT_BUDGET
    print("{}: {} pairs (shard {}{}), prompt {}, thinking {}, sampling {}, budget {}{}: {} calls to make, {} already in {}".format(
        args.judge, len(pairs), args.shard, "; {} own-family pairs of {} left out".format(skipped, n_all) if args.skip_own_family else "",
        args.prompt, thinking, sampling, budget, "" if args.no_retry else " (retry: 2x)", len(jobs), len(done), out_path), flush=True)
    if args.dry_run:
        return
    if not jobs:
        write(rows, out_path)
        return

    for k, v in (judge.get("env") or {}).items():          # e.g. allocator settings, before torch starts
        os.environ[k] = str(v)
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    kw = dict(model=path, tokenizer=path, dtype="bfloat16", tensor_parallel_size=args.tp or judge["tp"],
              max_model_len=judge["max_model_len"], gpu_memory_utilization=0.88, seed=0, max_num_seqs=128)
    kw.update(judge.get("vllm") or {})
    t0 = time.time()
    llm = LLM(**kw)
    signal.signal(signal.SIGTERM, on_term)                      # after vLLM, in case it installed its own handler
    tok = llm.get_tokenizer()
    engine = llm.llm_engine
    print("{} loaded in {:.0f}s".format(judge["name"], time.time() - t0), flush=True)

    thinking_kwargs = switches.get(thinking) or {}
    structured = thinking == "off" and judge["family"] != "openai"
    # Prompts are rendered here with the HF tokenizer (chat template plus the thinking switch) and sent as token
    # ids: the switch is applied verbatim and visible in the log (the first prompt's head and tail are printed),
    # and a prompt that leaves no room for the answer becomes a too_long row rather than a crash (Mistral
    # Large's 32k context). The response stage (collect-responses/sample.py) renders the same way.
    mistral = (judge.get("vllm") or {}).get("tokenizer_mode") == "mistral"
    if not mistral:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(path)

    def render(msgs):
        if mistral:
            return tok.apply_chat_template(msgs, add_generation_prompt=True, **thinking_kwargs)   # vLLM MistralTokenizer -> ids
        text = hf_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **thinking_kwargs)
        return hf_tok(text, add_special_tokens=False)["input_ids"]                   # the template has BOS already

    room = lambda n: judge["max_model_len"] - n - 8                                   # noqa: E731
    so = None
    if structured:
        try:
            from vllm.sampling_params import StructuredOutputsParams
            so = ("structured_outputs", StructuredOutputsParams(json=SCHEMA))
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams
            so = ("guided_decoding", GuidedDecodingParams(json=SCHEMA))
    sampling_txt = json.dumps(sampling, sort_keys=True)

    def params(base, attempt, n_prompt):
        seed = call_seed(base["pair_id"], base["order"], attempt)
        mt = min(budget * (2 if attempt else 1), room(n_prompt))
        sp_kw = dict(sampling, max_tokens=mt, seed=seed, logprobs=5, output_kind=RequestOutputKind.FINAL_ONLY,
                     skip_special_tokens=judge["family"] != "openai" and thinking == "off")   # keep the think tags
        if so:
            sp_kw[so[0]] = so[1]
        return SamplingParams(**sp_kw), seed, mt

    pending = collections.deque((base, msgs, None, 0, None) for base, msgs in jobs)
    inflight, alias, buf = {}, {}, []
    stats = collections.Counter()
    t0, last, n_tok = time.time(), time.time(), 0
    shown = False

    def flush():
        nonlocal buf, last
        if buf:
            rows.extend(buf)
            buf = []
            write(rows, out_path)
        last = time.time()

    def feed():
        nonlocal shown
        while pending and len(inflight) < args.window:
            base, msgs, ids, attempt, first = pending.popleft()
            if ids is None:
                ids = render(msgs)
                if room(len(ids)) < budget:
                    buf.append({**base, "status": "too_long", "verdict": None, "prefers": None, "prompt_tokens": len(ids)})
                    stats["too_long"] += 1
                    continue
            if not shown:
                print("rendered prompt head / tail: {} ... {}".format(repr(tok.decode(ids[:40])), repr(tok.decode(ids[-40:]))), flush=True)
                shown = True
            sp, seed, mt = params(base, attempt, len(ids))
            rid = "{}#{}#{}".format(base["pair_id"], base["order"], attempt)
            x = engine.add_request(rid, TokensPrompt(prompt_token_ids=ids), sp)
            if x is not None and x != rid:
                alias[x] = rid
            inflight[rid] = (base, msgs, ids, attempt, first, seed, mt)

    while True:
        if STOPPING:
            flush()
            print("{}: SIGTERM after {:.0f}s: {} calls written, {} in flight dropped (redone on resume)".format(
                judge["name"], time.time() - t0, len(rows), len(inflight)), flush=True)
            os._exit(0)
        feed()
        if not inflight and not engine.has_unfinished_requests():
            break
        for o in engine.step():
            if not o.finished:
                continue
            rid = o.request_id if o.request_id in inflight else alias.pop(o.request_id, None)
            if rid not in inflight:
                raise RuntimeError("unexpected output id {!r}".format(o.request_id))
            base, msgs, ids, attempt, first, seed, mt = inflight.pop(rid)
            c = o.outputs[0]
            n_tok += len(c.token_ids)
            status, obj, verdict = parse(c.text, judge, c.finish_reason, thinking)
            if verdict is None and attempt == 0 and not args.no_retry:
                pending.appendleft((base, msgs, ids, 1, (status, len(c.token_ids))))
                stats["retried"] += 1
                continue
            pa_, pb_ = verdict_probs(c, tok)
            buf.append({**base, "status": status, "verdict": verdict, "prefers": prefers(verdict, base["order"]),
                        "p_a": pa_, "p_b": pb_, "request": field(obj, "request"), "response_a": field(obj, "response_a"),
                        "response_b": field(obj, "response_b"), "comparison": field(obj, "comparison"), "raw": c.text,
                        "finish_reason": c.finish_reason, "prompt_tokens": len(ids), "completion_tokens": len(c.token_ids),
                        "attempt": attempt, "first_status": first[0] if first else None,
                        "first_completion_tokens": first[1] if first else None, "seed": seed, "max_tokens": mt,
                        "sampling": sampling_txt})
            stats[status] += 1
        if buf and (len(buf) >= args.flush_every or time.time() - last >= args.flush_secs):
            flush()
            el = time.time() - t0
            print("{}: {}/{} calls, {:.0f}s, {:.0f} out tok/s, {}".format(judge["name"], sum(stats.values()) - stats["retried"],
                                                                         len(jobs), el, n_tok / max(1, el), dict(stats)), flush=True)
    flush()
    print("{}: {} rows ({} ok, {} verdict by regex, {} parse errors, {} too long; {} first attempts retried), "
          "{} new in {:.0f}s, {:.0f} out tok/s -> {}".format(
              judge["name"], len(rows), sum(r["status"] == "ok" for r in rows), sum(r["status"] == "ok_regex" for r in rows),
              sum(r["status"] == "parse_error" for r in rows), sum(r["status"] == "too_long" for r in rows), stats["retried"],
              len(jobs), time.time() - t0, n_tok / max(1, time.time() - t0), out_path), flush=True)


if __name__ == "__main__":
    main()
