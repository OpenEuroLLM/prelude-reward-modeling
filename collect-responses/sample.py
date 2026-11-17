#!/usr/bin/env python3
"""Generate the responses assigned in data/responses/assignment.parquet (assign.py).

  collect-responses/run.sh sample.py plan [--gcds 32]   # CPU: split each model's requests into work units
  sbatch -N <nodes> collect-responses/slurm/sample.sbatch  # GPU: one `work` process per GCD (sample_node.sh)
  collect-responses/run.sh sample.py status             # CPU: progress per model
  collect-responses/run.sh sample.py assemble           # CPU: -> data/responses/responses.parquet

Units: each model's requests are split by request-id hash into shards of about equal expected work
(requests x mean completion tokens of that model and mode in the smoke test). Workers claim units
dynamically (atomic mkdir in samples/_claims/<job id>/), heaviest first and preferring units of the model
they already have loaded. Outputs are flushed to samples/<model>/<unit>-NNNNN.parquet every few hundred
rows (and on SIGTERM, which Slurm sends 30 s before killing a job at its time limit), so a job that hits its
time limit is simply resubmitted: finished requests are skipped and a unit is marked done
(samples/_done/<unit>) only when all its requests have an output. Output files are named per job and GCD,
so even overlapping jobs never overwrite each other's outputs (duplicates are dropped when reading).

Each request is rendered exactly as in the smoke test (smoke_test.prepare_messages / render: the model's
chat template, its mode's template arguments, the Prelude default_system where the template would inject
its own identity, system messages merged for templates that reject them) and sampled with the pool's
settings (temperature 1.0, top-p 1.0) and the request's own seed. The reasoning is split from the answer at
the mode's think_end token.
"""
import argparse
import collections
import gc
import glob
import hashlib
import json
import os
import signal
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from smoke_test import pool, prepare_messages, render, split_output, stop_ids  # noqa: E402

RESP = os.path.join(ROOT, "data", "responses")
SAMPLES = os.path.join(RESP, "samples")
PLAN = os.path.join(SAMPLES, "plan.json")


class Stop(BaseException):
    """raised by the SIGTERM handler: flush finished outputs and exit"""


STOPPING = []


def on_term(signum, frame):
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    STOPPING.append(time.time())
    raise Stop()


def worker_tag():
    return "{}.{}.g{}".format(os.environ.get("SLURM_JOB_ID", "local"), socket.gethostname(),
                              os.environ.get("HIP_VISIBLE_DEVICES", "x"))


def rid(r):
    return "{}#{}".format(r["id"], r["slot"])


def shard_of(request_id, n):
    return int(hashlib.md5(request_id.encode()).hexdigest(), 16) % n


def out_schema():
    import pyarrow as pa
    return pa.schema([("request_id", pa.string()), ("id", pa.string()), ("slot", pa.int8()), ("model", pa.string()),
                      ("mode", pa.string()), ("reasoning", pa.string()), ("response", pa.string()),
                      ("finish_reason", pa.string()), ("think_closed", pa.bool_()), ("leaks", pa.list_(pa.string())),
                      ("prompt_tokens", pa.int32()), ("completion_tokens", pa.int32()), ("max_tokens", pa.int32()),
                      ("seed", pa.int64())])


# ------------------------------------------------------------------ plan (CPU)
def smoke_tokens():
    """mean completion tokens per (model, mode) on the smoke test's general prompts"""
    out = {}
    for p in glob.glob(os.path.join(ROOT, "data", "responses", "smoke", "*.jsonl")):
        name = os.path.basename(p)[:-6]
        if "@" in name or name == "prompts":
            continue
        by = collections.defaultdict(list)
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            by[r["mode"]].append(r["completion_tokens"])
        for mode, xs in by.items():
            out[(name, mode)] = sum(xs) / len(xs)
    return out


def plan(args):
    import pyarrow.parquet as pq
    a = pq.read_table(os.path.join(RESP, "assignment.parquet")).to_pylist()
    est = smoke_tokens()
    work = collections.Counter()
    for r in a:
        work[r["model"]] += est.get((r["model"], r["mode"]), 1000) + r["prompt_tokens"] / 20
    total = sum(work.values())
    target = total / (args.gcds * args.units_per_gcd)
    units = []
    for model, w in work.items():
        n = max(1, round(w / target))
        for k in range(n):
            units.append({"unit": "{}__{:03d}of{:03d}".format(model, k, n), "model": model, "shard": k,
                          "num_shards": n, "est": w / n})
    units.sort(key=lambda u: -u["est"])
    os.makedirs(SAMPLES, exist_ok=True)
    json.dump({"units": units, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "gcds": args.gcds}, open(PLAN, "w"), indent=1)
    print("{} requests, {} units (target {:.2e} est. tokens each) -> {}".format(len(a), len(units), target, PLAN))
    for model, w in sorted(work.items(), key=lambda x: -x[1]):
        print("  {:36s} {:5.1f}% of the work, {} units".format(model, 100 * w / total,
                                                             sum(u["model"] == model for u in units)))


# ------------------------------------------------------------------ work (GPU)
def done_ids(model, unit):
    import pyarrow.parquet as pq
    ids = set()
    for p in glob.glob(os.path.join(SAMPLES, model, unit + "-*.parquet")):
        ids.update(pq.read_table(p, columns=["request_id"]).column("request_id").to_pylist())
    return ids


def write(rows, model, unit):
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = os.path.join(SAMPLES, model)
    os.makedirs(d, exist_ok=True)
    prefix = "{}-{}".format(unit, worker_tag())          # only this worker writes files with this prefix
    k = len(glob.glob(os.path.join(d, prefix + "-*.parquet")))
    path = os.path.join(d, "{}-{:05d}.parquet".format(prefix, k))
    pq.write_table(pa.Table.from_pylist(rows, schema=out_schema()), path + ".tmp")
    os.replace(path + ".tmp", path)


class Engine:
    """one loaded model; reloaded only when the next unit needs another model"""

    def __init__(self):
        self.name, self.llm = None, None

    def get(self, name, m):
        if self.name == name:
            return self.llm, self.tok, self.stops, self.turn
        self.close()
        from vllm import LLM
        path = os.path.join(ROOT, "data", "models", name)
        kw = dict(model=path, tokenizer=path, dtype="bfloat16", max_model_len=m["max_model_len"], tensor_parallel_size=1,
                  gpu_memory_utilization=0.88, seed=0, max_num_seqs=256)
        kw.update(m.get("vllm") or {})
        t0 = time.time()
        self.llm = LLM(**kw)
        signal.signal(signal.SIGTERM, on_term)           # in case vLLM installed its own handler
        self.tok = self.llm.get_tokenizer()
        self.stops, self.turn = stop_ids(self.tok, path)
        self.name = name
        print("{} loaded in {:.0f}s".format(name, time.time() - t0), flush=True)
        return self.llm, self.tok, self.stops, self.turn

    def close(self):
        if self.llm is not None:
            del self.llm
            self.llm = None
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        self.name = None


def run_unit(u, eng, p, models, prompts, assignment, args):
    from vllm import SamplingParams, TokensPrompt
    from vllm.sampling_params import RequestOutputKind
    name, m = u["model"], models[u["model"]]
    mine = [r for r in assignment[name] if shard_of(rid(r), u["num_shards"]) == u["shard"]]
    done = done_ids(name, u["unit"])
    todo = [r for r in mine if rid(r) not in done]
    print("{}: {} requests, {} done before".format(u["unit"], len(mine), len(mine) - len(todo)), flush=True)
    if not todo:
        return True
    llm, tok, stops, turn = eng.get(name, m)
    engine = llm.llm_engine
    sp_base = dict(p["defaults"]["sampling"])
    sp_base.update({k: v for k, v in (m.get("sampling") or {}).items() if k != "source"})
    todo.sort(key=lambda r: -r["max_tokens"])      # long budgets first: the tail starts early
    pending = iter(todo)
    waiting, info, buf, alias = set(), {}, [], {}
    n_done, toks_out, t0, last = 0, 0, time.time(), time.time()

    def feed():
        while len(waiting) < args.window:
            r = next(pending, None)
            if r is None:
                return
            req = rid(r)
            ids, _ = render(tok, prepare_messages(prompts[r["id"]], m), m["modes"][r["mode"]].get("template_kwargs") or {})
            mt = min(r["max_tokens"], m["max_model_len"] - len(ids))
            if mt < 256:
                buf.append({"request_id": req, "id": r["id"], "slot": r["slot"], "model": name, "mode": r["mode"],
                            "reasoning": "", "response": "", "finish_reason": "too_long", "think_closed": None,
                            "leaks": [], "prompt_tokens": len(ids), "completion_tokens": 0, "max_tokens": 0, "seed": r["seed"]})
                continue
            sp = SamplingParams(max_tokens=mt, seed=r["seed"], stop_token_ids=sorted(turn),
                                output_kind=RequestOutputKind.FINAL_ONLY, **sp_base)
            x = engine.add_request(req, TokensPrompt(prompt_token_ids=ids), sp)
            if x is not None and x != req:         # engine-internal id (as in llm_runner.py)
                alias[x] = req
            waiting.add(req)
            info[req] = (r, mt)

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
                r, mt = info.pop(req)
                c = o.outputs[0]
                think_end = m["modes"][r["mode"]].get("think_end") if r["mode"] == "think" else None
                reasoning, answer, closed, leaks = split_output(tok, list(c.token_ids), c.finish_reason, think_end, stops)
                buf.append({"request_id": req, "id": r["id"], "slot": r["slot"], "model": name, "mode": r["mode"],
                            "reasoning": reasoning, "response": answer, "finish_reason": c.finish_reason,
                            "think_closed": closed, "leaks": leaks, "prompt_tokens": len(o.prompt_token_ids),
                            "completion_tokens": len(c.token_ids), "max_tokens": mt, "seed": r["seed"]})
                toks_out += len(c.token_ids)
            if buf and (len(buf) >= args.flush_every or time.time() - last >= args.flush_secs):
                write(buf, name, u["unit"])
                n_done += len(buf)
                buf, last = [], time.time()
                el = time.time() - t0
                print("{}: {}/{} done, {:.0f}s, out {:.0f} tok/s".format(u["unit"], n_done, len(todo), el, toks_out / el),
                      flush=True)
    finally:                                          # also on SIGTERM (Stop) or an engine error
        if buf:
            write(buf, name, u["unit"])
            n_done += len(buf)
            buf = []
    print("{}: finished {} in {:.0f}s, out {:.0f} tok/s".format(u["unit"], n_done, time.time() - t0,
                                                              toks_out / max(1, time.time() - t0)), flush=True)
    return len(done_ids(name, u["unit"])) >= len(mine)


def work(args):
    import pyarrow.parquet as pq
    p, models = pool()
    units = json.load(open(PLAN))["units"]
    job = os.environ.get("SLURM_JOB_ID", "local")
    claims = os.path.join(SAMPLES, "_claims", job)
    done_dir = os.path.join(SAMPLES, "_done")
    os.makedirs(claims, exist_ok=True)
    os.makedirs(done_dir, exist_ok=True)
    assignment = collections.defaultdict(list)
    for r in pq.read_table(os.path.join(RESP, "assignment.parquet")).to_pylist():
        assignment[r["model"]].append(r)
    wanted = {r["id"] for rs in assignment.values() for r in rs}
    prompts = {}
    with open(os.path.join(ROOT, "data", "compiled", "prompts.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o["id"] in wanted:
                prompts[o["id"]] = {"id": o["id"], "messages": o["messages"]}
    eng = Engine()
    me = "{}:{}".format(socket.gethostname(), os.environ.get("HIP_VISIBLE_DEVICES", "?"))
    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    signal.signal(signal.SIGTERM, on_term)
    while True:
        if STOPPING:                                  # SIGTERM arrived while handling an engine error
            print(me, "SIGTERM: exiting", flush=True)
            os._exit(0)
        if deadline and time.time() > deadline:
            print(me, "stopping: time budget used", flush=True)
            break
        order = sorted(units, key=lambda u: (u["model"] != eng.name, -u["est"]))   # loaded model first
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
            complete = run_unit(unit, eng, p, models, prompts, assignment, args)
        except Stop:
            print(me, "SIGTERM: outputs flushed, exiting", flush=True)
            os._exit(0)                               # no engine shutdown: Slurm kills the job in 30 s
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
def load_outputs(columns=None):
    """request_id -> output row (a request written twice, e.g. by overlapping jobs, keeps one copy)"""
    import pyarrow.parquet as pq
    rows = {}
    for p in sorted(glob.glob(os.path.join(SAMPLES, "*", "*.parquet"))):
        for r in pq.read_table(p, columns=columns).to_pylist():
            rows.setdefault(r["request_id"], r)
    return rows


def status(args):
    import pyarrow.parquet as pq
    a = pq.read_table(os.path.join(RESP, "assignment.parquet")).to_pylist()
    out = load_outputs(["request_id", "completion_tokens", "finish_reason"])
    units = json.load(open(PLAN))["units"] if os.path.exists(PLAN) else []
    done_units = set(os.listdir(os.path.join(SAMPLES, "_done"))) if os.path.isdir(os.path.join(SAMPLES, "_done")) else set()
    by = collections.defaultdict(lambda: [0, 0, 0])
    for r in a:
        x = by[(r["model"], r["mode"])]
        x[0] += 1
        o = out.get(rid(r))
        if o:
            x[1] += 1
            x[2] += o["completion_tokens"]
    tot, got = len(a), sum(1 for r in a if rid(r) in out)
    print("{}/{} responses ({:.1%}); units done {}/{}".format(got, tot, got / tot, len(done_units), len(units)))
    for (model, mode), (n, d, t) in sorted(by.items()):
        print("  {:36s} {:6s} {:6d}/{:6d}  mean {:5.0f} tokens".format(model, mode, d, n, t / max(1, d)))
    fin = collections.Counter(o["finish_reason"] for o in out.values())
    print("finish reasons:", dict(fin))


def assemble(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    p, models = pool()
    a = pq.read_table(os.path.join(RESP, "assignment.parquet")).to_pylist()
    out = load_outputs()
    missing = [rid(r) for r in a if rid(r) not in out]
    assert not missing, "{} responses missing, e.g. {}".format(len(missing), missing[:5])
    rows = []
    for r in a:
        o = dict(out[rid(r)])
        o["model_hf"], o["model_revision"] = models[r["model"]]["hf"], models[r["model"]]["revision"]
        o["model_group"] = models[r["model"]]["group"]
        o["temperature"], o["top_p"] = p["defaults"]["sampling"]["temperature"], p["defaults"]["sampling"]["top_p"]
        rows.append(o)
    schema = out_schema()
    for field, typ in [("model_hf", pa.string()), ("model_revision", pa.string()), ("model_group", pa.string()),
                       ("temperature", pa.float64()), ("top_p", pa.float64())]:
        schema = schema.append(pa.field(field, typ))
    path = os.path.join(RESP, "responses.parquet")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")
    print("{} responses for {} prompts -> {}".format(len(rows), len({r['id'] for r in rows}), path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "work", "status", "assemble"])
    ap.add_argument("--gcds", type=int, default=32, help="plan: GCDs the units are sized for")
    ap.add_argument("--units-per-gcd", type=int, default=4, help="plan: units per GCD (smaller units, better balance)")
    ap.add_argument("--window", type=int, default=512, help="work: requests queued in the engine at a time")
    ap.add_argument("--flush-every", type=int, default=300)
    ap.add_argument("--flush-secs", type=float, default=300.0)
    ap.add_argument("--max-hours", type=float, help="work: claim no new unit after this many hours")
    ap.add_argument("--dir", default=os.environ.get("SAMPLE_DIR"),
                    help="responses directory (default data/responses; env SAMPLE_DIR; relative to the repository "
                         "root), e.g. for a test run")
    args = ap.parse_args()
    global RESP, SAMPLES, PLAN
    if args.dir:
        RESP = os.path.join(ROOT, args.dir)               # relative paths are relative to the repository root
        SAMPLES = os.path.join(RESP, "samples")
        PLAN = os.path.join(SAMPLES, "plan.json")
    {"plan": plan, "work": work, "status": status, "assemble": assemble}[args.cmd](args)


if __name__ == "__main__":
    main()
