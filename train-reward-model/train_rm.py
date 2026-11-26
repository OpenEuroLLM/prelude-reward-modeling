#!/usr/bin/env python3
"""Train a scalar reward model on the preference pairs: FSDP2 over every GPU of the job, packed sequences.

  srun --ntasks-per-node=8 ... run.sh train_rm.py --model gemma-4-31B-it --run NAME [options]

Each pair gives two sequences, the conversation rendered with the backbone's chat template followed by one
answer and the end-of-turn token. The reward is a linear head on the last hidden state of that token. The
loss is Bradley–Terry on the reward difference against the label (prepare_data.py; `p_a` is the share of
judge votes for answer a):
  soft            -p log σ(ra - rb) - (1 - p) log σ(rb - ra)      (the vote share as the target)
  hard            the same with the majority as the target; even splits (p = 0.5) are left out
  unanimous       only pairs whose votes agree (p = 0 or 1)
  weighted        majority target, each pair weighted by |2p - 1|
plus, optionally, --center c · (ra + rb)²  (keeps the rewards centred at 0, as TRL's center_rewards_coefficient).

Long sequences: the prompt keeps its first 16 and its last tokens up to half of --max-len, the answer is cut to
what remains, and the end-of-turn token always closes the sequence. A step takes --batch pairs (shuffled with
--seed), spreads them over the ranks by token count and packs each rank's share into forward passes of at most
--micro-tokens tokens (a pair is never split). Every rank runs the same number of passes (FSDP requires it);
ranks with fewer pairs run a tiny pass with weight 0.

Writes data/reward_model/runs/<run>/: args.json, log.jsonl (one line per step), val_<step>.parquet (rewards of
the validation pairs), and with --save the final model (bf16, the original checkpoint's key names plus
score.weight / score.bias).
"""
import argparse
import datetime
import gc
import json
import math
import os
import re
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import rm_model  # noqa: E402
from rm_model import PACK, set_pack  # noqa: E402

DATA = os.path.join(ROOT, "data", "reward_model")
OPUS = os.path.join(ROOT, "data", "preferences", "pilot", "v3", "opus", "votes.parquet")
START = time.time()


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="backbone directory under data/models")
    ap.add_argument("--run", required=True, help="run name: data/reward_model/runs/<run>")
    ap.add_argument("--loss", default="soft", choices=["soft", "hard", "unanimous", "weighted"])
    ap.add_argument("--center", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1, help="cosine decay to this fraction of --lr")
    ap.add_argument("--warmup", type=float, default=0.03, help="fraction of the steps")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=128, help="pairs per optimizer step")
    ap.add_argument("--max-len", type=int, default=16384, help="tokens per sequence (conversation + answer)")
    ap.add_argument("--micro-tokens", type=int, default=16384, help="tokens per packed forward pass")
    ap.add_argument("--train-limit", type=int, default=0, help="use only the first N shuffled training pairs")
    ap.add_argument("--lora", type=int, default=0, help="LoRA rank (0: full fine-tuning)")
    ap.add_argument("--lora-alpha", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--betas", default="0.9,0.95")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=0, help="steps between validation passes (0: only at the end)")
    ap.add_argument("--eval-first", action="store_true", help="also evaluate before training")
    ap.add_argument("--save", action="store_true", help="write the final model")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (smoke tests); the LR schedule "
                    "still spans the whole run")
    ap.add_argument("--time-limit", type=float, default=0.0, help="minutes of wall time for the whole job; "
                    "training stops early enough to evaluate and save")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--freeze-embed", action="store_true", help="keep the input embeddings frozen")
    ap.add_argument("--freeze-norms", action="store_true", help="keep the RMSNorm weights frozen (Gemma 4 has "
                    "post-attention norm weights of ~1e-4 that Adam would change by percents per step)")
    ap.add_argument("--fp32-residual", action="store_true", help="residual stream, norms and layer scalars in fp32, "
                    "matrix multiplications and attention in bf16 (Gemma 4: its bf16 gradients are mostly rounding noise)")
    ap.add_argument("--mem-headroom", type=float, default=4.0, help="GiB of each GPU kept out of PyTorch's caching "
                    "allocator for the ROCm runtime (kernel scratch, RCCL); 0: no cap. Without it Gemma 4 reserved 62.8 "
                    "of 64 GiB and the flash-attention backward aborted with HSA_STATUS_ERROR_OUT_OF_RESOURCES")
    ap.add_argument("--diagnose-grads", type=int, default=0, help="run N steps in bf16 without updates and print "
                    "the largest per-parameter gradient norms, then exit")
    ap.add_argument("--profile-step", type=int, default=0, help="profile the K-th training step of this process "
                    "(torch.profiler kernel table of rank 0 to runs/<run>/profile_<step>.txt, per-rank timings to the log)")
    ap.add_argument("--grad-report", type=float, default=100.0, help="when the gradient norm exceeds this, log the "
                    "parameters, parameter types and layers with the largest gradients (0: never)")
    ap.add_argument("--ckpt-every", type=int, default=0, help="steps between resumable checkpoints (model + "
                    "optimizer, sharded, runs/<run>/ckpt; 0: none)")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<run>/ckpt if it exists")
    ap.add_argument("--score", default="", help="comma-separated parquet files (id, prompt_ids, resp_ids) to score "
                    "with the final model; rewards go to runs/<run>/scores_<name>.parquet")
    return ap.parse_args()


# ------------------------------------------------------------------------------------------------ data

class Pairs:
    """the tokenized pairs of one split, as Arrow buffers (no Python lists of tokens)"""

    def __init__(self, model, split, suffix):
        t = pq.read_table(os.path.join(DATA, "tok", model, split + ".parquet"))
        self.n = t.num_rows
        self.ids = t.column("id").to_pylist()
        self.p = np.asarray(t.column("p_a").to_numpy(), dtype=np.float64)
        self.language = t.column("language").to_pylist()
        self.cols = {}
        for c in ("prompt_ids", "a_ids", "b_ids"):
            arr = t.column(c).combine_chunks()
            self.cols[c] = (np.asarray(arr.offsets.to_numpy()), np.asarray(arr.values.to_numpy()))
        self.suffix = np.asarray(suffix, dtype=np.int32)

    def get(self, col, i):
        off, val = self.cols[col]
        return val[off[i]:off[i + 1]]

    def lengths(self, col):
        off = self.cols[col][0]
        return np.diff(off)


def make_seq(prompt, resp, suffix, max_len):
    budget = max_len - len(suffix)
    if len(prompt) + len(resp) <= budget:
        return np.concatenate([prompt, resp, suffix])
    cap = max(budget // 2, budget - len(resp))
    if len(prompt) > cap:
        prompt = np.concatenate([prompt[:16], prompt[len(prompt) - (cap - 16):]])
    resp = resp[:budget - len(prompt)]
    return np.concatenate([prompt, resp, suffix])


def pair_seqs(data, i, max_len):
    pr = data.get("prompt_ids", i)
    return (make_seq(pr, data.get("a_ids", i), data.suffix, max_len),
            make_seq(pr, data.get("b_ids", i), data.suffix, max_len))


def seq_lens(data, max_len):
    """per pair: the lengths of its two (truncated) sequences, without building them"""
    lp, la, lb = data.lengths("prompt_ids"), data.lengths("a_ids"), data.lengths("b_ids")
    s = len(data.suffix)
    return np.minimum(lp + la + s, max_len), np.minimum(lp + lb + s, max_len)


def seq_len_est(data, max_len):
    """per pair: total tokens of its two (truncated) sequences"""
    a, b = seq_lens(data, max_len)
    return a + b


def plan_step(pair_idx, tokens, lens_a, lens_b, world, micro_tokens, nograd_cost=0.25):
    """-> (plans, halves). plans[r] is rank r's list of micro-batches, the same number on every rank (empty ones
    are dummy passes). A micro-batch is a list of items: ("p", i) runs both sequences of pair i; a pair longer than
    micro_tokens is split into ("a", i) and ("b", i), which may go to different ranks: each runs one sequence and
    takes its partner's reward as a constant from a no-grad pass (halves[r]: the halves of rank r) shared by an
    all-reduce. The gradients stay exact, and the busiest rank holds at most one long sequence instead of a whole
    long pair. Units are placed longest first onto the least loaded rank (a half also costs its no-grad forward)."""
    units = []
    for i in pair_idx:
        if tokens[i] > micro_tokens:
            units += [(int(lens_a[i]), ("a", int(i))), (int(lens_b[i]), ("b", int(i)))]
        else:
            units.append((int(tokens[i]), ("p", int(i))))
    units.sort(key=lambda u: -u[0])
    load = [0.0] * world
    share = [[] for _ in range(world)]
    for c, it in units:
        r = min(range(world), key=lambda k: load[k])
        share[r].append((c, it))
        load[r] += c * (1 + nograd_cost) if it[0] != "p" else c
    plans, halves = [], []
    for r in range(world):
        mbs, cur, cur_t = [], [], 0
        for c, it in share[r]:                       # in decreasing length
            if cur and cur_t + c > micro_tokens:
                mbs.append(cur)
                cur, cur_t = [], 0
            cur.append(it)
            cur_t += c
        if cur:
            mbs.append(cur)
        plans.append(mbs)
        halves.append([(c, it) for c, it in share[r] if it[0] != "p"])
    m = max(len(p) for p in plans)
    for p in plans:
        while len(p) < m:
            p.append([])                             # dummy pass
    return plans, halves


# ------------------------------------------------------------------------------------------------ loss

def pair_loss(args, ra, rb, p):
    d = ra - rb
    if args.loss == "soft":
        target, w = p, torch.ones_like(p)
    elif args.loss == "hard":
        target, w = (p > 0.5).float(), (p != 0.5).float()
    elif args.loss == "unanimous":
        target, w = (p > 0.5).float(), ((p == 0) | (p == 1)).float()
    elif args.loss == "weighted":
        target, w = (p > 0.5).float(), (2 * p - 1).abs()
    loss = F.binary_cross_entropy_with_logits(d, target, reduction="none") * w
    if args.center:
        loss = loss + args.center * (ra + rb) ** 2 * (w > 0).float()
    return loss, w


def weight_of(args, p):
    if args.loss == "soft":
        return 1.0
    if args.loss == "hard":
        return float(p != 0.5)
    if args.loss == "unanimous":
        return float(p in (0.0, 1.0))
    return abs(2 * p - 1)


# ------------------------------------------------------------------------------------------------ model

def setup_model(args, rank, world, device):
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.device_mesh import init_device_mesh
    dtype = torch.bfloat16 if (args.lora or args.diagnose_grads) else torch.float32
    rm, tcfg = rm_model.build(args.model, dtype=dtype, lora_rank=args.lora, lora_alpha=args.lora_alpha)
    if args.lora:
        for n, p in rm.named_parameters():
            p.requires_grad = ".lora_" in n or n.startswith("head.")
    if args.freeze_embed:
        for p in rm.backbone.embed_tokens.parameters():
            p.requires_grad = False
    if args.freeze_norms:
        for n, p in rm.backbone.named_parameters():
            if n.endswith("norm.weight"):
                p.requires_grad = False
    if not args.no_grad_ckpt:
        rm.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    mesh = init_device_mesh("cuda", (world,))
    fp32_residual = getattr(args, "fp32_residual", False)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                              cast_forward_inputs=not fp32_residual)
    if fp32_residual:
        rm.use_fp32_residual()
    for layer in rm_model.decoder_layers(rm):
        fully_shard(layer, mesh=mesh, mp_policy=mp)
    fully_shard(rm.head, mesh=mesh, mp_policy=MixedPrecisionPolicy(param_dtype=torch.float32,
                                                                   reduce_dtype=torch.float32))
    fully_shard(rm, mesh=mesh, mp_policy=mp)
    rm.to_empty(device=device)
    prefix = rm_model.load_weights(rm, args.model, rank, world)
    n_all = sum(p.numel() for p in rm.parameters())
    n_train = sum(p.numel() for p in rm.parameters() if p.requires_grad)
    return rm, tcfg, prefix, n_all, n_train


PROF = {"on": False, "fwd": 0.0, "fwd_nograd": 0.0, "passes": 0.0, "opt": 0.0}


def forward_pass(rm, seqs, device):
    if PROF["on"]:
        torch.cuda.synchronize()
        t = time.time()
        r = _forward_pass(rm, seqs, device)
        torch.cuda.synchronize()
        PROF["fwd" if torch.is_grad_enabled() else "fwd_nograd"] += time.time() - t
        return r
    return _forward_pass(rm, seqs, device)


def _forward_pass(rm, seqs, device):
    lens = [len(s) for s in seqs]
    ids = torch.from_numpy(np.concatenate(seqs).astype(np.int64)).to(device, non_blocking=True)[None]
    pos = torch.from_numpy(np.concatenate([np.arange(n) for n in lens])).to(device, non_blocking=True)[None]
    ends = torch.tensor(np.cumsum(lens) - 1, device=device)
    set_pack(lens, device)
    return rm(ids, pos, ends)


# ------------------------------------------------------------------------------------------------ evaluation

def score_sequences(rm, seqs, rank, world, device, micro_tokens):
    """rewards of a list of sequences (same on every rank); every rank runs the same number of passes"""
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    mine = order[rank::world]
    mbs, cur, cur_t = [], [], 0
    for i in mine:
        if cur and cur_t + len(seqs[i]) > micro_tokens:
            mbs.append(cur)
            cur, cur_t = [], 0
        cur.append(i)
        cur_t += len(seqs[i])
    if cur:
        mbs.append(cur)
    n = torch.tensor([len(mbs)], device=device)
    dist.all_reduce(n, op=dist.ReduceOp.MAX)
    out = {}
    rm.eval()
    with torch.no_grad():
        for k in range(int(n.item())):
            if k < len(mbs):
                r = forward_pass(rm, [seqs[i] for i in mbs[k]], device)
                out.update({i: float(v) for i, v in zip(mbs[k], r.tolist())})
            else:
                forward_pass(rm, [seqs[order[0]][:8]], device)
    rm.train()
    allout = [None] * world
    dist.all_gather_object(allout, out)
    res = {}
    for d in allout:
        res.update(d)
    return [res[i] for i in range(len(seqs))]


def opus_reference():
    ref = {}
    if not os.path.exists(OPUS):
        return ref
    rows = pq.read_table(OPUS, columns=["pair_id", "order", "prefers"]).to_pylist()
    d = {}
    for r in rows:
        d.setdefault(r["pair_id"], {})[r["order"]] = r["prefers"]
    for i, v in d.items():
        if v.get(0) in ("x", "y") and v.get(0) == v.get(1):
            ref[i] = "a" if v[0] == "x" else "b"
    return ref


def evaluate(rm, val, val_seqs, args, rank, world, device, step, run_dir, opus, bg=None):
    t = time.time()
    r = score_sequences(rm, val_seqs, rank, world, device, args.micro_tokens * 2)
    ra, rb = np.array(r[0::2]), np.array(r[1::2])
    if rank != 0:
        return None
    d = ra - rb
    p = val.p
    soft = float(np.mean(np.logaddexp(0, -d) * p + np.logaddexp(0, d) * (1 - p)))
    nt = p != 0.5
    acc = float(np.mean((d[nt] > 0) == (p[nt] > 0.5)))
    agree = float(np.mean(np.where(d > 0, p, np.where(d < 0, 1 - p, 0.5))))
    o = [(i, opus[pid]) for i, pid in enumerate(val.ids) if pid in opus]
    opus_acc = float(np.mean([(d[i] > 0) == (w == "a") for i, w in o])) if o else float("nan")
    la, lb = np.array([len(s) for s in val_seqs[0::2]]), np.array([len(s) for s in val_seqs[1::2]])
    ratio = np.maximum(la, lb) / np.maximum(1, np.minimum(la, lb))
    m = ratio >= 1.5
    longer_rm = float(np.mean(np.where(la[m] > lb[m], d[m] > 0, d[m] < 0)))
    longer_lab = float(np.mean(np.where(la[m] > lb[m], p[m], 1 - p[m])))
    res = {"step": step, "val_loss": soft, "val_acc": acc, "val_agree": agree, "opus_acc": opus_acc,
           "opus_n": len(o), "longer_wins_rm": longer_rm, "longer_wins_label": longer_lab, "len_pairs": int(m.sum()),
           "reward_mean": float(np.mean(np.concatenate([ra, rb]))), "reward_std": float(np.std(np.concatenate([ra, rb]))),
           "eval_seconds": time.time() - t}
    import pyarrow as pa
    table = pa.table({"id": val.ids, "r_a": ra, "r_b": rb, "p_a": p})
    out = os.path.join(run_dir, "val_{:05d}.parquet".format(step))
    if bg is not None:
        bg.call(pq.write_table, table, out)
    else:
        pq.write_table(table, out)
    return res


# ------------------------------------------------------------------------------------------------ rank-0 file I/O

class BackgroundIO:
    """rank 0's file work during training, off the training loop: Lustre can stall for many minutes, and a rank
    that blocks in a write or a stat while the others wait in a collective ends the job with an NCCL timeout.
    write() queues a line for log.jsonl, call() queues any function (e.g. a parquet write), stop is set when
    the STOP file appears (polled every 20 s)."""

    def __init__(self, log_path, stop_file):
        import queue
        import threading
        self.q = queue.Queue()
        self.stop = threading.Event()
        self.log_path, self.stop_file = log_path, stop_file
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()
        threading.Thread(target=self._poll, daemon=True).start()

    def write(self, line):
        self.q.put(("line", line))

    def flush(self):
        pass

    def call(self, fn, *a, **kw):
        self.q.put(("call", (fn, a, kw)))

    def _work(self):
        f = open(self.log_path, "a")
        while True:
            kind, x = self.q.get()
            try:
                if kind == "end":
                    f.close()
                    return
                if kind == "line":
                    f.write(x)
                else:
                    x[0](*x[1], **x[2])
                if self.q.empty():
                    f.flush()
            except Exception as e:                     # never let a file error kill the run
                print("background I/O error:", repr(e), flush=True)

    def _poll(self):
        while True:
            try:
                if os.path.exists(self.stop_file):
                    self.stop.set()
            except OSError:
                pass
            time.sleep(20)

    def close(self, timeout=None):
        self.q.put(("end", None))
        self.worker.join(timeout)


# ------------------------------------------------------------------------------------------------ diagnostics

def grad_report(rm, scale):
    """per-parameter gradient norms from the local shards (one all-reduce), largest parameters / types / layers"""
    names, sq = [], []
    for n, p in rm.named_parameters():
        if p.grad is not None:
            g = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
            names.append(n)
            sq.append(g.float().pow(2).sum())
    v = torch.stack(sq)
    dist.all_reduce(v)
    v = (v.sqrt() * scale).tolist()
    by_type, by_layer = {}, {}
    for n, x in zip(names, v):
        k = re.sub(r"layers\.\d+\.", "layers.*.", n)
        by_type[k] = by_type.get(k, 0.0) + x * x
        m = re.search(r"layers\.(\d+)\.", n)
        if m:
            by_layer[int(m.group(1))] = by_layer.get(int(m.group(1)), 0.0) + x * x
    top = lambda d, k: [(a, round(math.sqrt(b), 2)) for a, b in sorted(d.items(), key=lambda kv: -kv[1])[:k]]
    return {"params": [(n, round(x, 2)) for x, n in sorted(zip(v, names), reverse=True)[:8]],
            "types": top(by_type, 8), "layers": top(by_layer, 8)}


# ------------------------------------------------------------------------------------------------ checkpoints

def save_ckpt(rm, opt, run_dir, step, rank):
    import shutil
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict
    msd, osd = get_state_dict(rm, opt)
    tmp, final = os.path.join(run_dir, "ckpt_tmp"), os.path.join(run_dir, "ckpt")
    if rank == 0 and os.path.exists(tmp):
        shutil.rmtree(tmp)
    dist.barrier()
    dcp.save({"model": msd, "optim": osd}, checkpoint_id=tmp)
    dist.barrier()
    if rank == 0:
        with open(os.path.join(tmp, "step.json"), "w") as f:
            json.dump({"step": step}, f)
        if os.path.exists(final):
            shutil.rmtree(final)
        os.rename(tmp, final)
    dist.barrier()


def load_ckpt(rm, opt, run_dir):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    path = os.path.join(run_dir, "ckpt")
    if not os.path.exists(os.path.join(path, "step.json")):
        return 0
    msd, osd = get_state_dict(rm, opt)
    state = {"model": msd, "optim": osd}
    dcp.load(state, checkpoint_id=path)
    set_state_dict(rm, opt, model_state_dict=state["model"], optim_state_dict=state["optim"])
    return json.load(open(os.path.join(path, "step.json")))["step"]


# ------------------------------------------------------------------------------------------------ main

def main():
    args = parse()
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
    world = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))
    local = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if args.mem_headroom:          # above the cap the allocator frees its cache and retries instead of growing
        total = torch.cuda.get_device_properties(local).total_memory
        torch.cuda.set_per_process_memory_fraction(max(0.5, 1 - args.mem_headroom * 2**30 / total), local)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device,
                            timeout=datetime.timedelta(minutes=120))  # Lustre stalls
    run_dir = os.path.join(DATA, "runs", args.run)
    if rank == 0:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "args.json"), "w") as f:
            json.dump(vars(args) | {"world": world, "nodes": int(os.environ.get("SLURM_NNODES", 1)),
                                    "job": os.environ.get("SLURM_JOB_ID")}, f, indent=1)

    def log(*a):
        if rank == 0:
            print("[{:7.1f} min]".format((time.time() - START) / 60), *a, flush=True)

    meta = json.load(open(os.path.join(DATA, "tok", args.model, "meta.json")))
    train = Pairs(args.model, "train", meta["suffix_ids"])
    val = Pairs(args.model, "val", meta["suffix_ids"])
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(train.n)
    if args.train_limit:
        order = order[:args.train_limit]
    lens_a, lens_b = seq_lens(train, args.max_len)
    tokens = lens_a + lens_b
    steps_per_epoch = len(order) // args.batch
    sched_total = int(steps_per_epoch * args.epochs)      # the LR schedule always spans the whole run
    total = min(sched_total, args.max_steps) if args.max_steps else sched_total
    log("train pairs {} (of {}), val pairs {}, {} steps of {} pairs, {:.1f}M tokens per epoch".format(
        len(order), train.n, val.n, total, args.batch, tokens[order].sum() / 1e6))

    rm, tcfg, prefix, n_all, n_train = setup_model(args, rank, world, device)
    log("model {} ({:.2f}B parameters, {:.3f}B trained), checkpoint prefix '{}', {} ranks".format(
        args.model, n_all / 1e9, n_train / 1e9, prefix, world))
    params = [p for p in rm.parameters() if p.requires_grad]
    b1, b2 = (float(x) for x in args.betas.split(","))
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(b1, b2), eps=1e-8, weight_decay=args.weight_decay,
                            foreach=True)
    warm = max(1, int(args.warmup * sched_total))

    def lr_at(s):
        if s < warm:
            return args.lr * (s + 1) / warm
        q = (s - warm) / max(1, sched_total - warm)
        return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * q)))

    val_seqs = []
    for i in range(val.n):
        val_seqs += list(pair_seqs(val, i, args.max_len))
    opus = opus_reference()
    logf = BackgroundIO(os.path.join(run_dir, "log.jsonl"), os.path.join(run_dir, "STOP")) if rank == 0 else None

    timing = {"eval": None, "ckpt": None}
    for line in (open(os.path.join(run_dir, "log.jsonl")) if os.path.exists(os.path.join(run_dir, "log.jsonl")) else []):
        if '"eval"' in line:           # earlier chunks of this run: a first guess at the evaluation time
            try:
                timing["eval_prev"] = json.loads(line)["eval"]["eval_seconds"] * json.loads(line)["eval"].get("world", world) / world
            except (ValueError, KeyError, TypeError):
                pass
    val_tok = sum(len(s) for s in val_seqs)
    score_tok = 0
    if args.score:
        import pyarrow.compute as pc
        for f in args.score.split(","):
            t = pq.read_table(f, columns=["prompt_ids", "resp_ids"])
            score_tok += int(pc.sum(pc.list_value_length(t.column("prompt_ids"))).as_py() or 0)
            score_tok += int(pc.sum(pc.list_value_length(t.column("resp_ids"))).as_py() or 0)
    ckpt_on = bool(args.ckpt_every)
    stop_file = os.path.join(run_dir, "STOP")

    def final_need():
        """seconds for the final evaluation, the model save and the benchmark scoring"""
        ev = timing["eval"] or timing.get("eval_prev") or 300.0
        return ev * (1 + score_tok / max(val_tok, 1)) * 1.15 + (900 if args.save else 0) + 120

    def do_eval(step):
        t = time.time()
        res = evaluate(rm, val, val_seqs, args, rank, world, device, step, run_dir, opus, bg=logf)
        timing["eval"] = time.time() - t
        if rank == 0:
            res["world"] = world
            log("EVAL step {step}: val_loss {val_loss:.4f}  acc {val_acc:.4f}  agree {val_agree:.4f}  "
                "opus {opus_acc:.4f} (n={opus_n})  longer-wins rm {longer_wins_rm:.3f} vs labels "
                "{longer_wins_label:.3f} (n={len_pairs})  reward {reward_mean:+.2f}±{reward_std:.2f}  "
                "({eval_seconds:.0f}s)".format(**res))
            logf.write(json.dumps({"eval": res}) + "\n")
            logf.flush()

    def ckpt(step):
        t = time.time()
        save_ckpt(rm, opt, run_dir, step, rank)
        timing["ckpt"] = time.time() - t
        log("checkpoint at step {} ({:.0f}s)".format(step, timing["ckpt"]))

    def time_left():
        return args.time_limit * 60 - (time.time() - START) if args.time_limit else float("inf")

    def finish_early(why):
        """stop this job; the next one continues from the checkpoint (--resume)"""
        log("{}: stopping; rerun with --resume to continue".format(why))
        if rank == 0:
            logf.write(json.dumps({"stopped": why, "step": start_step_done[0], "job": os.environ.get("SLURM_JOB_ID")}) + "\n")
            logf.flush()
        dist.barrier()
        dist.destroy_process_group()
        if logf is not None:
            logf.close()

    start = 0
    if args.resume:
        t = time.time()
        start = load_ckpt(rm, opt, run_dir)
        # the load's temporary state dicts (about the size of the optimizer state) would otherwise stay reserved by
        # the caching allocator; Gemma 4 then left the ROCm runtime too little memory for the flash-attention
        # backward (HSA_STATUS_ERROR_OUT_OF_RESOURCES in the first step after a resume)
        gc.collect()
        torch.cuda.empty_cache()
        if start:
            log("resumed from the checkpoint at step {} ({:.0f}s)".format(start, time.time() - t))
    start_step_done = [start]
    if args.eval_first and start == 0:
        do_eval(0)
    rm.train()
    step_times = []
    early = None
    prof = None
    for step in range(start, total):
        if args.profile_step and step == start + args.profile_step - 1:
            from torch.profiler import ProfilerActivity, profile
            torch.cuda.synchronize()
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) if rank == 0 else None
            if prof is not None:
                prof.__enter__()
            PROF.update(on=True, fwd=0.0, fwd_nograd=0.0, passes=0.0, opt=0.0)
        t0 = time.time()
        idx = order[step * args.batch:(step + 1) * args.batch]
        plans, halves = plan_step(list(idx), tokens, lens_a, lens_b, world, args.micro_tokens)
        wsum = sum(weight_of(args, train.p[i]) for i in idx)
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        tot_loss = torch.zeros((), device=device)
        tot_acc = torch.zeros((), device=device)
        ntok = 0
        # split pairs: no-grad rewards of every half on the rank that holds it, shared by one all-reduce
        split_pairs = sorted({it[1] for h in halves for _, it in h})
        n_split = len(split_pairs)
        partner = None
        if n_split:
            pos = {i: k for k, i in enumerate(split_pairs)}
            ng = []                                  # per rank: no-grad passes of at most 2 * micro_tokens tokens
            for h in halves:
                passes, cur, cur_t = [], [], 0
                for c, it in h:
                    if cur and cur_t + c > 2 * args.micro_tokens:
                        passes.append(cur)
                        cur, cur_t = [], 0
                    cur.append(it)
                    cur_t += c
                if cur:
                    passes.append(cur)
                ng.append(passes)
            vec = torch.zeros(2 * n_split, device=device)
            with torch.no_grad():
                for k in range(max(len(x) for x in ng)):
                    if k < len(ng[rank]):
                        items = ng[rank][k]
                        seqs = [pair_seqs(train, i, args.max_len)[0 if h == "a" else 1] for h, i in items]
                        r = forward_pass(rm, seqs, device).float()
                        for (h, i), v in zip(items, r):
                            vec[2 * pos[i] + (h == "b")] = v
                    else:
                        pr = train.get("prompt_ids", int(idx[0]))[:8]
                        forward_pass(rm, [np.concatenate([pr, train.suffix])], device)
            dist.all_reduce(vec)
            partner = vec
        if PROF["on"]:
            torch.cuda.synchronize()
            t_pass = time.time()
        for mb in plans[rank]:
            if mb:
                seqs = []
                for h, i in mb:
                    sa, sb = pair_seqs(train, i, args.max_len)
                    seqs += [sa, sb] if h == "p" else [sa if h == "a" else sb]
                r = forward_pass(rm, seqs, device).float()
                RA, RB, cnt, j = [], [], [], 0
                if os.environ.get("RM_CHECK_PARTNER") and n_split:
                    jj = 0
                    for h, i in mb:
                        if h == "p":
                            jj += 2
                            continue
                        ref = partner[2 * pos[i] + (h == "b")].item()
                        print("PARTNER rank {} step {} pair {} {}: no-grad {:+.5f} grad {:+.5f} diff {:+.2e} len {}".format(
                            rank, step + 1, i, h, ref, r[jj].item(), r[jj].item() - ref, len(seqs[jj])), flush=True)
                        jj += 1
                for h, i in mb:
                    if h == "p":
                        RA.append(r[j]); RB.append(r[j + 1]); cnt.append(1.0); j += 2
                    elif h == "a":                   # counted once, here
                        RA.append(r[j]); RB.append(partner[2 * pos[i] + 1]); cnt.append(1.0); j += 1
                    else:
                        RA.append(partner[2 * pos[i]]); RB.append(r[j]); cnt.append(0.0); j += 1
                ra, rb = torch.stack(RA), torch.stack(RB)
                p = torch.tensor(train.p[[i for _, i in mb]], device=device, dtype=torch.float32)
                c = torch.tensor(cnt, device=device)
                loss, w = pair_loss(args, ra, rb, p)
                scaled = loss.sum() * world / max(wsum, 1e-9)
                tot_loss += (loss.detach() * c).sum()
                tot_acc += ((((ra - rb) > 0).float() == (p > 0.5).float()).float() * (p != 0.5).float() * c).sum().detach()
                ntok += sum(len(s) for s in seqs)
            else:
                pr = train.get("prompt_ids", int(idx[0]))[:8]
                seqs = [np.concatenate([pr, train.suffix])] * 2
                r = forward_pass(rm, seqs, device)
                scaled = r.sum() * 0.0
            scaled.backward()
        if args.diagnose_grads:
            norms = []
            for n, p in rm.named_parameters():
                if p.grad is not None:
                    gg = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
                    norms.append((torch.linalg.vector_norm(gg.float()).item(), n, tuple(p.shape)))
            if rank == 0:
                norms.sort(reverse=True)
                tot = math.sqrt(sum(x[0] ** 2 for x in norms))
                groups = {}
                for v, n, _ in norms:
                    k = "embed" if "embed" in n else ("head" if n.startswith("head") else re.sub(r"layers\.\d+\.", "layers.*.", n))
                    groups[k] = groups.get(k, 0.0) + v * v
                log("step {} grad norm {:.2f}; largest: {}".format(step + 1, tot, "; ".join(
                    "{} {:.2f}".format(n, v) for v, n, _ in norms[:8])))
                log("  by parameter type: " + "; ".join("{} {:.2f}".format(k, math.sqrt(v)) for k, v in
                                                         sorted(groups.items(), key=lambda kv: -kv[1])[:10]))
            opt.zero_grad(set_to_none=True)
            if step + 1 >= args.diagnose_grads:
                break
            continue
        if PROF["on"]:
            torch.cuda.synchronize()
            PROF["passes"] = time.time() - t_pass
            t_opt = time.time()
        g = rm.head.weight.grad
        ghead = torch.linalg.vector_norm(g.full_tensor() if hasattr(g, "full_tensor") else g).item() if g is not None else 0.0
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.clip)
        if args.grad_report and float(gnorm) > args.grad_report:
            rep = grad_report(rm, float(gnorm) / args.clip if args.clip and float(gnorm) > args.clip else 1.0)
            if rank == 0:
                log("step {} gradient norm {:.1f}: params {}; types {}; layers {}".format(
                    step + 1, float(gnorm), rep["params"], rep["types"], rep["layers"]))
                logf.write(json.dumps({"grad_report": rep, "step": step + 1, "gnorm": float(gnorm)}) + "\n")
        opt.step()
        opt.zero_grad(set_to_none=True)
        stats = torch.stack([tot_loss, tot_acc, torch.tensor(float(ntok), device=device)])
        dist.all_reduce(stats)
        ms = torch.cuda.memory_stats(device)
        mem = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30, torch.cuda.max_memory_reserved(device) / 2**30,
                            float(ms.get("num_alloc_retries", 0)), float(ms.get("num_ooms", 0))], device=device)
        dist.all_reduce(mem, op=dist.ReduceOp.MAX)
        torch.cuda.reset_peak_memory_stats(device)
        dt = time.time() - t0
        if PROF["on"]:
            torch.cuda.synchronize()
            PROF["opt"] = time.time() - t_opt
            PROF["on"] = False
            mine_t = torch.tensor([ntok, PROF["fwd"], PROF["fwd_nograd"], PROF["passes"], PROF["opt"], dt], device=device)
            allt = [torch.zeros_like(mine_t) for _ in range(world)]
            dist.all_gather(allt, mine_t)
            if prof is not None:
                prof.__exit__(None, None, None)
                ev = prof.key_averages()
                dev_t = lambda e: getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0.0)) / 1e6
                coll = sum(dev_t(e) for e in ev if re.search(r"nccl|rccl|AllGather|ReduceScatter|AllReduce", e.key, re.I))
                comp = sum(dev_t(e) for e in ev) - coll
                try:
                    table = ev.table(sort_by="self_device_time_total", row_limit=50)
                except Exception:
                    table = ev.table(sort_by="self_cuda_time_total", row_limit=50)
                logf.call(lambda t=table, n=step + 1: open(os.path.join(run_dir, "profile_{:05d}.txt".format(n)), "w").write(t))
                rows = torch.stack(allt).tolist()
                log("profile step {}: wall {:.1f}s; rank 0 GPU kernels: compute {:.1f}s, collectives {:.1f}s".format(
                    step + 1, dt, comp, coll))
                log("  per rank (tokens, fwd, no-grad fwd, all passes, clip+opt): " + "; ".join(
                    "{:.0f} {:.1f} {:.1f} {:.1f} {:.1f}".format(*r[:5]) for r in rows))
                logf.write(json.dumps({"profile": {"step": step + 1, "wall": dt, "compute": comp, "collectives": coll,
                                                   "ranks": rows}}) + "\n")
            prof = None
        step_times.append(dt)
        start_step_done[0] = step + 1
        n_nontie = sum(1 for i in idx if train.p[i] != 0.5)
        rec = {"step": step + 1, "lr": lr, "loss": stats[0].item() / max(wsum, 1e-9) if args.loss != "soft" else stats[0].item() / len(idx),
               "acc": stats[1].item() / max(1, n_nontie), "gnorm": float(gnorm), "gnorm_head": ghead,
               "tokens": int(stats[2].item()),
               "passes": len(plans[0]), "seconds": dt, "tok_per_s_per_gpu": stats[2].item() / dt / world,
               "mem_gb": round(mem[0].item(), 1), "reserved_gb": round(mem[1].item(), 1),
               "alloc_retries": int(mem[2].item()), "n_split": n_split}
        if rank == 0:
            logf.write(json.dumps(rec) + "\n")
            if step < 5 or (step + 1) % 10 == 0:
                log("step {step}/{tot}  loss {loss:.4f}  acc {acc:.3f}  gnorm {gnorm:.2f} (head {gnorm_head:.2f})  lr {lr:.2e}  "
                    "{tokens} tok in {passes} passes, {seconds:.1f}s ({tok_per_s_per_gpu:.0f} tok/s/gpu)".format(
                        tot=total, **rec))
                logf.flush()
        if args.eval_every and (step + 1) % args.eval_every == 0 and step + 1 < total:
            do_eval(step + 1)
        if ckpt_on and (step + 1) % args.ckpt_every == 0 and step + 1 < total:
            ckpt(step + 1)
        if step + 1 < total and (args.time_limit or ckpt_on):
            # stop when asked to (runs/<run>/STOP, another job takes the run over) or when the next step and a
            # checkpoint would no longer fit, unless the rest of the run, the evaluation and the scoring still fit
            left = time_left()
            mstep = float(np.mean(step_times[-20:]))
            if ckpt_on:
                # at least 10 minutes for the closing checkpoint: on a slow Lustre the 370 GB Gemma checkpoint took
                # 515 s and once over 9 minutes (usually 40 s), and Slurm killed the job during the write
                late = left < (total - step - 1) * mstep + final_need() and \
                    left < 2 * mstep + max(600.0, 2 * (timing["ckpt"] or 0.0)) + 120
            else:
                late = left < mstep * 2 + 600 + (900 if args.save else 0)
            asked = ckpt_on and rank == 0 and logf.stop.is_set()
            flags = torch.tensor([float(late), float(asked)], device=device)
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
            if flags.max().item() > 0:
                why = "asked to stop (STOP file)" if flags[1].item() > 0 else "time limit"
                log("{} after step {} of {}".format(why, step + 1, total))
                if ckpt_on:
                    ckpt(step + 1)
                    early = why
                else:
                    total = step + 1
                break
    if args.diagnose_grads:
        dist.barrier()
        dist.destroy_process_group()
        return
    if early:
        return finish_early(early)
    if ckpt_on and start_step_done[0] > start:
        # all steps done: checkpoint first, so that a job that runs out of time in the final phase loses nothing
        ckpt(total)
        if time_left() < final_need():
            return finish_early("not enough time left for the final evaluation and scoring")
    # the final phase is resumable too: final.json records what is done for this step (same decisions on all ranks)
    final_file = os.path.join(run_dir, "final.json")
    prev = json.load(open(final_file)) if os.path.exists(final_file) else {}
    dist.barrier()
    if prev.get("step") != total:
        prev = {}
    state = {"step": total, "eval": prev.get("eval", False), "saved": prev.get("saved", False),
             "scored": list(prev.get("scored", []))}

    def mark(**kw):
        state.update(kw)
        if rank == 0:
            with open(final_file + ".tmp", "w") as f:
                json.dump(state, f)
            os.replace(final_file + ".tmp", final_file)

    if not state["eval"]:
        do_eval(total)
        mark(eval=True)
    if args.save and not (state["saved"] and os.path.exists(os.path.join(run_dir, "model"))):
        t = time.time()
        rm_model.save(rm, os.path.join(run_dir, "model"), args.model, rank)
        log("saved model in {:.0f}s".format(time.time() - t))
        mark(saved=True)
    if args.score:
        import pyarrow as pa
        for f in args.score.split(","):
            name = os.path.basename(f).replace(".parquet", "")
            out = os.path.join(run_dir, "scores_{}.parquet".format(name))
            if name in state["scored"]:
                continue
            t = pq.read_table(f)
            seqs = [make_seq(np.asarray(a, dtype=np.int32), np.asarray(b, dtype=np.int32), train.suffix, args.max_len)
                    for a, b in zip(t.column("prompt_ids").to_pylist(), t.column("resp_ids").to_pylist())]
            r = score_sequences(rm, seqs, rank, world, device, args.micro_tokens * 2)
            if rank == 0:
                pq.write_table(pa.table({"id": t.column("id"), "reward": r}), out)
                log("scored {} ({} sequences)".format(name, len(seqs)))
            mark(scored=state["scored"] + [name])
    dist.barrier()
    if rank == 0:
        with open(os.path.join(run_dir, "DONE"), "w") as f:
            json.dump({"step": total, "job": os.environ.get("SLURM_JOB_ID")}, f)
    log("done")
    dist.destroy_process_group()
    if logf is not None:
        logf.close()


if __name__ == "__main__":
    main()
