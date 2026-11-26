#!/usr/bin/env python3
"""Checks of the training step on real data, without training: the gradient of one batch computed in several
ways, which must agree, and the forward pass against transformers' own attention.

  sbatch train-reward-model/slurm/grad_check.sbatch          (2 nodes; see there for the arguments)

Gradients of the first training batch (seed 1, the batch of step 1), with the training loss (soft labels,
--center):
  train32   the training step's own procedure (train_rm.py: plan_step, split pairs, partner rewards) with the
            plan of a 32-rank job; on 16 ranks each rank runs two plan slots in turn
  train16   the same with the plan of a 16-rank job
  single    the same procedure with every sequence alone in its forward pass (every pair split)
  coef      the exact gradient sum_s dL/dr_s * grad r_s, every sequence alone, without the split machinery
Short pairs (both sequences <= --short-len tokens, the next pairs of the shuffled order):
  packed    coefficients, sequences packed into rows of up to --micro-tokens (the patched attention)
  eager     coefficients, every sequence alone, transformers' eager attention with its own causal and sliding
            masks (the reference)
lm: next-token loss of the pretrained model on validation answers (tied embeddings, final logit softcap),
packed attention against eager attention.
update: the train32 gradient again (the noise floor), then AdamW steps at --update-lrs (the first LRs of the
run's warmup), each followed by the gradient of the same batch.
Every gradient is reported as its total, head and backbone norms and the largest parameter types and layers;
against a reference also the cosine, the norm ratio and the parameter types and layers that agree least.
"""
import argparse
import datetime
import json
import math
import os
import re
import sys
import time
import types

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rm_model  # noqa: E402
import train_rm as T  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--freeze-embed", action="store_true")
    ap.add_argument("--freeze-norms", action="store_true")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--micro-tokens", type=int, default=16384)
    ap.add_argument("--center", type=float, default=0.01)
    ap.add_argument("--short-len", type=int, default=4096)
    ap.add_argument("--tests", default="lm,train32,train16,single,coef,short,update")
    ap.add_argument("--update-lrs", default="6.8966e-08,1.3793e-07")
    ap.add_argument("--fp32-compute", action="store_true", help="FSDP computes in fp32 instead of bf16 (attn test: "
                    "only the reference attention and transformers' eager attention, flash-attention needs bf16)")
    ap.add_argument("--fp32-residual", action="store_true", help="train_rm.py --fp32-residual")
    ap.add_argument("--attn-variants", default="all", choices=["all", "localize"],
                    help="attn test: all paths (single and packed) or the packed paths one layer type at a time")
    ap.add_argument("--prec-variants", default="all", choices=["all", "coef"],
                    help="prec test: which parts need fp32 (all), or the training precision (bf16 matmuls, the "
                         "kernels) for other coefficients (coef: the pair coefficients, all of them positive, only "
                         "the first answers)")
    ap.add_argument("--out", default="")
    return ap.parse_args()


def main():
    a = parse()
    tests = a.tests.split(",")
    rank, world, local = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", rank=rank, world_size=world, device_id=device,
                            timeout=datetime.timedelta(minutes=60))
    t0 = time.time()
    summary = {"args": vars(a), "world": world}

    def log(*x):
        if rank == 0:
            print("[{:6.1f} min]".format((time.time() - t0) / 60), *x, flush=True)

    margs = types.SimpleNamespace(model=a.model, lora=0, lora_alpha=None, diagnose_grads=0,
                                  freeze_embed=a.freeze_embed, freeze_norms=a.freeze_norms, no_grad_ckpt=False,
                                  fp32_residual=a.fp32_residual)
    if a.fp32_compute:
        from torch.distributed.fsdp import MixedPrecisionPolicy
        orig_mp = MixedPrecisionPolicy.__init__

        def mp_fp32(self, param_dtype=None, reduce_dtype=None, **kw):
            orig_mp(self, param_dtype=torch.float32, reduce_dtype=torch.float32, **kw)
        MixedPrecisionPolicy.__init__ = mp_fp32         # setup_model's bf16 policy becomes fp32
    rm, tcfg, prefix, n_all, n_train = T.setup_model(margs, rank, world, device)
    rm.train()
    log("model {} ({:.2f}B parameters, {:.3f}B trained), {} ranks, freeze embed {} norms {}, compute {}".format(
        a.model, n_all / 1e9, n_train / 1e9, world, a.freeze_embed, a.freeze_norms,
        "fp32" if a.fp32_compute else ("bf16 matmuls, fp32 residual stream" if a.fp32_residual else "bf16")))
    meta = json.load(open(os.path.join(T.DATA, "tok", a.model, "meta.json")))
    train = T.Pairs(a.model, "train", meta["suffix_ids"])
    val = T.Pairs(a.model, "val", meta["suffix_ids"])
    order = np.random.default_rng(a.seed).permutation(train.n)
    lens_a, lens_b = T.seq_lens(train, a.max_len)
    tokens = lens_a + lens_b
    idx = [int(i) for i in order[:a.batch]]
    log("batch of step 1: {} pairs, {} tokens".format(len(idx), int(tokens[idx].sum())))
    largs = types.SimpleNamespace(loss="soft", center=a.center)
    named = [(n, p) for n, p in rm.named_parameters() if p.requires_grad]
    names = [n for n, _ in named]
    params = [p for _, p in named]
    pr0 = train.get("prompt_ids", idx[0])[:8]

    # ------------------------------------------------------------------ helpers
    def set_eager(on):
        impl = "eager" if on else "rm_packed"
        for m in rm.modules():
            c = getattr(m, "config", None)
            if c is not None and hasattr(c, "_attn_implementation"):
                c._attn_implementation = impl
        tcfg._attn_implementation = impl
        rm.is_gemma4 = (not on) and type(rm.backbone).__name__.startswith("Gemma4")

    def dummy(grad):
        r = T.forward_pass(rm, [np.concatenate([pr0, train.suffix])] * 2, device)
        if grad:
            (r.sum() * 0.0).backward()

    def local_grad(p):
        g = p.grad
        return None if g is None else (g.to_local() if hasattr(g, "to_local") else g)

    def snapshot():
        return {n: local_grad(p).detach().float().cpu().clone() for n, p in named if p.grad is not None}

    def stats(ref=None):
        """per trainable parameter, summed over ranks: |g|^2, |ref|^2, <g, ref>"""
        rows = []
        z = torch.zeros((), device=device)
        for n, p in named:
            g = local_grad(p)
            g = g.float() if g is not None else None
            s = g.pow(2).sum() if g is not None else z
            if ref is not None and n in ref:
                r = ref[n].to(device)
                rows.append(torch.stack([s, r.pow(2).sum(), (g * r).sum() if g is not None else z]))
            else:
                rows.append(torch.stack([s, z, z]))
        v = torch.stack(rows)
        dist.all_reduce(v)
        return v.double().cpu().numpy()

    def ptype(n):
        return "head" if n.startswith("head.") else re.sub(r"layers\.\d+\.", "layers.*.", n.replace("backbone.", ""))

    def layer_of(n):
        m = re.search(r"layers\.(\d+)\.", n)
        return int(m.group(1)) if m else None

    def report(tag, v, ref_tag=None):
        sq = v[:, 0]
        tot = math.sqrt(sq.sum())
        head = math.sqrt(sum(s for n, s in zip(names, sq) if n.startswith("head.")))
        body = math.sqrt(max(0.0, sq.sum() - head ** 2))
        nonorm = math.sqrt(sum(s for n, s in zip(names, sq) if not n.startswith("head.") and not n.endswith("norm.weight")))
        out = {"total": tot, "head": head, "backbone": body, "backbone_without_norm_weights": nonorm}
        by_type, by_layer = {}, {}
        for n, row in zip(names, v):
            by_type[ptype(n)] = by_type.get(ptype(n), 0) + row
            if layer_of(n) is not None:
                by_layer[layer_of(n)] = by_layer.get(layer_of(n), 0) + row
        out["types"] = {k: math.sqrt(x[0]) for k, x in by_type.items()}
        out["layers"] = {str(k): math.sqrt(x[0]) for k, x in by_layer.items()}
        line = "{}: gradient norm {:.3f} (head {:.3f}, backbone {:.3f}, backbone without norm weights {:.3f})".format(
            tag, tot, head, body, nonorm)
        if ref_tag:
            refn = math.sqrt(v[:, 1].sum())
            cos = v[:, 2].sum() / max(tot * refn, 1e-30)
            bb = np.array([not n.startswith("head.") for n in names])
            cos_bb = v[bb, 2].sum() / max(math.sqrt(v[bb, 0].sum() * v[bb, 1].sum()), 1e-30)
            out.update({"ref": ref_tag, "cos": cos, "cos_backbone": cos_bb, "ratio": tot / max(refn, 1e-30),
                        "ratio_backbone": math.sqrt(v[bb, 0].sum() / max(v[bb, 1].sum(), 1e-30))})
            line += "\n    vs {}: cosine {:.5f} (backbone {:.5f}), norm ratio {:.4f} (backbone {:.4f})".format(
                ref_tag, cos, cos_bb, tot / max(refn, 1e-30), out["ratio_backbone"])
            cs = lambda x: x[2] / math.sqrt(max(x[0] * x[1], 1e-60))  # noqa: E731
            tc = sorted(((k, cs(x), math.sqrt(x[0]), math.sqrt(x[1])) for k, x in by_type.items()), key=lambda t: t[1])
            lc = sorted(((k, cs(x), math.sqrt(x[0]), math.sqrt(x[1])) for k, x in by_layer.items()), key=lambda t: t[1])
            out["type_cos"] = {k: c for k, c, _, _ in tc}
            out["layer_cos"] = {str(k): c for k, c, _, _ in lc}
            line += "\n    least similar types: " + "; ".join("{} cos {:.4f} norm {:.3f} vs {:.3f}".format(*t) for t in tc[:6])
            line += "\n    least similar layers: " + "; ".join("{} cos {:.4f} norm {:.3f} vs {:.3f}".format(*t) for t in lc[:8])
        line += "\n    largest types: " + "; ".join("{} {:.3f}".format(k, x) for k, x in
                                                 sorted(out["types"].items(), key=lambda kv: -kv[1])[:8])
        line += "\n    largest layers: " + "; ".join("{} {:.3f}".format(k, x) for k, x in
                                                  sorted(out["layers"].items(), key=lambda kv: -kv[1])[:10])
        log(line)
        summary.setdefault("gradients", {})[tag] = out
        return out

    def reward_diff(tag, r, ref, ref_tag):
        d = (r - ref).abs()
        sd = ref.std().item()
        c = torch.corrcoef(torch.stack([r, ref]))[0, 1].item()
        log("rewards {} vs {}: max |diff| {:.2e}, mean |diff| {:.2e} (reward sd {:.2e}), correlation {:.6f}".format(
            tag, ref_tag, d.max().item(), d.mean().item(), sd, c))
        summary.setdefault("rewards", {})[tag + "_vs_" + ref_tag] = {"max": d.max().item(), "mean": d.mean().item(),
                                                                      "sd": sd, "corr": c}

    pos_of = {i: k for k, i in enumerate(idx)}

    def train_grad(plan_world, micro):
        """the training step's gradient as in train_rm.py's main loop, with the plan of a plan_world-rank job"""
        assert plan_world % world == 0
        slots = list(range(rank, plan_world, world))
        plans, halves = T.plan_step(idx, tokens, lens_a, lens_b, plan_world, micro)
        wsum = float(len(idx))
        split_pairs = sorted({it[1] for h in halves for _, it in h})
        pos = {i: k for k, i in enumerate(split_pairs)}
        partner = None
        if split_pairs:
            ng = []
            for h in halves:
                passes, cur, cur_t = [], [], 0
                for c, it in h:
                    if cur and cur_t + c > 2 * micro:
                        passes.append(cur)
                        cur, cur_t = [], 0
                    cur.append(it)
                    cur_t += c
                if cur:
                    passes.append(cur)
                ng.append(passes)
            mine = [ps for s in slots for ps in ng[s]]
            n = torch.tensor([len(mine)], device=device)
            dist.all_reduce(n, op=dist.ReduceOp.MAX)
            vec = torch.zeros(2 * len(split_pairs), device=device)
            with torch.no_grad():
                for k in range(int(n.item())):
                    if k < len(mine):
                        seqs = [T.pair_seqs(train, i, a.max_len)[0 if h == "a" else 1] for h, i in mine[k]]
                        r = T.forward_pass(rm, seqs, device).float()
                        for (h, i), x in zip(mine[k], r):
                            vec[2 * pos[i] + (h == "b")] = x
                    else:
                        dummy(False)
            dist.all_reduce(vec)
            partner = vec
        mbs = [mb for s in slots for mb in plans[s]]
        n = torch.tensor([len(mbs)], device=device)
        dist.all_reduce(n, op=dist.ReduceOp.MAX)
        rew = torch.zeros(2 * len(idx), device=device)
        for k in range(int(n.item())):
            mb = mbs[k] if k < len(mbs) else []
            if not mb:
                dummy(True)
                continue
            seqs = []
            for h, i in mb:
                sa, sb = T.pair_seqs(train, i, a.max_len)
                seqs += [sa, sb] if h == "p" else [sa if h == "a" else sb]
            r = T.forward_pass(rm, seqs, device).float()
            RA, RB, j = [], [], 0
            for h, i in mb:
                if h == "p":
                    RA.append(r[j]); RB.append(r[j + 1])
                    rew[2 * pos_of[i]], rew[2 * pos_of[i] + 1] = r[j].detach(), r[j + 1].detach()
                    j += 2
                elif h == "a":
                    RA.append(r[j]); RB.append(partner[2 * pos[i] + 1])
                    rew[2 * pos_of[i]] = r[j].detach()
                    j += 1
                else:
                    RA.append(partner[2 * pos[i]]); RB.append(r[j])
                    rew[2 * pos_of[i] + 1] = r[j].detach()
                    j += 1
            p = torch.tensor(train.p[[i for _, i in mb]], device=device, dtype=torch.float32)
            loss, _ = T.pair_loss(largs, torch.stack(RA), torch.stack(RB), p)
            (loss.sum() * world / wsum).backward()
        dist.all_reduce(rew)
        log("  {}-rank plan, micro-batches of {} tokens: {} passes per rank, {} split pairs".format(
            plan_world, micro, int(n.item()), len(split_pairs)))
        return rew

    def coefs(rew, pairs):
        ra, rb = rew[0::2], rew[1::2]
        p = torch.tensor(train.p[pairs], device=device, dtype=torch.float32)
        g = torch.sigmoid(ra - rb) - p
        cc = 2 * a.center * (ra + rb)
        return torch.stack([g + cc, -g + cc], 1).reshape(-1) / len(pairs)

    def coef_grad(pairs, coef, packed, grad=True):
        """sum_s coef_s * r_s backpropagated (exact gradient of the batch loss for coefficients dL/dr_s);
        sequences alone or packed up to micro_tokens; returns the rewards"""
        seqs = []
        for k, i in enumerate(pairs):
            sa, sb = T.pair_seqs(train, i, a.max_len)
            seqs += [(2 * k, sa), (2 * k + 1, sb)]
        seqs.sort(key=lambda x: -len(x[1]))
        load, share = [0] * world, [[] for _ in range(world)]
        for key, s in seqs:
            q = min(range(world), key=lambda q: load[q])
            share[q].append((key, s))
            load[q] += len(s)
        mbs = []
        if packed:
            cur, cur_t = [], 0
            for key, s in share[rank]:
                if cur and cur_t + len(s) > a.micro_tokens:
                    mbs.append(cur)
                    cur, cur_t = [], 0
                cur.append((key, s))
                cur_t += len(s)
            if cur:
                mbs.append(cur)
        else:
            mbs = [[x] for x in share[rank]]
        n = torch.tensor([len(mbs)], device=device)
        dist.all_reduce(n, op=dist.ReduceOp.MAX)
        rew = torch.zeros(2 * len(pairs), device=device)
        with torch.set_grad_enabled(grad):
            for k in range(int(n.item())):
                if k >= len(mbs):
                    dummy(grad)
                    continue
                keys = torch.tensor([key for key, _ in mbs[k]], device=device)
                r = T.forward_pass(rm, [s for _, s in mbs[k]], device).float()
                rew[keys] = r.detach()
                if grad:
                    ((r * coef[keys]).sum() * world).backward()
        dist.all_reduce(rew)
        return rew

    # ------------------------------------------------------------------ lm: forward pass against the LM objective
    if "lm" in tests:
        cap = getattr(tcfg, "final_logit_softcapping", None)
        hs = {}
        hook = rm.backbone.norm.register_forward_hook(lambda m, i, o: hs.__setitem__("h", o))
        W = rm.backbone.embed_tokens.weight
        Wb = (W.full_tensor() if hasattr(W, "full_tensor") else W).to(torch.bfloat16)
        cand = []
        for i in range(val.n):
            lp = len(val.get("prompt_ids", i))
            for s in T.pair_seqs(val, i, a.max_len):
                if len(s) <= 8192 and len(s) - lp >= 32:
                    cand.append((s, lp))
            if len(cand) >= 2 * world:
                break
        mine = cand[rank:2 * world:world]
        res = {}
        for mode in ("packed", "eager"):
            set_eager(mode == "eager")
            per = []
            with torch.no_grad():
                for g in ([mine] if mode == "packed" else [[x] for x in mine]):
                    T.forward_pass(rm, [s for s, _ in g], device)
                    h, off = hs["h"][0], 0
                    for s, lp in g:
                        hh = h[off + lp - 1: off + len(s) - 1]
                        tgt = torch.from_numpy(s[lp:].astype(np.int64)).to(device)
                        nll = 0.0
                        for c0 in range(0, len(tgt), 2048):
                            lg = (hh[c0:c0 + 2048].to(Wb.dtype) @ Wb.T).float()
                            if cap:
                                lg = cap * torch.tanh(lg / cap)
                            nll += F.cross_entropy(lg, tgt[c0:c0 + 2048], reduction="sum").item()
                        per.append((nll, len(tgt)))
                        off += len(s)
            t = torch.tensor([sum(x for x, _ in per), sum(n for _, n in per)], device=device, dtype=torch.float64)
            dist.all_reduce(t)
            res[mode] = (t[0] / t[1]).item()
            res[mode + "_per"] = [x / n for x, n in per]
        set_eager(False)
        hook.remove()
        del Wb
        dmax = torch.tensor([max(abs(x - y) for x, y in zip(res["packed_per"], res["eager_per"]))], device=device)
        dist.all_reduce(dmax, op=dist.ReduceOp.MAX)
        log("LM check, next-token loss on {} validation answers: packed attention {:.4f} nats/token, transformers "
            "eager attention {:.4f}; largest per-answer difference {:.4f}".format(2 * world, res["packed"], res["eager"],
                                                                                 dmax.item()))
        summary["lm"] = {"packed": res["packed"], "eager": res["eager"], "max_diff": dmax.item()}

    # ------------------------------------------------------------------ gradients of the step-1 batch
    G0 = None
    rew0 = None
    g0_tag = None
    plan_tests = [t for t in tests if re.fullmatch(r"train\d+", t)] + (["single"] if "single" in tests else [])
    for tag in plan_tests:                          # trainN: the plan of an N-rank job; single: sequences alone
        pw, micro = (int(tag[5:]), a.micro_tokens) if tag != "single" else (world, 1)
        if pw % world:
            log("{}: skipped, {} ranks cannot run a {}-rank plan".format(tag, world, pw))
            continue
        rm.zero_grad(set_to_none=True)
        rew = train_grad(pw, micro)
        report(tag, stats(G0), g0_tag)
        if G0 is None:
            G0, rew0, g0_tag = snapshot(), rew, tag
        else:
            reward_diff(tag, rew, rew0, g0_tag)
        if tag == "single":
            rew_single = rew
        torch.cuda.empty_cache()
    if "coef" in tests:
        rm.zero_grad(set_to_none=True)
        base = rew_single if "single" in tests else rew0
        rew = coef_grad(idx, coefs(base, idx), packed=False)
        report("coef", stats(G0), g0_tag)
        if rew0 is not None:
            reward_diff("coef", rew, rew0, g0_tag)

    # ------------------------------------------------------------------ attention paths, one at a time
    # Short pairs, the same coefficients for every variant. Attention per layer type: "kernel" is what training
    # uses (flash-attention varlen for the sliding layers, SDPA per sequence for the global ones), "ref" an
    # explicit bf16 matmul with an fp32 softmax and its own causal (and sliding-window) mask per sequence.
    if "attn" in tests:
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface
        choice = {"sliding": "kernel", "global": "kernel"}

        def ref_attn(q, k, v, scaling, window):
            if q.dtype == torch.float32 and not a.fp32_compute:     # fp32 residual stream: bf16 like the kernels
                q, k, v = (x.to(torch.bfloat16) for x in (q, k, v))
            rep = q.shape[1] // k.shape[1]
            k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
            s = torch.matmul(q, k.transpose(-1, -2)) * (scaling if scaling is not None else q.shape[-1] ** -0.5)
            i = torch.arange(q.shape[2], device=q.device)
            m = i[None, :] <= i[:, None]
            if window:
                m = m & ((i[:, None] - i[None, :]) < window)
            s = s.masked_fill(~m, float("-inf"))
            return torch.matmul(torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype), v)

        def hybrid(module, query, key, value, attention_mask, scaling=None, dropout=0.0, sliding_window=None,
                   softcap=None, **kw):
            how = choice["sliding" if sliding_window else "global"]
            if how == "ref":
                outs = [ref_attn(query[:, :, s:e], key[:, :, s:e], value[:, :, s:e], scaling, sliding_window)
                        for s, e in rm_model.PACK.segments]
                return torch.cat(outs, dim=2).transpose(1, 2).contiguous(), None
            if how == "contig":                     # SDPA per sequence on contiguous copies (global layers)
                if query.dtype == torch.float32:
                    query, key, value = (x.to(torch.bfloat16) for x in (query, key, value))
                outs = [F.scaled_dot_product_attention(query[:, :, s:e].contiguous(), key[:, :, s:e].contiguous(),
                                                       value[:, :, s:e].contiguous(), is_causal=True, scale=scaling,
                                                       enable_gqa=query.shape[1] != key.shape[1])
                        for s, e in rm_model.PACK.segments]
                return torch.cat(outs, dim=2).transpose(1, 2).contiguous(), None
            if how == "contig-flash" and sliding_window:   # flash varlen on explicitly contiguous inputs
                return rm_model.packed_attention(module, query.contiguous(), key.contiguous(), value.contiguous(),
                                                 attention_mask, scaling=scaling, dropout=dropout,
                                                 sliding_window=sliding_window, softcap=softcap, **kw)
            return rm_model.packed_attention(module, query, key, value, attention_mask, scaling=scaling,
                                             dropout=dropout, sliding_window=sliding_window, softcap=softcap, **kw)

        AttentionInterface.register("rm_hybrid", hybrid)
        AttentionMaskInterface.register("rm_hybrid", rm_model._no_mask)

        def set_impl(impl):
            for m in rm.modules():
                c = getattr(m, "config", None)
                if c is not None and hasattr(c, "_attn_implementation"):
                    c._attn_implementation = impl
            tcfg._attn_implementation = impl

        short = [int(i) for i in order[a.batch:] if lens_a[i] <= a.short_len and lens_b[i] <= a.short_len][:a.batch]
        log("attention paths on {} short pairs ({} tokens, longest sequence {})".format(
            len(short), int(tokens[short].sum()), int(max(max(lens_a[short]), max(lens_b[short])))))
        rm.zero_grad(set_to_none=True)
        if a.fp32_compute:                           # flash-attention needs bf16
            choice.update(sliding="ref", **{"global": "ref"})
            set_impl("rm_hybrid")
        c = coefs(coef_grad(short, None, packed=True, grad=False), short)
        set_impl("rm_packed")
        refs, rews = {}, {}
        variants = [("single ref/ref", False, "ref", "ref", None),
                    ("packed ref/ref", True, "ref", "ref", "single ref/ref"),
                    ("single kernel/kernel", False, "kernel", "kernel", "single ref/ref"),
                    ("packed kernel/kernel", True, "kernel", "kernel", "single kernel/kernel"),
                    ("single kernel-sliding/ref-global", False, "kernel", "ref", "single ref/ref"),
                    ("single ref-sliding/kernel-global", False, "ref", "kernel", "single ref/ref"),
                    ("single transformers-eager", False, None, None, "single ref/ref")]
        if a.attn_variants == "localize":
            variants = [("packed ref/ref", True, "ref", "ref", None),
                        ("packed kernel-sliding/ref-global", True, "kernel", "ref", "packed ref/ref"),
                        ("packed ref-sliding/kernel-global", True, "ref", "kernel", "packed ref/ref"),
                        ("packed ref-sliding/contiguous-SDPA-global", True, "ref", "contig", "packed ref/ref"),
                        ("packed contiguous-flash-sliding/ref-global", True, "contig-flash", "ref", "packed ref/ref"),
                        ("packed kernel-sliding/contiguous-SDPA-global", True, "kernel", "contig", "packed ref/ref")]
        if a.fp32_compute:
            variants = [v for v in variants if "kernel" not in (v[2], v[3]) and v[2] != "contig-flash"]
        for tag, packed, sl, gl, ref_tag in variants:
            rm.zero_grad(set_to_none=True)
            if sl is None:
                set_eager(True)
            else:
                choice.update(sliding=sl, **{"global": gl})
                set_impl("rm_hybrid")
            rews[tag] = coef_grad(short, c, packed=packed)
            set_eager(False)
            torch.cuda.empty_cache()
            report(tag, stats(refs.get(ref_tag)), ref_tag if ref_tag in refs else None)
            if ref_tag in rews:
                reward_diff(tag, rews[tag], rews[ref_tag], ref_tag)
            if tag in ("single ref/ref", "single kernel/kernel"):
                refs[tag] = snapshot()
        set_impl("rm_packed")
        refs.clear()

    # ------------------------------------------------------------------ which parts need fp32 (with --fp32-compute)
    # Short pairs, every sequence alone, exact attention. Against the all-fp32 gradient, each variant computes one
    # group of operations in bf16 (autocast): the attention core (q k^T, softmax, p v), the attention projections
    # (q, k, v, o), the MLP, or all of them in the early or late layers.
    if "prec" in tests and a.fp32_compute and type(rm.backbone).__name__.startswith("Gemma4"):
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface
        from transformers.models.gemma4 import modeling_gemma4 as G
        prec = {"core": False, "proj": False, "mlp": False, "layers": range(10 ** 6)}
        for i, layer in enumerate(rm.backbone.layers):
            layer.self_attn._rm_layer = i
            layer.mlp._rm_layer = i

        def wrap(orig, kind):
            def f(self, *args, **kw):
                on = prec[kind] and self._rm_layer in prec["layers"]
                # no weight cache: inside RewardModel.forward's (disabled) autocast context it would keep a bf16
                # copy of every unsharded fp32 weight until the end of the forward pass (out of memory)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=on, cache_enabled=False):
                    return orig(self, *args, **kw)
            return f
        G.Gemma4TextAttention.forward = wrap(G.Gemma4TextAttention.forward, "proj")
        G.Gemma4TextMLP.forward = wrap(G.Gemma4TextMLP.forward, "mlp")

        def exact(module, query, key, value, attention_mask, scaling=None, dropout=0.0, sliding_window=None, **kw):
            bf = prec["core"] and module.layer_idx in prec["layers"]
            dt = torch.bfloat16 if bf else torch.float32
            outs = []
            for s0, e0 in rm_model.PACK.segments:
                q, k, v = (x[:, :, s0:e0].to(dt) for x in (query, key, value))
                rep = q.shape[1] // k.shape[1]
                k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
                with torch.autocast("cuda", enabled=False):
                    sc = torch.matmul(q, k.transpose(-1, -2)) * (scaling if scaling is not None else q.shape[-1] ** -0.5)
                    i = torch.arange(q.shape[2], device=q.device)
                    m = i[None, :] <= i[:, None]
                    if sliding_window:
                        m = m & ((i[:, None] - i[None, :]) < sliding_window)
                    pr = torch.softmax(sc.masked_fill(~m, float("-inf")), dim=-1, dtype=torch.float32).to(dt)
                    outs.append(torch.matmul(pr, v).float())
            return torch.cat(outs, dim=2).transpose(1, 2).contiguous(), None
        AttentionInterface.register("rm_exact", exact)
        AttentionMaskInterface.register("rm_exact", rm_model._no_mask)

        def set_impl2(impl):
            for m in rm.modules():
                c = getattr(m, "config", None)
                if c is not None and hasattr(c, "_attn_implementation"):
                    c._attn_implementation = impl
            tcfg._attn_implementation = impl

        short = [int(i) for i in order[a.batch:] if lens_a[i] <= a.short_len and lens_b[i] <= a.short_len][:a.batch]
        set_impl2("rm_exact")
        rm.zero_grad(set_to_none=True)
        c = coefs(coef_grad(short, None, packed=True, grad=False), short)
        L = len(rm.backbone.layers)
        # coefficients per sequence: the loss's own (pairs), all of them positive, the first answers only
        cset = {"pair": c, "plus": c.abs(), "first": c * (torch.arange(len(c), device=c.device) % 2 == 0)}
        # (tag, autocast groups, attention: exact per sequence or the training kernels, coefficients)
        variants = [("all fp32", {}, "rm_exact", "pair"),
                    ("attention core bf16", {"core": True}, "rm_exact", "pair"),
                    ("attention projections bf16", {"proj": True}, "rm_exact", "pair"),
                    ("MLP bf16", {"mlp": True}, "rm_exact", "pair"),
                    ("everything bf16 in layers 0-14", {"core": True, "proj": True, "mlp": True, "layers": range(15)},
                     "rm_exact", "pair"),
                    ("everything bf16 in layers 15-59", {"core": True, "proj": True, "mlp": True,
                                                         "layers": range(15, L)}, "rm_exact", "pair"),
                    ("everything bf16", {"core": True, "proj": True, "mlp": True}, "rm_exact", "pair")]
        if a.prec_variants == "coef":
            tp = {"proj": True, "mlp": True}         # with the kernels: the precision of train_rm.py --fp32-residual
            variants = []
            for ck, what in (("pair", "pair coefficients"), ("plus", "all coefficients positive"),
                             ("first", "first answers only")):
                variants += [("all fp32, " + what, {}, "rm_exact", ck),
                             ("training precision (bf16 matmuls, kernels), " + what, tp, "rm_packed", ck)]
        truths = {}
        for tag, cfg, impl, ck in variants:
            prec.update({"core": False, "proj": False, "mlp": False, "layers": range(10 ** 6)})
            prec.update(cfg)
            set_impl2(impl)
            rm.zero_grad(set_to_none=True)
            r = coef_grad(short, cset[ck], packed=False)
            truth = truths.get(ck)
            report("precision: " + tag, stats(truth[0] if truth else None), truth[2] if truth else None)
            if truth is None:
                truths[ck] = (snapshot(), r, tag)
            else:
                reward_diff(tag, r, truth[1], truth[2])
            torch.cuda.empty_cache()
        set_impl2("rm_packed")
        del truths

    # ------------------------------------------------------------------ short pairs: patched attention vs eager
    if "short" in tests:
        short = [int(i) for i in order[a.batch:] if lens_a[i] <= a.short_len and lens_b[i] <= a.short_len][:a.batch]
        log("short pairs: {} pairs, {} tokens, longest sequence {}".format(
            len(short), int(tokens[short].sum()), int(max(max(lens_a[short]), max(lens_b[short])))))
        rm.zero_grad(set_to_none=True)
        r_nog = coef_grad(short, None, packed=True, grad=False)
        c = coefs(r_nog, short)
        rm.zero_grad(set_to_none=True)
        r_packed = coef_grad(short, c, packed=True)
        report("short-packed", stats())
        Gs = snapshot()
        rm.zero_grad(set_to_none=True)
        set_eager(True)
        r_eager = coef_grad(short, c, packed=False)
        set_eager(False)
        report("short-eager", stats(Gs), "short-packed")
        reward_diff("short-eager", r_eager, r_packed, "short-packed")
        del Gs

    # ------------------------------------------------------------------ update sensitivity
    if "update" in tests and G0 is not None and g0_tag != "single":
        lrs = [float(x) for x in a.update_lrs.split(",")]
        opt = torch.optim.AdamW(params, lr=lrs[0], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, foreach=True)
        for k in range(len(lrs) + 1):
            rm.zero_grad(set_to_none=True)
            train_grad(int(g0_tag[5:]), a.micro_tokens)
            report("after {} update{}".format(k, "" if k == 1 else "s") + (" (repeat of {})".format(g0_tag) if k == 0 else ""),
                   stats(G0), g0_tag)
            if k < len(lrs):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                for g in opt.param_groups:
                    g["lr"] = lrs[k]
                opt.step()
        rm.zero_grad(set_to_none=True)

    if rank == 0 and a.out:
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=1)
    log("done")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
