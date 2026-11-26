#!/usr/bin/env python3
"""Qwen3.8 linear-attention layer on a packed row: the varlen forward (projections over all tokens; fla's short
convolution and gated delta rule with cu_seqlens) against the per-sequence loop, on one GPU.
Compares outputs and gradients (input, in_proj_qkv, conv1d, A_log, out_proj) and times forward + backward.
Lengths are drawn from the training data's sequence lengths (median about 730 tokens) up to 16k per row."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rm_model  # noqa: E402

rm_model.patch_gated_deltanet()
from transformers.models.qwen3_5 import modeling_qwen3_5 as M  # noqa: E402

dev = torch.device("cuda", 0)
_, tcfg = rm_model.text_config("qwen3.8-27b")
torch.manual_seed(0)
layer = M.Qwen3_5GatedDeltaNet(tcfg, layer_idx=0)
with torch.no_grad():
    for n, p in layer.named_parameters():
        if n == "A_log":
            p.copy_(torch.log(torch.empty_like(p).uniform_(1, 16)))
        elif n == "dt_bias":
            p.fill_(0.5)
        elif "norm" in n:
            p.fill_(1.0)
        else:
            p.normal_(0, 0.02)
layer = layer.to(dev, torch.bfloat16)
cls = M.Qwen3_5GatedDeltaNet


def run(fn, x, w):
    for p in layer.parameters():
        p.grad = None
    xi = x.clone().requires_grad_(True)
    o = fn(layer, xi)
    (o.float() * w).sum().backward()
    grads = {"input": xi.grad.float().clone()}
    for n in ("in_proj_qkv.weight", "conv1d.weight", "A_log", "out_proj.weight"):
        grads[n] = dict(layer.named_parameters())[n].grad.float().clone()
    return o.float(), grads


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


rng = torch.Generator().manual_seed(1)
for trial, total in enumerate((4000, 16384, 16384)):
    lens = []
    while sum(lens) < total:
        lens.append(int(min(16384, max(8, torch.distributions.LogNormal(6.6, 1.0).sample().item()))))
    lens[-1] -= sum(lens) - total
    lens = [n for n in lens if n > 0]
    if trial == 2:
        lens = [5, 1, 3, 2000, 7, 16384 - 2016]          # tiny segments (shorter than the conv kernel)
    rm_model.set_pack(lens, dev)
    T = sum(lens)
    x = torch.randn(1, T, tcfg.hidden_size, device=dev, dtype=torch.bfloat16)
    w = torch.randn(1, T, tcfg.hidden_size, device=dev)
    o1, g1 = run(cls._forward_loop, x, w)
    o2, g2 = run(cls._forward_varlen, x, w)
    print("T={} in {} sequences: out {:.1e}; grads {}".format(
        T, len(lens), rel(o2, o1), ", ".join("{} {:.1e}".format(k, rel(g2[k], g1[k])) for k in g1)), flush=True)
    for name, fn in (("loop", cls._forward_loop), ("varlen", cls._forward_varlen)):
        run(fn, x, w)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(3):
            run(fn, x, w)
        torch.cuda.synchronize()
        print("   {:6s} fwd+bwd {:.3f}s".format(name, (time.time() - t) / 3), flush=True)
