#!/usr/bin/env python3
"""Correctness of the packed attention paths on one GPU, outputs and gradients, against an fp32 reference that
processes each sequence on its own with an explicit mask:
  sliding  flash-attention varlen, head dim 256, causal window 1024 (Gemma 4 sliding layers)
  full     flash-attention varlen, head dim 256, causal (Qwen3.8 / Prelude full attention)
  global   SDPA per sequence, head dim 512, causal (Gemma 4 global layers)
  gdr      flash-linear-attention chunk_gated_delta_rule vs the PyTorch reference (same output weighting this time)
Relative errors are ||x - ref|| / ||ref||; bf16 inputs give about 1e-2 at most.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
dev = torch.device("cuda", 0)


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()


def ref_attn(q, k, v, window=None, scale=None):
    """q [L, H, D], k/v [L, KVH, D] fp32 -> [L, H, D]"""
    L, H, D = q.shape
    rep = H // k.shape[1]
    k, v = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    s = torch.einsum("ihd,jhd->hij", q, k) * (scale if scale is not None else D ** -0.5)
    i = torch.arange(L, device=q.device)
    m = i[None, :] <= i[:, None]
    if window:
        m &= (i[:, None] - i[None, :]) < window
    s = s.masked_fill(~m[None], float("-inf"))
    return torch.einsum("hij,jhd->ihd", s.softmax(-1), v)


def check_attn(kind, lengths, H, KVH, D, window=None):
    from rm_model import set_pack, packed_attention
    torch.manual_seed(len(lengths) * 7 + D)
    T = sum(lengths)
    q0 = torch.randn(1, H, T, D, device=dev, dtype=torch.bfloat16)
    k0 = torch.randn(1, KVH, T, D, device=dev, dtype=torch.bfloat16)
    v0 = torch.randn(1, KVH, T, D, device=dev, dtype=torch.bfloat16)
    w = torch.randn(1, T, H, D, device=dev)
    q, k, v = (x.clone().requires_grad_(True) for x in (q0, k0, v0))
    set_pack(lengths, dev)
    o, _ = packed_attention(None, q, k, v, None, scaling=D ** -0.5, sliding_window=window)
    (o.float() * w).sum().backward()
    qr, kr, vr = (x.float().clone().requires_grad_(True) for x in (q0, k0, v0))
    outs, s = [], 0
    for n in lengths:
        outs.append(ref_attn(qr[0, :, s:s + n].transpose(0, 1), kr[0, :, s:s + n].transpose(0, 1),
                             vr[0, :, s:s + n].transpose(0, 1), window=window))
        s += n
    orf = torch.cat(outs, 0)[None]
    (orf * w).sum().backward()
    print("{:8} lengths {}: out {:.1e}  grad q {:.1e} k {:.1e} v {:.1e}".format(
        kind, lengths, rel(o, orf), rel(q.grad, qr.grad), rel(k.grad, kr.grad), rel(v.grad, vr.grad)), flush=True)


def check_real(kind, lengths, H, KVH, D, window=None, contig=False, kscale=0.1, reps=2):
    """the model's own layout and scales: projections give [1, T, heads, D] that the attention receives as
    [1, heads, T, D] views (transposed, not contiguous); q and k RMS-normalised as Gemma 4 does (k scaled by its
    norm weight), attention scale 1.0. contig: SDPA on contiguous copies of each sequence. The kernel runs reps
    times on the same inputs (nondeterminism), each against the fp32 reference."""
    from rm_model import set_pack, packed_attention
    torch.manual_seed(sum(lengths) + D)
    T = sum(lengths)
    q0 = (F.normalize(torch.randn(1, T, H, D, device=dev), dim=-1) * D ** 0.5).bfloat16()
    k0 = (F.normalize(torch.randn(1, T, KVH, D, device=dev), dim=-1) * D ** 0.5 * kscale).bfloat16()
    v0 = (F.normalize(torch.randn(1, T, KVH, D, device=dev), dim=-1) * D ** 0.5).bfloat16()
    w = torch.randn(1, T, H, D, device=dev)
    qr, kr, vr = (x.float().clone().requires_grad_(True) for x in (q0, k0, v0))
    outs, s0 = [], 0
    for n in lengths:
        outs.append(ref_attn(qr[0, s0:s0 + n], kr[0, s0:s0 + n], vr[0, s0:s0 + n], window=window, scale=1.0))
        s0 += n
    orf = torch.cat(outs, 0)[None]
    (orf * w).sum().backward()
    res = []
    for _ in range(reps):
        q, k, v = (x.clone().requires_grad_(True) for x in (q0, k0, v0))
        set_pack(lengths, dev)
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if contig and not window:
            o = torch.cat([F.scaled_dot_product_attention(qt[:, :, a:b].contiguous(), kt[:, :, a:b].contiguous(),
                                                          vt[:, :, a:b].contiguous(), is_causal=True, scale=1.0,
                                                          enable_gqa=H != KVH)
                           for a, b in zip([0] + list(np.cumsum(lengths)[:-1]), np.cumsum(lengths))], dim=2).transpose(1, 2)
        else:
            o, _ = packed_attention(None, qt, kt, vt, None, scaling=1.0, sliding_window=window)
        (o.float() * w).sum().backward()
        res.append("out {:.1e} grad q {:.1e} k {:.1e} v {:.1e} (|grad k| {:.3g} vs {:.3g})".format(
            rel(o, orf), rel(q.grad, qr.grad), rel(k.grad, kr.grad), rel(v.grad, vr.grad),
            k.grad.float().norm().item(), kr.grad.norm().item()))
    print("{:14} lengths {}: {}".format(kind + ("-contig" if contig else ""), lengths, " | ".join(res)), flush=True)


def check_gdr():
    import rm_model  # noqa: F401  (fla path)
    import gdn_fast
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_gdr
    torch.manual_seed(0)
    for T in (1000, 4096):
        H, D = 48, 128
        x0 = [F.normalize(torch.randn(1, T, H, D, device=dev), dim=-1), F.normalize(torch.randn(1, T, H, D, device=dev), dim=-1),
              torch.randn(1, T, H, D, device=dev), -F.softplus(torch.randn(1, T, H, device=dev)),
              torch.sigmoid(torch.randn(1, T, H, device=dev))]
        w = torch.randn(1, T, H, D, device=dev)
        res = []
        for fn, dt in ((gdn_fast.chunk_gated_delta_rule, torch.float32), (fla_gdr, torch.bfloat16)):
            ins = [x.to(dt if j != 3 else torch.float32).clone().requires_grad_(True) for j, x in enumerate(x0)]
            o, _ = fn(*ins, use_qk_l2norm_in_kernel=True)
            (o.float() * w).sum().backward()
            res.append((o, [x.grad for x in ins]))
        (o1, g1), (o2, g2) = res
        print("gdr      T={}: out {:.1e}  grad q {:.1e} k {:.1e} v {:.1e} g {:.1e} beta {:.1e}".format(
            T, rel(o2, o1), *[rel(b, a) for a, b in zip(g1, g2)]), flush=True)


if __name__ == "__main__":
    if "--real" in sys.argv:
        import numpy as np
        for lengths in ([3000], [37, 1500, 3000, 900], [6000, 200], [1, 4000, 3, 3000], [8000], [2000] * 4):
            check_real("global-real", lengths, 32, 4, 512)
            check_real("global-real", lengths, 32, 4, 512, contig=True)
        for lengths in ([3000], [37, 1500, 3000, 900], [6000, 200], [1, 4000, 3, 3000], [8000], [2000] * 4):
            check_real("sliding-real", lengths, 32, 16, 256, window=1024)
        sys.exit(0)
    for lengths in ([3000], [37, 1500, 3000, 900], [5, 1100, 2047, 1024, 3], [6000, 200]):
        check_attn("sliding", lengths, 32, 16, 256, window=1024)
    for lengths in ([37, 1500, 3000, 900], [6000, 200]):
        check_attn("full", lengths, 24, 4, 256)
    for lengths in ([3000], [37, 1500, 3000, 900], [5, 1100, 2047, 3], [6000, 200]):
        check_attn("global", lengths, 32, 4, 512)
    check_gdr()
