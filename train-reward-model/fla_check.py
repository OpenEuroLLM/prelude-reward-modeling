#!/usr/bin/env python3
"""flash-linear-attention's chunk_gated_delta_rule against the PyTorch reference (gdn_fast.py) on one GPU: outputs
and gradients on random inputs shaped like Qwen3.8-27B's linear attention (16 key / 48 value heads, dim 128)."""
import os
import sys
import time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rm_model  # noqa: F401  (adds the fla path)
import gdn_fast
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_gdr
import fla
print("fla", fla.__version__, fla.__file__, flush=True)
dev = torch.device("cuda", 0)
torch.manual_seed(0)
for T in (1000, 4096):
    H, D = 48, 128
    q = F.normalize(torch.randn(1, T, H, D, device=dev), dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(1, T, H, D, device=dev), dim=-1).to(torch.bfloat16)
    v = torch.randn(1, T, H, D, device=dev, dtype=torch.bfloat16)
    g = (-F.softplus(torch.randn(1, T, H, device=dev))).float()
    b = torch.sigmoid(torch.randn(1, T, H, device=dev)).to(torch.bfloat16)
    res = {}
    for name, fn in (("torch", gdn_fast.chunk_gated_delta_rule), ("fla", fla_gdr)):
        ins = [x.clone().requires_grad_(True) for x in (q, k, v, g, b)]
        o, _ = fn(*ins, use_qk_l2norm_in_kernel=True) if name == "torch" else fn(*ins, use_qk_l2norm_in_kernel=True)
        w = torch.randn_like(o.float())
        (o.float() * w).sum().backward()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(3):
            ins2 = [x.clone().requires_grad_(True) for x in (q, k, v, g, b)]
            o2, _ = fn(*ins2, use_qk_l2norm_in_kernel=True)
            (o2.float() * w).sum().backward()
        torch.cuda.synchronize()
        res[name] = (o.float(), [x.grad.float() for x in ins], (time.time() - t) / 3)
    (o1, g1, t1), (o2, g2, t2) = res["torch"], res["fla"]
    rel = lambda a, b: ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    print("T={}: output rel err {:.2e}; grad rel err q {:.2e} k {:.2e} v {:.2e} g {:.2e} beta {:.2e}; "
          "fwd+bwd torch {:.3f}s, fla {:.3f}s".format(T, rel(o2, o1), *[rel(b2, b1) for b1, b2 in zip(g1, g2)], t1, t2),
          flush=True)
