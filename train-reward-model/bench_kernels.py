#!/usr/bin/env python3
"""Kernel and layer benchmarks on one MI250X GCD, to choose the attention paths for reward-model training.

  srun ... train-reward-model/run.sh bench_kernels.py

1. scaled_dot_product_attention with head dim 512 (Gemma 4's global layers): which backend runs, time, memory.
2. flash-attention varlen with a sliding window, head dim 256 (Gemma 4's sliding layers, Qwen's full layers).
3. One decoder layer of each type of Qwen3.8-27B and Gemma 4 31B (random weights, bf16), forward + backward
   on one packed sequence, to extrapolate tokens per second for the whole model.
"""
import json
import os
import time

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dev = torch.device("cuda", 0)


def timed(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / reps, torch.cuda.max_memory_allocated() / 2**30


def sdpa_512():
    from torch.nn.attention import SDPBackend, sdpa_kernel
    for L in (2048, 4096, 8192, 16384):
        q = torch.randn(1, 32, L, 512, device=dev, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 4, L, 512, device=dev, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, 4, L, 512, device=dev, dtype=torch.bfloat16, requires_grad=True)
        for name, be in [("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                         ("math", SDPBackend.MATH)]:
            def run():
                with sdpa_kernel(be):
                    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                o.sum().backward()
            try:
                s, mem = timed(run, reps=2)
                print("sdpa d=512 L={:>6} {:>9}: {:.3f}s fwd+bwd, peak {:.1f} GiB".format(L, name, s, mem), flush=True)
            except Exception as e:
                print("sdpa d=512 L={:>6} {:>9}: FAILED {}".format(L, name, str(e).split("\n")[0][:150]), flush=True)
            torch.cuda.empty_cache()
        del q, k, v


def flash_window():
    from flash_attn import flash_attn_varlen_func
    for L, w in ((8192, 1024), (8192, None)):
        q = torch.randn(L, 32, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(L, 16, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(L, 16, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
        cu = torch.tensor([0, L // 4, L // 2, L], device=dev, dtype=torch.int32)
        def run():
            o = flash_attn_varlen_func(q, k, v, cu, cu, L // 2, L // 2, causal=True,
                                       window_size=(w - 1, 0) if w else (-1, -1))
            o.sum().backward()
        try:
            s, mem = timed(run)
            print("flash varlen d=256 L={} window={}: {:.3f}s fwd+bwd, peak {:.1f} GiB".format(L, w, s, mem), flush=True)
        except Exception as e:
            print("flash varlen window={}: FAILED {}".format(w, str(e)[:200]), flush=True)


def layer_bench(model, kinds):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(os.path.join(ROOT, "data", "models", model)).text_config
    cfg._attn_implementation = "flash_attention_2"
    if model.startswith("qwen"):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import rm_model
        if os.environ.get("GDN_FAST", "1") == "1":
            rm_model.patch_gated_deltanet()       # the triangular-solve gated delta rule (gdn_fast.py)
        from transformers.models.qwen3_5 import modeling_qwen3_5 as M
        Layer, Rot = M.Qwen3_5DecoderLayer, M.Qwen3_5TextRotaryEmbedding
    else:
        from transformers.models.gemma4 import modeling_gemma4 as M
        Layer, Rot = M.Gemma4TextDecoderLayer, M.Gemma4TextRotaryEmbedding
    rot = Rot(config=cfg).to(dev)
    out = {}
    for kind in kinds:
        idx = cfg.layer_types.index(kind)
        layer = Layer(cfg, idx).to(dev, torch.bfloat16)
        n = sum(p.numel() for p in layer.parameters())
        for L in (4096, 8192):
            x = torch.randn(1, L, cfg.hidden_size, device=dev, dtype=torch.bfloat16, requires_grad=True)
            pos = torch.arange(L, device=dev)[None]
            if model.startswith("qwen"):
                pe = rot(x, pos[None].expand(3, 1, L))
                kw = dict(position_embeddings=pe, attention_mask=None, position_ids=pos)
            else:
                pe = rot(x, pos, kind)
                kw = dict(position_embeddings=pe, attention_mask=None, position_ids=pos, shared_kv_states={})

            def run():
                y = layer(x, **kw)
                y = y[0] if isinstance(y, tuple) else y
                y.float().pow(2).mean().backward()
            try:
                s, mem = timed(run)
                tflops = 6 * n * L / s / 1e12
                print("{} layer {} ({:.0f}M params) L={}: {:.3f}s fwd+bwd, {:.0f} matmul TFLOP/s, peak {:.1f} GiB".format(
                    model, kind, n / 1e6, L, s, tflops, mem), flush=True)
                out[(kind, L)] = s
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("{} layer {} L={}: FAILED {}".format(model, kind, L, str(e)[:300]), flush=True)
            del x
            torch.cuda.empty_cache()
        del layer
    counts = {k: cfg.layer_types.count(k) for k in kinds}
    for L in (4096, 8192):
        if all((k, L) in out for k in kinds):
            tot = sum(out[(k, L)] * counts[k] for k in kinds)
            print("{} whole model (layers only, no recompute) L={}: {:.1f}s per sequence -> {:.0f} tokens/s per GCD".format(
                model, L, tot, L / tot), flush=True)


if __name__ == "__main__":
    import sys
    print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0), flush=True)
    if "--layers-only" not in sys.argv:
        flash_window()
        sdpa_512()
    layer_bench("qwen3.8-27b", ["linear_attention", "full_attention"])
    if "--layers-only" not in sys.argv:
        layer_bench("gemma-4-31B-it", ["sliding_attention"])
