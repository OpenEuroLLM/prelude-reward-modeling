import torch, time, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
dev = "cuda"
H, KVH, D = 32, 4, 512
for L in (4096, 16384):
    for name, be in (("default", None), ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("flash", SDPBackend.FLASH_ATTENTION), ("math", SDPBackend.MATH)):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        q = torch.randn(1, H, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, KVH, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, KVH, L, D, device=dev, dtype=torch.bfloat16, requires_grad=True)
        base = torch.cuda.max_memory_allocated()
        try:
            t = time.time()
            if be is None:
                o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:
                with sdpa_kernel(be):
                    o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            o.float().sum().backward(); torch.cuda.synchronize()
            print(f"L={L} {name:9s} ok  peak extra {(torch.cuda.max_memory_allocated()-base)/1e9:.1f} GB  {time.time()-t:.2f}s", flush=True)
        except Exception as e:
            print(f"L={L} {name:9s} FAIL {type(e).__name__}: {str(e)[:120]}", flush=True)
        del q, k, v
        o = None
