#!/usr/bin/env python3
"""The reward model: a decoder backbone (text part only) plus a linear value head on the last token of each answer.

Training packs several sequences into one row without padding. Every attention layer then needs the
sequence boundaries:
  - head dim <= 256 (Qwen3.8 full attention, Gemma 4 sliding layers, Prelude/Qwen3): flash-attention varlen
    with the boundaries (and the sliding window where the layer has one);
  - head dim 512 (Gemma 4 global layers): PyTorch SDPA per sequence (its flash backend supports 512 on MI250X);
  - Qwen3.5/3.8 Gated DeltaNet layers: the transformers implementation run once per sequence, so that no state
    crosses a boundary. There are no flash-linear-attention kernels in the container, so the PyTorch fallback is
    used, with its per-chunk matrix inversion done by one triangular solve (gdn_fast.py, same result).
The boundaries are set in PACK before each forward pass and read by these patched layers (also during the
recomputation of gradient checkpointing, which happens before the next forward).

Weights are created on the meta device, sharded with FSDP2 and filled shard by shard from the safetensors
files, so that no process holds the full model. LoRA (optional) is a plain wrapper around nn.Linear.
"""
import json
import math
import os
import re
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "data", "models")
# flash-linear-attention (Triton kernels for Qwen3.5/3.8's Gated DeltaNet), from the v0.6.0 source release
# (github.com/fla-org/flash-linear-attention, MIT), unpacked outside the container and the repository. The
# scripts run with `python -I`, which ignores PYTHONPATH, so the path is added here, before transformers loads
# the Qwen code. FLA_PATH= (empty) disables it; then gdn_fast.py's PyTorch version is used.
FLA_PATH = os.environ.get("FLA_PATH", "/pfs/lustrep3/scratch/project_465003466/dasamuel/vendor/fla-v0.6.0/"
                          "flash-linear-attention-0.6.0")
if FLA_PATH and os.path.isdir(os.path.join(FLA_PATH, "fla")) and FLA_PATH not in sys.path:
    sys.path.insert(0, FLA_PATH)


class _Pack:
    segments = None      # list of (start, end) in the packed row
    cu = None            # int32 cumulative boundaries on the device
    cu64 = None          # the same as int64 (flash-linear-attention)
    max_len = 0


PACK = _Pack()


def set_pack(lengths, device):
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    PACK.segments = list(zip(starts[:-1], starts[1:]))
    PACK.cu = torch.tensor(starts, dtype=torch.int32, device=device)
    PACK.cu64 = PACK.cu.long()
    PACK.max_len = max(lengths)


def packed_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, sliding_window=None,
                     softcap=None, **kwargs):
    """query [1, H, T, D], key/value [1, KVH, T, D] -> ([1, T, H, D], None)"""
    if query.dtype == torch.float32:        # fp32 residual stream: RoPE promotes q/k to fp32; the kernels run bf16
        query, key, value = (x.to(torch.bfloat16) for x in (query, key, value))
    d = query.shape[-1]
    if d <= 256:
        from flash_attn import flash_attn_varlen_func
        q = query[0].transpose(0, 1)
        k = key[0].transpose(0, 1)
        v = value[0].transpose(0, 1)
        win = (sliding_window - 1, 0) if sliding_window else (-1, -1)
        o = flash_attn_varlen_func(q, k, v, PACK.cu, PACK.cu, PACK.max_len, PACK.max_len, dropout_p=0.0,
                                   softmax_scale=scaling, causal=True, window_size=win,
                                   softcap=softcap if softcap else 0.0)
        return o.unsqueeze(0), None
    if sliding_window:
        raise NotImplementedError("sliding window with head dim > 256")
    outs = []
    for s, e in PACK.segments:
        outs.append(F.scaled_dot_product_attention(query[:, :, s:e], key[:, :, s:e], value[:, :, s:e], is_causal=True,
                                                   scale=scaling, enable_gqa=query.shape[1] != key.shape[1]))
    return torch.cat(outs, dim=2).transpose(1, 2).contiguous(), None


def _no_mask(*args, **kwargs):
    return None


def register_attention():
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    AttentionInterface.register("rm_packed", packed_attention)
    AttentionMaskInterface.register("rm_packed", _no_mask)


def patch_gated_deltanet():
    """run Qwen3.5/3.8 linear-attention layers once per packed sequence"""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    import gdn_fast
    cls = M.Qwen3_5GatedDeltaNet
    if getattr(cls, "_rm_patched", False):
        return
    if M.chunk_gated_delta_rule is None:          # no flash-linear-attention: the triangular-solve version
        M.torch_chunk_gated_delta_rule = gdn_fast.chunk_gated_delta_rule
    # fla's fused gated RMSNorm is created directly on the GPU (device= in its constructor), which breaks
    # meta-device initialisation; the plain PyTorch norm computes the same thing
    M.FusedRMSNormGated = None
    orig = cls.forward

    def forward_loop(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        """the whole layer once per packed sequence (fallback without flash-linear-attention)"""
        segs = PACK.segments
        if not segs or len(segs) == 1:
            return orig(self, hidden_states, cache_params=None, attention_mask=None)
        return torch.cat([orig(self, hidden_states[:, s:e], cache_params=None, attention_mask=None) for s, e in segs],
                         dim=1)

    try:
        from fla.modules.causal_conv1d import causal_conv1d as fla_conv
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_gdr
    except ImportError:
        fla_conv = fla_gdr = None

    def forward_varlen(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        """the layer once on the packed row: projections over all tokens, the short convolution and the gated delta
        rule with the sequence boundaries (flash-linear-attention varlen kernels, cu_seqlens)"""
        B, T, _ = hidden_states.shape
        cu = PACK.cu64 if PACK.segments else torch.tensor([0, T], dtype=torch.long, device=hidden_states.device)
        mixed_qkv = self.in_proj_qkv(hidden_states)
        z = self.in_proj_z(hidden_states).reshape(B, T, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        mixed_qkv, _ = fla_conv(mixed_qkv, weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias,
                                activation=self.activation, cu_seqlens=cu)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(B, T, -1, self.head_k_dim)
        key = key.reshape(B, T, -1, self.head_k_dim)
        value = value.reshape(B, T, -1, self.head_v_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        core, _ = fla_gdr(query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False,
                          use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
        core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        return self.out_proj(core.reshape(B, T, -1))

    use_varlen = fla_conv is not None and fla_gdr is not None and os.environ.get("GDN_LOOP") != "1"
    cls.forward = forward_varlen if use_varlen else forward_loop
    cls._forward_loop, cls._forward_varlen = forward_loop, forward_varlen
    cls._rm_patched = True


# ---------------------------------------------------------------------------------------------------- LoRA

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.empty(base.out_features, r, dtype=torch.float32, device=base.weight.device))

    def reset_lora(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.scale


LORA_TARGETS = r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|in_proj_qkv|in_proj_z|out_proj)$"


def add_lora(module, r, alpha, targets=LORA_TARGETS):
    n = 0
    for name, child in list(module.named_modules()):
        for cname, c in list(child.named_children()):
            if isinstance(c, nn.Linear) and re.search(targets, cname):
                setattr(child, cname, LoRALinear(c, r, alpha))
                n += 1
    return n


# ---------------------------------------------------------------------------------------------------- model

class RewardModel(nn.Module):
    def __init__(self, backbone, hidden):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden, 1)
        self.is_gemma4 = type(backbone).__name__.startswith("Gemma4")
        self.fp32_residual = False

    def use_fp32_residual(self):
        """keep the residual stream (embeddings, the residual additions, the layer scalars, the norms) in fp32 and
        run only the matrix multiplications and attention in bf16 (autocast). Needs FSDP without casting the
        layers' inputs (cast_forward_inputs=False). For Gemma 4: its early layers add attention outputs scaled by
        norm weights of ~1e-4 to a residual stream of order 1-100, below bf16 resolution."""
        self.fp32_residual = True
        self.backbone.embed_tokens.register_forward_hook(lambda m, i, o: o.float())

    def forward(self, input_ids, position_ids, reward_index):
        """input_ids, position_ids [1, T] (packed); reward_index [n]: position of each sequence's last token"""
        kw = {}
        if self.is_gemma4:
            kw["attention_mask"] = {"full_attention": None, "sliding_attention": None}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.fp32_residual):
            h = self.backbone(input_ids=input_ids, position_ids=position_ids, use_cache=False, **kw).last_hidden_state
        return self.head(h[0, reward_index].float()).squeeze(-1).float()


def text_config(model):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(os.path.join(MODELS, model))
    return cfg, getattr(cfg, "text_config", None) or cfg


def build(model, dtype=torch.float32, lora_rank=0, lora_alpha=None):
    """-> RewardModel on the meta device"""
    register_attention()
    cfg, tcfg = text_config(model)
    tcfg._attn_implementation = "rm_packed"
    tcfg.use_cache = False
    if cfg.model_type == "qwen3_5":
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel as Cls
        patch_gated_deltanet()
    elif cfg.model_type == "gemma4":
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel as Cls
    else:
        from transformers import AutoModel
        Cls = None
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with torch.device("meta"):
            backbone = Cls(tcfg) if Cls is not None else AutoModel.from_config(tcfg, attn_implementation="rm_packed")
            rm = RewardModel(backbone, tcfg.hidden_size)
            if lora_rank:
                add_lora(rm.backbone, lora_rank, lora_alpha or lora_rank)
    finally:
        torch.set_default_dtype(prev)
    rm.head.float()
    assert backbone.config._attn_implementation == "rm_packed", backbone.config._attn_implementation
    return rm, tcfg


def checkpoint_index(model):
    d = os.path.join(MODELS, model)
    idx = os.path.join(d, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(d, f) for k, f in wm.items()}
    from safetensors import safe_open
    f = os.path.join(d, "model.safetensors")
    with safe_open(f, "pt") as h:
        return {k: f for k in h.keys()}


def find_prefix(names, keys):
    for pre in ("model.language_model.", "model.", ""):
        if all(pre + n in keys for n in names[:50]):
            return pre
    raise SystemExit("cannot map model parameters to checkpoint keys, e.g. {}".format(names[:3]))


def _ckpt_name(name):
    """parameter name inside the backbone -> name in the original checkpoint (LoRA wraps .base)"""
    return name.replace(".base.", ".")


def load_weights(rm, model, rank, world, head_std=None):
    """fill the (FSDP2-sharded, materialised) parameters and buffers from the safetensors files"""
    from safetensors import safe_open
    from torch.distributed.tensor import DTensor
    index = checkpoint_index(model)
    names = [n for n, _ in rm.backbone.named_parameters() if ".lora_" not in n]
    prefix = find_prefix([_ckpt_name(n) for n in names], index)
    handles = {}

    def tensor_slice(key, start, end):
        f = index[key]
        if f not in handles:
            handles[f] = safe_open(f, "pt", device="cpu")
        return handles[f].get_slice(key)[start:end]

    with torch.no_grad():
        for name, p in rm.backbone.named_parameters():
            if ".lora_" in name:
                continue
            key = prefix + _ckpt_name(name)
            if isinstance(p, DTensor):
                local = p.to_local()
                full0 = p.shape[0]
                chunk = math.ceil(full0 / world)
                start = min(rank * chunk, full0)
                end = min(start + local.shape[0], full0)
                if local.numel():
                    local.copy_(tensor_slice(key, start, end).to(local.dtype))
            else:
                p.copy_(handles.setdefault(index[key], safe_open(index[key], "pt", device="cpu")).get_tensor(key).to(p.dtype))
        # buffers: from the checkpoint when it has them (Gemma 4 layer scalars), else recomputed
        for mname, mod in rm.backbone.named_modules():
            if "RotaryEmbedding" in type(mod).__name__:
                fresh = type(mod)(config=rm.backbone.config)
                for bname, b in fresh.named_buffers(recurse=False):
                    mod._buffers[bname] = b.to(device=getattr(mod, bname).device)
        rot = {n for n, m in rm.backbone.named_modules() if "RotaryEmbedding" in type(m).__name__}
        for name, b in rm.backbone.named_buffers():
            owner = name.rsplit(".", 1)[0] if "." in name else ""
            if owner in rot:
                continue
            key = prefix + name
            if key in index:
                b.copy_(handles.setdefault(index[key], safe_open(index[key], "pt", device="cpu")).get_tensor(key).to(b.dtype))
            elif name.endswith("embed_scale"):
                b.fill_(rm.backbone.config.hidden_size ** 0.5)
            else:
                raise SystemExit("buffer {} has no source".format(name))
        # LoRA and the head
        for m in rm.modules():
            if isinstance(m, LoRALinear):
                _init_dtensor(m.lora_A, lambda t: nn.init.kaiming_uniform_(t, a=math.sqrt(5)), rank)
                _init_dtensor(m.lora_B, nn.init.zeros_, rank)
        hidden = rm.head.weight.shape[-1]
        std = head_std if head_std is not None else 1.0 / (hidden + 1)
        _init_dtensor(rm.head.weight, lambda t: nn.init.normal_(t, 0.0, std), rank, seed=1234)
        _init_dtensor(rm.head.bias, nn.init.zeros_, rank)
    return prefix


def _init_dtensor(p, fn, rank, seed=4321):
    """initialise a (possibly sharded) parameter identically on every rank: build the full tensor with a fixed
    seed on the CPU, then copy this rank's shard"""
    from torch.distributed.tensor import DTensor
    g = torch.random.get_rng_state()
    torch.manual_seed(seed + p.numel())
    full = torch.empty(p.shape, dtype=torch.float32)
    fn(full)
    torch.random.set_rng_state(g)
    if isinstance(p, DTensor):
        local = p.to_local()
        chunk = math.ceil(p.shape[0] / max(1, p.device_mesh.size()))
        start = min(p.device_mesh.get_local_rank() * chunk, p.shape[0])
        if local.numel():
            local.copy_(full[start:start + local.shape[0]].to(local.dtype))
    else:
        p.copy_(full.to(p.dtype))


def decoder_layers(rm):
    return list(rm.backbone.layers)


def save(rm, out_dir, model, rank, merge_lora=True, shard_bytes=5 * 2**30):
    """full bf16 weights in the original checkpoint's key layout (backbone) plus score.weight / score.bias,
    written by rank 0, one tensor gathered at a time"""
    from safetensors.torch import save_file
    from torch.distributed.tensor import DTensor
    index = checkpoint_index(model)
    names = [n for n, _ in rm.backbone.named_parameters() if ".lora_" not in n]
    prefix = find_prefix([_ckpt_name(n) for n in names], index)

    def full(p):
        return p.full_tensor() if isinstance(p, DTensor) else p.detach()

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    shard, size, files, wmap = {}, 0, [], {}

    def flush():
        nonlocal shard, size
        if rank == 0 and shard:
            fn = "model-{:05d}.safetensors".format(len(files) + 1)
            save_file(shard, os.path.join(out_dir, fn), metadata={"format": "pt"})
            files.append(fn)
            wmap.update({k: fn for k in shard})
        shard, size = {}, 0

    lora = {n: m for n, m in rm.backbone.named_modules() if isinstance(m, LoRALinear)}
    with torch.no_grad():
        for name, p in rm.backbone.named_parameters():
            if ".lora_" in name:
                continue
            t = full(p).float()
            owner = name.rsplit(".", 2)[0] if name.endswith(".base.weight") else None
            if merge_lora and owner in lora:
                m = lora[owner]
                t = t + (full(m.lora_B).float() @ full(m.lora_A).float()) * m.scale
            if rank == 0:
                shard[prefix + _ckpt_name(name)] = t.to(torch.bfloat16).cpu().contiguous()
                size += t.numel() * 2
            if size > shard_bytes:
                flush()
        for name, b in rm.backbone.named_buffers():
            if rank == 0 and (prefix + name) in index:
                shard[prefix + name] = b.detach().cpu().contiguous()
        w, bias = full(rm.head.weight).float().cpu(), full(rm.head.bias).float().cpu()
        if rank == 0:
            shard["score.weight"], shard["score.bias"] = w.contiguous(), bias.contiguous()
        flush()
    if rank == 0:
        with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": wmap}, f, indent=1)
        src = os.path.join(MODELS, model)
        for fn in os.listdir(src):
            if fn.endswith((".json", ".jinja", ".txt", ".model")) and not fn.startswith("model.safetensors"):
                if fn not in ("_download_info.json",):
                    import shutil
                    shutil.copy(os.path.join(src, fn), os.path.join(out_dir, fn))
        with open(os.path.join(out_dir, "reward_model.json"), "w") as f:
            json.dump({"base_model": model, "head": "score.weight / score.bias (float32) on the last hidden state "
                       "of the answer's end-of-turn token", "prefix": prefix}, f, indent=1)
