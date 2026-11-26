#!/usr/bin/env python3
"""A faster PyTorch version of transformers' torch_chunk_gated_delta_rule (Qwen3.5/3.8 linear attention).

The reference implementation inverts the unit lower-triangular matrix of each chunk with a 63-step Python loop
(forward substitution, with an autograd node per step). The result of that loop is (I - L)^-1 for the
strictly lower-triangular L it starts from, so one batched triangular solve gives the same matrix. Everything
else is the reference code unchanged. Checked against the reference on random inputs (check()).
"""
import torch
import torch.nn.functional as F


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
                           use_qk_l2norm_in_kernel=False, **kwargs):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                         for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    attn = torch.linalg.solve_triangular(eye - attn, eye.expand_as(attn), upper=False, unitriangular=True)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype,
                                        device=value.device) if initial_state is None else initial_state.to(value))
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    outs = []
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        outs.append(attn_inter + attn @ v_new)
        last_recurrent_state = (last_recurrent_state * g[:, :, i, -1, None, None].exp()
                                + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new)
    core_attn_out = torch.stack(outs, dim=2)
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def check(device="cpu", T=300, H=4, D=32):
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule as ref
    torch.manual_seed(0)
    q, k = torch.randn(1, T, H, D, device=device), torch.randn(1, T, H, D, device=device)
    v = torch.randn(1, T, H, 2 * D, device=device)
    g = -F.softplus(torch.randn(1, T, H, device=device))
    b = torch.sigmoid(torch.randn(1, T, H, device=device))
    ins = [x.clone().requires_grad_(True) for x in (q, k, v, g, b)]
    ins2 = [x.clone().requires_grad_(True) for x in (q, k, v, g, b)]
    o1, _ = ref(*ins, use_qk_l2norm_in_kernel=True)
    o2, _ = chunk_gated_delta_rule(*ins2, use_qk_l2norm_in_kernel=True)
    w = torch.randn_like(o1)
    (o1 * w).sum().backward()
    (o2 * w).sum().backward()
    err = (o1 - o2).abs().max().item() / o1.abs().max().item()
    gerr = max((a.grad - b.grad).abs().max().item() / max(1e-12, a.grad.abs().max().item()) for a, b in zip(ins, ins2))
    return err, gerr


if __name__ == "__main__":
    print("relative max error: output {:.2e}, gradients {:.2e}".format(*check()))
