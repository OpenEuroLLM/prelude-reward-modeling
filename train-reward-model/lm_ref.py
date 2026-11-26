#!/usr/bin/env python3
"""Reference check of the forward pass: transformers' own from_pretrained model (device_map over one node's GPUs)
against the pieces the reward model replaces, on validation answers.

  sbatch train-reward-model/slurm/lm_ref.sbatch           (one node, one process)

Next-token loss of the answers (nats per token) for:
  hf          the checkpoint as transformers loads it: eager attention with its own masks, the model's lm_head
              and final logit softcap (the official computation)
  hf-sdpa     the same with SDPA attention
  hf-packed   the same weights through the reward model's packed attention and mask bypass (rm_packed)
  hf-fresh    hf-packed after replacing the rotary and embed-scale buffers the way rm_model.load_weights does
  ours-state  hf-packed after loading every parameter and buffer with rm_model's own loader code path
Also: one sequence through model(...).logits against the manual lm_head path, a greedy answer to a short
question, and the differences between transformers' buffers and the ones rm_model computes.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rm_model  # noqa: E402
import train_rm as T  # noqa: E402
from rm_model import PACK, set_pack  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-31B-it")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=8192)
    a = ap.parse_args()
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface
    path = os.path.join(rm_model.MODELS, a.model)
    tok = AutoTokenizer.from_pretrained(path)

    meta = json.load(open(os.path.join(T.DATA, "tok", a.model, "meta.json")))
    val = T.Pairs(a.model, "val", meta["suffix_ids"])
    cand = []
    for i in range(val.n):
        lp = len(val.get("prompt_ids", i))
        for s in T.pair_seqs(val, i, 16384):
            if len(s) <= a.max_seq and len(s) - lp >= 32:
                cand.append((s, lp))
        if len(cand) >= a.n:
            break
    cand = cand[:a.n]
    print("{} validation answers, {} tokens".format(len(cand), sum(len(s) for s, _ in cand)), flush=True)

    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="auto",
                                                        attn_implementation="eager")
    model.eval()
    lm = model.model.language_model
    cap = model.config.get_text_config().final_logit_softcapping
    dev0 = lm.embed_tokens.weight.device
    print("loaded {} ({}), softcap {}, text model on {}".format(type(model).__name__, type(lm).__name__, cap,
                                                              sorted({str(p.device) for p in lm.parameters()})),
          flush=True)

    def head_nll(h, s, lp):
        """h [L, H] final hidden states -> summed NLL of s[lp:] through the model's lm_head and softcap"""
        tgt = torch.from_numpy(s[lp:].astype(np.int64)).to(model.lm_head.weight.device)
        hh = h[lp - 1:len(s) - 1].to(model.lm_head.weight.device)
        tot = 0.0
        for c0 in range(0, len(tgt), 1024):
            lg = model.lm_head(hh[c0:c0 + 1024]).float()
            if cap:
                lg = cap * torch.tanh(lg / cap)
            tot += F.cross_entropy(lg, tgt[c0:c0 + 1024], reduction="sum").item()
        return tot, len(tgt)

    def run(tag, packed=False, pack_two=False):
        tot = n = 0
        per = []
        with torch.no_grad():
            groups = [cand[k:k + 2] for k in range(0, len(cand), 2)] if pack_two else [[x] for x in cand]
            for g in groups:
                seqs = [s for s, _ in g]
                ids = torch.from_numpy(np.concatenate(seqs).astype(np.int64))[None].to(dev0)
                pos = torch.from_numpy(np.concatenate([np.arange(len(s)) for s in seqs]))[None].to(dev0)
                kw = {}
                if packed:
                    set_pack([len(s) for s in seqs], dev0)
                    kw["attention_mask"] = {"full_attention": None, "sliding_attention": None}
                h = lm(input_ids=ids, position_ids=pos, use_cache=False, **kw).last_hidden_state[0]
                off = 0
                for s, lp in g:
                    x, m = head_nll(h[off:off + len(s)], s, lp)
                    per.append(x / m)
                    tot += x
                    n += m
                    off += len(s)
        print("{:11s} next-token loss {:.4f} nats/token  (per answer: min {:.3f}, median {:.3f}, max {:.3f})".format(
            tag, tot / n, min(per), float(np.median(per)), max(per)), flush=True)
        return per

    def set_impl(impl):
        for m in model.modules():
            c = getattr(m, "config", None)
            if c is not None and hasattr(c, "_attn_implementation"):
                try:
                    c._attn_implementation = impl
                except Exception:
                    pass

    # the official path on one short sequence: model(...).logits against the manual lm_head path
    s, lp = min(cand, key=lambda x: len(x[0]))
    with torch.no_grad():
        ids = torch.from_numpy(s.astype(np.int64))[None].to(dev0)
        lg = model(input_ids=ids, use_cache=False).logits[0].float()
        tgt = torch.from_numpy(s[lp:].astype(np.int64)).to(lg.device)
        off_nll = F.cross_entropy(lg[lp - 1:len(s) - 1], tgt).item()
        h = lm(input_ids=ids, use_cache=False).last_hidden_state[0]
        man = head_nll(h, s, lp)
    print("one answer ({} tokens): model(...).logits {:.4f}, manual lm_head path {:.4f}".format(
        len(s), off_nll, man[0] / man[1]), flush=True)

    q = "<bos><|turn>user\nWhat is the capital of France? Answer in one sentence.<turn|>\n<|turn>model\n"
    with torch.no_grad():
        ids = tok(q, add_special_tokens=False, return_tensors="pt").input_ids.to(dev0)
        out = model.generate(ids, max_new_tokens=30, do_sample=False)
    print("greedy answer:", repr(tok.decode(out[0, ids.shape[1]:])), flush=True)

    hf = run("hf")
    set_impl("sdpa")
    run("hf-sdpa")

    # the reward model's attention (packed, mask bypass); PACK follows the query's device (device_map)
    def packed_any_device(module, query, *args, **kw):
        if PACK.cu.device != query.device:
            PACK.cu, PACK.cu64 = PACK.cu.to(query.device), PACK.cu64.to(query.device)
        return rm_model.packed_attention(module, query, *args, **kw)
    AttentionInterface.register("rm_packed", packed_any_device)
    AttentionMaskInterface.register("rm_packed", rm_model._no_mask)
    set_impl("rm_packed")
    run("hf-packed", packed=True)
    run("hf-packed2", packed=True, pack_two=True)

    # buffers as rm_model.load_weights makes them
    print("buffers in the transformers model:", {n: (str(b.dtype), tuple(b.shape)) for n, b in lm.named_buffers()
                                                 if "layers.1." not in n and not n.startswith("layers.") or n.endswith("layers.0.layer_scalar")},
          flush=True)
    for mname, mod in lm.named_modules():
        if "RotaryEmbedding" in type(mod).__name__:
            fresh = type(mod)(config=lm.config)
            for bname, b in fresh.named_buffers(recurse=False):
                old = getattr(mod, bname)
                print("  {}.{}: max |transformers - rm_model| {:.3e} ({} vs {})".format(
                    mname, bname, (old.float().cpu() - b.float()).abs().max().item(), old.dtype, b.dtype), flush=True)
                mod._buffers[bname] = b.to(device=old.device)
    es = lm.embed_tokens.embed_scale
    print("  embed_scale: transformers {} ({}), rm_model {}".format(es.item(), es.dtype, lm.config.hidden_size ** 0.5))
    es.fill_(lm.config.hidden_size ** 0.5)
    run("hf-fresh", packed=True)

    # every parameter and buffer through rm_model's loader code path (world 1: plain copies)
    from safetensors import safe_open
    index = rm_model.checkpoint_index(a.model)
    names = [n for n, _ in lm.named_parameters()]
    prefix = rm_model.find_prefix(names, index)
    handles = {}
    nd = 0
    with torch.no_grad():
        for name, p in lm.named_parameters():
            key = prefix + name
            t = handles.setdefault(index[key], safe_open(index[key], "pt", device="cpu")).get_tensor(key)
            if not torch.equal(t.to(p.dtype).cpu(), p.detach().cpu()):
                nd += 1
                if nd <= 10:
                    print("  parameter differs from the checkpoint:", name, flush=True)
            p.copy_(t.to(p.dtype))
        for name, b in lm.named_buffers():
            key = prefix + name
            if key in index:
                t = handles.setdefault(index[key], safe_open(index[key], "pt", device="cpu")).get_tensor(key)
                if not torch.equal(t.to(b.dtype).cpu(), b.detach().cpu()):
                    print("  buffer differs from the checkpoint:", name, t.flatten()[:3].tolist(), b.flatten()[:3].tolist())
                b.copy_(t.to(b.dtype))
    print("parameters that differ between transformers' model and the raw checkpoint: {}".format(nd), flush=True)
    run("ours-state", packed=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
