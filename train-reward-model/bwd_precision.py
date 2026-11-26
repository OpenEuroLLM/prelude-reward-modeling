#!/usr/bin/env python3
"""Where Gemma 4's backward pass loses its precision. transformers' own model on one node (device_map over the
GPUs, one process), eager attention, a few validation sequences one at a time, each in several precisions:

  fp32       fp32 weights and compute: the reference (run twice on the first sequence: must be identical)
  residual   bf16 autocast over fp32 weights: bf16 matmuls (projections, MLP, attention scores and values), fp32
             residual stream, norms and softmax (the precision of the training run's --fp32-residual)
  core32     residual, but the attention core (scores, softmax, values) in fp32
  band0/1/2  autocast only in layers 0-19 / 20-39 / 40-59, fp32 elsewhere
  bf16       bf16 weights and compute

Two scalar objectives per sequence: a reward (a fixed random unit vector on the final hidden state of the last
token, as the value head reads it) and the mean next-token loss of the answer (fp32 logits in every mode).
Per layer, against fp32: the per-token cosine of the layer output and of the branch outputs (attention before
post_attention_layernorm, MLP before post_feedforward_layernorm), and the cosine and norm ratio of the gradient
of the layer output (all tokens) and of the post-norm weight gradients. For fp32 itself: the size of the branch
outputs and how much the per-token terms of the post_attention_layernorm weight gradient cancel.

  sbatch train-reward-model/slurm/bwd_precision.sbatch
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rm_model  # noqa: E402
import train_rm as T  # noqa: E402

BANDS = {"band0": (0, 20), "band1": (20, 40), "band2": (40, 60)}
CUR = {"mode": "fp32", "loss": None, "eager_calls": 0}
T0 = time.time()


def log(*a):
    print("[{:6.1f} min]".format((time.time() - T0) / 60), *a, flush=True)


def autocast_layer(l):
    m = CUR["mode"]
    if m in ("residual", "core32"):
        return True
    if m in BANDS:
        return BANDS[m][0] <= l < BANDS[m][1]
    return False


def tokcos(a, b):
    return F.cosine_similarity(a.double(), b.double(), dim=-1)


def flatcos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    na, nb = a.norm(), b.norm()
    return float(a @ b / (na * nb).clamp_min(1e-300)), float(na / nb.clamp_min(1e-300))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma-4-31B-it")
    ap.add_argument("--lengths", default="400-800,800-1400,1400-2048", help="one sequence per token-length range")
    ap.add_argument("--modes", default="residual,core32,band0,band1,band2,bf16")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from transformers import AutoModelForImageTextToText
    import transformers.models.gemma4.modeling_gemma4 as G

    meta = json.load(open(os.path.join(T.DATA, "tok", a.model, "meta.json")))
    val = T.Pairs(a.model, "val", meta["suffix_ids"])
    ranges = [tuple(int(x) for x in r.split("-")) for r in a.lengths.split(",")]
    seqs = [None] * len(ranges)
    for i in range(val.n):
        lp = len(val.get("prompt_ids", i))
        for s in T.pair_seqs(val, i, 16384):
            for k, (lo, hi) in enumerate(ranges):
                if seqs[k] is None and lo <= len(s) < hi and len(s) - lp >= 64:
                    seqs[k] = (val.ids[i], s, lp)
        if all(x is not None for x in seqs):
            break
    log("sequences:", [(pid, len(s), len(s) - lp) for pid, s, lp in seqs], "(pair, tokens, answer tokens)")

    path = os.path.join(rm_model.MODELS, a.model)
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.float32, device_map="auto",
                                                        attn_implementation="eager")
    model.eval()
    lm = model.model.language_model
    cfg = lm.config
    cap = cfg.final_logit_softcapping
    H = cfg.hidden_size
    L = cfg.num_hidden_layers
    kinds = ["G" if t == "full_attention" else "S" for t in cfg.layer_types]
    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for l, layer in enumerate(lm.layers):
        for nm in ("post_attention_layernorm", "post_feedforward_layernorm"):
            w = getattr(layer, nm).weight
            w.requires_grad_(True)
            params.append((l, nm, w))
    W32 = model.lm_head.weight.detach().clone().float()       # the tied embedding, fp32, for the LM loss in every mode
    V = torch.randn(H, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    V = (V / V.norm()).float()
    log("loaded: {} layers, global layers {}, devices {}".format(
        L, [l for l in range(L) if kinds[l] == "G"], sorted({str(p.device) for p in lm.parameters()})))

    # precision switches: autocast per decoder layer, fp32 attention core for core32
    for l, layer in enumerate(lm.layers):
        orig = layer.forward

        def fwd(*args, _orig=orig, _l=l, **kw):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_layer(_l)):
                return _orig(*args, **kw)
        layer.forward = fwd
    orig_eager = G.eager_attention_forward

    def eager(module, query, key, value, attention_mask, **kw):
        CUR["eager_calls"] += 1
        if CUR["mode"] == "core32":
            with torch.autocast("cuda", enabled=False):
                m = attention_mask.float() if attention_mask is not None and attention_mask.is_floating_point() \
                    else attention_mask
                return orig_eager(module, query.float(), key.float(), value.float(), m, **kw)
        return orig_eager(module, query, key, value, attention_mask, **kw)
    G.eager_attention_forward = eager

    # captures
    ST = {}

    def put(key, t):
        ST[key] = t.detach().float().cpu()

    def on_layer(l, out):
        h = out[0] if isinstance(out, tuple) else out
        put(("h", l), h[0])
        if h.requires_grad:
            h.register_hook(lambda g, l=l: put((CUR["loss"], "g", l), g[0]))

    def on_mid(l, out):
        if CUR["mode"] == "fp32" and out.requires_grad:
            out.register_hook(lambda g, l=l: put(("reward", "gmid", l), g[0]) if CUR["loss"] == "reward" else None)

    for l, layer in enumerate(lm.layers):
        layer.register_forward_hook(lambda mod, inp, out, l=l: on_layer(l, out))
        layer.self_attn.register_forward_hook(lambda mod, inp, out, l=l: put(("attn", l), out[0][0]))
        layer.mlp.register_forward_hook(lambda mod, inp, out, l=l: put(("mlp", l), out[0]))
        layer.post_attention_layernorm.register_forward_hook(lambda mod, inp, out, l=l: on_mid(l, out))

    def run(seq, lp):
        ST.clear()
        dev0 = lm.embed_tokens.weight.device
        ids = torch.from_numpy(seq.astype(np.int64))[None].to(dev0)
        pos = torch.arange(len(seq), device=dev0)[None]
        out = lm(input_ids=ids, position_ids=pos, use_cache=False).last_hidden_state[0]
        hl = out[-1].float()
        r = (hl * V.to(hl.device)).sum()
        with torch.autocast("cuda", enabled=False):
            hh = out[lp - 1:len(seq) - 1].float().to(W32.device)
            lg = hh @ W32.T
            lg = cap * torch.tanh(lg / cap)
            nll = F.cross_entropy(lg, ids[0, lp:].to(W32.device))
        ws = [w for _, _, w in params]
        CUR["loss"] = "reward"
        gR = torch.autograd.grad(r, ws, retain_graph=True)
        CUR["loss"] = "lm"
        gL = torch.autograd.grad(nll, ws)
        CUR["loss"] = None
        res = dict(ST)
        res["wR"] = [g.detach().float().cpu() for g in gR]
        res["wL"] = [g.detach().float().cpu() for g in gL]
        res["r"], res["nll"] = r.item(), nll.item()
        ST.clear()
        return res

    def facts(ref):
        """fp32 only: branch sizes and the cancellation in the post_attention_layernorm weight gradient"""
        rows = []
        for l in range(L):
            at, ml, h = ref[("attn", l)], ref[("mlp", l)], ref[("h", l)]
            g = ref[("reward", "gmid", l)]
            xh = at / at.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
            terms = (g * xh).double()
            canc = float(terms.sum(0).norm() / terms.abs().sum(0).norm().clamp_min(1e-300))
            rows.append({"attn_rms": float(at.pow(2).mean(-1).sqrt().median()),
                         "mlp_rms": float(ml.pow(2).mean(-1).sqrt().median()),
                         "h_rms": float(h.pow(2).mean(-1).sqrt().median()),
                         "h_rms_max": float(h.pow(2).mean(-1).sqrt().max()),
                         "gR_norm": float(ref[("reward", "g", l)].norm()),
                         "gL_norm": float(ref[("lm", "g", l)].norm()),
                         "canc_postattn_R": canc})
        return rows

    def compare(res, ref):
        rows = []
        for l in range(L):
            ca = tokcos(res[("attn", l)], ref[("attn", l)])
            rel = ((res[("attn", l)] - ref[("attn", l)]).norm(dim=-1) / ref[("attn", l)].norm(dim=-1).clamp_min(1e-30))
            gr = flatcos(res[("reward", "g", l)], ref[("reward", "g", l)])
            gl = flatcos(res[("lm", "g", l)], ref[("lm", "g", l)])
            last = flatcos(res[("reward", "g", l)][-1], ref[("reward", "g", l)][-1])
            k = 2 * l
            rows.append({"h_cos": float(tokcos(res[("h", l)], ref[("h", l)]).mean()),
                         "attn_cos": float(ca.mean()), "attn_cos_p05": float(ca.quantile(0.05)),
                         "attn_relerr_med": float(rel.median()),
                         "mlp_cos": float(tokcos(res[("mlp", l)], ref[("mlp", l)]).mean()),
                         "gR_cos": gr[0], "gR_ratio": gr[1], "gR_last_cos": last[0],
                         "gL_cos": gl[0], "gL_ratio": gl[1],
                         "wR_postattn": flatcos(res["wR"][k], ref["wR"][k])[0],
                         "wR_postff": flatcos(res["wR"][k + 1], ref["wR"][k + 1])[0],
                         "wL_postattn": flatcos(res["wL"][k], ref["wL"][k])[0],
                         "wL_postff": flatcos(res["wL"][k + 1], ref["wL"][k + 1])[0]})
        allR = flatcos(torch.cat([g.flatten() for g in res["wR"]]), torch.cat([g.flatten() for g in ref["wR"]]))
        allL = flatcos(torch.cat([g.flatten() for g in res["wL"]]), torch.cat([g.flatten() for g in ref["wL"]]))
        return {"layers": rows, "wR_all": allR, "wL_all": allL, "dr": res["r"] - ref["r"], "r": ref["r"],
                "dnll": res["nll"] - ref["nll"], "nll": ref["nll"]}

    out = {"sequences": [(pid, len(s), len(s) - lp) for pid, s, lp in seqs], "modes": {}}
    refs = []
    CUR["mode"] = "fp32"
    for pid, s, lp in seqs:
        refs.append(run(s, lp))
        log("fp32 {}: reward {:.4f}, nll {:.4f}".format(pid, refs[-1]["r"], refs[-1]["nll"]))
    if CUR["eager_calls"] == 0:
        raise SystemExit("the patched eager attention was never called: the core32 switch would do nothing")
    out["facts"] = [facts(r) for r in refs]
    rep = compare(run(seqs[0][1], seqs[0][2]), refs[0])
    log("fp32 repeat: reward diff {:.2e}, gradient cos min over layers {:.6f}, norm-weight grads cos {:.6f}/{:.6f}".format(
        rep["dr"], min(x["gR_cos"] for x in rep["layers"]), rep["wR_all"][0], rep["wL_all"][0]))
    out["modes"]["fp32-repeat"] = [rep]

    modes = a.modes.split(",")
    for mode in modes:
        if mode == "bf16":
            for p in model.parameters():
                p.data = p.data.to(torch.bfloat16)
            for layer in lm.layers:
                layer.layer_scalar.data = layer.layer_scalar.data.to(torch.bfloat16)
            torch.cuda.empty_cache()
        CUR["mode"] = mode
        out["modes"][mode] = []
        for (pid, s, lp), ref in zip(seqs, refs):
            c = compare(run(s, lp), ref)
            out["modes"][mode].append(c)
            log("{:9s} {}: reward diff {:+.2e} (reward {:.3f}), nll diff {:+.2e}; norm-weight grads vs fp32: reward cos {:.4f} "
                "(norm ratio {:.3f}), LM cos {:.4f} (ratio {:.3f}); layer-output grads: reward cos at layer 59/30/0 "
                "{:.3f}/{:.3f}/{:.3f}, LM {:.3f}/{:.3f}/{:.3f}".format(
                    mode, pid, c["dr"], c["r"], c["dnll"], c["wR_all"][0], c["wR_all"][1], c["wL_all"][0], c["wL_all"][1],
                    c["layers"][59]["gR_cos"], c["layers"][30]["gR_cos"], c["layers"][0]["gR_cos"],
                    c["layers"][59]["gL_cos"], c["layers"][30]["gL_cos"], c["layers"][0]["gL_cos"]))

    # tables: averages over the sequences
    def avg(rows_per_seq, key, l):
        return float(np.mean([r["layers"][l][key] for r in rows_per_seq]))
    F_ = out["facts"]
    print("\n== fp32: branch sizes (median token rms), residual-gradient norms, cancellation of the post_attention_layernorm "
          "weight gradient (|sum of per-token terms| / sum of |terms|), mean over sequences")
    print("layer    attn_rms   mlp_rms     h_rms   h_rms_max  |gR|      |gL|      cancel")
    for l in range(L):
        m = {k: float(np.mean([f[l][k] for f in F_])) for k in F_[0][l]}
        print("{:3d} {}  {:9.2e} {:9.2e} {:9.2e} {:9.2e} {:9.2e} {:9.2e} {:9.2e}".format(
            l, kinds[l], m["attn_rms"], m["mlp_rms"], m["h_rms"], m["h_rms_max"], m["gR_norm"], m["gL_norm"],
            m["canc_postattn_R"]))
    for mode in modes:
        R = out["modes"][mode]
        print("\n== {}: per layer, mean over sequences (cos with fp32)".format(mode))
        print("layer  h_cos   attn_cos(p05)  attn_relerr mlp_cos | gR_cos  gR_ratio gR_last | gL_cos  gL_ratio | "
              "w_postattn R/L   w_postff R/L")
        for l in range(L):
            print("{:3d} {}  {:.4f}  {:.4f}({:.3f})  {:9.2e}  {:.4f} | {:+.3f}  {:7.2f}  {:+.3f} | {:+.3f}  {:7.2f} | "
                  "{:+.3f}/{:+.3f}  {:+.3f}/{:+.3f}".format(
                      l, kinds[l], avg(R, "h_cos", l), avg(R, "attn_cos", l), avg(R, "attn_cos_p05", l),
                      avg(R, "attn_relerr_med", l), avg(R, "mlp_cos", l), avg(R, "gR_cos", l), avg(R, "gR_ratio", l),
                      avg(R, "gR_last_cos", l), avg(R, "gL_cos", l), avg(R, "gL_ratio", l), avg(R, "wR_postattn", l),
                      avg(R, "wL_postattn", l), avg(R, "wR_postff", l), avg(R, "wL_postff", l)))
    print("\n== summary: cosine with fp32 of all post-norm weight gradients (reward / LM), mean over sequences")
    for mode in ["fp32-repeat"] + modes:
        R = out["modes"][mode]
        print("  {:11s} reward {:+.4f}  LM {:+.4f}   |dr| {:.2e}  |dnll| {:.2e}".format(
            mode, np.mean([r["wR_all"][0] for r in R]), np.mean([r["wL_all"][0] for r in R]),
            np.mean([abs(r["dr"]) for r in R]), np.mean([abs(r["dnll"]) for r in R])))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f)
    log("done")


if __name__ == "__main__":
    main()
