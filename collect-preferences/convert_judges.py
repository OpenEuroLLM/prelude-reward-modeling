#!/usr/bin/env python3
"""Download and convert the two judge candidates that vLLM 0.26 on LUMI cannot load as published (login node, CPU).

  collect-preferences/run.sh convert_judges.py mistral-small-4     # Mistral Small 4 119B: FP8 -> bf16
  collect-preferences/run.sh convert_judges.py apertus-1.5         # Apertus v1.5 70B: text-only Apertus v1 layout

mistral-small-4 (mistralai/Mistral-Small-4-119B-2603). Published in FP8 (E4M3, one scale per tensor), which the
MI250X cannot compute with, and in HF format as a `mistral4` model that vLLM 0.26 does not know. Its native
format (params.json, consolidated-*.safetensors, tekken.json) maps to vLLM's MistralLarge3ForCausalLM (MLA
attention, mixture of experts with a shared expert), which vLLM 0.26 supports. Every FP8 weight is multiplied by
its qscale_weight (vLLM reads that as weight_scale: dequantised weight = FP8 weight x scale) and stored in bf16;
the activation scales (qscale_act) are dropped and params.json loses its "quantization" block. Nothing else
changes. Run with tokenizer_mode, config_format and load_format "mistral".

apertus-1.5 (swiss-ai/Apertus-v1.5-70B). Published as Apertus1p5ForConditionalGeneration with vision and audio
tokenizers, which vLLM 0.26 does not know. Its language model has exactly the tensors of Apertus v1
(ApertusForCausalLM: xIELU MLP, QK-norm), so the text-only checkpoint renames model.language_model.* to model.*,
keeps lm_head, drops the vision and audio tokenizers and cuts the input embedding to the 131,072 text tokens:
the rows above are image and audio codes that a text prompt never contains, and the output layer has only
131,072 rows. config.json is the Apertus v1 layout (taken from data/models/apertus-8b-2509) with v1.5's
dimensions and RoPE. The script checks that every special token of the chat template lies below 131,072.

Sources go to data/models/_src/<name>/ and are kept; the result goes to data/models/<name>/ with
_download_info.json (repo, revision, what was converted) marked complete, as fetch_judges.py writes it.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "0"      # run_gpu.sh sets it to 1; huggingface_hub reads it when first imported

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "data", "models")
SPECS = {
    "mistral-small-4": dict(name="mistral-small-4-119b", repo="mistralai/Mistral-Small-4-119B-2603",
                            revision="a11f36bebf709121056b1dbcc943d1c6afbe494d",
                            allow=["params.json", "consolidated-*.safetensors", "consolidated.safetensors.index.json",
                                   "tekken.json", "generation_config.json", "chat_template.jinja", "SYSTEM_PROMPT.txt",
                                   "README.md", "processor_config.json"]),
    "apertus-1.5": dict(name="apertus-1.5-70b", repo="swiss-ai/Apertus-v1.5-70B",
                        revision="b212ebf8758bcd8200cc26b83a3f14759a94aa56",
                        allow=["config.json", "generation_config.json", "chat_template.jinja", "tokenizer.json",
                               "tokenizer_config.json", "special_tokens_map.json", "model.safetensors.index.json",
                               "model-apertus-model-*.safetensors", "LICENSE.txt", "README.md"]),
}
TEXT_VOCAB = 131072


def download(spec):
    from huggingface_hub import snapshot_download
    src = os.path.join(MODELS, "_src", spec["name"])
    t0 = time.time()
    snapshot_download(repo_id=spec["repo"], revision=spec["revision"], local_dir=src, max_workers=16,
                      allow_patterns=spec["allow"])
    print("{}: downloaded {} @ {} in {:.0f} min -> {}".format(spec["name"], spec["repo"], spec["revision"][:12],
                                                             (time.time() - t0) / 60, src), flush=True)
    return src


def finish(spec, dst, what):
    json.dump({"repo": spec["repo"], "revision": spec["revision"], "converted_by": "collect-preferences/convert_judges.py",
               "conversion": what, "downloaded": time.strftime("%Y-%m-%dT%H:%M:%S"), "complete": True},
              open(os.path.join(dst, "_download_info.json"), "w"), indent=1)
    size = sum(os.path.getsize(p) for p in glob.glob(os.path.join(dst, "*")) if os.path.isfile(p)) / 1e9
    print("{}: {:.0f} GB in {}".format(spec["name"], size, dst), flush=True)


class Reader:
    """one tensor at a time from a safetensors file, read with seek/read rather than a memory map: the login node
    limits a process to 64 GiB of virtual memory, and a mapped 48 GB shard plus its converted copy exceeded it"""
    DT = None

    def __init__(self, path):
        import struct
        import torch
        Reader.DT = Reader.DT or {"F8_E4M3": torch.float8_e4m3fn, "BF16": torch.bfloat16, "F16": torch.float16,
                                  "F32": torch.float32, "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8,
                                  "BOOL": torch.bool}
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.header.pop("__metadata__", None)
        self.base = 8 + n

    def keys(self):
        return list(self.header)

    def dtype(self, k):
        return self.header[k]["dtype"]

    def get(self, k):
        import torch
        h = self.header[k]
        a, b = h["data_offsets"]
        self.f.seek(self.base + a)
        buf = bytearray(self.f.read(b - a))
        t = torch.frombuffer(buf, dtype=torch.uint8) if buf else torch.empty(0, dtype=torch.uint8)
        return t.view(self.DT[h["dtype"]]).reshape(h["shape"])


class Writer:
    """collects tensors into shards of about `limit` bytes, written as soon as they are full, then renamed to
    <prefix>-NNNNN-of-NNNNN.safetensors with an index"""

    def __init__(self, dst, prefix, index_name, limit=4e9):
        self.dst, self.prefix, self.index_name, self.limit = dst, prefix, index_name, limit
        self.cur, self.size, self.parts, self.total = {}, 0, [], 0

    def add(self, k, t):
        n = t.numel() * t.element_size()
        if self.cur and self.size + n > self.limit:
            self.flush()
        self.cur[k] = t.contiguous()
        self.size += n
        self.total += n

    def flush(self):
        from safetensors.torch import save_file
        if not self.cur:
            return
        tmp = os.path.join(self.dst, "{}.part{:05d}.tmp".format(self.prefix, len(self.parts)))
        save_file(self.cur, tmp, metadata={"format": "pt"})
        os.chmod(tmp, 0o660)  # save_file creates 0600; the rest of data/models is group-readable
        self.parts.append((tmp, list(self.cur)))
        print("  part {}: {} tensors, {:.1f} GB".format(len(self.parts), len(self.cur), self.size / 1e9), flush=True)
        self.cur, self.size = {}, 0

    def close(self):
        self.flush()
        wm = {}
        for i, (tmp, keys) in enumerate(self.parts):
            name = "{}-{:05d}-of-{:05d}.safetensors".format(self.prefix, i + 1, len(self.parts))
            os.replace(tmp, os.path.join(self.dst, name))
            wm.update({k: name for k in keys})
        json.dump({"metadata": {"total_size": self.total}, "weight_map": wm},
                  open(os.path.join(self.dst, self.index_name), "w"), indent=1)
        return wm


def clean(dst, patterns):
    for pat in patterns:
        for p in glob.glob(os.path.join(dst, pat)):
            os.remove(p)


def mistral_small_4(spec):
    import torch
    src = download(spec)
    dst = os.path.join(MODELS, spec["name"])
    os.makedirs(dst, exist_ok=True)
    clean(dst, ["consolidated*", "*.tmp"])
    wm = json.load(open(os.path.join(src, "consolidated.safetensors.index.json")))["weight_map"]
    readers = {f: Reader(os.path.join(src, f)) for f in sorted(set(wm.values()))}
    w = Writer(dst, "consolidated", "consolidated.safetensors.index.json")
    n_fp8 = n_dropped = 0
    for k in sorted(wm):
        if k.endswith(".qscale_weight") or k.endswith(".qscale_act"):
            n_dropped += 1
            continue
        r = readers[wm[k]]
        t = r.get(k)
        if r.dtype(k) == "F8_E4M3":
            sk = k[:-len("weight")] + "qscale_weight"
            assert k.endswith(".weight") and sk in wm, "no scale for " + k
            s = readers[wm[sk]].get(sk).float()
            assert s.numel() == 1, (sk, s.shape)
            t = (t.float() * s).to(torch.bfloat16)
            assert torch.isfinite(t).all(), k
            n_fp8 += 1
        w.add(k, t)
    new_map = w.close()
    kept = [k for k in wm if not (k.endswith(".qscale_weight") or k.endswith(".qscale_act"))]
    assert set(new_map) == set(kept), "tensors lost"
    params = json.load(open(os.path.join(src, "params.json")))
    q = params.pop("quantization", None)
    json.dump(params, open(os.path.join(dst, "params.json"), "w"), indent=2)
    for f in ("tekken.json", "generation_config.json", "chat_template.jinja", "SYSTEM_PROMPT.txt", "README.md", "processor_config.json"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    print("{}: {} FP8 weights dequantised to bf16, {} scale tensors dropped, {} tensors kept; quantization block removed: {}".format(
        spec["name"], n_fp8, n_dropped, len(new_map), q), flush=True)
    finish(spec, dst, "FP8 (E4M3, per-tensor qscale_weight) -> bf16 weight x scale; qscale_act dropped; "
                      "params.json without 'quantization'; native Mistral format (consolidated-*.safetensors)")


def apertus_15(spec):
    from transformers import AutoTokenizer
    src = download(spec)
    dst = os.path.join(MODELS, spec["name"])
    os.makedirs(dst, exist_ok=True)
    clean(dst, ["model-*.safetensors", "*.tmp"])
    c15 = json.load(open(os.path.join(src, "config.json")))
    t15 = c15["text_config"]
    cfg = json.load(open(os.path.join(MODELS, "apertus-8b-2509", "config.json")))         # Apertus v1 layout
    for k in list(cfg):
        if k in t15 and k not in ("architectures", "model_type", "transformers_version", "torch_dtype", "dtype"):
            cfg[k] = t15[k]
    rp = t15["rope_parameters"]
    cfg["rope_theta"] = rp["rope_theta"]
    cfg["rope_scaling"] = {"factor": rp["factor"], "high_freq_factor": rp["high_freq_factor"], "low_freq_factor": rp["low_freq_factor"],
                           "original_max_position_embeddings": rp["original_max_position_embeddings"],
                           "rope_type": rp["rope_type"], "type": rp["rope_type"]}
    cfg["vocab_size"] = TEXT_VOCAB
    cfg["tie_word_embeddings"] = bool(c15.get("tie_word_embeddings"))
    cfg["architectures"], cfg["model_type"] = ["ApertusForCausalLM"], "apertus"
    cfg.pop("rope_parameters", None)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json"):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    # every token the chat template can produce, and the end-of-sequence ids, must be text tokens
    tok = AutoTokenizer.from_pretrained(src)
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}, {"role": "assistant", "content": "A"},
            {"role": "user", "content": "V"}]
    hi = 0
    for kw in ({"enable_thinking": True}, {"enable_thinking": False}):
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
        hi = max(hi, max(tok(text, add_special_tokens=False)["input_ids"]))
    marks = {m: tok.convert_tokens_to_ids(m) for m in ("<|inner_prefix|>", "<|inner_suffix|>", "<|assistant_end|>")}
    eos = cfg.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    assert hi < TEXT_VOCAB and max(marks.values()) < TEXT_VOCAB and max(eos) < TEXT_VOCAB, (hi, marks, eos)
    print("{}: template token ids up to {}, thinking markers {}, eos {}: all text tokens".format(spec["name"], hi, marks, eos), flush=True)
    wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    pre = "model.language_model."
    want = sorted(k for k in wm if k.startswith(pre) or k == "lm_head.weight")
    readers = {f: Reader(os.path.join(src, f)) for f in sorted({wm[k] for k in want})}
    w = Writer(dst, "model", "model.safetensors.index.json")
    for k in want:
        t = readers[wm[k]].get(k)
        nk = "model." + k[len(pre):] if k.startswith(pre) else k
        if nk == "model.embed_tokens.weight":
            t = t[:TEXT_VOCAB].clone()
        if nk == "lm_head.weight":
            assert t.shape[0] == TEXT_VOCAB, t.shape
        w.add(nk, t)
    new_map = w.close()
    assert len(new_map) == len(want)
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    print("{}: {} language-model tensors kept, {} vision/audio tensors dropped; config {}".format(
        spec["name"], len(new_map), len(wm) - len(want), {k: cfg[k] for k in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                                                                              "num_key_value_heads", "vocab_size", "rope_theta")}), flush=True)
    finish(spec, dst, "Apertus1p5ForConditionalGeneration -> text-only ApertusForCausalLM: model.language_model.* -> model.*, "
                      "lm_head kept, vision/audio tokenizers dropped, input embedding cut to the 131,072 text tokens, "
                      "Apertus v1 config layout with v1.5 dimensions and RoPE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=sorted(SPECS))
    args = ap.parse_args()
    {"mistral-small-4": mistral_small_4, "apertus-1.5": apertus_15}[args.model](SPECS[args.model])


if __name__ == "__main__":
    sys.exit(main())
