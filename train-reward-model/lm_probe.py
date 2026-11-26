#!/usr/bin/env python3
"""Where the next-token loss of validation answers comes from (transformers' own model, one node, one process):
per-token losses of the best, the median and the worst answer, the model's top predictions at the first answer
positions, and the loss with the answer placed after Gemma 4's empty thinking block (the non-thinking format)."""
import json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rm_model  # noqa: E402
import train_rm as T  # noqa: E402

def main():
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    name = sys.argv[1] if len(sys.argv) > 1 else "gemma-4-31B-it"
    path = os.path.join(rm_model.MODELS, name)
    tok = AutoTokenizer.from_pretrained(path)
    meta = json.load(open(os.path.join(T.DATA, "tok", name, "meta.json")))
    val = T.Pairs(name, "val", meta["suffix_ids"])
    cand = []
    for i in range(val.n):
        pr = val.get("prompt_ids", i)
        for col in ("a_ids", "b_ids"):
            r = val.get(col, i)
            if len(pr) + len(r) + 1 <= 8192 and len(r) + 1 >= 32:
                cand.append((i, col, pr, r))
        if len(cand) >= 32:
            break
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    model.eval()
    dev0 = model.model.language_model.embed_tokens.weight.device
    think = tok("<|channel>thought\n<channel|>", add_special_tokens=False)["input_ids"]
    print("empty thinking block ids:", think, [tok.convert_ids_to_tokens(x) for x in think], flush=True)

    def nll(seq, start):
        with torch.no_grad():
            ids = torch.tensor(seq, dtype=torch.long)[None].to(dev0)
            lg = model(input_ids=ids, use_cache=False).logits[0].float()
            tgt = torch.tensor(seq[start:], dtype=torch.long, device=lg.device)
            return F.cross_entropy(lg[start - 1:len(seq) - 1], tgt, reduction="none").cpu().numpy(), lg[start - 1:start + 4].cpu()

    rows = []
    for i, col, pr, r in cand:
        seq = list(pr) + list(r) + meta["suffix_ids"]
        per, first = nll(seq, len(pr))
        seq2 = list(pr) + think + list(r) + meta["suffix_ids"]
        per2, _ = nll(seq2, len(pr) + len(think))
        rows.append((per.mean(), per2.mean(), i, col, pr, r, per, first))
    m1 = np.mean([np.concatenate([x[6]]).mean() for x in rows])
    tot1 = sum(x[6].sum() for x in rows) / sum(len(x[6]) for x in rows)
    tot2 = sum(x[1] * len(x[6]) for x in rows) / sum(len(x[6]) for x in rows)
    print("next-token loss of {} answers: as in the RM data {:.4f} nats/token; after the empty thinking block {:.4f}".format(len(rows), tot1, tot2), flush=True)
    rows.sort(key=lambda x: x[0])
    for lab, x in (("best", rows[0]), ("median", rows[len(rows) // 2]), ("worst", rows[-1])):
        mean1, mean2, i, col, pr, r, per, first = x
        print("\n==== {} answer: val pair {} {} ({} answer tokens): loss {:.3f}, after the thinking block {:.3f}".format(lab, i, col, len(r), mean1, mean2))
        print("prompt end:", repr(tok.decode(pr[-30:])))
        print("answer start:", repr(tok.decode(r[:60])))
        top = torch.topk(first[0], 5)
        print("model's top predictions for the first answer token:", [(tok.convert_ids_to_tokens(int(k)), round(float(v), 1)) for v, k in zip(top.values, top.indices)])
        q = np.percentile(per, [10, 50, 90])
        print("per-token loss percentiles 10/50/90: {:.2f} / {:.2f} / {:.2f}; first 10 tokens: {}".format(*q, np.round(per[:10], 1).tolist()))
        worst = np.argsort(-per)[:8]
        print("highest-loss tokens:", [(int(k), tok.convert_ids_to_tokens(int((list(r) + meta['suffix_ids'])[k])), round(float(per[k]), 1)) for k in sorted(worst)])
    print("done", flush=True)

if __name__ == "__main__":
    main()
