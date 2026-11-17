#!/usr/bin/env python3
"""Smoke test of the response-sampling pool (models.yaml): does every model load in vLLM, render our
prompts with its pool settings in each of its modes, stop properly and answer in the prompt's language?

  collect-responses/run.sh smoke_test.py prompts                    # CPU: pick the test prompts
  collect-responses/run_gpu.sh smoke_test.py templates              # CPU (GPU container): template check
  sbatch collect-responses/slurm/smoke.sbatch                       # GPU: `run` for every model
  collect-responses/run.sh smoke_test.py report                     # CPU: checks per model and mode

Test prompts (data/responses/smoke/prompts.jsonl): one short general prompt per language in LANGS, plus
system-message, multi-turn, verifiable instruction-following and safety_benign prompts (no safety_harmful
ones); all short enough for the 4k-context models.
`templates` renders conversations with every model's chat template, removes the message contents and
checks that the remaining frame (template text, injected system prompts, control tokens) names no model
or maker other than Prelude / OpenEuroLLM. Models with `default_system` get that prompt when the
conversation has no system message, so the template's own identity prompt is never used.
`run` answers every prompt in each of the model's modes and writes <model>.jsonl (reasoning and answer
split at the mode's think_end token) and <model>.meta.json; `report` adds language ID (OpenLID-v3) and
writes REPORT.txt.
"""
import argparse
import collections
import datetime
import gc
import json
import os
import random
import re
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "responses", "smoke")
LANGS = ["en", "de", "fr", "es", "it", "pl", "cs", "fi", "sv", "lv", "lt", "et", "hu", "el", "uk", "bg",
         "sr", "sl", "mt", "ga", "is", "ka", "eu", "ltg", "sq", "mk"]
MARKERS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "[INST]", "[/INST]", "<start_of_turn>", "<end_of_turn>",
           "<|user|>", "<|assistant|>", "<|system|>", "<|eot_id|>", "<|turn>", "<turn|>", "<think>", "</think>",
           "<|channel>", "<channel|>", "<|think|>"]
TURN_STARTS = ["<|im_start|>", "<start_of_turn>", "<|turn>"]
# model and maker names that must not appear in a template frame (Prelude / OpenEuroLLM are fine)
OTHER_NAMES = re.compile(r"qwen|alibaba|tongyi|apertus|swiss ?ai|salamandra|barcelona|\bbsc\b|generalitat|mistral|ministral|"
                         r"gemma|google|deepmind|openai|chatgpt|\bgpt|claude|anthropic|llama|\bmeta\b|(?<!open)eurollm|unbabel|olmo|"
                         r"allen ?ai|\bai2\b|zephyr|\bh4\b|kimi|moonshot|deepseek|cohere|\baya\b|granite|\bibm\b|"
                         r"falcon|teuken|nemotron|nvidia", re.I)


def pool():
    with open(os.path.join(HERE, "models.yaml"), encoding="utf-8") as f:
        p = yaml.safe_load(f)
    return p, {m["name"]: m for m in p["models"]}


# ------------------------------------------------------------------ prompts (CPU)
def make_prompts(seed=7):
    rows = []
    with open(os.path.join(ROOT, "data", "compiled", "prompts.jsonl"), encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o["category"] == "safety_harmful":
                continue
            chars = sum(len(m["content"] or "") for m in o["messages"])
            if chars > 2400:
                continue
            o["_chars"] = chars
            rows.append(o)
    rng = random.Random(seed)
    rng.shuffle(rows)
    picked, used = [], set()

    def take(pred, n, why):
        k = 0
        for o in rows:
            if k == n:
                break
            if o["id"] in used or not pred(o):
                continue
            used.add(o["id"])
            picked.append(dict(o, smoke_case=why))
            k += 1

    for lang in LANGS:
        take(lambda o, lang=lang: (o["language"] == lang and o["category"] == "general" and len(o["messages"]) == 1
                                   and o["locale_dependence"] != "language_specific" and o["task_type"] != "translation"
                                   and 80 <= o["_chars"] <= 1200), 1, "general")
    take(lambda o: o["messages"][0]["role"] == "system" and o["language"] != "en", 2, "system")
    take(lambda o: o["messages"][0]["role"] == "system" and o["language"] == "en", 1, "system")
    take(lambda o: sum(m["role"] == "user" for m in o["messages"]) >= 2 and o["messages"][0]["role"] != "system", 3, "multi_turn")
    take(lambda o: o["verifiable_instructions"] is not None and o["language"] != "en", 3, "if")
    take(lambda o: o["category"] == "safety_benign", 2, "safety_benign")
    os.makedirs(OUT, exist_ok=True)
    keep = ("id", "language", "messages", "category", "task_type", "locale_dependence", "smoke_case")
    with open(os.path.join(OUT, "prompts.jsonl"), "w", encoding="utf-8") as f:
        for o in picked:
            f.write(json.dumps({k: o[k] for k in keep}, ensure_ascii=False) + "\n")
    print("{} test prompts {}".format(len(picked), dict(collections.Counter(o["smoke_case"] for o in picked))))


# ------------------------------------------------------------------ rendering
def prepare_messages(p, m, date=None):
    """the conversation as it goes to the template: the pool's Prelude default system prompt for templates
    that would otherwise inject their own identity; system messages merged into the first user turn for
    templates that reject them. `date` (YYYY-MM-DD, default today) fills a default system prompt's {date}."""
    msgs = [dict(role=x["role"], content=x["content"]) for x in p["messages"]]
    if msgs[0]["role"] != "system" and m.get("default_system"):
        text = m["default_system"].replace("{date}", date or datetime.date.today().isoformat())
        msgs.insert(0, {"role": "system", "content": text})
    if m.get("system", "native") == "merge" and msgs[0]["role"] == "system":
        sys_text = msgs.pop(0)["content"].strip()
        i = next(k for k, x in enumerate(msgs) if x["role"] == "user")
        msgs[i]["content"] = sys_text + "\n\n" + msgs[i]["content"]
    return msgs


def render(tok, msgs, kwargs):
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kwargs)
    return tok.encode(text, add_special_tokens=False), text


def check_templates(names):
    """every rendered frame (template text minus message contents) must name no other model or maker"""
    from transformers import AutoTokenizer
    p, models = pool()
    convs = {"user": [{"role": "user", "content": "Hallo, wie geht es dir?"}],
             "system+user": [{"role": "system", "content": "Antworte kurz."}, {"role": "user", "content": "Hallo, wie geht es dir?"}],
             "multi-turn": [{"role": "user", "content": "Hallo."}, {"role": "assistant", "content": "Hallo! Wie kann ich helfen?"},
                            {"role": "user", "content": "Erzähl einen Witz."}]}
    bad = 0
    for name in names:
        m = models[name]
        tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "data", "models", name))
        for mode, cfg in m["modes"].items():
            for cname, conv in convs.items():
                msgs = prepare_messages({"messages": conv}, m)
                _, text = render(tok, msgs, cfg.get("template_kwargs") or {})
                frame = text
                for x in conv:
                    frame = frame.replace(x["content"], "")
                hits = sorted(set(h.lower() for h in OTHER_NAMES.findall(frame)))
                if hits:
                    bad += 1
                    print("FAIL {:36s} {:6s} {:12s} frame names {}".format(name, mode, cname, hits))
                if m.get("default_system") and cname == "user" and "Prelude" not in text:
                    bad += 1
                    print("FAIL {:36s} {:6s} {:12s} default_system not rendered".format(name, mode, cname))
        print("{:36s} ok{}".format(name, " (default_system: Prelude)" if m.get("default_system") else ""))
    print("{} problems".format(bad))
    return bad


# ------------------------------------------------------------------ run (GPU)
def find_sub(ids, sub):
    n = len(sub)
    for k in range(len(ids) - n + 1):
        if ids[k:k + n] == sub:
            return k
    return -1


def stop_ids(tok, path):
    """ids that end an answer: generation_config / tokenizer EOS, plus turn starts (a model that starts a new
    turn without ending its own must stop there)"""
    eos, turn = set(), set()
    gc_path = os.path.join(path, "generation_config.json")
    if os.path.exists(gc_path):
        e = json.load(open(gc_path)).get("eos_token_id")
        eos.update(e if isinstance(e, list) else [e] if e is not None else [])
    if getattr(tok, "eos_token_id", None) is not None:
        eos.add(tok.eos_token_id)
    for t in TURN_STARTS:
        try:
            i = tok.convert_tokens_to_ids(t)
        except Exception:
            continue
        if isinstance(i, int) and i >= 0 and i != getattr(tok, "unk_token_id", None):
            turn.add(i)
    return eos | turn, turn


def split_output(tok, ids, finish, think_end, stops):
    """-> reasoning, answer, think_closed (None in direct mode), markers leaked into the answer"""
    special = set(getattr(tok, "all_special_ids", []) or []) | stops
    if finish == "stop":
        while ids and ids[-1] in special:
            ids = ids[:-1]
    if think_end is None:
        raw = tok.decode(ids, skip_special_tokens=False)
        return "", tok.decode(ids, skip_special_tokens=True).strip(), None, [x for x in MARKERS if x in raw]
    end = tok.encode(think_end, add_special_tokens=False)
    k = find_sub(ids, end) if end else -1
    if k < 0:
        return tok.decode(ids, skip_special_tokens=True).strip(), "", False, []
    reasoning = tok.decode(ids[:k], skip_special_tokens=True)
    reasoning = re.sub(r"^\s*(<think>|thought\s*\n)", "", reasoning).strip()
    rest = ids[k + len(end):]
    raw = tok.decode(rest, skip_special_tokens=False)
    return reasoning, tok.decode(rest, skip_special_tokens=True).strip(), True, [x for x in MARKERS if x in raw]


def run_model(name, p, m, prompts, tp, eager, override=None, tag=""):
    from vllm import LLM, SamplingParams
    defaults = p["defaults"]
    path = os.path.join(ROOT, "data", "models", name)
    t0 = time.time()
    kw = dict(model=path, tokenizer=path, dtype="bfloat16", max_model_len=m["max_model_len"], tensor_parallel_size=tp,
              gpu_memory_utilization=0.88, seed=0, enforce_eager=eager, max_num_seqs=128)
    kw.update(m.get("vllm") or {})
    llm = LLM(**kw)
    load_s = time.time() - t0
    tok = llm.get_tokenizer()
    stops, turn = stop_ids(tok, path)
    sp = dict(defaults["sampling"])
    sp.update({k: v for k, v in (m.get("sampling") or {}).items() if k != "source"})
    for k, v in (override or {}).items():      # a value of null removes the parameter
        if v is None:
            sp.pop(k, None)
        else:
            sp[k] = v
    budgets = dict(defaults["max_new_tokens"])
    budgets.update(m.get("max_new_tokens") or {})
    jobs = []      # (prompt, mode, token ids, max_tokens, text)
    for mode, cfg in m["modes"].items():
        for pr in prompts:
            ids, text = render(tok, prepare_messages(pr, m), cfg.get("template_kwargs") or {})
            jobs.append((pr, mode, ids, min(budgets[mode], m["max_model_len"] - len(ids)), text))
    t1 = time.time()
    outs = llm.generate([{"prompt_token_ids": j[2]} for j in jobs],
                        [SamplingParams(max_tokens=j[3], seed=1, stop_token_ids=sorted(turn), **sp) for j in jobs])
    gen_s = time.time() - t1
    rows = []
    for (pr, mode, ids, max_new, text), o in zip(jobs, outs):
        c = o.outputs[0]
        think_end = m["modes"][mode].get("think_end") if mode == "think" else None
        reasoning, answer, closed, leaks = split_output(tok, list(c.token_ids), c.finish_reason, think_end, stops)
        rows.append({"model": name, "mode": mode, "id": pr["id"], "language": pr["language"],
                     "smoke_case": pr["smoke_case"], "category": pr["category"], "task_type": pr["task_type"],
                     "locale_dependence": pr["locale_dependence"], "finish_reason": c.finish_reason,
                     "think_closed": closed, "leaks": leaks, "prompt_tokens": len(ids),
                     "completion_tokens": len(c.token_ids), "reasoning": reasoning, "response": answer})
    with open(os.path.join(OUT, name + tag + ".jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    renders = {}
    for pr, mode, ids, max_new, text in jobs:
        key = "{}/{}".format(mode, pr["smoke_case"])
        if key not in renders:
            renders[key] = text[-800:]
    meta = {"model": name, "hf": m["hf"], "revision": m["revision"], "load_s": round(load_s, 1),
            "gen_s": round(gen_s, 1), "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "sampling": sp, "budgets": budgets, "modes": m["modes"], "default_system": m.get("default_system"),
            "stop_ids": sorted(stops), "turn_start_ids": sorted(turn), "renders": renders}
    with open(os.path.join(OUT, name + tag + ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("{}: load {:.0f}s, {} responses, {} tokens in {:.0f}s".format(name, load_s, len(rows),
          meta["completion_tokens"], gen_s), flush=True)
    del llm
    gc.collect()


def run(names, tp, eager, override=None, tag=""):
    p, models = pool()
    prompts = [json.loads(l) for l in open(os.path.join(OUT, "prompts.jsonl"), encoding="utf-8")]
    for name in names:
        if os.path.exists(os.path.join(OUT, name + tag + ".meta.json")):
            print(name + ": done before, skipped", flush=True)
            continue
        info = os.path.join(ROOT, "data", "models", name, "_download_info.json")
        for _ in range(120):                 # wait up to an hour for fetch_pool.py to finish this model
            if os.path.exists(info) and json.load(open(info))["revision"] == models[name]["revision"]:
                break
            time.sleep(30)
        try:
            run_model(name, p, models[name], prompts, tp, eager, override, tag)
        except Exception as e:     # one broken model must not stop the others
            import traceback
            traceback.print_exc()
            with open(os.path.join(OUT, name + ".error.txt"), "w") as f:
                f.write("{}: {}\n".format(type(e).__name__, e))
            print("{}: FAILED {}: {}".format(name, type(e).__name__, str(e)[:300]), flush=True)


# ------------------------------------------------------------------ report (CPU)
SAME = [{"bs", "hr", "sr", "sr-Latn", "me"}, {"nb", "nn", "no"}, {"lv", "ltg"}, {"et", "ekk"}, {"sq", "als"}]


def same_lang(a, b):
    return a == b or any(a in g and b in g for g in SAME)


def pct(xs, f):
    xs = list(xs)
    return "{:4.0f}%".format(100 * sum(map(f, xs)) / len(xs)) if xs else "    -"


def report():
    sys.path.insert(0, os.path.join(ROOT, "collect-prompts", "compile"))
    import build_pool as B
    B.load_model(os.path.join(ROOT, "data", "models", "openlid-v3", "openlid-v3.bin"), {})
    p, models = pool()
    lines = ["{:36s} {:6s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s} {:>6s} {:>6s}".format(
        "model", "mode", "stop", "cut", "open", "leak", "lang", "tok", "rtok")]
    for name, m in models.items():
        path = os.path.join(OUT, name + ".jsonl")
        if not os.path.exists(path):
            err = os.path.join(OUT, name + ".error.txt")
            lines.append("{:36s} {}".format(name, open(err).read().strip()[:140] if os.path.exists(err) else "not run"))
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        for mode in m["modes"]:
            gen = [r for r in rows if r["mode"] == mode and r["smoke_case"] not in ("identity", "overtrigger")]
            lang_rows = [r for r in gen if r["locale_dependence"] != "language_specific" and r["task_type"] != "translation"
                         and r["response"]]
            for r in lang_rows:
                r["lid"] = B.lid(B.clean_for_lid(r["response"], 3000))[0]
            lines.append("{:36s} {:6s} {} {} {} {} {} {:6.0f} {:6.0f}".format(
                name, mode, pct(gen, lambda r: r["finish_reason"] == "stop" and r["think_closed"] is not False),
                pct(gen, lambda r: r["finish_reason"] == "length"),
                pct(gen, lambda r: r["think_closed"] is False) if mode == "think" else "    -",
                pct(gen, lambda r: bool(r["leaks"])),
                pct(lang_rows, lambda r: bool(r.get("lid")) and same_lang(r["lid"], r["language"])),
                sum(len(r["response"]) for r in gen) / max(1, len(gen)) / 4,
                sum(len(r["reasoning"]) for r in gen) / max(1, len(gen)) / 4))
    text = ("\n".join(lines) + "\n\nstop: ended by itself (and closed its reasoning); cut: hit the token budget; "
            "open: reasoning never closed; leak: chat-template or reasoning markers in the answer; lang: OpenLID "
            "language = prompt language (translation and language_specific prompts excluded; bs/hr/sr, nb/nn, lv/ltg "
            "count as equal); tok / rtok: mean answer / reasoning length in tokens (chars / 4).\n")
    print(text)
    with open(os.path.join(OUT, "REPORT.txt"), "w") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompts", "templates", "run", "report"])
    ap.add_argument("--models", help="comma-separated pool names (templates, run; default all)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--eager", action="store_true", help="no graph capture (faster start-up, slower decoding)")
    ap.add_argument("--override", help='JSON merged into the sampling parameters (run), e.g. {"top_p": 0.95}')
    ap.add_argument("--tag", default="", help="suffix of the output files (run), e.g. @top_p0.95")
    args = ap.parse_args()
    names = args.models.split(",") if args.models else list(pool()[1])
    if args.cmd == "prompts":
        make_prompts()
    elif args.cmd == "templates":
        sys.exit(1 if check_templates(names) else 0)
    elif args.cmd == "run":
        run(names, args.tp, args.eager, json.loads(args.override) if args.override else None, args.tag)
    else:
        report()


if __name__ == "__main__":
    main()
