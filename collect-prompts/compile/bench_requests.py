#!/usr/bin/env python3
"""Stage 5i: translator benchmark (FLORES+) and every_prompt judge -- requests.

  ./run.sh compile/bench_requests.py [--n 200]

1. FLORES+ devtest, the same --n sentences for every target language (English
   source; French source for English), translated by
     gm::<lang>::<id>  Gemma-4-31B-it with the pipeline's own translation prompt
                       (translate_prompt.messages + JSON schema, as in stage 5)
     tg::<lang>::<id>  TranslateGemma-27B
   -> bench_score.py: chrF++ per language and system, paired bootstrap.
2. every_prompt judge (Gemma-4, JSON): each lang-uk/every_prompt row's last user
   turn -> is_request / self_contained / spam (see JUDGE). ids ep::<row id>
Outputs: data/llm/bench/{tg,gemma}/requests.parquet
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import mt_common as M  # noqa: E402
import translate_prompt as T  # noqa: E402

OUT = os.path.join(M.DATA, "llm", "bench")
TARGETS = [l for l in M.FLORES_CODE if l != "ltg"]

JUDGE = """You check user prompts for a dataset of requests that people send to an AI assistant. The prompt below was scraped from the FAQ section of a web page. Judge it as a message a user would type to an assistant, in any language.

Answer three questions:
- is_request: Is it a question or a request that states what the user wants (a question, an instruction, or a short how-to title such as "remove a virus from an Android phone")? Answer "no" for bare keyword strings, product or brand names, page titles or URL slugs without a clear question or task.
- self_contained: Can it be understood and answered without the web page it came from? Answer "no" if it relies on context that is missing (e.g. "the module", "our service", "do you also deliver to X?" addressed to an unnamed business, or a numbered FAQ item that only makes sense on its page).
- spam: Does it promote or ask about a specific betting, gambling or casino site or bonus, an escort, dating-for-pay or adult service, a psychic, fortune-telling or occult service, or is it advertising for a specific business? General questions about a topic (e.g. "are online slot machines rigged?") are not spam.

Return JSON: {"is_request": "yes"|"no", "self_contained": "yes"|"no", "spam": "yes"|"no"}

Prompt:
"""
JUDGE_SCHEMA = {"type": "object", "properties": {k: {"type": "string", "enum": ["yes", "no"]}
                                                 for k in ("is_request", "self_contained", "spam")},
                "required": ["is_request", "self_contained", "spam"], "additionalProperties": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    ids = M.eval_ids(args.n, seed=20261110)
    reqs = {"tg": [], "gemma": []}
    base = {"prompt": None, "prompt_ids": None, "stop": None}
    for lang in TARGETS:
        src_lang = "fr" if lang == "en" else "en"
        S = M.flores("devtest", src_lang)
        for fid in ids:
            s = S[fid]
            reqs["gemma"].append(dict(base, id="gm::{}::{}".format(lang, fid), max_tokens=768,
                                      messages=json.dumps(T.messages([{"role": "user", "content": s}], src_lang, lang),
                                                          ensure_ascii=False),
                                      schema=json.dumps(T.schema(1))))
            reqs["tg"].append(dict(base, id="tg::{}::{}".format(lang, fid), max_tokens=768, schema=None,
                                   messages=json.dumps(M.tg_messages(s, src_lang, lang), ensure_ascii=False)))
    rows = pq.read_table(os.path.join(M.DATA, "compiled", "prompts.parquet"),
                         columns=["id", "source", "context"]).to_pylist()
    n_ep = 0
    for r in rows:
        if r["source"] != "lang-uk/every_prompt":
            continue
        n_ep += 1
        u = r["context"][-1]["content"]
        reqs["gemma"].append(dict(base, id="ep::" + r["id"], max_tokens=64, schema=json.dumps(JUDGE_SCHEMA),
                                  messages=json.dumps([{"role": "user", "content": JUDGE + u}], ensure_ascii=False)))
    for name, rs in reqs.items():
        os.makedirs(os.path.join(OUT, name), exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rs), os.path.join(OUT, name, "requests.parquet"))
    print("languages:", len(TARGETS), "| FLORES sentences per language:", len(ids),
          "| requests tg:", len(reqs["tg"]), "gemma:", len(reqs["gemma"]), "(every_prompt judge:", n_ep, ")")


if __name__ == "__main__":
    main()
