#!/usr/bin/env python3
"""Stage 5l: second translator benchmark (FLORES+ devtest, chrF++) -- requests.

  ./run.sh compile/bench2_requests.py [--n 200]

Candidates for the languages where Gemma-4 (and EuroMultiSynth, which Gemma generated)
read badly (manual_check/ems_review/VERDICTS.md). Same --n English sentences as stage 5i
(bench_requests.py, seed 20261110), so Gemma / TranslateGemma scores are comparable.
  eurollm::  EuroLLM-22B-Instruct-2512 (Apache-2.0)       chat, Tower-style prompt
  salta::    SalamandraTA-7b-instruct (GPL-3.0)           chat, its documented prompt
  tower::    Tower-Plus-72B (CC-BY-NC-SA-4.0)             chat, its documented prompt
  tilde::    TildeOpen-30b (CC-BY-4.0, base model)        8-shot, exemplars = other devtest
             sentences (FLORES+ dev is not available for all languages here); SentencePiece
             prompt_ids, token ids saved (see the tokenizer note in mt_common).
Each system only for the languages its model card lists. Outputs: data/llm/bench2/<sys>/requests.parquet
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import mt_common as M  # noqa: E402

OUT = os.path.join(M.DATA, "llm", "bench2")
NAME = {"lv": "Latvian", "et": "Estonian", "lt": "Lithuanian", "mt": "Maltese", "eu": "Basque",
        "is": "Icelandic", "ga": "Irish", "sl": "Slovenian", "hu": "Hungarian", "hr": "Croatian",
        "gl": "Galician", "cs": "Czech", "el": "Greek", "sk": "Slovak", "mk": "Macedonian"}
LANGS = {"eurollm": "lv et lt mt ga sl hu hr gl cs el sk",
         "salta": "lv et lt mt eu is ga sl hu hr gl cs el sk",
         "tower": "is hu cs",
         "tilde": "lv et lt mt is ga sl hu hr cs sk mk"}
BASE = {"messages": None, "prompt": None, "prompt_ids": None, "stop": None, "schema": None, "max_tokens": 512}


def chat(text, lang, system):
    if system == "salta":
        c = "Translate the following text from English into {0}.\nEnglish: {1} \n{0}:".format(NAME[lang], text)
    else:
        c = "Translate the following English source text to {0}:\nEnglish: {1}\n{0}: ".format(NAME[lang], text)
    return json.dumps([{"role": "user", "content": c}], ensure_ascii=False)


def tilde_prompt(text, lang, ex):
    S, L, n = M.flores("devtest", "en"), M.flores("devtest", lang), NAME[lang]
    head = "English texts with their translations into {}.\n\n".format(n)
    shots = "".join("<English>\n{}\n</English>\n<{n}>\n{}\n</{n}>\n\n".format(S[e], L[e], n=n) for e in ex)
    return head + shots + "<English>\n{}\n</English>\n<{n}>\n".format(text.strip(), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--shots", type=int, default=8)
    args = ap.parse_args()
    ids = M.eval_ids(args.n, seed=20261110)
    ex = random.Random(8).sample(sorted(set(M.flores("devtest", "en")) - set(ids)), args.shots)
    S = M.flores("devtest", "en")
    for system, langs in LANGS.items():
        reqs = []
        for lang in langs.split():
            for fid in ids:
                rid = "{}::{}::{}".format(system, lang, fid)
                if system == "tilde":
                    p = tilde_prompt(S[fid], lang, ex)
                    reqs.append(dict(BASE, id=rid, prompt=p, prompt_ids=M.tilde_encode(p),
                                     stop=json.dumps(["</{}>".format(NAME[lang])])))
                else:
                    reqs.append(dict(BASE, id=rid, messages=chat(S[fid], lang, system)))
        os.makedirs(os.path.join(OUT, system), exist_ok=True)
        pq.write_table(pa.Table.from_pylist(reqs), os.path.join(OUT, system, "requests.parquet"))
        print(system, len(langs.split()), "languages,", len(reqs), "requests")
    with open(os.path.join(OUT, "exemplars.json"), "w") as f:
        json.dump({"eval_ids": ids, "tilde_exemplar_ids": ex}, f)


if __name__ == "__main__":
    main()
