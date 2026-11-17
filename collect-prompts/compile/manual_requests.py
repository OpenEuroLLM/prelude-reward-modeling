#!/usr/bin/env python3
"""Stage 5n: manual re-translation (by Claude, in Claude Code) of the translated rows in the
weakest languages, Icelandic and Irish, where every MT system tried stays weak (final spot
check: SalamandraTA 7/23 problem rows).

  ./run.sh compile/manual_requests.py --tag A [--prompts data/compiled/prompts.parquet]
  python3 collect-prompts/compile/manual_collect.py --check data/llm/translate_manual/runA/out/is_01.txt
  ./run.sh compile/manual_collect.py         # all runs -> data/llm/translate_manual/translations.parquet

Rows: fill == translated in --langs, except harmful-safety prompts, rows with IF verifier params
(their kwargs would have to match the new wording), code-heavy prose, IT-security / attack
vocabulary (SECURITY; the model's safeguards stop translators on such text), rows failed in
force_fail.parquet (being replaced), and rows an earlier run already covers. They are translated
from the pool source, every message (earlier assistant turns too), under the stage-5 rules
(translate_prompt.SYSTEM). Batch files (data/llm/translate_manual/run<tag>/batches/) hold whole
rows, up to --batch-chars source characters:
  ##### ROW <id> | <lang> | from <source language> | <n> messages
  === <id> | msg <i> | <role>
  <source text>
The translator writes run<tag>/out/<batch>.txt with the same lines, each message's text replaced
by its translation. rows.json keeps slot, uid, languages and the source context per row.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow.parquet as pq  # noqa: E402

import mt_common as M  # noqa: E402
from mt2_requests import load_sources  # noqa: E402
from translate_prompt import LANG_NAMES  # noqa: E402

OUT = os.path.join(M.DATA, "llm", "translate_manual")
HEADER = re.compile(r"^(=== mlp128k-\d+ \| msg \d+ \| \w+|##### ROW )", re.M)
# IT-security / attack vocabulary: such rows stay on their machine translation (the model's
# safeguards stopped a translator on a batch with an SQL-injection prompt; broad on purpose).
SECURITY = re.compile(r"\b(hack\w*|exploit\w*|malware|ransomware|trojan|keylog\w*|rootkit|botnet|ddos|phishing|"
                      r"payload|shellcode|reverse shell|backdoor|privilege escalation|sql injection|injection attack|"
                      r"xss|cross-site|brute[- ]?forc\w*|crack\w*|bypass\w*|jailbreak\w*|penetration test\w*|"
                      r"pentest\w*|metasploit|nmap|burp|wireshark|vulnerabilit\w*|cve-\d+|zero[- ]day|"
                      r"buffer overflow|obfuscat\w*|evasion|evade|antivirus|spoof\w*|sniff\w*|stealer|"
                      r"cryptojack\w*|aircrack|kali|password|credential\w*|exfiltrat\w*|command and control|"
                      r"spyware|token grabber|carding|darknet|dark web|onion|deobfuscat\w*|decompil\w*|"
                      r"reverse engineer\w*|disassembl\w*|firewall|intrusion|red team|cyber\w*)\b", re.I)


def covered():
    """(row id, source uid) pairs an earlier run holds (a refilled slot keeps its id, not its uid)"""
    done = set()
    for p in glob.glob(os.path.join(OUT, "run*", "rows.json")):
        done |= {(rid, m["uid"]) for rid, m in json.load(open(p)).items()}
        ex = os.path.join(os.path.dirname(p), "excluded.json")
        if os.path.exists(ex):      # excluded rows keep their uid in the batch's history only
            done |= {(rid, None) for rid in json.load(open(ex))}
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--langs", default="is,ga")
    ap.add_argument("--batch-chars", type=int, default=25000)
    ap.add_argument("--prompts", default=os.path.join(M.DATA, "compiled", "prompts.parquet"))
    args = ap.parse_args()
    langs = args.langs.split(",")
    out_dir = os.path.join(OUT, "run" + args.tag)
    if os.path.exists(os.path.join(out_dir, "rows.json")):
        sys.exit(out_dir + " exists")
    ff = {(r["slot"], r["uid"]) for r in pq.read_table(os.path.join(M.DATA, "compiled", "force_fail.parquet")).to_pylist()}
    done = covered()
    rows = [r for r in pq.read_table(args.prompts, columns=["id", "lang", "fill", "source_uid", "source_lang",
            "purpose", "if_kwargs", "translator"]).to_pylist() if r["fill"] == "translated" and r["lang"] in langs]
    src = load_sources({r["source_uid"] for r in rows})
    skip = collections.Counter()
    keep = []
    for r in sorted(rows, key=lambda r: (langs.index(r["lang"]), r["id"])):
        slot = int(r["id"].split("-")[1])
        ctx = src[r["source_uid"]]
        why = ("being_replaced" if (slot, r["source_uid"]) in ff else
               "covered" if (r["id"], r["source_uid"]) in done else
               "safety_harmful" if r["purpose"] == "safety_harmful" else
               "if_params" if r["if_kwargs"] else
               "code_heavy" if any(M.codeish(m["content"]) for m in ctx) else
               "security_terms" if any(SECURITY.search(m["content"] or "") for m in ctx) else
               "delimiter" if any(HEADER.search(m["content"] or "") for m in ctx) else None)
        if why:
            skip[why] += 1
            continue
        keep.append((r, slot, ctx))
    batches = collections.defaultdict(list)
    for lang in langs:
        cur, size, k = [], 0, 1
        for r, slot, ctx in (x for x in keep if x[0]["lang"] == lang):
            n = sum(len(m["content"] or "") for m in ctx)
            if cur and size + n > args.batch_chars:
                batches["{}_{:02d}".format(lang, k)] = cur
                cur, size, k = [], 0, k + 1
            cur.append((r, slot, ctx))
            size += n
        if cur:
            batches["{}_{:02d}".format(lang, k)] = cur
    os.makedirs(os.path.join(out_dir, "batches"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "out"), exist_ok=True)
    meta = {}
    for name, items in batches.items():
        blocks = []
        for r, slot, ctx in items:
            blocks.append("##### ROW {} | {} | from {} | {} messages".format(
                r["id"], LANG_NAMES[r["lang"]], LANG_NAMES.get(r["source_lang"], r["source_lang"]), len(ctx)))
            for i, m in enumerate(ctx):
                blocks.append("=== {} | msg {} | {}\n{}".format(r["id"], i, m["role"], (m["content"] or "").strip()))
            meta[r["id"]] = {"slot": slot, "uid": r["source_uid"], "lang": r["lang"], "src_lang": r["source_lang"],
                             "batch": name, "prev_translator": r["translator"],
                             "context": [{"role": m["role"], "content": m["content"] or ""} for m in ctx]}
        with open(os.path.join(out_dir, "batches", name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(blocks) + "\n")
    json.dump(meta, open(os.path.join(out_dir, "rows.json"), "w"), ensure_ascii=False)
    print("rows:", len(meta), dict(collections.Counter(m["lang"] for m in meta.values())), "| skipped:", dict(skip))
    for name, items in batches.items():
        print("  {}: {} rows, {} chars".format(name, len(items), sum(len(m["content"] or "") for _, _, c in items for m in c)))


if __name__ == "__main__":
    main()
