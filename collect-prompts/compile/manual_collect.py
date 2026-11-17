#!/usr/bin/env python3
"""Stage 5n: collect the manual translations (manual_requests.py batches).

  python3 collect-prompts/compile/manual_collect.py --check data/llm/translate_manual/runA/out/is_01.txt
  ./run.sh compile/manual_collect.py      # every run -> data/llm/translate_manual/translations.parquet

--check (standard library only, for the translator): per row and message, the problems
assembly would reject (missing or extra messages, role mismatch, empty text, inline code /
URLs / digits / brackets not kept, a translator preamble) and warnings worth a second look
(length ratio, words of two scripts, long runs of copied source words).
Collect: parses run*/out/*.txt; a row is usable when every message is present and has no
problem; its text keeps the source message's leading/trailing whitespace. A later run
replaces an earlier one per row id.
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

import mt_common as M  # noqa: E402  (standard library only)

OUT = os.path.join(M.DATA, "llm", "translate_manual")
HEAD = re.compile(r"^=== (mlp128k-\d+) \| msg (\d+) \| (\w+)[ \t]*$", re.M)
ROWLINE = re.compile(r"^##### ROW .*$\n?", re.M)
WORD = re.compile(r"[^\W\d_]{5,}")


def parse(path):
    """-> {row id: {msg index: (role, text)}}, duplicates"""
    text = ROWLINE.sub("", open(path, encoding="utf-8").read())
    heads = list(HEAD.finditer(text))
    out, dup = collections.defaultdict(dict), []
    for k, h in enumerate(heads):
        body = text[h.end():heads[k + 1].start() if k + 1 < len(heads) else len(text)]
        rid, i, role = h.group(1), int(h.group(2)), h.group(3)
        if i in out[rid]:
            dup.append((rid, i))
        out[rid][i] = (role, body.strip("\n").strip())
    return out, dup


def restore_ws(src, tr):
    s = src or ""
    return s[:len(s) - len(s.lstrip())] + tr + s[len(s.rstrip()):]


def check_row(meta, got):
    """-> (problems, warnings) for one row"""
    probs, warns = [], []
    ctx = meta["context"]
    if set(got) != set(range(len(ctx))):
        probs.append("messages: expected {}, got {}".format(len(ctx), sorted(got)))
    for i, m in enumerate(ctx):
        if i not in got:
            continue
        role, tr = got[i]
        src = (m["content"] or "").strip()
        if role != m["role"]:
            probs.append("msg {}: role {} != {}".format(i, role, m["role"]))
        if src and not tr:
            probs.append("msg {}: empty".format(i))
            continue
        lp = M.literal_problems(src, tr)
        if lp:
            probs.append("msg {}: {}".format(i, ", ".join(lp)))
        if len(src) > 200 and not 0.6 <= len(tr) / len(src) <= 1.8:
            warns.append("msg {}: length ratio {:.2f}".format(i, len(tr) / len(src)))
        mixed = sorted(set(M.mixed_script_words(tr)) - set(M.mixed_script_words(src)))
        if mixed:
            warns.append("msg {}: mixed-script words {}".format(i, mixed[:5]))
        sw = set(w.lower() for w in WORD.findall(M.FENCE.sub(" ", src)))
        tw = [w.lower() for w in WORD.findall(M.FENCE.sub(" ", tr))]
        if len(tw) >= 20 and sum(w in sw for w in tw) / len(tw) > 0.5:
            warns.append("msg {}: {:.0%} of the words copied from the source".format(i, sum(w in sw for w in tw) / len(tw)))
    return probs, warns


def restore_code(src, tr):
    """Fenced code blocks that equal the source's apart from trailing whitespace (the translators'
    Write tool strips it) -> the source's exact block. -> (text, blocks restored)"""
    sb, tb = M.FENCE.findall(src or ""), M.FENCE.findall(tr or "")
    if not sb or len(sb) != len(tb):
        return tr, 0
    norm = lambda b: "\n".join(ln.rstrip() for ln in b.split("\n"))
    out, pos, n = [], 0, 0
    for s_, t_ in zip(sb, tb):
        i = tr.index(t_, pos)
        same = s_ != t_ and norm(s_) == norm(t_)
        out += [tr[pos:i], s_ if same else t_]
        n += same
        pos = i + len(t_)
    out.append(tr[pos:])
    return "".join(out), n


def run_rows(run_dir):
    return json.load(open(os.path.join(run_dir, "rows.json"), encoding="utf-8"))


def check_file(path):
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    rows = run_rows(run_dir)
    batch = os.path.splitext(os.path.basename(path))[0]
    want = [rid for rid, m in rows.items() if m["batch"] == batch]
    got, dup = parse(path)
    bad = 0
    for rid, i in dup:
        print("DUPLICATE {} msg {}".format(rid, i))
        bad += 1
    for rid in got:
        if rid not in rows or rows[rid]["batch"] != batch:
            print("EXTRA row {} (not in this batch)".format(rid))
            bad += 1
    for rid in want:
        if rid not in got:
            print("MISSING row", rid)
            bad += 1
            continue
        probs, warns = check_row(rows[rid], got[rid])
        for p in probs:
            print("PROBLEM {} {}".format(rid, p))
        for w in warns:
            print("warning {} {}".format(rid, w))
        bad += bool(probs)
    print("{}: {} rows expected, {} parsed, {} with problems".format(batch, len(want), len(got), bad))
    return bad


def collect():
    import pyarrow as pa
    import pyarrow.parquet as pq
    res, stats = {}, collections.Counter()
    for run_dir in sorted(glob.glob(os.path.join(OUT, "run*"))):
        rows = run_rows(run_dir)
        for path in sorted(glob.glob(os.path.join(run_dir, "out", "*.txt"))):
            got, _ = parse(path)
            for rid, msgs in got.items():
                meta = rows.get(rid)
                if meta is None:
                    stats["unknown_row"] += 1
                    continue
                probs, warns = check_row(meta, msgs)
                if probs:
                    stats["rejected"] += 1
                    continue
                ctx = []
                for i, m in enumerate(meta["context"]):
                    text, n = restore_code(m["content"], msgs[i][1])
                    stats["code_blocks_restored"] += n
                    ctx.append({"role": m["role"], "content": restore_ws(m["content"], text)})
                # keyed like assembly: a refilled slot keeps its id, so an id can be in several runs
                res[(meta["slot"], meta["uid"])] = {"id": rid, "slot": meta["slot"], "uid": meta["uid"], "lang": meta["lang"],
                            "run": os.path.basename(run_dir), "context": json.dumps(ctx, ensure_ascii=False),
                            "warnings": json.dumps(warns, ensure_ascii=False)}
        stats["expected"] += len(rows)
    stats["usable"] = len(res)
    pq.write_table(pa.Table.from_pylist(list(res.values())), os.path.join(OUT, "translations.parquet"))
    print(dict(stats), dict(collections.Counter(r["lang"] for r in res.values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check")
    args = ap.parse_args()
    if args.check:
        sys.exit(1 if check_file(args.check) else 0)
    collect()


if __name__ == "__main__":
    main()
