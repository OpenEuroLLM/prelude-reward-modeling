#!/usr/bin/env python3
"""Extract instruction-following verification metadata for IF candidates.

  ./run.sh compile/if_meta.py
Reads data/pool/candidates.parquet (purpose == "if"), fetches each record's
meta from data/extracted/<part>, writes data/pool/if_meta.parquet:
  uid, instruction_ids (list<str>), kwargs (JSON list aligned with ids, or
  null), transferable (bool: every constraint can be carried over to another
  language by translating its string kwargs; see TRANSFERABLE)
"""
import ast
import bisect
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(HERE))
EXT = os.path.join(ROOT, "data", "extracted")
POOL = os.path.join(ROOT, "data", "pool")

# IFEval constraints whose checkers stay valid after translating string kwargs
# (keywords, phrases, markers). Excluded on purpose: change_case:english_* and
# language:response_language (force English), keywords:letter_frequency,
# detectable_format:constrained_response (fixed English answer strings), and
# IFBench ids that depend on letters/alphabet/character positions (letter
# counting, palindromes, no_adjacent_consecutive, new:copy_span_idx).
TRANSFERABLE = {
    "length_constraints:number_words", "length_constraints:number_sentences",
    "length_constraints:number_paragraphs", "length_constraints:nth_paragraph_first_word",
    "detectable_format:number_bullet_lists", "detectable_format:number_highlighted_sections",
    "detectable_format:multiple_sections", "detectable_format:json_format",
    "detectable_format:title", "detectable_content:number_placeholders",
    "detectable_content:postscript", "keywords:existence", "keywords:frequency",
    "keywords:forbidden_words", "startend:end_checker", "startend:quotation",
    "combination:repeat_prompt", "combination:two_responses", "punctuation:no_comma",
    "change_case:capital_word_frequency",
    # IFBench training constraints that only count words / check given phrases
    "keywords:word_once", "keywords:exclude_word_harder", "keywords:keyword_specific_position",
    "keywords:start_end", "count:count_increment_word", "count:count_unique",
    "count:counting_composition", "count:lowercase_counting",
    "first_word:first_word_answer", "first_word:first_word_sent",
    "last_word:last_word_answer", "last_word:last_word_sent",
    "paragraphs:paragraphs", "paragraphs:paragraphs2",
    "detectable_format:sentence_hyphens", "detectable_format:bigram_wrapping",
    "detectable_format:square_brackets", "punctuation:punctuation_exclamation",
    "punctuation:punctuation_dot", "copy:repeat_phrase", "copy:copying_simple",
    "copy:copying_multiple", "copy:copy",
}
CASE_IDS = {"change_case:capital_word_frequency", "count:lowercase_counting"}
NO_CASE_LANGS = {"ka"}   # scripts without letter case


def parse_kwargs(meta):
    kw = meta.get("kwargs")
    if isinstance(kw, str):
        try:
            kw = json.loads(kw)
        except Exception:
            try:
                kw = ast.literal_eval(kw)
            except Exception:
                kw = None
    return kw


def fetch(part, wanted):
    pf = pq.ParquetFile(os.path.join(EXT, part))
    starts, acc = [], 0
    for i in range(pf.num_row_groups):
        starts.append(acc)
        acc += pf.metadata.row_group(i).num_rows
    by_rg = collections.defaultdict(list)
    for row, uid in wanted:
        by_rg[bisect.bisect_right(starts, row) - 1].append((row, uid))
    out = []
    for rg, ws in by_rg.items():
        t = pf.read_row_group(rg, columns=["uid", "meta"])
        uids, metas = t.column("uid").to_pylist(), t.column("meta").to_pylist()
        for row, uid in ws:
            j = row - starts[rg]
            assert uids[j] == uid, (part, row)
            out.append((uid, json.loads(metas[j]) if metas[j] else {}))
    return out


def main():
    c = pq.read_table(os.path.join(POOL, "candidates.parquet"),
                      columns=["uid", "part", "row", "purpose"]).to_pylist()
    tasks = collections.defaultdict(list)
    for r in c:
        if r["purpose"] == "if":
            tasks[r["part"]].append((r["row"], r["uid"]))
    rows = []
    stats = collections.Counter()
    for part, wanted in sorted(tasks.items()):
        for uid, meta in fetch(part, wanted):
            ids = meta.get("instruction_id_list")
            if isinstance(ids, str):
                try:
                    ids = ast.literal_eval(ids)
                except Exception:
                    ids = None
            ids = [str(x) for x in ids] if isinstance(ids, (list, tuple)) else []
            kw = parse_kwargs(meta) if ids else None
            transferable = all(i in TRANSFERABLE for i in ids)
            stats["with_ids" if ids else "no_ids"] += 1
            stats["transferable" if transferable else "not_transferable"] += 1
            rows.append({"uid": uid, "instruction_ids": ids,
                         "kwargs": json.dumps(kw, ensure_ascii=False) if kw is not None else None,
                         "transferable": transferable,
                         "has_case_constraint": any(i.startswith("change_case:") or i in CASE_IDS
                                                    for i in ids)})
    pq.write_table(pa.Table.from_pylist(rows), os.path.join(POOL, "if_meta.parquet"))
    print(dict(stats), "->", os.path.join(POOL, "if_meta.parquet"))


if __name__ == "__main__":
    main()
