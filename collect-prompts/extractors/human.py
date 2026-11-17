"""Human-written / human-labelled sources.

  lmsys/lmsys-chat-1m                  real LMSYS chat logs; one record per
                                       conversation (like WildChat).
  OpenAssistant/oasst2                 message trees; one record per prompter
                                       message (see below).
  facebook/community-alignment-dataset best-of-4 choices per turn.
  CohereLabs/aya_dataset               human instruction/response pairs.
  AnnikaSimonsen/TrustLLM-reformulation-prompts   prompt-only.
  pelcra/PLLuMIC                       Polish expert-written conversations.
  NASK-PIB/PLLuM-Align                 Polish human preference data.
  ctu-aic/questions_ujc_cas_cs         Czech language-consultancy Q&A.
  projecte-aina/MentorES               Spanish Dolly-style instructions.

Conventions specific to this module
-----------------------------------
* Responses written by people (not models) have model="human".
* Extra preference type "best_of_n":
      {"type": "best_of_n", "chosen": i, "raw": ...}
  The annotator picked response i as best among all `responses`; there is no
  order among the others. Pairs to train on are (i, j) for every j != i.

OASST2 (2023-11-05_oasst2_all.messages.jsonl.gz, all 208,584 messages,
including the prompt_lottery_waiting prompt-only trees and spam): one record
per prompter message. context = path from the root to that message
(prompter -> user), responses = all assistant children (ordered by
created_date), preference = {"type": "ranking", "ranks": [...]} from the
children's `rank` (0 = best, None = unranked) if >= 2 children and at least one
is ranked. Deleted / review-failed messages are kept and flagged (record meta
`deleted`, `review_result`, `context_deleted`, `context_review_failed`;
response meta has the same per-reply fields). subset = tree_state. split =
the tree's split in the HF parquet (validation trees -> validation), other
trees -> train.

Community Alignment: one record per turn (up to 4 per conversation),
responses = candidates A-D (empty ones dropped), preference = best_of_n.
The source does not document which response carries the history into the
next turn. We assume the annotator's preferred one (meta
`history_assumption`); turns after a turn without a preferred response
cannot be built and are skipped ("history_unknown_no_preference").

PLLuM-Align: ranking.jsonl pairs share the chosen reply per prompt and are
regrouped into one ranking per prompt (chosen = 0, rejected ordered by the
rank number in their id). rating.jsonl -> scores. dialogs.jsonl pairs are
grouped per (dialogue, turn) -> best_of_n (chosen beats each rejected
alternative). The repo has no test split, so everything is train.
"""
import collections
import csv
import hashlib
import os
import re
import sys

import pyarrow.parquet as pq

from common import (iter_json, iter_jsonl, iter_parquet, msg, register,
                    response, split_last_turn)


def _dedup_id(seen, sid):
    seen[sid] = seen.get(sid, 0) + 1
    return sid if seen[sid] == 1 else "{}#{}".format(sid, seen[sid])


def _split_from_name(path):
    name = os.path.basename(path).lower()
    if name.startswith("test"):
        return "test"
    if name.startswith(("validation", "valid", "dev")):
        return "validation"
    return "train"


def _nonempty(s):
    return s is not None and str(s).strip() != ""


# ------------------------------------------------------------ lmsys-chat-1m

def _flagged(moderation, idx=None):
    out = set()
    for i, m in enumerate(moderation or []):
        if idx is not None and i not in idx:
            continue
        for cat, v in ((m or {}).get("categories") or {}).items():
            if v:
                out.add(cat.replace("/", "_").replace("-", "_"))
    return sorted(out)


@register("lmsys/lmsys-chat-1m")
def lmsys_chat(ctx):
    seen = {}
    for path, row in iter_parquet(ctx.files("data/*.parquet")):
        conv = row["conversation"] or []
        messages = [msg(t["role"], t["content"]) for t in conv]
        context, reply = split_last_turn(messages)
        if context is None or not any(_nonempty(m["content"]) for m in context
                                      if m["role"] == "user"):
            ctx.skip("no_user_turn")
            continue
        mod = row.get("openai_moderation") or []
        user_idx = {i for i, m in enumerate(messages) if m["role"] == "user"}
        yield ctx.record(
            subset="default", split="train",
            source_id=_dedup_id(seen, row["conversation_id"]),
            context=context,
            responses=[response(reply, model=row.get("model"))]
            if reply is not None else [],
            language=row.get("language"),
            meta={
                "conversation_id": row["conversation_id"],
                "model": row.get("model"),
                "turn": row.get("turn"),
                "redacted": row.get("redacted"),
                "openai_flagged_any": any((m or {}).get("flagged") for m in mod),
                "openai_flagged": _flagged(mod),
                "openai_flagged_user": _flagged(mod, user_idx),
            })


# ------------------------------------------------------------------ oasst2

_OASST_MSG_KEYS = ("message_id", "user_id", "created_date", "review_count",
                   "review_result", "deleted", "synthetic", "model_name",
                   "labels", "emojis", "detoxify")


@register("OpenAssistant/oasst2")
def oasst2(ctx):
    path = ctx.files("*_oasst2_all.messages.jsonl.gz")[0]
    by_id, order = {}, []
    children = collections.defaultdict(list)
    for _, m in iter_jsonl([path]):
        by_id[m["message_id"]] = m
        order.append(m["message_id"])
        if m.get("parent_id"):
            children[m["parent_id"]].append(m["message_id"])

    # split membership of the ready_for_export trees (HF parquet, by tree)
    val_trees = set()
    for p in ctx.files("data/validation-*.parquet"):
        val_trees.update(pq.read_table(p, columns=["message_tree_id"])
                         .column("message_tree_id").to_pylist())

    def resp_meta(c):
        d = {k: c.get(k) for k in _OASST_MSG_KEYS}
        d["rank"] = c.get("rank")
        d["lang"] = c.get("lang")
        return d

    for mid in order:
        m = by_id[mid]
        if m["role"] != "prompter":
            continue
        if not _nonempty(m.get("text")):
            ctx.skip("empty_prompt")
            continue
        path_msgs, cur, broken = [], m, False
        while cur is not None:
            path_msgs.append(cur)
            pid = cur.get("parent_id")
            if pid is None:
                break
            cur = by_id.get(pid)
            if cur is None:
                broken = True
        if broken:
            ctx.skip("missing_ancestor")
            continue
        path_msgs.reverse()
        context = [msg(x["role"], x["text"]) for x in path_msgs]
        kids = sorted((by_id[c] for c in children.get(mid, [])
                       if by_id[c]["role"] == "assistant"),
                      key=lambda c: (c.get("created_date") or "", c["message_id"]))
        responses = [response(c["text"],
                              model=c.get("model_name") if c.get("synthetic")
                              else "human",
                              **resp_meta(c)) for c in kids]
        ranks = [c.get("rank") for c in kids]
        preference = None
        if len(kids) >= 2 and any(r is not None for r in ranks):
            preference = {"type": "ranking", "ranks": ranks, "raw": ranks}
        anc = path_msgs[:-1]
        tree = m.get("message_tree_id")
        meta = {k: m.get(k) for k in _OASST_MSG_KEYS}
        meta.update({
            "parent_id": m.get("parent_id"),
            "message_tree_id": tree,
            "tree_state": m.get("tree_state"),
            "depth": len(path_msgs) - 1,
            "context_deleted": any(bool(x.get("deleted")) for x in anc),
            "context_review_failed": any(x.get("review_result") is False
                                         for x in anc),
        })
        yield ctx.record(
            subset=m.get("tree_state") or "unknown",
            split="validation" if tree in val_trees else "train",
            source_id=mid, context=context, responses=responses,
            preference=preference, language=m.get("lang"), meta=meta)


# ------------------------------------------------- community alignment

_CA_TURNS = ("first", "second", "third", "fourth")
_CA_DEMO = ("annotator_age", "annotator_gender", "annotator_education_level",
            "annotator_political", "annotator_ethnicity", "annotator_country")


def _bool(s):
    if s in ("True", "true", "1"):
        return True
    if s in ("False", "false", "0"):
        return False
    return None if s in ("", None) else s


@register("facebook/community-alignment-dataset")
def community_alignment(ctx):
    csv.field_size_limit(sys.maxsize)
    seen = {}
    for path in ctx.files("*.csv"):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cid = row["conversation_id"]
                base_meta = {
                    "conversation_id": cid,
                    "annotator_id": row.get("annotator_id"),
                    "wave": row.get("wave"),
                    "assigned_lang": row.get("assigned_lang"),
                    "is_pregenerated_first_prompt":
                        _bool(row.get("is_pregenerated_first_prompt")),
                    "in_balanced_subset": _bool(row.get("in_balanced_subset")),
                    "in_balanced_subset_10":
                        _bool(row.get("in_balanced_subset_10")),
                }
                base_meta.update({k: row.get(k) or None for k in _CA_DEMO})
                history, broken = [], False
                for k, turn in enumerate(_CA_TURNS):
                    prompt = row.get(turn + "_turn_prompt")
                    if not _nonempty(prompt):
                        break  # turns are contiguous in this release
                    if broken:
                        ctx.skip("history_unknown_no_preference")
                        continue
                    letters = [x for x in "abcd"
                               if _nonempty(row.get(turn + "_turn_response_" + x))]
                    responses = [response(row[turn + "_turn_response_" + x],
                                          label="response_" + x)
                                 for x in letters]
                    pref_raw = row.get(turn + "_turn_preferred_response") or None
                    chosen = None
                    if pref_raw and pref_raw.startswith("response_") \
                            and pref_raw[-1] in letters:
                        chosen = letters.index(pref_raw[-1])
                    preference = None
                    if chosen is not None and len(letters) >= 2:
                        preference = {"type": "best_of_n", "chosen": chosen,
                                      "raw": pref_raw}
                    context = history + [msg("user", prompt)]
                    meta = dict(base_meta)
                    meta.update({
                        "turn": k + 1,
                        "prompt_author": "pregenerated" if k == 0 and
                        base_meta["is_pregenerated_first_prompt"] else "annotator",
                        "preferred_response": pref_raw,
                        "feedback": row.get(turn + "_turn_feedback") or None,
                        "history_assumption": "preferred response of each "
                        "earlier turn (not documented by the source)"
                        if k else None,
                        "response_generator": "Llama-3.3-70B-Instruct (paper)",
                    })
                    yield ctx.record(
                        subset="default", split="train",
                        source_id=_dedup_id(seen, "{}_t{}".format(cid, k + 1)),
                        context=context, responses=responses,
                        preference=preference,
                        language=row.get("assigned_lang") or None, meta=meta)
                    if chosen is None:
                        broken = True
                    else:
                        history = context + [
                            msg("assistant", row[turn + "_turn_" + pref_raw])]


# -------------------------------------------------------------------- aya

@register("CohereLabs/aya_dataset")
def aya_dataset(ctx):
    seen = {}
    # demographics/ holds annotator profiles, not prompts: skipped
    for path, row in iter_parquet(ctx.files("data/*.parquet")):
        split = _split_from_name(path)
        if not _nonempty(row.get("inputs")):
            ctx.skip("empty_prompt")
            continue
        key = "\x1f".join(str(row.get(k)) for k in
                          ("inputs", "targets", "language_code", "user_id"))
        sid = "{}-{}".format(split, hashlib.sha1(key.encode("utf-8"))
                             .hexdigest()[:16])
        yield ctx.record(
            subset="default", split=split, source_id=_dedup_id(seen, sid),
            context=[msg("user", row["inputs"])],
            responses=[response(row["targets"], model="human")]
            if _nonempty(row.get("targets")) else [],
            language=row.get("language_code"),
            meta={"user_id": row.get("user_id"),
                  "language_name": row.get("language"),
                  "annotation_type": row.get("annotation_type")})


# ---------------------------------------------------------------- TrustLLM

@register("AnnikaSimonsen/TrustLLM-reformulation-prompts")
def trustllm_reformulation(ctx):
    # `id` is neither unique nor always set (NaN for 1,068 rows), so the row
    # index within the (single) parquet file is the source id.
    # The release has no English original / OASST link column.
    i = -1
    for path, row in iter_parquet(ctx.files("data/*.parquet")):
        i += 1
        if not _nonempty(row.get("original_text")):
            ctx.skip("empty_prompt")
            continue
        pid = row.get("id")
        pid = int(pid) if pid is not None and pid == pid else None
        yield ctx.record(
            subset="default", split="train", source_id="row{:05d}".format(i),
            context=[msg("user", row["original_text"])],
            language=row.get("language"),
            meta={"platform_id": pid, "type": row.get("type"),
                  "created_by": row.get("created_by"),
                  "created_at": row.get("created_at")})


# ----------------------------------------------------------------- PLLuMIC

@register("pelcra/PLLuMIC")
def pllumic(ctx):
    seen = {}
    for path, conv in iter_json(ctx.files("pllumic.json")):
        raw = conv.get("messages") or []
        kept = [m for m in raw
                if not (m["role"] == "system" and not _nonempty(m.get("content")))]
        messages = [msg(m["role"], m.get("content")) for m in kept]
        context, reply = split_last_turn(messages)
        if context is None or not _nonempty(context[-1]["content"]):
            ctx.skip("no_user_turn")
            continue
        last_user = [m for m in kept if m["role"] == "user"][-1]
        users = [m for m in kept if m["role"] == "user"]
        yield ctx.record(
            subset="default", split="train",
            source_id=_dedup_id(seen, conv["conv_id"]),
            context=context,
            responses=[response(reply, model="human")]
            if reply is not None else [],
            language=last_user.get("language"),
            meta={
                "conv_id": conv["conv_id"],
                "dataset_source": conv.get("dataset_source"),
                "instruction_id": last_user.get("instruction_id"),
                "type": last_user.get("type"),
                "subtype": last_user.get("subtype"),
                "topic": last_user.get("topic"),
                "sources": last_user.get("source"),
                "types_all_turns": [u.get("type") for u in users],
                "has_system_prompt": any(m["role"] == "system" for m in kept),
            })


# -------------------------------------------------------------- PLLuM-Align

def _pllum_category(pid):
    return re.sub(r"[-_]?\d.*$", "", pid)


@register("NASK-PIB/PLLuM-Align")
def pllum_align(ctx):
    def last(conv):
        return conv[-1]["content"]

    def ctx_of(conv):
        return [msg(m["role"], m["content"]) for m in conv[:-1]]

    # ranking.jsonl: pairs <prompt>_<rank of rejected>, chosen shared
    groups = collections.OrderedDict()
    for _, r in iter_jsonl(ctx.files("ranking.jsonl")):
        pid, rk = r["id"].rsplit("_", 1)
        groups.setdefault(pid, []).append((int(rk), r))
    for pid, pairs in groups.items():
        pairs.sort(key=lambda x: x[0])
        chosen = pairs[0][1]["chosen"]
        if any(last(p["chosen"]) != last(chosen) for _, p in pairs):
            ctx.skip("ranking_inconsistent_chosen")
            continue
        context = ctx_of(chosen)
        if not context or context[-1]["role"] != "user":
            ctx.skip("no_user_turn")
            continue
        rks = [rk for rk, _ in pairs]
        annot = 1 in rks  # rank-1 output among rejected -> chosen is annotator's
        responses = [response(last(chosen), model="human" if annot else None,
                              chosen_is_annotator_answer=annot or None)]
        responses += [response(last(p["rejected"]), source_rank=rk)
                      for rk, p in pairs]
        yield ctx.record(
            subset="ranking", split="train", source_id="ranking:" + pid,
            context=context, responses=responses,
            preference={"type": "ranking",
                        "ranks": list(range(len(responses))),
                        "raw": {"pair_ids": [p["id"] for _, p in pairs],
                                "rejected_ranks": rks}},
            language="pl",
            meta={"prompt_id": pid, "category": _pllum_category(pid)})

    # rating.jsonl: one pair per prompt with mean ratings
    for _, r in iter_jsonl(ctx.files("rating.jsonl")):
        context = ctx_of(r["chosen"])
        if not context or context[-1]["role"] != "user":
            ctx.skip("no_user_turn")
            continue
        yield ctx.record(
            subset="rating", split="train", source_id="rating:" + r["id"],
            context=context,
            responses=[response(last(r["chosen"])),
                       response(last(r["rejected"]))],
            preference={"type": "scores",
                        "scores": [r.get("chosen_rating"),
                                   r.get("rejected_rating")],
                        "scale": "1-5, mean of 7 criteria",
                        "details": [r.get("chosen_rating_details"),
                                    r.get("rejected_rating_details")],
                        "raw": {"chosen": 0, "rejected": 1}},
            language="pl",
            meta={"prompt_id": r["id"], "category": _pllum_category(r["id"])})

    # dialogs.jsonl: <dialogue>_<turn>_<pair>, same history + chosen per turn
    turns = collections.OrderedDict()
    for _, r in iter_jsonl(ctx.files("dialogs.jsonl")):
        d, t, p = r["id"].split("_")
        turns.setdefault((d, int(t)), []).append((int(p), r))
    for (d, t), pairs in turns.items():
        pairs.sort(key=lambda x: x[0])
        chosen = pairs[0][1]["chosen"]
        if any(p["chosen"] != chosen or p["rejected"][:-1] != chosen[:-1]
               for _, p in pairs):
            ctx.skip("dialog_inconsistent_pair")
            continue
        context = ctx_of(chosen)
        if not context or context[-1]["role"] != "user":
            ctx.skip("no_user_turn")
            continue
        texts = [last(chosen)]
        for _, p in pairs:
            if last(p["rejected"]) not in texts:
                texts.append(last(p["rejected"]))
        yield ctx.record(
            subset="dialogs", split="train",
            source_id="dialogs:{}_{}".format(d, t),
            context=context, responses=[response(x) for x in texts],
            preference={"type": "best_of_n", "chosen": 0,
                        "raw": [p["id"] for _, p in pairs]},
            language="pl",
            meta={"dialogue_id": d, "turn_index": t})


# ----------------------------------------------------- questions_ujc_cas_cs

@register("ctu-aic/questions_ujc_cas_cs")
def questions_ujc(ctx):
    # conversations[] is the curated (URL-stripped) user/answer pair; it is
    # empty for 27 rows, where we fall back to specific_question/question.
    for path, row in iter_parquet(ctx.files("data/*.parquet")):
        conv = row.get("conversations") or []
        if conv:
            messages = [msg(m["role"], m["content"]) for m in conv]
            context, reply = split_last_turn(messages)
        else:
            q = row.get("specific_question") or row.get("question")
            context = [msg("user", q)] if _nonempty(q) else None
            reply = row.get("answer")
        if not context or not _nonempty(context[-1]["content"]):
            ctx.skip("no_user_turn")
            continue
        lu = row.get("last_usage")
        yield ctx.record(
            subset="default", split="train", source_id=row["id"],
            context=context,
            responses=[response(reply, model="human")]
            if _nonempty(reply) else [],
            language="cs",
            meta={"question_category": row.get("question"),
                  "keyword": row.get("keyword"),
                  "variants": row.get("variants"),
                  "last_usage": str(lu) if lu is not None else None,
                  "specific_question_missing":
                      not _nonempty(row.get("specific_question")),
                  "conversations_missing": not conv})


# ---------------------------------------------------------------- MentorES

@register("projecte-aina/MentorES")
def mentor_es(ctx):
    # Dolly format; the reference text (if any) follows the instruction.
    # topic_list.csv is a topic description table, not prompts.
    # MentorCA is a separate HF repo and is not part of this download.
    seen = {}
    for path, row in iter_jsonl(ctx.files("*.jsonl")):
        instr = row.get("instruction")
        if not _nonempty(instr):
            ctx.skip("empty_prompt")
            continue
        ref = row.get("context")
        text = instr.strip() + ("\n\n" + ref.strip() if _nonempty(ref) else "")
        yield ctx.record(
            subset="default", split=_split_from_name(path),
            source_id=_dedup_id(seen, row["id"]),
            context=[msg("user", text)],
            responses=[response(row["response"], model="human")]
            if _nonempty(row.get("response")) else [],
            language="es",
            meta={"category": row.get("category"),
                  "has_reference_text": _nonempty(ref)})
