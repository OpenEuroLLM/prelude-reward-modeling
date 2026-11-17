"""Human/synthetic preference sets, large prompt pools and instruction-
following (IF) sets with verification info.

Preference sources
------------------
nvidia/HelpSteer3 (configs = subsets; `validation` split kept as validation)
  preference: responses = [response1, response2];
      preference = {"type": "score_diff", "value": overall_preference,
                    "scale": "-3..3; <0 means responses[0] better",
                    "winner": 0 | 1 | "tie",
                    "raw": {"overall_preference", "individual_preference"}}
      ("score_diff" is a signed graded pairwise label; individual_preference
      is kept as given: per-annotator score, reasoning, feedback1/feedback2).
      Response meta {"helpfulness": [...]}: per-annotator helpfulness 1..5
      parsed from that annotator's feedback text "The response is
      {not|slightly|partially|mostly|perfectly} helpful" (None if no match).
  feedback: same contexts/response pairs (mostly) with a separately selected
      set of up to 3 free-text feedbacks per response:
      preference = {"type": "scores", "scores": [mean helpfulness r1, r2],
                    "scale": "1..5 mean annotator helpfulness ...",
                    "raw": {"helpfulness": [[...], [...]]}};
      response meta {"feedback": [texts as given], "helpfulness": [...]}.
  edit / edit_quality / principle are not extracted (same prompts; signals
      are edit-model training data or LLM-derived principle labels).
  `language` = source label for natural languages; for domain "code" the
  source label is a programming language -> language None, kept in
  meta.code_language. The source has many exact duplicate rows: they are
  kept (one record each) and marked with meta.exact_dup_of = source_id of
  the first identical row.
nvidia/Nemotron-Cascade-RM-Training: chosen/rejected conversations with a
  shared prefix -> responses [chosen, rejected], "chosen_rejected";
  subset = the row's `source` field. Exact duplicates marked as above.
nvidia/Nemotron-Cascade-RL-RLHF: prompt only; subset = data_source,
  meta keeps index/category/cat.
allenai/tulu-3-wildchat-unused: single-turn WildChat prompts with an
  UltraFeedback-judged chosen/rejected pair -> "chosen_rejected". The rows
  carry no WildChat hash/model names (only a running id).
allenai/Dolci-Instruct-SFT: one record per conversation, subset =
  source_dataset, meta {source_dataset, domain}. Tool-use rows (roles
  system/user/assistant/environment) are linearised as in the OLMo 3 chat
  template: system content + " <functions>{functions}</functions>",
  assistant content + "<function_calls>{function_calls}</function_calls>".
  Environment (tool output) turns cannot be represented in the 3-role
  schema, so the conversation is cut before the first environment turn:
  context = up to the last user turn before it, response = the assistant
  turn right after that user turn (usually a function call); meta
  {"tool_use": true, "dropped_user_turns": n}.

Instruction-following sources (verification info kept in meta as given)
------------------------------------------------------------------------
allenai/IF_multi_constraints_upto5: prompt only. meta.ground_truth = the
  original string (a Python literal, consumed with ast.literal_eval by the
  open-instruct IFEval verifier), plus instruction_id_list / kwargs parsed
  from it, key, dataset, constraint_type, constraint (texts, tab-separated).
nvidia/Nemotron-RL-instruction_following: prompt only (WildChat prompts +
  IFEval-style instructions); meta.instruction_id_list / kwargs as given.
allenai/tulu-3-sft-personas-instruction-following: prompt + SFT response;
  meta.constraints = the list of constraint names as given (taxonomy names,
  no kwargs -> not directly machine-checkable). language "en" (card).
openeurollm/EU-Instruct-Following-Synthetic: prompt + SFT response per
  language dir (subset = dir, language = row column). The source has no
  verification info.
LumiOpen/AutoIF-FI: 2-3 turn Finnish chats (language "fi" from the card),
  test split -> test. Each later user turn repeats all earlier constraints,
  so every constraint applies to the final turn. meta.constraints = list as
  given (constraint_id, constraint, template, category, eval_funcs =
  candidate `evaluate(response, **kwargs)` sources, placeholders); the
  placeholder `values_json` string is stored parsed as `values`.
"""
import ast
import hashlib
import json
import os
import re

from common import (iter_jsonl, iter_parquet, msg, norm_messages, register,
                    response, split_last_turn)

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.S)
HELPFUL_RE = re.compile(
    r"^\W*the response is\W+(not|slightly|partially|mostly|perfectly)\W+helpful",
    re.I)
HELPFUL = {"not": 1, "slightly": 2, "partially": 3, "mostly": 4, "perfectly": 5}


def _resp(text, model=None, **meta):
    """response() with <think> blocks stripped (marked in meta)."""
    text = "" if text is None else str(text)
    if "<think>" in text:
        stripped = THINK_RE.sub("", text)
        if stripped != text:
            text = stripped
            meta["reasoning_dropped"] = True
    return response(text, model=model, **meta)


def _has_user_text(context):
    return context is not None and any(
        m["role"] == "user" and m["content"].strip() for m in context)


class _DupMarker:
    """Returns the source_id of the first identical row (or None)."""

    def __init__(self):
        self.first = {}

    def __call__(self, obj, sid):
        h = hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False)
                         .encode("utf-8")).hexdigest()
        first = self.first.setdefault(h, sid)
        return None if first == sid else first


def _nonnull(d):
    return {k: v for k, v in d.items() if v is not None}


# ------------------------------------------------------------- HelpSteer3

def _helpfulness(text):
    m = HELPFUL_RE.match(text or "")
    return HELPFUL[m.group(1).lower()] if m else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


@register("nvidia/HelpSteer3")
def helpsteer3(ctx):
    for cfg in ("preference", "feedback"):
        dup = _DupMarker()
        for split in ("train", "validation"):
            path = os.path.join(ctx.raw_dir, cfg, split + ".jsonl.gz")
            for i, (_, row) in enumerate(iter_jsonl([path])):
                sid = "{}/{}/{}".format(cfg, split, i)
                context = norm_messages(row["context"])
                if not context or context[-1]["role"] != "user" \
                        or not _has_user_text(context):
                    ctx.skip("{}_no_user_turn".format(cfg))
                    continue
                code = row["domain"] == "code"
                meta = _nonnull({
                    "domain": row["domain"],
                    "language": row["language"],
                    "code_language": row["language"] if code else None,
                    "exact_dup_of": dup(row, sid),
                })
                if cfg == "preference":
                    ip = row.get("individual_preference") or []
                    responses = [
                        _resp(row["response1"], helpfulness=[
                            _helpfulness(p.get("feedback1")) for p in ip]),
                        _resp(row["response2"], helpfulness=[
                            _helpfulness(p.get("feedback2")) for p in ip]),
                    ]
                    v = row["overall_preference"]
                    pref = {"type": "score_diff", "value": v,
                            "scale": "-3..3; <0 means responses[0] better",
                            "winner": 0 if v < 0 else 1 if v > 0 else "tie",
                            "raw": {"overall_preference": v,
                                    "individual_preference": ip}}
                else:
                    h1 = [_helpfulness(t) for t in row["feedback1"] or []]
                    h2 = [_helpfulness(t) for t in row["feedback2"] or []]
                    responses = [
                        _resp(row["response1"], feedback=row["feedback1"],
                              helpfulness=h1),
                        _resp(row["response2"], feedback=row["feedback2"],
                              helpfulness=h2),
                    ]
                    pref = {"type": "scores", "scores": [_mean(h1), _mean(h2)],
                            "scale": "1..5 mean annotator helpfulness parsed "
                                     "from feedback 'The response is {not|"
                                     "slightly|partially|mostly|perfectly} "
                                     "helpful'",
                            "raw": {"helpfulness": [h1, h2]}}
                yield ctx.record(
                    subset=cfg, split=split, source_id=sid, context=context,
                    responses=responses, preference=pref,
                    language=None if code else row["language"], meta=meta)


# ------------------------------------------------------ chosen / rejected

def _chosen_rejected(ctx, chosen, rejected):
    """-> (context, [chosen, rejected] responses) or None (counted skip)."""
    c_ctx, c_reply = split_last_turn(norm_messages(chosen))
    r_ctx, r_reply = split_last_turn(norm_messages(rejected))
    if not _has_user_text(c_ctx):
        ctx.skip("no_user_turn")
        return None
    if c_ctx != r_ctx:
        ctx.skip("chosen_rejected_context_mismatch")
        return None
    if c_reply is None or r_reply is None:
        ctx.skip("missing_chosen_or_rejected_reply")
        return None
    return c_ctx, [_resp(c_reply), _resp(r_reply)]


CHOSEN_REJECTED = {"type": "chosen_rejected", "chosen": 0, "rejected": 1,
                   "raw": "chosen/rejected columns"}


@register("nvidia/Nemotron-Cascade-RM-Training")
def nemotron_cascade_rm(ctx):
    dup = _DupMarker()
    for i, (_, row) in enumerate(iter_jsonl(ctx.files("rm_training_data.jsonl"))):
        out = _chosen_rejected(ctx, row["chosen"], row["rejected"])
        if out is None:
            continue
        sid = str(i)
        yield ctx.record(
            subset=row.get("source") or "unknown", split="train",
            source_id=sid, context=out[0], responses=out[1],
            preference=CHOSEN_REJECTED,
            meta=_nonnull({"data_source": row.get("source"),
                           "exact_dup_of": dup(row, sid)}))


@register("nvidia/Nemotron-Cascade-RL-RLHF")
def nemotron_cascade_rlhf(ctx):
    for _, row in iter_parquet(ctx.files("rlhf_train_data.parquet")):
        context = norm_messages(row["prompt"])
        if not context or context[-1]["role"] != "user" \
                or not _has_user_text(context):
            ctx.skip("no_user_turn")
            continue
        yield ctx.record(
            subset=row["data_source"], split="train",
            source_id=str(row["index"]), context=context,
            meta={"data_source": row["data_source"], "index": row["index"],
                  "category": row["category"], "cat": row["cat"]})


@register("allenai/tulu-3-wildchat-unused")
def tulu3_wildchat_unused(ctx):
    for _, row in iter_parquet(ctx.files("data/*.parquet")):
        out = _chosen_rejected(ctx, row["chosen"], row["rejected"])
        if out is None:
            continue
        yield ctx.record(
            subset="default", split="train", source_id=row["id"],
            context=out[0], responses=out[1], preference=CHOSEN_REJECTED)


# ------------------------------------------------------------------ Dolci

def _dolci_messages(raw):
    """Normalize Dolci messages; environment turns become None markers."""
    out = []
    for m in raw or []:
        role = m["role"]
        if role == "environment":
            out.append(None)
            continue
        content = m.get("content") or ""
        if role == "system" and m.get("functions") is not None:
            content += " <functions>" + m["functions"] + "</functions>"
        if role == "assistant" and m.get("function_calls") is not None:
            content += ("<function_calls>" + m["function_calls"]
                        + "</function_calls>")
        out.append(msg(role, content))
    return out


@register("allenai/Dolci-Instruct-SFT")
def dolci_instruct_sft(ctx):
    cols = ["id", "messages", "source_dataset", "domain"]
    for _, row in iter_parquet(ctx.files("data/*.parquet"), columns=cols):
        messages = _dolci_messages(row["messages"])
        tool_use = any(m.get("functions") or m.get("function_calls")
                       or m["role"] == "environment"
                       for m in row["messages"] or [])
        env = next((i for i, m in enumerate(messages) if m is None), None)
        prefix = messages if env is None else messages[:env]
        context, reply = split_last_turn(prefix)
        if not _has_user_text(context):
            ctx.skip("no_user_turn")
            continue
        meta = {"source_dataset": row["source_dataset"], "domain": row["domain"]}
        if tool_use:
            meta["tool_use"] = True
            n_all = sum(1 for m in messages if m and m["role"] == "user")
            meta["dropped_user_turns"] = n_all - sum(
                1 for m in context if m["role"] == "user")
        yield ctx.record(
            subset=row["source_dataset"], split="train", source_id=row["id"],
            context=context,
            responses=[_resp(reply)] if reply is not None else [],
            meta=meta)


# ---------------------------------------------------- instruction following

@register("allenai/IF_multi_constraints_upto5")
def if_multi_constraints(ctx):
    seen = {}
    for _, row in iter_parquet(ctx.files("data/*.parquet")):
        context, reply = split_last_turn(norm_messages(row["messages"]))
        if not _has_user_text(context):
            ctx.skip("no_user_turn")
            continue
        # `key` repeats for a few hard_coded_* rows; disambiguate repeats
        k = row["key"]
        seen[k] = seen.get(k, 0) + 1
        sid = k if seen[k] == 1 else "{}#{}".format(k, seen[k])
        meta = {"key": k, "dataset": row["dataset"],
                "constraint_type": row["constraint_type"],
                "constraint": row["constraint"],
                "ground_truth": row["ground_truth"]}
        try:
            gt = ast.literal_eval(row["ground_truth"])
            if isinstance(gt, list) and len(gt) == 1 and isinstance(gt[0], dict):
                meta["instruction_id_list"] = gt[0].get("instruction_id")
                meta["kwargs"] = gt[0].get("kwargs")
        except (ValueError, SyntaxError):
            pass  # raw ground_truth string is kept either way
        yield ctx.record(
            subset="default", split="train", source_id=sid, context=context,
            responses=[_resp(reply)] if reply is not None else [], meta=meta)


@register("nvidia/Nemotron-RL-instruction_following")
def nemotron_rl_if(ctx):
    for _, row in iter_jsonl(ctx.files("instruction_following.jsonl")):
        rcp = row.get("responses_create_params") or {}
        if rcp.get("input"):
            context = norm_messages(rcp["input"])
        else:
            context = [msg("user", row["prompt"])]
        if not context or context[-1]["role"] != "user" \
                or not _has_user_text(context):
            ctx.skip("no_user_turn")
            continue
        meta = {"instruction_id_list": row["instruction_id_list"],
                "kwargs": row["kwargs"]}
        extra = {k: v for k, v in rcp.items()
                 if k != "input" and v not in (None, [], False)}
        if extra:
            meta["responses_create_params"] = extra
        yield ctx.record(
            subset="default", split="train", source_id=str(row["id"]),
            context=context, meta=meta)


@register("allenai/tulu-3-sft-personas-instruction-following")
def tulu3_personas_if(ctx):
    for _, row in iter_parquet(ctx.files("data/*.parquet")):
        context, reply = split_last_turn(norm_messages(row["messages"]))
        if not _has_user_text(context):
            ctx.skip("no_user_turn")
            continue
        yield ctx.record(
            subset="default", split="train", source_id=row["id"],
            context=context,
            responses=[_resp(reply)] if reply is not None else [],
            language="en", meta={"constraints": row["constraints"]})


@register("openeurollm/EU-Instruct-Following-Synthetic")
def eu_instruct_following(ctx):
    for path in ctx.files("*/train.parquet"):
        lang_dir = os.path.basename(os.path.dirname(path))
        for i, (_, row) in enumerate(iter_parquet([path])):
            context, reply = split_last_turn(norm_messages(row["messages"]))
            if not _has_user_text(context):
                ctx.skip("no_user_turn")
                continue
            yield ctx.record(
                subset=lang_dir, split="train",
                source_id="{}/{}".format(lang_dir, i), context=context,
                responses=[_resp(reply)] if reply is not None else [],
                language=row.get("language") or lang_dir)


def _autoif_constraints(constraints):
    out = []
    for c in constraints or []:
        c = dict(c)
        phs = []
        for p in c.get("placeholders") or []:
            p = dict(p)
            vj = p.pop("values_json", None)
            p["values"] = json.loads(vj) if vj else None
            phs.append(p)
        c["placeholders"] = phs
        out.append(c)
    return out


@register("LumiOpen/AutoIF-FI")
def autoif_fi(ctx):
    for split in ("train", "test"):
        path = os.path.join(ctx.raw_dir, split + ".parquet")
        for i, (_, row) in enumerate(iter_parquet([path])):
            context, reply = split_last_turn(norm_messages(row["messages"]))
            if not _has_user_text(context):
                ctx.skip("no_user_turn")
                continue
            yield ctx.record(
                subset="default", split=split,
                source_id="{}/{}".format(split, i), context=context,
                responses=[_resp(reply)] if reply is not None else [],
                language="fi",
                meta={"constraints": _autoif_constraints(row["constraints"])})
