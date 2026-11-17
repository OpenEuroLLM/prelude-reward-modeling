#!/usr/bin/env python3
"""The judge prompt: one overall pairwise preference with a forced binary verdict, written analysis first.

build(pair, order) -> (system, user)   order 0: A = response_x, B = response_y;  order 1: A = response_y, B = response_x
SCHEMA                                 JSON schema of the judge's answer (analysis fields before the verdict)
extract(text)                          -> dict or None: the last JSON object with a verdict in a raw model output

The system prompt is the text of judge_prompt.txt (version v3, the one every vote of the dataset was made with;
the version id is stored with every vote). The judge sees the conversation and the two final answers verbatim
and nothing else: no model names, no hidden reasoning, no metadata. Instructions and analysis are in English
(judges are more reliable that way), the content stays in its own language. The pilot's earlier versions (v1 and
v2: the same criteria at twice the length, and stated facts about the language and length limits in the user
message) are no longer in the repository; the repository README describes the pilot.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT_VERSION = "v3"
with open(os.path.join(HERE, "judge_prompt.txt"), encoding="utf-8") as _f:
    SYSTEM = _f.read().rstrip("\n")

SCHEMA = {
    "type": "object",
    "properties": {
        "request": {"type": "string"},
        "response_a": {"type": "string"},
        "response_b": {"type": "string"},
        "comparison": {"type": "string"},
        "verdict": {"type": "string", "enum": ["A", "B"]},
    },
    "required": ["request", "response_a", "response_b", "comparison", "verdict"],
    "additionalProperties": False,
}

# language-ID siblings that count as the same language (as in collect-responses/smoke_test.py)
SAME = [{"bs", "hr", "sr", "sr-Latn", "me"}, {"nb", "nn", "no"}, {"lv", "ltg"}, {"et", "ekk"}, {"sq", "als"}]


def same_lang(a, b):
    return a == b or any(a in g and b in g for g in SAME)


def render_conversation(messages):
    parts = []
    for m in messages:
        parts.append("[{}]\n{}".format(m["role"], (m.get("content") or "").strip()))
    return "\n\n".join(parts)


def build(pair, order):
    """-> (system, user). pair: a row of pairs.parquet (dict). order 0: A = x, B = y; order 1: A = y, B = x."""
    messages = json.loads(pair["messages"]) if isinstance(pair["messages"], str) else pair["messages"]
    a, b = ("x", "y") if order == 0 else ("y", "x")
    resp = {"A": pair["response_" + a], "B": pair["response_" + b]}
    body = ["<conversation>", render_conversation(messages), "</conversation>", ""]
    for label in ("A", "B"):
        text = resp[label].strip() or "(empty response)"
        body += ["<response_{}>".format(label), text, "</response_{}>".format(label), ""]
    body.append("Which response is better, A or B? Answer with the JSON object described in your instructions.")
    return SYSTEM, "\n".join(body)


def extract_span(text):
    """-> (obj, end): the last JSON object in `text` that has a verdict and the index just after it, or
    (None, None). Tolerates thinking or channel text around it."""
    if not text:
        return None, None
    dec = json.JSONDecoder()
    for m in reversed(list(re.finditer(r"\{", text))):
        try:
            obj, end = dec.raw_decode(text, m.start())
        except ValueError:
            continue
        if isinstance(obj, dict) and obj.get("verdict") in ("A", "B"):
            return obj, end
    return None, None


def extract(text):
    """the last JSON object in `text` that has a verdict"""
    return extract_span(text)[0]
