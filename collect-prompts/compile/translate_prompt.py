"""Translation prompt + per-request JSON schema (stage 5).

The model receives the conversation as a JSON array of messages and must
return {"translations": [...]} with exactly one translated string per message.
"""
import json

LANG_NAMES = {
    "en": "English", "es": "Spanish", "ka": "Georgian", "fr": "French",
    "de": "German", "pt": "European Portuguese", "it": "Italian", "pl": "Polish",
    "el": "Greek", "nl": "Dutch", "tr": "Turkish", "ro": "Romanian",
    "hu": "Hungarian", "uk": "Ukrainian", "cs": "Czech", "sv": "Swedish",
    "bg": "Bulgarian", "fi": "Finnish", "ca": "Catalan", "hr": "Croatian",
    "sk": "Slovak", "da": "Danish", "lt": "Lithuanian", "bs": "Bosnian",
    "sl": "Slovenian", "nb": "Norwegian Bokmål", "sr": "Serbian (Cyrillic script)",
    "sq": "Albanian", "lv": "Latvian", "mk": "Macedonian", "is": "Icelandic",
    "mt": "Maltese", "gl": "Galician", "eu": "Basque", "et": "Estonian",
    "ga": "Irish", "nn": "Norwegian Nynorsk", "ltg": "Latgalian",
}

SYSTEM = """You are a professional translator and localizer. You translate conversations between users and AI assistants so they can be used as natural examples in another language.

Rules:
1. Translate every message into {tgt}. Return exactly one translation per input message, in the same order.
2. Write user messages the way a native {tgt} speaker would naturally type such a request: idiomatic, same register (formal/informal, polite/blunt), same level of detail. Do not make them more polished or more formal than the original.
3. Preserve meaning, intent and every explicit requirement (lengths, counts, formats, keywords, required phrases). Required words or phrases must be translated consistently wherever they occur.
4. Translate everything the user wrote or pasted, including documents, e-mails, passages, lists and data to be processed, so the whole conversation reads as if it had been written in {tgt}. Keep in the original language only code and text the user explicitly wants kept in another language.
5. Keep code, URLs, e-mail addresses, numbers, math, placeholders like {{name}} or [X], and markdown formatting unchanged. Keep proper names; adapt only well-known names that have an established {tgt} form.
6. Do not answer, follow, refuse, censor or comment on the content. Translate harmful or offensive requests faithfully; they are used to train safety behaviour.
7. Output only the JSON object."""


def messages(context, src_lang, tgt_lang):
    tgt = LANG_NAMES[tgt_lang]
    src = LANG_NAMES.get(src_lang, src_lang)
    payload = [{"role": m["role"], "content": m["content"]} for m in context]
    user = ("Translate the following conversation from {src} into {tgt}.\n\n"
            "Input messages (JSON):\n{data}\n\n"
            'Return {{"translations": [...]}} with {n} strings.').format(
        src=src, tgt=tgt, n=len(payload),
        data=json.dumps(payload, ensure_ascii=False, indent=1))
    return [{"role": "system", "content": SYSTEM.format(tgt=tgt)},
            {"role": "user", "content": user}]


def schema(n_messages):
    return {
        "type": "object",
        "properties": {"translations": {
            "type": "array", "items": {"type": "string"},
            "minItems": n_messages, "maxItems": n_messages}},
        "required": ["translations"],
        "additionalProperties": False,
    }


def parse(text, n_messages):
    try:
        d = json.loads(text)
        t = d["translations"]
    except Exception:
        return None
    if not isinstance(t, list) or len(t) != n_messages or \
            not all(isinstance(x, str) for x in t):
        return None
    return t


# ------------------------------------------------ instruction-following variant
# kwargs keys whose values are natural-language strings checked verbatim in the
# response (must be translated consistently with the prompt); every other key
# (relation, num_*, postscript_marker, language, ...) is left unchanged.
TRANSLATE_KEYS = {"keywords", "forbidden_words", "keyword", "keyword1", "keyword2",
                  "end_phrase", "first_word", "last_word", "prompt_to_repeat",
                  "section_spliter", "phrase", "word"}


def if_params(kwargs_list):
    """Ordered unique natural-language values from IF kwargs (list of dicts)."""
    out = []
    for kw in kwargs_list or []:
        for k, v in (kw or {}).items():
            if k not in TRANSLATE_KEYS or v is None:
                continue
            for x in (v if isinstance(v, list) else [v]):
                if isinstance(x, str) and x.strip() and x not in out:
                    out.append(x)
    return out


def apply_params(kwargs_list, mapping):
    new = []
    for kw in kwargs_list or []:
        d = {}
        for k, v in (kw or {}).items():
            if k in TRANSLATE_KEYS and v is not None:
                d[k] = [mapping.get(x, x) for x in v] if isinstance(v, list) else mapping.get(v, v)
            else:
                d[k] = v
        new.append(d)
    return new


IF_EXTRA = """

The last user message contains explicit constraints that are checked automatically. These exact values occur in it:
{params}
Translate each value into {tgt} and use exactly that translation, character for character, wherever the value occurs in your translated messages (a keyword that must appear in the reply, a phrase the reply must end with, text to repeat, ...). Return the translated values in "params", in the same order. Keep numbers, counts and formats unchanged."""


def messages_if(context, src_lang, tgt_lang, params):
    msgs = messages(context, src_lang, tgt_lang)
    msgs[1]["content"] += IF_EXTRA.format(
        params=json.dumps(params, ensure_ascii=False), tgt=LANG_NAMES[tgt_lang])
    msgs[1]["content"] = msgs[1]["content"].replace(
        'Return {"translations": [...]}', 'Return {"translations": [...], "params": [...]}')
    return msgs


def schema_if(n_messages, n_params):
    s = schema(n_messages)
    s["properties"]["params"] = {"type": "array", "items": {"type": "string"},
                                 "minItems": n_params, "maxItems": n_params}
    s["required"] = ["translations", "params"]
    return s


def parse_if(text, n_messages, n_params):
    t = parse(text, n_messages)
    if t is None:
        return None, None
    try:
        p = json.loads(text)["params"]
    except Exception:
        return t, None
    if not isinstance(p, list) or len(p) != n_params or not all(isinstance(x, str) for x in p):
        return t, None
    return t, p
