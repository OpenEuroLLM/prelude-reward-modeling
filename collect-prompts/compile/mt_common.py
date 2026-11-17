"""Shared helpers for per-text translators (TranslateGemma, few-shot Latgalian MT).

segments(text) splits a message into pieces whose contents concatenate back to
the original text exactly:
  ("text", s)  prose to translate (paragraph chunks of at most max_chars)
  ("code", s)  fenced code block, kept verbatim (an unterminated ``` counts only when it
               opens a code block: ``` or ```lang at the end of its line; an inline mention
               like "use triple backticks (```) for code" is prose, see dangling_fence)
  ("ws", s)    whitespace between pieces, kept verbatim
  ("keep", s)  text without letters (numbers, URLs, symbols), kept verbatim

Also: TranslateGemma language codes, a code/JSON detector (items whose prose
is mostly code are left to the conversation-level translator), the FLORES+
evaluation sample and few-shot exemplars for Latgalian, and an exotic-script
detector (stray Hangul/Thai/Kana/Indic glyphs = MT artifacts).
"""
import collections
import functools
import json
import os
import random
import re
import unicodedata as ud

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")

FENCE = re.compile(r"```.*?(?:```|\Z)", re.S)
LETTER = re.compile(r"[^\W\d_]")
CODELINE = re.compile(r'^\s*([\[\]{}]|"[^"\n]*"\s*:|(def|class|import|from|function|var|let|const|'
                      r'public|private|return|#include|SELECT|INSERT|UPDATE)\b)|[;{]\s*$')

# our language code -> TranslateGemma code (the target variant we want)
TG_TARGET = {"pt": "pt-PT", "sr": "sr-Cyrl", "bs": "bs-Latn"}
NAMES = {"en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
         "pt": "Portuguese", "pl": "Polish", "nl": "Dutch", "lv": "Latvian", "ltg": "Latgalian"}
FLORES_CODE = {"en": "eng_Latn", "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
               "it": "ita_Latn", "pt": "por_Latn", "pl": "pol_Latn", "nl": "nld_Latn",
               "lv": "lvs_Latn", "ltg": "ltg_Latn",
               "ka": "kat_Geor", "el": "ell_Grek", "tr": "tur_Latn", "ro": "ron_Latn",
               "hu": "hun_Latn", "uk": "ukr_Cyrl", "cs": "ces_Latn", "sv": "swe_Latn",
               "bg": "bul_Cyrl", "fi": "fin_Latn", "ca": "cat_Latn", "hr": "hrv_Latn",
               "sk": "slk_Latn", "da": "dan_Latn", "lt": "lit_Latn", "bs": "bos_Latn",
               "sl": "slv_Latn", "nb": "nob_Latn", "sr": "srp_Cyrl", "sq": "als_Latn",
               "mk": "mkd_Cyrl", "is": "isl_Latn", "mt": "mlt_Latn", "gl": "glg_Latn",
               "eu": "eus_Latn", "et": "ekk_Latn", "ga": "gle_Latn", "nn": "nno_Latn"}


def tg_code(lang, target=True):
    return TG_TARGET.get(lang, lang) if target else lang


def codeish(text):
    """Prose outside code fences that is mostly code/JSON lines."""
    lines = [ln for ln in FENCE.sub("", text or "").split("\n") if ln.strip()]
    n = sum(bool(CODELINE.search(ln)) for ln in lines)
    return n >= 5 and n >= 0.3 * len(lines)


def _text_pieces(region, max_chars):
    if not region:
        return []
    out = []
    body = region.strip()
    if not body:
        return [("ws", region)]
    lead = region[:len(region) - len(region.lstrip())]
    trail = region[len(region.rstrip()):]
    if lead:
        out.append(("ws", lead))
    parts = re.split(r"(\n[ \t]*\n\s*)", body)
    paras, seps = parts[0::2], parts[1::2]
    chunk = paras[0]
    for sep, p in zip(seps, paras[1:]):
        if len(chunk) + len(sep) + len(p) > max_chars:
            out.append(("text", chunk))
            out.append(("ws", sep))
            chunk = p
        else:
            chunk += sep + p
    out.append(("text", chunk))
    if trail:
        out.append(("ws", trail))
    return [("keep", s) if k == "text" and not LETTER.search(s) else (k, s) for k, s in out]


def _dangling(fence):
    """An unterminated FENCE match whose opening line is not ``` / ```lang."""
    if len(fence) >= 6 and fence.endswith("```"):
        return False
    return not re.fullmatch(r"[\w+#.-]{0,20}[ \t]*", fence[3:].split("\n", 1)[0])


def dangling_fence(text):
    """True if the text has a ``` that FENCE takes for an unterminated code block although it
    is an inline mention (segmentations made before 2026-11-10 kept the rest of such a
    message verbatim, untranslated)."""
    return any(_dangling(m.group(0)) for m in FENCE.finditer(text or ""))


def segments(text, max_chars=2000):
    text = text or ""
    out, pos = [], 0
    for m in FENCE.finditer(text):
        if _dangling(m.group(0)):       # runs to the end of the text: the rest is prose
            break
        out += _text_pieces(text[pos:m.start()], max_chars)
        out.append(("code", m.group(0)))
        pos = m.end()
    out += _text_pieces(text[pos:], max_chars)
    assert "".join(s for _, s in out) == text
    return out


def tg_messages(text, src, tgt):
    return [{"role": "user", "content": [{"type": "text", "source_lang_code": tg_code(src, False),
                                          "target_lang_code": tg_code(tgt), "text": text}]}]


# ---- exotic-script artifacts --------------------------------------------------
EXOTIC = ("HANGUL", "THAI", "HIRAGANA", "KATAKANA", "DEVANAGARI", "BENGALI", "TAMIL", "ARMENIAN")
_CODE = re.compile(r"`[^`]*`|```.*?```", re.S)


def exotic_counts(text):
    c = {}
    for ch in _CODE.sub(" ", text or ""):
        if ch.isalpha() and ord(ch) > 0x2FF:
            try:
                name = ud.name(ch)
            except ValueError:
                continue
            for s in EXOTIC:
                if s in name:
                    c[s] = c.get(s, 0) + 1
    return c


_SCRIPTS = ("LATIN", "CYRILLIC", "GREEK", "GEORGIAN", "CJK", "HIRAGANA", "KATAKANA", "HANGUL",
            "ARABIC", "HEBREW", "THAI", "DEVANAGARI", "ARMENIAN", "BENGALI", "TAMIL")
_WORD = re.compile(r"[^\W\d_]{2,}")
_CODEURL = re.compile(r"`[^`]*`|```.*?```|https?://\S+", re.S)


@functools.lru_cache(maxsize=65536)
def _script(ch):
    n = ud.name(ch, "")
    return next((s for s in _SCRIPTS if s in n), None)


def mixed_script_words(text):
    """Words mixing letters of two scripts (e.g. 'προ젝τέρ', 'eレットronikus', 'Сledя') --
    generation artifacts. A single Greek letter in a Latin word (kΩ, cosφ) is not counted."""
    out = []
    for w in _WORD.findall(_CODEURL.sub(" ", text or "")):
        s = collections.Counter(_script(c) for c in w)
        s.pop(None, None)
        if len(s) < 2:
            continue
        if set(s) == {"LATIN", "GREEK"} and s["GREEK"] == 1:
            continue
        if set(s) == {"CYRILLIC", "LATIN"} and all(c in "àèìòùÀÈÌÒÙ" for c in w if _script(c) == "LATIN"):
            continue        # Macedonian/Bulgarian 'сè', 'нè' with a Latin grave vowel
        out.append(w)
    return out


_LIT_CODE = re.compile(r"`[^`\n]+`")
_URL = re.compile(r"https?://\S+")
_DIGITS = re.compile(r"\d+")
_PREAMBLE = re.compile(r"^\s*(here is|here's|translation\s*:|sure[,!])", re.I)


def literal_problems(src, tr):
    """Literal material a translation must keep (used for the text-level translators,
    stage 5m): inline code and URLs verbatim, every digit run of the source, at least as
    many << >> [ ] { } as the source (IF-style title / placeholder formats), and no
    translator preamble ("Here is the translation:"). Code fences are ignored (kept
    verbatim by segments()). -> list of problems, empty when the translation is fine."""
    s, t = FENCE.sub(" ", src or ""), FENCE.sub(" ", tr or "")
    probs = []
    for x in _LIT_CODE.findall(s) + [u.rstrip(".,;:!?)\"'") for u in _URL.findall(s)]:
        if x not in t:
            probs.append("missing:" + x[:40])
    lost = collections.Counter(_DIGITS.findall(s)) - collections.Counter(_DIGITS.findall(t))
    if lost:
        probs.append("digits:" + ",".join(sorted(lost)[:5]))
    for tok in ("<<", ">>", "[", "]", "{", "}"):
        if t.count(tok) < s.count(tok):
            probs.append("brackets:" + tok)
    if _PREAMBLE.match(t) and not _PREAMBLE.match(s):
        probs.append("preamble")
    return probs


def injected(src_text, tr_text):
    """Exotic scripts in a translation that are absent (or rarer) in its source."""
    a, b = exotic_counts(src_text), exotic_counts(tr_text)
    return {s: n for s, n in b.items() if n > a.get(s, 0)}


# ---- TildeOpen tokenization ------------------------------------------------------
# transformers 5 (vLLM container) converts TildeOpen's SentencePiece model into a
# tokenizer that encodes every space twice ('▁Jis', ' ', '▁sace', ...); prompts are
# therefore encoded and outputs decoded with the SentencePiece model itself.
TILDE = os.path.join(DATA, "models", "TildeOpen-30b")
TILDE_DOC_SEP = 48          # <|endoftext|>: document separator / BOS / EOS


@functools.lru_cache(maxsize=None)
def tilde_sp():
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=os.path.join(TILDE, "tokenizer.model"))
    assert sp.id_to_piece(TILDE_DOC_SEP) == "<|endoftext|>", sp.id_to_piece(TILDE_DOC_SEP)
    return sp


def tilde_encode(prompt):
    return [TILDE_DOC_SEP] + tilde_sp().encode(prompt)


# ---- FLORES+ for the Latgalian evaluation --------------------------------------
FLORES = os.path.join(DATA, "ltg", "flores_plus")


@functools.lru_cache(maxsize=None)
def flores(split, lang):
    path = os.path.join(FLORES, split, FLORES_CODE[lang] + ".jsonl")
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r["text"] for r in map(json.loads, f)}


def eval_ids(n=300, seed=20261109):
    ids = sorted(flores("devtest", "ltg"))
    return sorted(random.Random(seed).sample(ids, n))


def exemplar_ids(k=24, seed=7):
    ids = sorted(flores("dev", "ltg"))
    return random.Random(seed).sample(ids, k)   # the first 8 are the k=8 set


LTG_GUIDE = (
    "Latgalian (latgalīšu volūda) is the written regional language of Latgale in eastern Latvia. "
    "It is not Latvian: it has its own spelling, vocabulary and grammar, and a translation into "
    "standard Latvian is wrong. Typical correspondences with Latvian (in many, not all, words): "
    "ā -> uo (zinātnieki -> zynuotnīki), ie -> ī (diena -> dīna, pētnieki -> pietnīki), "
    "ī -> ei (Rīga -> Reiga, dzīvot -> dzeivuot), a -> o (saka -> soka, gads -> gods), "
    "o -> ū or uo (to -> tū), i -> y after hard consonants (visi -> vysi, zināt -> zynuot); "
    "genitive -as/-es -> -ys/-is (valodas -> volūdys, Latvijas -> Latvejis), "
    "dative plural -iem -> -im; un -> i, kā -> kai, viņš -> jis, viņa -> jei, bija -> beja. "
    "Follow the example translations closely.")
