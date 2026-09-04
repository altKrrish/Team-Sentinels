"""
sentinel.text_norm
==================
Script-aware normalisation for Assamese / Bengali / Hindi / Hinglish field text.

Why not "translate to English first"
------------------------------------
The original plan called for an MT pass (IndicBERT) before vectorisation.
Two problems:

1. IndicBERT is an *encoder*, not a translation model. It cannot translate.
   (The translation model you'd actually want is IndicTrans2.)
2. An MT hop in front of a safety classifier is a silent failure mode: MT
   errors are unlogged, unversioned, and land on the *high-recall* path. A
   mistranslated "কাৰেণ্ট লগা" (got an electric shock) into "current came" wipes
   out the single most important token in the report.

So this layer is deterministic and lossless-by-addition:

    original text  ->  (a) script-tagged segments
                       (b) rule-transliterated Latin form
                       (c) phonetic key stream for fuzzy lexicon matching

All three are retained. MT, if you want it, becomes an *optional extra channel*
(`translated_text`) that is concatenated, never substituted. Nothing is ever
thrown away before the classifier sees it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------
_RANGES = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali": (0x0980, 0x09FF),   # covers Assamese (Bengali-Assamese script)
    "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF),
    "Oriya": (0x0B00, 0x0B7F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Latin": (0x0041, 0x024F),
}


def detect_scripts(text: str) -> Set[str]:
    found: Set[str] = set()
    for ch in text:
        cp = ord(ch)
        for name, (lo, hi) in _RANGES.items():
            if lo <= cp <= hi:
                found.add(name)
                break
    return found


# ---------------------------------------------------------------------------
# Rule transliteration (Devanagari + Bengali/Assamese -> Latin)
# ---------------------------------------------------------------------------
_DEV = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh",
    "स": "s", "ह": "h", "ळ": "l", "क़": "q", "ख़": "kh", "ग़": "g",
    "ज़": "z", "ड़": "r", "ढ़": "rh", "फ़": "f",
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ं": "n", "ः": "h", "ँ": "n", "्": "", "ऽ": "",
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}

_BEN = {
    "অ": "o", "আ": "aa", "ই": "i", "ঈ": "ii", "উ": "u", "ঊ": "uu",
    "ঋ": "ri", "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "ch", "ছ": "chh", "জ": "j", "ঝ": "jh", "ঞ": "ny",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "ph", "ব": "b", "ভ": "bh", "ম": "m",
    "য": "j", "র": "r", "ৰ": "r", "ল": "l", "ৱ": "w",
    "শ": "sh", "ষ": "sh", "স": "s", "হ": "h",
    "ড়": "r", "ঢ়": "rh", "য়": "y", "ৎ": "t",
    "া": "aa", "ি": "i", "ী": "ii", "ু": "u", "ূ": "uu", "ৃ": "ri",
    "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou",
    "ং": "ng", "ঃ": "h", "ঁ": "n", "্": "", "়": "",
    "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
    "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
}

_TRANSLIT: Dict[str, str] = {}
_TRANSLIT.update(_DEV)
_TRANSLIT.update(_BEN)

# Dependent-vowel signs suppress the inherent 'a' of the preceding consonant.
_VOWEL_SIGNS = set("ािीुूृेैोौ" + "ািীুূৃেৈোৌ")
_VIRAMA = {"्", "্"}
_CONSONANTS = {c for c in list(_DEV) + list(_BEN)
               if c not in _VOWEL_SIGNS and c not in _VIRAMA
               and not c.isdigit() and _TRANSLIT.get(c, "") not in
               ("a", "aa", "i", "ii", "u", "uu", "e", "ai", "o", "au",
                "oi", "ou", "ri", "n", "h", "ng", "")}


def transliterate(text: str) -> str:
    """Rule transliteration of Devanagari / Bengali-Assamese into Latin.

    Handles the inherent vowel: a consonant gets a trailing 'a' unless it is
    followed by a virama or an explicit vowel sign.
    """
    out: List[str] = []
    chars = list(text)
    for i, ch in enumerate(chars):
        mapped = _TRANSLIT.get(ch)
        if mapped is None:
            out.append(ch)
            continue
        out.append(mapped)
        if ch in _CONSONANTS:
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt not in _VOWEL_SIGNS and nxt not in _VIRAMA:
                out.append("a")
    # collapse the inherent-vowel over-generation ("kaka" style tails)
    s = "".join(out)
    s = re.sub(r"a\b", "", s)          # drop word-final schwa (Hindi behaviour)
    s = re.sub(r"([aeiou])\1{2,}", r"\1\1", s)
    return s


# ---------------------------------------------------------------------------
# Latin-side normalisation
# ---------------------------------------------------------------------------
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "@": "a"})

_ABBREV = {
    "wrkr": "worker", "wrk": "work", "eqpt": "equipment", "eqp": "equipment",
    "pmp": "pump", "vlv": "valve", "prsr": "pressure", "press": "pressure",
    "temp": "temperature", "hgt": "height", "ht": "height",
    "elec": "electrical", "mech": "mechanical", "hyd": "hydraulic",
    "loto": "lockout tagout", "ptw": "permit to work", "cse": "confined space entry",
    "wah": "working at height", "sif": "serious injury fatality",
    "lti": "lost time injury", "mtc": "medical treatment case",
    "gs": "gathering station", "ocs": "oil collecting station",
    "ews": "early warning signal", "bop": "blowout preventer",
    "h2s": "h2s", "hc": "hydrocarbon", "kv": "kv", "hv": "high voltage",
    "lv": "low voltage", "ng": "natural gas", "cp": "cathodic protection",
    "sop": "standard operating procedure", "jsa": "job safety analysis",
    "nm": "near miss", "ua": "unsafe act", "uc": "unsafe condition",
}


def expand_abbreviations(text: str) -> str:
    def _sub(m: re.Match) -> str:
        w = m.group(0).lower()
        return _ABBREV.get(w, m.group(0))
    return re.sub(r"[A-Za-z0-9]+", _sub, text)


def normalize(text: str, *, translit: bool = True, expand: bool = True) -> str:
    """Canonical normalised form used for all lexicon matching."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    if translit and detect_scripts(s) - {"Latin"}:
        s = transliterate(s)
    s = s.lower().translate(_LEET)
    if expand:
        s = expand_abbreviations(s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)     # "leeeaak" -> "leeak"
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Phonetic keying for romanized Indic shorthand
# ---------------------------------------------------------------------------
# Double-metaphone was the original proposal. It is tuned for English/European
# name phonotactics and mangles Hinglish (it has no notion of aspirated stops
# or retroflex/dental collapse), so it produces keys that collide across
# unrelated safety terms. These rules target the actual variation seen in
# romanized Hindi/Assamese/Bengali field text instead.
_PH_RULES: List[Tuple[str, str]] = [
    (r"aa+", "a"), (r"ee+", "i"), (r"ii+", "i"), (r"oo+", "u"), (r"uu+", "u"),
    (r"ph", "f"), (r"bh", "b"), (r"dh", "d"), (r"gh", "g"), (r"jh", "j"),
    (r"kh", "k"), (r"th", "t"), (r"chh", "c"), (r"ch", "c"), (r"sh", "s"),
    (r"ck", "k"), (r"q", "k"), (r"x", "ks"), (r"z", "j"), (r"w", "v"),
    (r"y(?=[aeiou])", "i"), (r"ee\b", "i"), (r"ie\b", "i"),
]


def phonetic_key(token: str) -> str:
    """Deterministic phonetic key for a single romanized token."""
    t = re.sub(r"[^a-z]", "", token.lower())
    if not t:
        return ""
    for pat, rep in _PH_RULES:
        t = re.sub(pat, rep, t)
    head = t[0]
    tail = re.sub(r"[aeiou]", "", t[1:])
    t = head + tail
    t = re.sub(r"(.)\1+", r"\1", t)
    return t


def phonetic_phrase_key(text: str) -> str:
    return " ".join(k for k in (phonetic_key(w) for w in text.split()) if k)


def damerau_levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    """Bounded OSA edit distance. Returns max_dist+1 if it exceeds the bound."""
    if abs(len(a) - len(b)) > max_dist:
        return max_dist + 1
    prev2: List[int] = []
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        best = cur[0]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                v = min(v, prev2[j - 2] + 1)
            cur[j] = v
            best = min(best, v)
        if best > max_dist:
            return max_dist + 1
        prev2, prev = prev, cur
    return prev[-1] if prev[-1] <= max_dist else max_dist + 1


__all__ = [
    "detect_scripts", "transliterate", "normalize", "expand_abbreviations",
    "phonetic_key", "phonetic_phrase_key", "damerau_levenshtein",
]
