"""
Preprocessing: the standardisation step that runs before anything is vectorised.

Two jobs:

1. Expand industry shorthand to one canonical form, so `LOTO`, `L.O.T.O.` and
   "lockout tagout" all land on the same tokens. Without this, the vectoriser
   treats an abbreviation and its expansion as unrelated features and every
   report written in shorthand is scored on a sparser vector than it deserves.

2. Strip report boilerplate (form headers, "Sl. No.", trailing sign-off lines)
   that carries no safety signal but does carry site-specific noise the model
   would happily overfit to.

`expansions_in()` is the visible half: the UI shows which terms were expanded so
a reviewer can see the normalisation happened rather than take it on faith.

Mirrored (ids and labels only) in `src/lib/model.js`. This file is the
authoritative copy - it is what actually touches the text.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# glossary
# ---------------------------------------------------------------------------

#: Shorthand -> canonical expansion. Order is irrelevant here; matching is done
#: longest-first below so that a short key cannot consume part of a longer one.
GLOSSARY: dict[str, str] = {
    "LOTO": "Lockout Tagout",
    "PTW": "Permit to Work",
    "BOP": "Blowout Preventer",
    "JSA": "Job Safety Analysis",
    "LEL": "Lower Explosive Limit",
    "H2S": "Hydrogen Sulphide",
    "PPE": "Personal Protective Equipment",
    "MCC": "Motor Control Centre",
    "GCS": "Gas Collecting Station",
    "OCS": "Oil Collecting Station",
    "GGS": "Group Gathering Station",
    "ROW": "Right of Way",
    "CSE": "Confined Space Entry",
    "WAH": "Work at Height",
    "SIMOPS": "Simultaneous Operations",
    "TBT": "Toolbox Talk",
    "MOC": "Management of Change",
    "DCS": "Distributed Control System",
}

#: Spelling variants that should collapse onto a glossary key before expansion.
#: Real reports are written by hand at the end of a shift; these are the forms
#: that actually turn up.
_VARIANTS: dict[str, str] = {
    "L.O.T.O.": "LOTO",
    "L O T O": "LOTO",
    "lock out tag out": "LOTO",
    "lock-out tag-out": "LOTO",
    "lockout-tagout": "LOTO",
    "lock out": "LOTO",
    "lock-out": "LOTO",
    "P.T.W.": "PTW",
    "work permit": "PTW",
    "hot work permit": "hot work PTW",
    "H2 S": "H2S",
    "H-2-S": "H2S",
    "hydrogen sulfide": "H2S",
    "sour gas": "H2S",
    "B.O.P.": "BOP",
    "J.S.A.": "JSA",
    "P.P.E.": "PPE",
    "M.C.C.": "MCC",
    "right of way": "ROW",
    "confined space entry": "CSE",
    "work at height": "WAH",
    "working at height": "WAH",
    "tool box talk": "TBT",
    "toolbox talk": "TBT",
}

_BOILERPLATE = [
    re.compile(r"^\s*(sl\.?\s*no\.?|s\.?\s*no\.?|ref\.?\s*no\.?)\s*[:.\-]?\s*\S+", re.I | re.M),
    re.compile(r"^\s*(observation|remarks?|description|details)\s*[:\-]\s*", re.I | re.M),
    re.compile(r"^\s*(reported by|raised by|observed by|closed by)\s*[:\-].*$", re.I | re.M),
    re.compile(r"\b(signature|sign\.?|countersigned)\s*[:\-].*$", re.I | re.M),
    re.compile(r"\[?\s*(end of report|report ends)\s*\]?\.?", re.I),
]

_WS = re.compile(r"[ \t ]+")
_NEWLINES = re.compile(r"\s*\n\s*")


def _word_boundary_pattern(term: str) -> re.Pattern[str]:
    """Boundary-aware pattern for a term that may contain dots and spaces.

    `\\b` is unreliable around `.` and digits (``H2S``, ``L.O.T.O.``), so the
    boundary is expressed explicitly as "not a letter or digit".
    """
    body = re.escape(term).replace(r"\ ", r"[\s\-]+")
    return re.compile(rf"(?<![A-Za-z0-9])({body})(?![A-Za-z0-9])", re.I)


# Longest key first: "hot work permit" must be tried before "work permit", and
# "confined space entry" before any substring of it.
_VARIANT_PATTERNS = [
    (_word_boundary_pattern(k), v)
    for k, v in sorted(_VARIANTS.items(), key=lambda kv: -len(kv[0]))
]
_GLOSSARY_PATTERNS = [
    (_word_boundary_pattern(k), k, v)
    for k, v in sorted(GLOSSARY.items(), key=lambda kv: -len(kv[0]))
]


def strip_boilerplate(text: str) -> str:
    """Remove form scaffolding that carries no safety signal."""
    out = text
    for pat in _BOILERPLATE:
        out = pat.sub(" ", out)
    return out


def normalize(text: str) -> str:
    """Full preprocessing: unicode-fold, de-boilerplate, expand shorthand, fold case.

    The expansion keeps BOTH forms - ``"LOTO"`` becomes ``"LOTO Lockout Tagout"``.
    Dropping the abbreviation would throw away a real signal (shorthand
    correlates with hurried reporting), and keeping both lets the char n-gram
    vectoriser see the acronym while the word vectoriser sees the phrase.
    """
    if not text:
        return ""

    out = unicodedata.normalize("NFKC", text)
    out = out.replace("’", "'").replace("‘", "'")
    out = out.replace("“", '"').replace("”", '"')
    out = out.replace("–", "-").replace("—", "-")
    out = strip_boilerplate(out)

    for pat, canon in _VARIANT_PATTERNS:
        out = pat.sub(canon, out)

    for pat, key, expansion in _GLOSSARY_PATTERNS:
        out = pat.sub(f"{key} {expansion}", out)

    out = _NEWLINES.sub(" ", out)
    out = _WS.sub(" ", out)
    return out.strip().lower()


def expansions_in(text: str) -> list[dict[str, str]]:
    """Which glossary terms a narrative actually contains, for the UI to show.

    Runs variant-folding first, so a report that wrote "lock out tag out" is
    correctly reported as having had LOTO standardised.

    :returns: ``[{"from": "LOTO", "to": "Lockout Tagout"}, ...]``
    """
    if not text:
        return []

    folded = unicodedata.normalize("NFKC", text)
    for pat, canon in _VARIANT_PATTERNS:
        folded = pat.sub(canon, folded)

    seen: list[dict[str, str]] = []
    for pat, key, expansion in _GLOSSARY_PATTERNS:
        if pat.search(folded):
            seen.append({"from": key, "to": expansion})
    # Present in the declaration order of GLOSSARY, which reads more naturally
    # than longest-first.
    order = list(GLOSSARY)
    seen.sort(key=lambda e: order.index(e["from"]))
    return seen
