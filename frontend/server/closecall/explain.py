"""
Explainability.

Because every head is a linear model, an explanation is not an approximation:
the push a feature applies to the SIF logit is exactly ``coefficient x value``.
There is no surrogate model, no sampling, and no gap between what the model used
and what the reviewer is shown.

Two things are produced:

`feature_contributions`
    The six engineered families, each with its activation and its signed push on
    the logit. This is the "why" an HSE officer can argue with - "it flagged this
    because the barrier family fired and there is a negation attached to a named
    control" is a reviewable claim.

`evidence_spans`
    The word n-grams that contributed most, mapped back to verbatim substrings of
    the ORIGINAL text so the UI can highlight them in place. Spans are only
    emitted when they genuinely occur in the text as written - a term that exists
    only after acronym expansion is looked up through its acronym instead, and if
    neither can be located it is dropped rather than invented.
"""

from __future__ import annotations

import re

import numpy as np

from .features import FAMILY_SLICES, extract, family_activation
from .normalize import GLOSSARY

#: Terms too generic to be worth highlighting even when the coefficient is high.
#:
#: Prepositions and light adverbs are here for the *bigram* case below: "before
#: the" carries a real coefficient but explains nothing, and it only gets dropped
#: if both of its tokens are listed. A content word beside one of these still
#: survives, so "still lined up" and "no permit" are unaffected.
_STOPISH = {
    "the", "a", "an", "and", "or", "but", "was", "were", "is", "are", "be", "been",
    "at", "in", "on", "of", "to", "for", "with", "by", "from", "as", "that", "this",
    "it", "its", "his", "her", "their", "he", "she", "they", "had", "has", "have",
    "during", "while", "which", "there", "then", "than", "also", "been", "being",
    "observed", "found", "noticed", "seen", "reported", "raised", "time", "place",
    "before", "after", "about", "around", "still", "instead", "again", "when",
    "into", "onto", "over", "under", "up", "down", "out", "off", "same", "other",
}

#: expansion (lowercased) -> acronym, so "lockout tagout" can be found in a report
#: that only ever wrote "LOTO".
_EXPANSION_TO_ACRONYM = {v.lower(): k for k, v in GLOSSARY.items()}

#: Spans weaker than this fraction of the strongest one are dropped rather than
#: highlighted. A drawer with six faint highlights explains nothing.
MIN_SPAN_WEIGHT = 0.20

#: A highlight must not run across one of these.
_SENTENCE_BREAK = re.compile(r"[.;:!?\n]")


def _flexible_pattern(term: str) -> re.Pattern[str]:
    """Match a term allowing any whitespace/hyphen run between its words."""
    parts = [re.escape(p) for p in term.split()]
    body = r"[\s\-]+".join(parts)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.I)


def _locate(term: str, text: str) -> tuple[int, int] | None:
    """Find `term` in the original text, falling back to its acronym.

    :returns: ``(start, end)`` character offsets, or ``None`` if the term cannot
        be located honestly.
    """
    m = _flexible_pattern(term).search(text)
    if m:
        return m.start(), m.end()

    # The term may only exist because normalisation expanded an acronym.
    acronym = _EXPANSION_TO_ACRONYM.get(term.lower())
    if acronym:
        m = _flexible_pattern(acronym).search(text)
        if m:
            return m.start(), m.end()

    # A bigram whose second half is an expansion, e.g. "no lockout".
    words = term.split()
    if len(words) == 2:
        for lo, hi in ((0, 1), (1, 0)):
            acronym = _EXPANSION_TO_ACRONYM.get(words[hi].lower())
            if not acronym:
                continue
            m = _flexible_pattern(f"{words[lo]} {acronym}" if lo < hi else f"{acronym} {words[lo]}").search(text)
            if m:
                return m.start(), m.end()
    return None


def _merge(intervals: list[tuple[int, int, float]], text: str, gap: int = 3) -> list[tuple[int, int, float]]:
    """Merge overlapping or near-touching spans, summing their contributions.

    Without this, "no" and "no gas" and "gas test" become three ragged highlights
    over the same six words instead of one readable phrase.

    Merging stops at a full stop. Two adjacent terms either side of one belong to
    different facts, and bridging them produces highlights like "wearing a
    harness. Nobody was hurt." - which reads as though the model called the
    no-harm sentence evidence of fatal potential. It did not; the merger did.
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda iv: iv[0])
    merged = [list(intervals[0])]
    for start, end, weight in intervals[1:]:
        last = merged[-1]
        touching = start <= last[1] + gap
        crosses_sentence = bool(_SENTENCE_BREAK.search(text[last[1] : start]))
        if touching and not crosses_sentence:
            last[1] = max(last[1], end)
            last[2] += weight
        else:
            merged.append([start, end, weight])
    return [(int(s), int(e), float(w)) for s, e, w in merged]


def evidence_spans(engine, text: str, row=None, top_terms: int = 40, top_spans: int = 6) -> list[dict]:
    """Verbatim substrings of `text` that drove the SIF score.

    :param engine: a fitted :class:`~closecall.model.SifEngine`
    :param row: the already-transformed sparse row, if the caller has one
    :returns: ``[{"span": str, "weight": float}, ...]`` sorted strongest first,
        weights normalised to ``(0, 1]``.
    """
    if row is None:
        row = engine.transform([text])

    w_start, w_stop = engine.block_offsets["word"]
    coef = engine.sif_head.coef_.ravel()

    csr = row.tocsr()
    cols = csr.indices
    vals = csr.data

    scored: list[tuple[float, str]] = []
    for col, value in zip(cols, vals):
        if not (w_start <= col < w_stop):
            continue
        push = float(value) * float(coef[col])
        if push <= 0:
            continue
        term = engine.word_vocab_inv.get(int(col))
        if not term:
            continue
        if term in _STOPISH:
            continue
        # A bigram of two stopwords ("at the") passes the check above but says
        # nothing a reviewer can act on.
        if all(word in _STOPISH for word in term.split()):
            continue
        if len(term.split()) == 1 and len(term) < 3:
            continue
        scored.append((push, term))

    scored.sort(key=lambda t: -t[0])

    intervals: list[tuple[int, int, float]] = []
    for push, term in scored[:top_terms]:
        found = _locate(term, text)
        if found:
            intervals.append((found[0], found[1], push))

    merged = _merge(intervals, text)
    merged.sort(key=lambda iv: -iv[2])
    merged = merged[:top_spans]
    if not merged:
        return []

    peak = max(iv[2] for iv in merged) or 1.0
    out = []
    for start, end, weight in merged:
        # Spans worth a fifth of the strongest one are noise on screen. Showing
        # them dilutes the highlight that actually explains the verdict.
        relative = weight / peak
        if relative < MIN_SPAN_WEIGHT:
            continue
        span = text[start:end].strip()
        if span:
            out.append({"span": span, "weight": round(min(1.0, relative), 3)})
    return out


def feature_contributions(engine, text: str, row=None) -> list[dict]:
    """Per-family activation and signed push on the SIF logit.

    :returns: ``[{"group": str, "value": float, "contribution": float}, ...]`` in
        the declared family order, with ``contribution`` in ``[-1, 1]``.
    """
    if row is None:
        row = engine.transform([text])

    start, stop = engine.block_offsets["eng"]
    coef = engine.sif_head.coef_.ravel()[start:stop]
    scaled = row[:, start:stop].toarray().ravel()

    # `value` is the RAW activation, not the MaxAbsScaler output - the scaler's
    # units are an implementation detail and would not mean anything on screen.
    raw = family_activation(extract(text))

    out = []
    for family, cols in FAMILY_SLICES.items():
        idx = np.asarray(cols)
        push = float(scaled[idx] @ coef[idx]) / engine.contribution_scale
        out.append(
            {
                "group": family,
                "value": round(float(raw[family]), 3),
                "contribution": round(float(np.clip(push, -1.0, 1.0)), 3),
            }
        )
    return out
