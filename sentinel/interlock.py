"""
sentinel.interlock
==================
Deterministic, auditable SIF override layer.

This sits IN FRONT OF the ensemble output, not behind it. Order matters:
the interlock is a hard gate that can force a SIF flag; it never suppresses
one the model already raised, and it always explains itself.

Key correctness properties (each has a test in tests/test_interlock.py):

1. Determinism: same input text -> same output, no randomness, no model calls.
2. Fuzzy matching is bounded and phrase-scoped: a lexicon phrase is matched
   against a sliding window of the ORIGINAL token count (not the whole
   document), so "the fall protection training video" does not fire
   `fall_from_height` just because "fall" appears near unrelated words.
3. Negation/hypothetical scoping: "no arc flash occurred", "arc flash drill",
   "arc flash training" must NOT fire the interlock. A small negation/
   irrealis window is checked before accepting a match.
4. Every fired rule returns a `MatchTrace` (span, canonical id, energy class,
   matched surface, edit distance) so the safety officer sees exactly why an
   interlock fired -- this is a Zero-Tolerance Safety Gate requirement, not
   an add-on.
5. INTERLOCK-tier phrases fire alone. CORROBORATE-tier phrases only fire
   the interlock in combination with either (a) another CORROBORATE/
   INTERLOCK hit from a different energy class, or (b) a metadata energy
   breach (see sentinel.energy_metadata).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from .lexicon import LEXICON, LexEntry, TIER_CORROBORATE, TIER_INTERLOCK
from .text_norm import damerau_levenshtein, normalize, phonetic_phrase_key

# Negation / irrealis markers that, within NEGATION_WINDOW tokens before a
# match, cancel it. Deliberately narrow -- broad negation scoping is a known
# source of false negatives in safety NLP (it swallows real hazards phrased
# as "worker did not have fall protection when they fell").
_NEGATION_MARKERS = {
    "no", "not", "without any", "avoided", "prevented", "near miss", "nearly",
    "almost", "could have", "toolbox talk", "poster", "as per procedure",
    "successfully", "narrowly avoided", "was averted", "did not occur",
}
# "near" alone is deliberately excluded: it is the ordinary spatial
# preposition in hazard reports ("leak near pump", "worker near panel"), and
# including it as a negation cue silently cancelled real matches -- caught
# by tests/test_interlock.py::test_two_corroborate_hits_different_energy_classes_fire.

# Forward-looking markers: words like "drill"/"training" typically follow the
# hazard phrase they neutralise ("fall from height drill", "arc flash
# training"), so they need a forward window, not just a backward one.
_FORWARD_NEGATION_MARKERS = {
    "drill", "training", "simulation", "exercise", "video", "refresher",
    "awareness session", "mock drill", "tabletop exercise",
}
FORWARD_NEGATION_WINDOW = 3
# Markers that specifically neutralise "near miss"-shaped sentences EXCEPT
# where the sentence also contains an outcome marker (fatality/hospitalised),
# in which case something did happen and negation must not apply.
_OUTCOME_OVERRIDE = {"fatality", "hospitalisation_major", "amputation", "amputated"}

NEGATION_WINDOW = 4
FUZZY_MAX_EDIT = 2
FUZZY_MIN_TOKEN_LEN = 5


@dataclass
class MatchTrace:
    canonical: str
    energy_class: str
    tier: str
    matched_surface: str
    matched_span_text: str
    start_token: int
    end_token: int
    method: str          # "exact" | "fuzzy" | "phonetic"
    edit_distance: int = 0
    negated: bool = False


@dataclass
class InterlockResult:
    fired: bool
    forced_label: Optional[str]           # e.g. "SIF" or None
    matches: List[MatchTrace] = field(default_factory=list)
    corroborate_only: List[MatchTrace] = field(default_factory=list)
    energy_classes_hit: Set[str] = field(default_factory=set)
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "fired": self.fired,
            "forced_label": self.forced_label,
            "reason": self.reason,
            "energy_classes_hit": sorted(self.energy_classes_hit),
            "matches": [m.__dict__ for m in self.matches],
            "corroborate_only": [m.__dict__ for m in self.corroborate_only],
        }


def _tokens(text: str) -> List[str]:
    return text.split()


def _phrase_len(surface: str) -> int:
    return len(surface.split())


def _window_has_negation(tokens: Sequence[str], start: int, end: int) -> bool:
    lo = max(0, start - NEGATION_WINDOW)
    back_window = " ".join(tokens[lo:start])
    if any(neg in back_window for neg in _NEGATION_MARKERS):
        return True
    hi = min(len(tokens), end + FORWARD_NEGATION_WINDOW)
    fwd_window = " ".join(tokens[end:hi])
    if any(neg in fwd_window for neg in _FORWARD_NEGATION_MARKERS):
        return True
    return False


def _find_exact(norm_text: str, tokens: List[str], surface: str) -> Optional[tuple]:
    """Return (start_token, end_token) of an exact phrase match, or None."""
    idx = norm_text.find(surface)
    if idx == -1:
        return None
    # translate char index -> token index
    prefix = norm_text[:idx]
    start = len(prefix.split())
    end = start + _phrase_len(surface)
    return start, end


@dataclass(frozen=True)
class _CompiledSurface:
    entry: Any
    surface: str
    surf_norm: str
    surf_tokens: List[str]
    surf_key: Optional[str]
    surf_key_tokens: List[str]


_COMPILED_SURFACES: List[_CompiledSurface] = []
for _e in LEXICON.entries:
    for _s in _e.surfaces:
        _sn = normalize(_s)
        _st = _sn.split()
        _sk = phonetic_phrase_key(_sn)
        _skt = _sk.split() if _sk else []
        _COMPILED_SURFACES.append(_CompiledSurface(
            entry=_e, surface=_s, surf_norm=_sn,
            surf_tokens=_st, surf_key=_sk, surf_key_tokens=_skt,
        ))


def _find_fuzzy_tokens(tokens: List[str], surf_tokens: List[str]) -> Optional[tuple]:
    n = len(surf_tokens)
    if n == 0 or len(tokens) < n:
        return None
    for i in range(0, len(tokens) - n + 1):
        window = tokens[i:i + n]
        total = 0
        ok = True
        for wt, st in zip(window, surf_tokens):
            if len(wt) < FUZZY_MIN_TOKEN_LEN and len(st) < FUZZY_MIN_TOKEN_LEN:
                if wt != st:
                    ok = False
                    break
                continue
            d = damerau_levenshtein(wt, st, max_dist=FUZZY_MAX_EDIT)
            if d > FUZZY_MAX_EDIT:
                ok = False
                break
            total += d
        if ok:
            return i, i + n, total
    return None


def _find_phonetic_keys(text_keys: List[str], surf_tokens: List[str]) -> Optional[tuple]:
    n = len(surf_tokens)
    if n == 0 or len(text_keys) < n:
        return None
    for i in range(0, len(text_keys) - n + 1):
        if text_keys[i:i + n] == surf_tokens:
            return i, i + n
    return None


def scan(text: str, *, allow_fuzzy: bool = True, allow_phonetic: bool = True) -> InterlockResult:
    norm = normalize(text)
    tokens = _tokens(norm)
    text_keys = [phonetic_phrase_key(t) for t in tokens] if allow_phonetic else []
    matches: List[MatchTrace] = []
    corroborate_hits: List[MatchTrace] = []
    energy_hit: Set[str] = set()
    seen_spans: Set[tuple] = set()

    for cs in _COMPILED_SURFACES:
        entry = cs.entry
        surf_norm = cs.surf_norm
        span = None
        method = "exact"
        edit_dist = 0

        # Fast exact search
        if surf_norm in norm:
            span = _find_exact(norm, tokens, surf_norm)

        if span is None and allow_fuzzy and entry.allow_fuzzy:
            r = _find_fuzzy_tokens(tokens, cs.surf_tokens)
            if r is not None:
                span = (r[0], r[1])
                edit_dist = r[2]
                method = "fuzzy"

        if span is None and allow_phonetic and entry.allow_fuzzy and cs.surf_key_tokens:
            r = _find_phonetic_keys(text_keys, cs.surf_key_tokens)
            if r is not None:
                span = r
                method = "phonetic"

        if span is None:
            continue

        start, end = span
        key = (entry.canonical, start, end)
        if key in seen_spans:
            continue
        seen_spans.add(key)

        negated = _window_has_negation(tokens, start, end)
        if negated and entry.canonical not in _OUTCOME_OVERRIDE:
            # still record for audit, but do not let it count toward firing
            trace = MatchTrace(
                entry.canonical, entry.energy_class, entry.tier, cs.surface,
                " ".join(tokens[start:end]), start, end, method, edit_dist,
                negated=True,
            )
            corroborate_hits.append(trace)
            continue

        trace = MatchTrace(
            entry.canonical, entry.energy_class, entry.tier, cs.surface,
            " ".join(tokens[start:end]), start, end, method, edit_dist,
        )
        if entry.tier == TIER_INTERLOCK:
            matches.append(trace)
            energy_hit.add(entry.energy_class)
        elif entry.tier == TIER_CORROBORATE:
            corroborate_hits.append(trace)
            energy_hit.add(entry.energy_class)

    fired = len(matches) > 0
    reason = ""
    if fired:
        canon = sorted({m.canonical for m in matches})
        reason = f"INTERLOCK: matched high-energy phrase(s) {canon}"
    else:
        # two-or-more distinct-energy-class corroborate hits also fire,
        # since e.g. "gas leak" + "hot work without permit" together
        # describe a credible SIF precursor even though neither alone does.
        corrob_classes = {m.energy_class for m in corroborate_hits if not m.negated}
        if len(corrob_classes) >= 2:
            fired = True
            reason = (
                "INTERLOCK: multiple corroborating high-energy indicators "
                f"across energy classes {sorted(corrob_classes)}"
            )

    return InterlockResult(
        fired=fired,
        forced_label="SIF" if fired else None,
        matches=matches,
        corroborate_only=corroborate_hits,
        energy_classes_hit=energy_hit,
        reason=reason,
    )


__all__ = ["scan", "InterlockResult", "MatchTrace", "NEGATION_WINDOW"]
