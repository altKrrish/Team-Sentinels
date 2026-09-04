"""
sentinel.form_guidance
=======================
Server-side counterpart to "Dynamic Form Guidance". A client-side minimum-word
check alone is not a control -- any direct API call, retry-with-shorter-text,
or offline-sync client bypasses it. This module re-validates on ingest and
returns a structured, actionable prompt (not a bare rejection) so the mobile/
web client can show the worker exactly what's missing.

It does NOT block ingestion outright: safety reports must never be silently
dropped because a stressed or injured worker typed six words. Instead:
  - under-threshold reports are still accepted and still scored,
  - they are tagged `needs_detail=True` and given a same-session, one-time
    "add more detail" prompt,
  - and — critically — sparse reports are what the low-word-count feature in
    `sentinel.features` and the metadata cross-reference in
    `sentinel.energy_metadata` exist to compensate for. Guidance improves
    input quality over time; it must never be the *only* mitigation for a
    class of report the model already handles by other means.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

MIN_WORDS_DEFAULT = 8

# Slot -> (regex hints, prompt shown to the worker). Keep prompts short;
# they render on a phone screen in the field.
_SLOT_PATTERNS: Dict[str, List[str]] = {
    "equipment_state": [
        r"\brunning\b", r"\bstopped\b", r"\bisolated\b", r"\bde-?energi[sz]ed\b",
        r"\benergi[sz]ed\b", r"\blocked out\b", r"\bstandby\b", r"\bidle\b",
    ],
    "voltage_or_pressure": [
        r"\d+\s*(v|kv|volt|psi|bar|kpa|mpa)\b",
    ],
    "location": [
        r"\b(rig|gs|gathering station|flowline|wellhead|platform|unit|bay|panel)\b",
    ],
    "working_height": [
        r"\d+\s*(m|meter|metre|ft|feet)\b", r"\bground level\b",
    ],
}

_SLOT_PROMPTS = {
    "equipment_state": "What state was the equipment in (running / isolated / locked out)?",
    "voltage_or_pressure": "What was the voltage or pressure involved (e.g. '11kV', '1200 psi')?",
    "location": "Which asset or location (rig, gathering station, flowline, panel)?",
    "working_height": "Was this at height? If so, roughly how high?",
}


@dataclass
class GuidanceResult:
    word_count: int
    meets_minimum: bool
    needs_detail: bool
    missing_slots: List[str] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "word_count": self.word_count,
            "meets_minimum": self.meets_minimum,
            "needs_detail": self.needs_detail,
            "missing_slots": self.missing_slots,
            "prompts": self.prompts,
        }


def evaluate(
    text: str,
    *,
    min_words: int = MIN_WORDS_DEFAULT,
    relevant_slots: Optional[List[str]] = None,
) -> GuidanceResult:
    """
    relevant_slots: restrict which slots are checked, e.g. an electrical-panel
    observation card only needs equipment_state + voltage_or_pressure, not
    working_height. If None, checks all slots (used for free-text reports
    without a known asset type).
    """
    words = re.findall(r"\S+", text or "")
    wc = len(words)
    meets_min = wc >= min_words

    slots = relevant_slots if relevant_slots is not None else list(_SLOT_PATTERNS)
    missing = []
    low = (text or "").lower()
    for slot in slots:
        patterns = _SLOT_PATTERNS.get(slot, [])
        if not any(re.search(p, low) for p in patterns):
            missing.append(slot)

    needs_detail = (not meets_min) or bool(missing)
    prompts = [_SLOT_PROMPTS[s] for s in missing if s in _SLOT_PROMPTS]
    if not meets_min:
        prompts.insert(0, f"Please add a bit more detail (at least {min_words} words).")

    return GuidanceResult(
        word_count=wc, meets_minimum=meets_min, needs_detail=needs_detail,
        missing_slots=missing, prompts=prompts,
    )


__all__ = ["evaluate", "GuidanceResult", "MIN_WORDS_DEFAULT"]
