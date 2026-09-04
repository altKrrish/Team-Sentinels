"""
The 16 engineered domain features, in 6 explainable families.

Why these exist at all: TF-IDF sees "no" and "gas test" as two independent
features. A safety officer reads "no gas test" as a *barrier that was never
applied* - one fact, and the most important one in the sentence. These 16
features encode what the officer reads for and the bag of words cannot express:
whether a control was absent, bypassed or merely unverified; whether a negation
attaches to a named control; how much energy the measurements imply; and whether
the work happened in a handover window when supervision is thinnest.

Every feature returns a value in ``[0, 1]``. Counts are put through a saturating
transform rather than left raw, because "three negations" and "eight negations"
mean the same thing operationally, and an unbounded count would dominate the
L2-normalised TF-IDF block it is concatenated with.

Families are what the UI reports, because a family is explainable and a single
feature index is not. Mirrored (names and labels only) in `src/lib/model.js`.
"""

from __future__ import annotations

import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .normalize import normalize

# ---------------------------------------------------------------------------
# lexicons
# ---------------------------------------------------------------------------

#: Harm terms graded by seriousness. The rank matters more than the hit count -
#: one "fatality" outranks five "bruises" - so both are emitted as features.
SEVERITY_LEXICON: list[tuple[int, tuple[str, ...]]] = [
    (1, ("discomfort", "bruise", "bruised", "graze", "grazed", "scratch", "minor cut", "first aid")),
    (2, ("laceration", "sprain", "strain", "abrasion", "stitches", "cut his", "cut her", "burn to the hand")),
    (3, ("fracture", "fractured", "broken arm", "broken leg", "broken ribs", "dislocated", "concussion",
         "second degree burn", "hospitalised", "hospitalized", "lost time")),
    (4, ("amputation", "amputated", "crush injury", "crushed hand", "crushed leg", "third degree burn",
         "unconscious", "asphyxiation", "electrocution", "permanent disability", "life altering",
         "life-altering", "critical condition", "intensive care")),
    (5, ("fatality", "fatal", "died", "death", "deceased", "killed", "loss of life")),
]

_INJURY_OCCURRED = (
    "was injured", "were injured", "sustained", "suffered", "received treatment",
    "taken to hospital", "shifted to hospital", "referred to the hospital",
    "reported to the dispensary", "first aid was given", "was hurt", "were hurt",
)

#: Phrases that explicitly say nobody was harmed.
#:
#: These veto `_INJURY_OCCURRED`, which is a substring match and would otherwise
#: read "nobody was hurt" as an injury. Near-miss reports say this in almost
#: every closing line, so getting it wrong mislabels most of the corpus.
_NO_INJURY = (
    "no injury", "no injuries", "nobody was hurt", "nobody was injured",
    "no one was hurt", "no one was injured", "no person was injured",
    "was not injured", "were not injured", "not hurt", "no harm",
    "without injury", "escaped unhurt", "unhurt", "no personnel were affected",
    "before anyone was exposed", "nobody in the line of fire",
)

#: A control that was never there in the first place.
_BARRIER_ABSENT = (
    "no permit", "without a permit", "without permit", "no PTW", "without PTW",
    "no gas test", "without a gas test", "without gas test", "no gas testing",
    "no lock", "no lockout", "no LOTO", "without LOTO", "no tag", "no tag was fitted",
    "not fitted", "no isolation", "without isolation", "no barricade", "not barricaded",
    "no barricading", "no guard", "guard was missing", "guard missing", "no handrail",
    "no railing", "missing railing", "no lifeline", "no anchor point", "no fall arrest",
    "without fall arrest", "no harness", "not wearing a harness", "no attendant",
    "no standby", "no fire watch", "without a fire watch", "no continuous monitoring",
    "no gas monitor", "no rescue plan", "no spotter", "no signaller", "no banksman",
    "not available", "was absent", "were absent", "none was provided", "not provided",
    "no supervision", "unsupervised", "no earthing", "not earthed", "no blinding",
    "not blinded", "no JSA", "without JSA", "no TBT", "without TBT",
)

#: A control that existed and was deliberately defeated.
_BARRIER_BYPASS = (
    "bypassed", "bypassing", "by-passed", "jumpered", "jumped out", "jumper was placed",
    "overridden", "override", "overrode", "defeated", "inhibited", "inhibit was placed",
    "forced open", "tied back", "tied down", "wedged open", "propped open", "blocked open",
    "unclipped", "unhooked", "disconnected the", "disabled", "switched off the alarm",
    "muted the alarm", "silenced the alarm", "isolated the trip", "trip was inhibited",
    "interlock was", "removed the guard", "guard was removed", "guard had been removed",
    "cut the lock", "broke the lock", "seal was broken", "tag was removed",
    "lock was removed", "climbed over", "crossed the barricade", "entered the barricaded",
    "worked around", "short circuited", "shorted out",
)

#: A control assumed to be working that nobody proved was working.
_BARRIER_VERIFY_GAP = (
    "not verified", "was not verified", "were not verified", "unverified",
    "without verifying", "did not verify", "no try-out", "without try out",
    "without try-out", "not cross checked", "not cross-checked", "not witnessed",
    "no witness", "assumed", "presumed", "believed to be", "taken as",
    "not confirmed", "did not confirm", "unconfirmed", "not tested",
    "not calibrated", "calibration was not", "out of calibration", "not inspected",
    "no inspection record", "not signed", "without signature", "closing signature",
    "not logged", "no record", "no entry in the register", "register was not",
    "not checked", "did not check", "without checking", "not rechecked",
    # "the gas test had not been done" - a verification that was skipped is
    # phrased far more often as an undone action than as an unverified state.
    "had not been done", "was not done", "were not done", "not been done",
    "not carried out", "was not carried out", "not performed", "not done before",
    "no gas test was done", "test was not done", "check was not done",
    "not taken before", "was yet to be done", "still pending", "yet to be carried out",
)

_PERMIT_VIOLATION = (
    "permit had expired", "permit expired", "expired permit", "permit was expired",
    "permit not valid", "invalid permit", "permit for a different", "wrong permit",
    "permit mismatch", "permit did not cover", "beyond the permit", "outside the permit",
    "permit closed", "permit was not closed", "no permit", "without a permit",
    "unauthorised", "unauthorized", "not authorised", "not authorized",
    "permit conditions were not", "violating the permit", "lapsed permit",
    "PTW had expired", "PTW was not", "extension was not taken",
)

#: "no hot work permit was raised", "without a valid confined space clearance".
#:
#: A phrase list cannot catch these because the qualifier sits between the
#: negation and the noun, and that is how people actually write it. Matched
#: against the NORMALISED text, where acronyms have been expanded, so "no hot
#: work PTW" reaches this as "no hot work ptw permit to work".
_PERMIT_GAP = re.compile(
    r"(?:no|without(?:\s+a|\s+any|\s+valid)?|not\s+have\s+a|did\s+not\s+have)"
    r"(?:\s+\w+){0,4}?"
    r"\s+(?:permit|ptw|clearance|authorisation|authorization|work\s+order|jsa|tbt)",
    re.I,
)

_PPE_VIOLATION = (
    "without PPE", "no PPE", "not wearing", "without wearing", "removed his helmet",
    "removed her helmet", "no helmet", "without a helmet", "no safety shoes",
    "no gloves", "without gloves", "no goggles", "no face shield", "no ear protection",
    "cloth mask", "cotton mask", "no respirator", "without a respirator",
    "no breathing apparatus", "without SCBA", "no SCBA", "wrong gloves",
    "damaged helmet", "harness was not worn", "not worn", "chin strap was not",
)

_PROCEDURE_DEVIATION = (
    "not as per procedure", "against the procedure", "contrary to the procedure",
    "deviation from", "deviated from", "short cut", "shortcut", "short-cut",
    "not following", "did not follow", "without following", "own method",
    "improvised", "makeshift", "jugaad", "temporary arrangement", "not as per SOP",
    "SOP was not", "no MOC", "without MOC", "no risk assessment",
    "without a risk assessment", "not as per the drawing", "hand tight",
    "by hand instead", "used a pipe wrench as", "wrong tool", "unsuitable tool",
    "standing on a drum", "standing on the railing", "climbed the pipe rack",
    "rode on the load", "rode on the forks",
)

_NEGATIONS = (
    "no", "not", "never", "without", "none", "neither", "nor", "failed to",
    "omitted", "absent", "missing", "lacking", "unavailable", "nil", "didn't",
    "did not", "wasn't", "was not", "weren't", "were not", "hadn't", "had not",
    "couldn't", "could not", "isn't", "is not",
)

#: Named controls. A negation inside the scope window of one of these is a
#: qualitatively different fact from a negation floating free in the sentence.
_CONTROLS = (
    "permit", "ptw", "loto", "lockout", "lock", "tag", "isolation", "isolated",
    "gas test", "gas testing", "gas monitor", "monitoring", "barricade",
    "barricading", "guard", "handrail", "railing", "lifeline", "anchor",
    "harness", "fall arrest", "attendant", "standby", "fire watch", "spotter",
    "signaller", "banksman", "earthing", "earth", "blinding", "blind", "jsa",
    "tbt", "toolbox talk", "risk assessment", "interlock", "trip", "alarm",
    "relief valve", "psv", "rupture disc", "bop", "esd", "supervision",
    "supervisor", "certificate", "inspection", "calibration", "signature",
    "ppe", "helmet", "harness", "scba", "respirator", "moc", "sop", "procedure",
)

_HANDOVER = (
    "night shift", "general shift", "second shift", "third shift", "graveyard",
    "shift change", "shift handover", "handover", "hand over", "crew change",
    "change of crew", "meal break", "lunch break", "tea break", "after the break",
    "resumed after", "end of shift", "start of shift", "just before shift",
    "relieving", "reliever", "stand-in", "temporarily deputed", "new to the job",
    "first day", "sunday", "holiday", "skeleton staff", "single operator",
)

# ---------------------------------------------------------------------------
# quantity patterns
# ---------------------------------------------------------------------------

_PRESSURE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kg\s*/?\s*cm2|kg\s*/?\s*cm\^?2|kgf?\s*/?\s*cm2|bar(?:g)?\b|psi(?:g)?\b|mpa\b|kpa\b)",
    re.I,
)
_HEIGHT = re.compile(
    r"(\d+(?:\.\d+)?)\s*(m\b|mtr\b|mtrs\b|meter[s]?\b|metre[s]?\b|ft\b|feet\b|foot\b)",
    re.I,
)
_GAS = re.compile(
    r"(\d+(?:\.\d+)?)\s*(%\s*lel|percent\s*lel|%\s*of\s*lel|ppm|%\s*o2|%\s*oxygen|percent\s*oxygen|%\s*ch4)",
    re.I,
)
_OVERDUE = re.compile(
    r"(?:overdue|past due|due since|not\s+(?:done|tested|inspected|calibrated)\s+(?:for|since))"
    r"[^.]{0,40}?(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)",
    re.I,
)
_OVERDUE_BARE = re.compile(
    r"\b(overdue|past due|expired|lapsed|beyond the due date|due for renewal|"
    r"validity had ended|out of date|time barred)\b",
    re.I,
)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-/]*")

# ---------------------------------------------------------------------------
# feature schema
# ---------------------------------------------------------------------------

#: family id -> ordered feature names. The concatenation of these lists, in this
#: order, is the column order of the matrix `EngineeredFeatures` emits.
FEATURE_FAMILIES: dict[str, list[str]] = {
    "severity": ["sev_lexicon_hits", "max_sev_term_rank", "injury_outcome_flag"],
    "barrier": ["barrier_absent_cue", "barrier_bypass_cue", "barrier_verify_gap"],
    "violation": ["permit_violation_cue", "ppe_violation_cue", "procedure_deviation_cue"],
    "negation": ["negation_count", "negated_control_scope"],
    "measurement": ["pressure_qty", "height_qty", "gas_conc_qty"],
    "temporal": ["overdue_interval", "shift_handover_cue"],
}

FEATURE_NAMES: list[str] = [n for names in FEATURE_FAMILIES.values() for n in names]
FEATURE_COUNT: int = len(FEATURE_NAMES)
FEATURE_INDEX: dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}

#: Which columns belong to which family, for grouped attribution.
FAMILY_SLICES: dict[str, list[int]] = {
    family: [FEATURE_INDEX[n] for n in names] for family, names in FEATURE_FAMILIES.items()
}

FAMILY_LABEL: dict[str, str] = {
    "severity": "Severity indicators",
    "barrier": "Barrier failures",
    "violation": "Rule violations",
    "negation": "Negation handling",
    "measurement": "Measurements",
    "temporal": "Temporal patterns",
}

assert FEATURE_COUNT == 16, f"the pitch says 16 engineered features, found {FEATURE_COUNT}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _saturate(count: int, k: float) -> float:
    """Diminishing-returns count -> [0, 1]. Two hits already mean "yes"."""
    if count <= 0:
        return 0.0
    return float(min(1.0, count / k))


def _count_phrases(haystack: str, needles: tuple[str, ...]) -> int:
    return sum(1 for n in needles if n in haystack)


def _max_quantity(text: str, pattern: re.Pattern[str], ceiling: float) -> float:
    """Largest matched magnitude, scaled by a physically meaningful ceiling."""
    best = 0.0
    for m in pattern.finditer(text):
        try:
            value = float(m.group(1))
        except (TypeError, ValueError):
            continue
        unit = (m.group(2) or "").lower().replace(" ", "")
        # fold everything to one unit per family so magnitudes are comparable
        if unit.startswith("psi"):
            value /= 14.5
        elif unit.startswith("mpa"):
            value *= 10.0
        elif unit.startswith("kpa"):
            value /= 100.0
        elif unit in ("ft", "feet", "foot"):
            value *= 0.3048
        best = max(best, value)
    if best <= 0:
        return 0.0
    return float(min(1.0, best / ceiling))


def _negated_control_scope(tokens: list[str], window: int = 4) -> float:
    """Does a negation marker govern a *named control* within `window` tokens?

    This is the feature that separates "no gas test was carried out" from "no
    damage was observed". Both contain a negation; only the first describes a
    missing barrier.
    """
    simple_negations = {"no", "not", "never", "without", "none", "nil", "missing", "absent"}
    control_heads = {c.split()[0] for c in _CONTROLS}

    hits = 0
    for i, tok in enumerate(tokens):
        if tok not in simple_negations:
            continue
        for j in range(i + 1, min(i + 1 + window, len(tokens))):
            if tokens[j] in control_heads:
                hits += 1
                break
    return _saturate(hits, 2.0)


def _overdue_interval(text: str) -> float:
    """How far past due, in days, saturating at a quarter."""
    days = 0.0
    for m in _OVERDUE.finditer(text):
        try:
            value = float(m.group(1))
        except (TypeError, ValueError):
            continue
        unit = m.group(2).lower()
        if unit.startswith("week"):
            value *= 7
        elif unit.startswith("month"):
            value *= 30
        elif unit.startswith("year"):
            value *= 365
        days = max(days, value)
    if days > 0:
        return float(min(1.0, days / 90.0))
    # A bare "overdue" with no interval still carries the fact, at lower weight.
    return 0.4 if _OVERDUE_BARE.search(text) else 0.0


def extract(text: str) -> dict[str, float]:
    """Compute all 16 features for one narrative. Keyed by feature name."""
    norm = normalize(text)
    tokens = _TOKEN.findall(norm)

    sev_hits = 0
    max_rank = 0
    for rank, terms in SEVERITY_LEXICON:
        found = _count_phrases(norm, terms)
        if found:
            sev_hits += found
            max_rank = max(max_rank, rank)

    return {
        # severity indicators
        "sev_lexicon_hits": _saturate(sev_hits, 3.0),
        "max_sev_term_rank": max_rank / 5.0,
        "injury_outcome_flag": (
            0.0
            if _count_phrases(norm, _NO_INJURY)
            else 1.0
            if _count_phrases(norm, _INJURY_OCCURRED)
            else 0.0
        ),
        # barrier failures
        "barrier_absent_cue": _saturate(_count_phrases(norm, _BARRIER_ABSENT), 2.0),
        "barrier_bypass_cue": _saturate(_count_phrases(norm, _BARRIER_BYPASS), 2.0),
        "barrier_verify_gap": _saturate(_count_phrases(norm, _BARRIER_VERIFY_GAP), 2.0),
        # rule violations
        "permit_violation_cue": _saturate(
            _count_phrases(norm, _PERMIT_VIOLATION) + len(_PERMIT_GAP.findall(norm)), 2.0
        ),
        "ppe_violation_cue": _saturate(_count_phrases(norm, _PPE_VIOLATION), 2.0),
        "procedure_deviation_cue": _saturate(_count_phrases(norm, _PROCEDURE_DEVIATION), 2.0),
        # negation handling
        "negation_count": _saturate(_count_phrases(norm, _NEGATIONS), 4.0),
        "negated_control_scope": _negated_control_scope(tokens),
        # measurements  (ceilings: 100 kg/cm2, 30 m, 100 %LEL-equivalent)
        "pressure_qty": _max_quantity(norm, _PRESSURE, 100.0),
        "height_qty": _max_quantity(norm, _HEIGHT, 30.0),
        "gas_conc_qty": _max_quantity(norm, _GAS, 100.0),
        # temporal patterns
        "overdue_interval": _overdue_interval(norm),
        "shift_handover_cue": _saturate(_count_phrases(norm, _HANDOVER), 2.0),
    }


def family_activation(values: dict[str, float]) -> dict[str, float]:
    """Mean activation per family, which is what the UI shows as `value`."""
    return {
        family: float(np.mean([values[n] for n in names]))
        for family, names in FEATURE_FAMILIES.items()
    }


class EngineeredFeatures(BaseEstimator, TransformerMixin):
    """sklearn transformer wrapping :func:`extract`.

    Stateless - ``fit`` exists only to satisfy the estimator API, so this can sit
    inside a ``FeatureUnion`` beside the two TF-IDF vectorisers without any
    train/serve skew.
    """

    def fit(self, X, y=None):  # noqa: N803 - sklearn's argument name
        return self

    def transform(self, X):  # noqa: N803
        rows = np.zeros((len(X), FEATURE_COUNT), dtype=np.float64)
        for i, text in enumerate(X):
            values = extract(text)
            for name, idx in FEATURE_INDEX.items():
                rows[i, idx] = values[name]
        return rows

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)
