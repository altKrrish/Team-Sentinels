"""
sentinel.pattern_extractor
============================
Unsupervised Precursor Pattern & Entity Extractor.

Addresses SIH Problem Statement 26165, requirement (c):
"Surfaces recurring precursor patterns (activity, location, barrier failure)
via a dashboard."

Extracts three entity slots from observation narrative text:
  - Activity  (e.g., Tripping pipe, Welding, Crane lifting)
  - Location  (e.g., Drill floor, Substation, Mud tank)
  - Barrier Failure (e.g., No harness, Bypassed interlock, Worn sling)

Uses deterministic regex slot extraction (not spaCy NER) to maintain:
  1. Zero additional heavyweight dependencies (keeps <2ms inference budget)
  2. Full DGMS auditability (every extraction traceable to a pattern)
  3. Domain-specific oilfield vocabulary coverage

Provides:
  - extract_slots(text) -> PatternResult  (single-report extraction)
  - rank_patterns(reports) -> list[PatternTriad]  (batch pattern surfacing)
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .text_norm import normalize

# ---------------------------------------------------------------------------
# Slot Taxonomy: Activity, Location, Barrier Failure
# ---------------------------------------------------------------------------
# Each slot is a list of (canonical_label, compiled_regex) pairs.
# Patterns are applied to NORMALIZED text (lowercase Latin after transliteration).

_ACTIVITY_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("Drilling", re.compile(r"\b(drilling|drill(?:ing)?\s+operation|rotary\s+drilling)\b")),
    ("Tripping Pipe", re.compile(r"\b(tripping\s+(?:pipe|in|out)|pipe\s+tripping|pulling\s+out|running\s+in)\b")),
    ("Wireline Logging", re.compile(r"\b(wireline|logging\s+tool|well\s+logging|slickline)\b")),
    ("Welding", re.compile(r"\b(welding|weld(?:ed)?|grinding|cutting\s+torch|hot\s+work|gas\s+cutting)\b")),
    ("Crane Lifting", re.compile(r"\b(crane\s+lift|crane\s+operation|lifting|rigging|hoisting|mechanical\s+lift)\b")),
    ("Rig Skidding", re.compile(r"\b(rig\s+skid|skidding|rig\s+move|rig\s+up|rig\s+down|rig\s+mobilization)\b")),
    ("Confined Space Entry", re.compile(r"\b(confined\s+space|tank\s+entry|vessel\s+entry|entry\s+into\s+tank)\b")),
    ("Electrical Work", re.compile(r"\b(electrical\s+work|panel\s+work|switchgear|cable\s+laying|energi[sz]ed|de.?energi[sz])\b")),
    ("Scaffold Work", re.compile(r"\b(scaffold|scaffolding|scaffold\s+erection|scaffold\s+dismantl)\b")),
    ("Pipeline Work", re.compile(r"\b(pipeline\s+(?:repair|work|laying|pig)|pigging|flowline\s+work)\b")),
    ("Pressure Testing", re.compile(r"\b(pressure\s+test|hydro(?:static)?\s+test|leak\s+test|pneumatic\s+test)\b")),
    ("Vehicle Operation", re.compile(r"\b(driving|vehicle|truck|tanker|transport|reversing|loading\s+truck)\b")),
    ("Maintenance", re.compile(r"\b(maintenance|repair|overhaul|shutdown\s+work|turnaround)\b")),
    ("Chemical Handling", re.compile(r"\b(chemical\s+handling|acid\s+job|acid(?:izing|isation)|cementing|mud\s+mixing)\b")),
    ("Well Control", re.compile(r"\b(well\s+control|well\s+kick|kill\s+operation|blowout\s+preventer|bop\s+test)\b")),
    ("Excavation", re.compile(r"\b(excavat|digging|trenching|earthwork|ground\s+disturbance)\b")),
]

_LOCATION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("Drill Floor", re.compile(r"\b(drill\s+floor|rig\s+floor|rotary\s+table|mouse\s+hole|rat\s+hole)\b")),
    ("Derrick", re.compile(r"\b(derrick|monkey\s+board|crown\s+block|mast|v.?door)\b")),
    ("Mud Tank", re.compile(r"\b(mud\s+tank|mud\s+pit|shale\s+shaker|mud\s+pump)\b")),
    ("Substation", re.compile(r"\b(substation|transformer|switchyard|panel\s+room|mcc\s+room)\b")),
    ("Wellhead", re.compile(r"\b(wellhead|well\s+head|christmas\s+tree|x.?mas\s+tree|well\s+cellar)\b")),
    ("Gathering Station", re.compile(r"\b(gathering\s+station|group\s+gathering|oil\s+collecting|ocs|production\s+facility)\b")),
    ("Pipeline ROW", re.compile(r"\b(pipeline|flowline|right\s+of\s+way|row|pig\s+launcher|pig\s+receiver)\b")),
    ("Tank Farm", re.compile(r"\b(tank\s+farm|storage\s+tank|oil\s+tank|water\s+tank|separator)\b")),
    ("Workshop", re.compile(r"\b(workshop|fabrication\s+yard|workshop\s+area|store|warehouse)\b")),
    ("Manifold", re.compile(r"\b(manifold|header|production\s+manifold|test\s+manifold)\b")),
    ("Pump House", re.compile(r"\b(pump\s+house|pump\s+station|compressor\s+station|booster\s+station)\b")),
    ("Confined Vessel", re.compile(r"\b(vessel|tank\s+interior|inside\s+tank|drum|column)\b")),
    ("Road / Access", re.compile(r"\b(road|access\s+road|highway|intersection|approach\s+road)\b")),
    ("Camp / Office", re.compile(r"\b(camp|office|admin|control\s+room|accommodation)\b")),
]

_BARRIER_FAILURE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("No Harness / Fall Protection", re.compile(
        r"\b(without\s+harness|no\s+harness|no\s+fall\s+(?:arrest|protection)|unhooked\s+lanyard|"
        r"harness\s+not\s+(?:worn|used|attached)|did\s+not\s+(?:wear|use)\s+harness)\b")),
    ("No Permit / PTW Violation", re.compile(
        r"\b(without\s+permit|no\s+permit|permit\s+(?:not|expired|violated)|"
        r"ptw\s+(?:not|expired|violated|bypassed)|no\s+ptw|permit\s+to\s+work\s+not)\b")),
    ("LOTO Violation", re.compile(
        r"\b(without\s+(?:loto|lockout|isolation)|no\s+(?:loto|lockout|isolation)|"
        r"loto\s+(?:not|bypassed|violated)|lockout\s+tagout\s+not|"
        r"not\s+(?:isolated|de.?energi[sz]ed)|without\s+lockout\s+tagout)\b")),
    ("Bypassed Safety Control", re.compile(
        r"\b(bypassed?\s+(?:interlock|guard|alarm|safety|control)|"
        r"safety\s+(?:device|guard|interlock)\s+(?:removed|disabled|bypassed|defeated)|"
        r"override\s+(?:alarm|interlock|safety)|tampered)\b")),
    ("Worn / Corroded Equipment", re.compile(
        r"\b(corroded|corrosion|worn\s+(?:out|sling|rope|cable|hose)|"
        r"frayed|deteriorated|rusted|pitting|wall\s+thinning|"
        r"defective\s+(?:sling|rope|hose|valve|gasket|flange))\b")),
    ("No PPE / Inadequate PPE", re.compile(
        r"\b(without\s+(?:ppe|helmet|gloves|goggles|safety\s+(?:shoes|glasses|gear))|"
        r"no\s+(?:ppe|helmet|gloves|goggles|safety\s+(?:shoes|glasses|gear))|"
        r"ppe\s+not\s+(?:worn|used|available)|improper\s+ppe)\b")),
    ("No Gas Test / Atmospheric Monitoring", re.compile(
        r"\b(without\s+gas\s+test|no\s+gas\s+test|gas\s+test\s+not|"
        r"no\s+atmospheric\s+(?:monitoring|check|test)|lel\s+not\s+(?:checked|tested))\b")),
    ("Barrier Failure / Breach", re.compile(
        r"\b(barrier\s+(?:failure|breach|compromised)|guard\s+rail\s+(?:missing|broken|removed)|"
        r"barricade\s+(?:missing|removed|not\s+(?:placed|erected))|"
        r"safety\s+net\s+(?:missing|torn|not))\b")),
    ("Procedure Violation / SOP Deviation", re.compile(
        r"\b((?:deviated|deviation)\s+from\s+(?:sop|procedure|protocol)|"
        r"(?:violated|violation)\s+(?:of\s+)?(?:sop|procedure|protocol)|"
        r"did\s+not\s+follow\s+(?:sop|procedure)|unauthorized|"
        r"not\s+as\s+per\s+(?:sop|procedure))\b")),
    ("Missing Supervision", re.compile(
        r"\b(no\s+supervision|unsupervised|without\s+supervision|"
        r"supervisor\s+(?:absent|not\s+present)|lone\s+work(?:er|ing)\s+without)\b")),
]


@dataclass
class PatternResult:
    """Extracted entity slots from a single observation report."""
    activities: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    barrier_failures: List[str] = field(default_factory=list)
    raw_matches: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def has_full_triad(self) -> bool:
        return bool(self.activities) and bool(self.locations) and bool(self.barrier_failures)

    def to_dict(self) -> Dict:
        return {
            "activities": self.activities,
            "locations": self.locations,
            "barrier_failures": self.barrier_failures,
            "has_full_triad": self.has_full_triad,
        }


@dataclass
class PatternTriad:
    """A ranked recurring pattern triad surfaced from batch analysis."""
    activity: str
    location: str
    barrier_failure: str
    count: int
    avg_severity: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "activity": self.activity,
            "location": self.location,
            "barrier_failure": self.barrier_failure,
            "count": self.count,
            "avg_severity": round(self.avg_severity, 4),
        }


def _extract_slot(text: str, patterns: List[Tuple[str, re.Pattern]]) -> List[str]:
    """Extract all matching canonical labels from text for a given slot."""
    found = []
    seen = set()
    for label, pat in patterns:
        if label not in seen and pat.search(text):
            found.append(label)
            seen.add(label)
    return found


def extract_slots(text: str) -> PatternResult:
    """Extract activity, location, and barrier failure slots from a single
    observation narrative. Text is normalized before extraction."""
    norm = normalize(text)
    activities = _extract_slot(norm, _ACTIVITY_PATTERNS)
    locations = _extract_slot(norm, _LOCATION_PATTERNS)
    barriers = _extract_slot(norm, _BARRIER_FAILURE_PATTERNS)

    return PatternResult(
        activities=activities,
        locations=locations,
        barrier_failures=barriers,
        raw_matches={
            "activities": activities,
            "locations": locations,
            "barrier_failures": barriers,
        },
    )


def rank_patterns(
    reports: Sequence[str],
    *,
    severity_scores: Optional[Sequence[float]] = None,
    top_k: int = 20,
) -> List[PatternTriad]:
    """Batch analysis: extract slots from all reports, count co-occurring
    (activity, location, barrier_failure) triads, and rank by frequency
    weighted by average severity.

    Args:
        reports: List of observation narrative strings.
        severity_scores: Optional parallel list of severity scores (0.0-1.0)
            from the severity regressor. If provided, triads are ranked by
            count * avg_severity.
        top_k: Maximum number of triads to return.

    Returns:
        Ranked list of PatternTriad objects (highest risk first).
    """
    triad_counts: Counter = Counter()
    triad_severities: Dict[Tuple[str, str, str], List[float]] = {}

    for idx, report in enumerate(reports):
        result = extract_slots(report)
        sev = severity_scores[idx] if severity_scores is not None else 0.5

        # Generate all (activity, location, barrier) triads
        acts = result.activities or ["Unknown Activity"]
        locs = result.locations or ["Unknown Location"]
        bars = result.barrier_failures or ["Unknown Barrier"]

        for a in acts:
            for l in locs:
                for b in bars:
                    key = (a, l, b)
                    triad_counts[key] += 1
                    triad_severities.setdefault(key, []).append(sev)

    # Rank by count * avg_severity
    ranked = []
    for (a, l, b), count in triad_counts.most_common(top_k * 3):
        sevs = triad_severities[(a, l, b)]
        avg_sev = sum(sevs) / len(sevs) if sevs else 0.0
        ranked.append(PatternTriad(
            activity=a, location=l, barrier_failure=b,
            count=count, avg_severity=avg_sev,
        ))

    # Sort by weighted score: count * avg_severity (descending)
    ranked.sort(key=lambda t: t.count * t.avg_severity, reverse=True)
    return ranked[:top_k]


__all__ = ["extract_slots", "rank_patterns", "PatternResult", "PatternTriad"]
