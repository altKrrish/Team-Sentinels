"""
Loading REAL labelled reports, when there are any.

This is the seam between the prototype and the real thing. `train.py` calls
:func:`load` first; if a CSV is there it trains on that and the synthetic
generator is never touched. Nothing else in the codebase changes - the
vectorisers, the seven heads, the explainability and the served JSON all work off
`(texts, labels)` and do not care where the rows came from.

Why the corpus is synthetic today
---------------------------------
OIL's UA/UC, near-miss and incident narratives are internal HSSE-platform
records. They are not published - OIL's Annual Report and BRSR carry aggregate
safety figures only (LTIFR, fatalities, man-hours), no free text. There is
nothing to download, so the prototype learns the DEKRA / EEI precursor rule from
generated narratives instead.

Public corpora that DO carry real narratives, if you want real text before OIL's
export arrives:

    OSHA Severe Injury Reports      ~100k rows, `Final Narrative` column
    MSHA accident / injury / illness  `NARRATIVE` column
    PHMSA pipeline incident data    oil & gas specific, narrative description
    BSEE offshore incident data     oil & gas, panel reports
    NASA ASRS                       ~1M voluntary near-miss self-reports

None of them are drop-in, for one reason: they have no `sif`, no Life-Saving Rule
tag, no hazard energy and no barrier state. Those four fields are precisely what
this model predicts, so real text still has to be labelled by a reviewer before
it can train anything. They are excellent for VALIDATION though - score them with
the trained engine and read the verdicts.

CSV format
----------
One row per report. Only `text` and `sif` are required::

    text                 the raw narrative, verbatim - do not pre-clean it
    sif                  1/0, true/false, yes/no
    rules                Life-Saving Rule ids, pipe- or semicolon-separated
    severity             0-10 continuous, optional (derived if absent)
    severity_actual      1-5, what did happen
    severity_potential   1-5, what could have happened
    energy               hazard energy
    barrier              barrier state
    reported_at          ISO date - if present, train.py splits BY DATE

Aliases are accepted for every column (`narrative` for `text`, `sifPotential`
for `sif`, and so on) so an export can usually be fed in unrenamed.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

#: The nine IOGP Life-Saving Rules, as the frontend spells them in `src/lib/lsr.js`.
LSR_IDS = (
    "bypassing-safety-controls",
    "confined-space",
    "driving",
    "energy-isolation",
    "hot-work",
    "line-of-fire",
    "safe-mechanical-lifting",
    "work-authorisation",
    "working-at-height",
)

HAZARD_ENERGIES = (
    "gravity", "pressure", "electrical", "thermal", "mechanical", "chemical", "motion",
)

BARRIER_STATES = ("absent", "failed", "bypassed", "inadequate", "not-verified")

#: Where `train.py` looks when no path is given.
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "reports.csv"

#: canonical column -> accepted header spellings (lowercased, non-alnum stripped)
_ALIASES: dict[str, tuple[str, ...]] = {
    "text": ("text", "narrative", "description", "observation", "finalnarrative", "details", "remarks"),
    "sif": ("sif", "sifpotential", "sifflag", "issif", "label", "target"),
    "rules": ("rules", "lsr", "lsrtags", "lifesavingrules", "rule"),
    "severity": ("severity", "severityscore", "risk", "riskscore"),
    "severity_actual": ("severityactual", "actualseverity", "actual"),
    "severity_potential": ("severitypotential", "potentialseverity", "potential"),
    "energy": ("energy", "hazardenergy", "energysource"),
    "barrier": ("barrier", "barrierfailure", "barrierstate", "controlfailure"),
    "reported_at": ("reportedat", "date", "reporteddate", "incidentdate", "eventdate"),
}

_TRUE = {"1", "true", "t", "yes", "y", "sif", "sif-potential", "high"}
_FALSE = {"0", "false", "f", "no", "n", "non-sif", "low", ""}

_SPLIT_RULES = re.compile(r"[|;,]")


def _key(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", header.strip().lower())


def _resolve_columns(fieldnames) -> dict[str, str]:
    """Map canonical names onto the header spellings actually present."""
    seen = {_key(h): h for h in fieldnames or ()}
    found: dict[str, str] = {}
    for canonical, spellings in _ALIASES.items():
        for spelling in spellings:
            if spelling in seen:
                found[canonical] = seen[spelling]
                break
    return found


def _as_bool(raw: str, row_no: int) -> bool:
    value = (raw or "").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(
        f"row {row_no}: cannot read {raw!r} as a SIF label. "
        f"Use 1/0, true/false or yes/no."
    )


def _as_rules(raw: str, row_no: int, strict: bool) -> list[str]:
    ids = [p.strip().lower() for p in _SPLIT_RULES.split(raw or "") if p.strip()]
    unknown = sorted(set(ids) - set(LSR_IDS))
    if unknown:
        message = (
            f"row {row_no}: unknown Life-Saving Rule id(s) {unknown}. "
            f"Valid ids are {list(LSR_IDS)}."
        )
        if strict:
            raise ValueError(message)
        ids = [i for i in ids if i in LSR_IDS]
    return sorted(set(ids))


def _as_band(raw: str, default: int) -> int:
    try:
        return max(1, min(5, int(round(float(raw)))))
    except (TypeError, ValueError):
        return default


def _as_enum(raw: str, allowed: tuple[str, ...], default: str) -> str:
    value = (raw or "").strip().lower().replace(" ", "-").replace("_", "-")
    return value if value in allowed else default


def load(path: str | Path | None = None, strict: bool = True) -> tuple[list[str], list[dict]] | None:
    """Read labelled reports from CSV.

    :param path: CSV to read. Defaults to ``server/data/reports.csv``.
    :param strict: raise on an unknown Life-Saving Rule id rather than dropping it.
    :returns: ``(texts, labels)`` in exactly the shape
        :func:`closecall.corpus.synthetic_corpus` returns, or ``None`` if the file
        does not exist - so the caller can fall back to the generator.
    :raises ValueError: if the file exists but is missing `text` or `sif`, or a
        row cannot be read. A malformed real dataset is an error worth stopping
        on; silently training on half of it is worse than not training.
    """
    csv_path = Path(path) if path else DEFAULT_PATH
    if not csv_path.exists():
        return None

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        cols = _resolve_columns(reader.fieldnames)

        missing = [c for c in ("text", "sif") if c not in cols]
        if missing:
            raise ValueError(
                f"{csv_path.name}: required column(s) {missing} not found. "
                f"Headers present: {list(reader.fieldnames or ())}. "
                f"Accepted spellings for `text`: {list(_ALIASES['text'])}; "
                f"for `sif`: {list(_ALIASES['sif'])}."
            )

        texts: list[str] = []
        labels: list[dict] = []

        for row_no, row in enumerate(reader, start=2):  # row 1 is the header
            text = (row.get(cols["text"]) or "").strip()
            if not text:
                continue

            sif = _as_bool(row.get(cols["sif"], ""), row_no)
            rules = (
                _as_rules(row.get(cols["rules"], ""), row_no, strict)
                if "rules" in cols
                else []
            )
            actual = _as_band(row.get(cols.get("severity_actual", ""), ""), 1)
            potential = _as_band(
                row.get(cols.get("severity_potential", ""), ""), 5 if sif else 2
            )
            potential = max(potential, actual)

            raw_severity = row.get(cols.get("severity", ""), "")
            try:
                severity = max(0.0, min(10.0, float(raw_severity)))
            except (TypeError, ValueError):
                # No score given: reconstruct one on the same 0-10 scale the
                # ridge head is trained against.
                severity = min(10.0, 1.05 * potential + 1.15 * (actual - 1) + (1.4 if sif else 0.0))

            labels.append(
                {
                    "sif": sif,
                    "rules": rules,
                    "severity": severity,
                    "severity_actual": actual,
                    "severity_potential": potential,
                    "energy": _as_enum(
                        row.get(cols.get("energy", ""), ""), HAZARD_ENERGIES, "mechanical"
                    ),
                    "barrier": _as_enum(
                        row.get(cols.get("barrier", ""), ""), BARRIER_STATES, "not-verified"
                    )
                    if sif
                    else _as_enum(row.get(cols.get("barrier", ""), ""), BARRIER_STATES, "intact"),
                    "reported_at": (row.get(cols.get("reported_at", ""), "") or "").strip(),
                    "source": "real",
                }
            )
            texts.append(text)

    if not texts:
        raise ValueError(f"{csv_path.name}: no rows with a non-empty narrative.")

    untagged = sum(1 for l in labels if not l["rules"])
    if untagged == len(labels):
        raise ValueError(
            f"{csv_path.name}: no Life-Saving Rule tags in any row. The rule head "
            f"cannot be trained. Add a `rules` column (pipe-separated ids from "
            f"{list(LSR_IDS)})."
        )

    return texts, labels


def describe(labels: list[dict]) -> str:
    """One line about what was loaded, for the training log."""
    positives = sum(1 for l in labels if l["sif"])
    tagged = sum(1 for l in labels if l["rules"])
    dated = sum(1 for l in labels if l.get("reported_at"))
    return (
        f"{len(labels)} rows, {positives / len(labels):.1%} SIF-positive, "
        f"{tagged} rule-tagged, {dated} dated"
    )
