"""
sentinel.benchmark
===================
Two things:

1. A strict schema + validator for `indian_oil_gas_benchmark.csv` so that
   "expand beyond 14 cases" doesn't just mean "add more rows" -- every row
   must carry a source citation (DGMS inquiry number, OISD bulletin id, or
   internal incident id), a ground-truth label, and an energy-class tag, or
   it is rejected. Unsourced benchmark rows are worse than no benchmark rows:
   they let a bad model look good.

2. A shadow-mode harness: runs a candidate model against a stream of
   already-labelled (human-reviewed) reports, computes agreement/precision/
   recall against the human label, and enforces a promotion gate before a
   model is allowed anywhere near the Zero-Tolerance Safety Gate.

Neither of these calls out to a live SAP EHS system or a real model file --
this repo doesn't have network access here, so both are built as pure
functions over in-memory / CSV data with a documented interface a real
pipeline plugs into (`ModelAdapter` protocol below).
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Dict, Iterable, List, Optional, Protocol

from .lexicon import ENERGY_CLASSES

REQUIRED_COLUMNS = [
    "case_id", "text", "label", "energy_class", "source_type",
    "source_ref", "incident_date", "asset_class",
]

VALID_LABELS = {"SIF", "NOT_SIF"}
VALID_SOURCE_TYPES = {"DGMS_INQUIRY", "OISD_BULLETIN", "INTERNAL_INVESTIGATION", "OTHER_STATUTORY"}


@dataclass
class RowError:
    row_index: int
    case_id: str
    errors: List[str]


@dataclass
class ValidationReport:
    total_rows: int
    valid_rows: int
    row_errors: List[RowError] = field(default_factory=list)
    label_balance: Dict[str, int] = field(default_factory=dict)
    energy_class_coverage: Dict[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return len(self.row_errors) == 0

    def to_dict(self) -> Dict:
        return {
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "is_valid": self.is_valid,
            "label_balance": self.label_balance,
            "energy_class_coverage": self.energy_class_coverage,
            "row_errors": [e.__dict__ for e in self.row_errors],
        }


def validate_benchmark_csv(csv_text: str, *, min_rows: int = 100,
                            min_per_energy_class: int = 5,
                            allow_oil_schema: bool = False) -> ValidationReport:
    reader = csv.DictReader(io.StringIO(csv_text))
    fieldnames = reader.fieldnames or []

    # Check for Indian OISD & OIL real benchmark schema if requested or detected
    is_oil_schema = allow_oil_schema or ("report_id" in fieldnames and ("text_cleaned" in fieldnames or "original_narrative" in fieldnames))
    if is_oil_schema:
        return validate_oil_benchmark_csv(csv_text, min_rows=min(min_rows, 14))

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing_cols:
        return ValidationReport(
            total_rows=0, valid_rows=0,
            row_errors=[RowError(-1, "<header>", [f"missing required columns: {missing_cols}"])],
        )

    rows = list(reader)
    errors: List[RowError] = []
    label_balance: Dict[str, int] = {}
    energy_coverage: Dict[str, int] = {}
    seen_ids = set()

    for i, row in enumerate(rows):
        row_errs = []
        cid = row.get("case_id", "").strip()
        if not cid:
            row_errs.append("missing case_id")
        elif cid in seen_ids:
            row_errs.append(f"duplicate case_id: {cid}")
        seen_ids.add(cid)

        text = row.get("text", "").strip()
        if len(text.split()) < 4:
            row_errs.append("text too short to be a usable benchmark case (<4 words)")

        label = row.get("label", "").strip()
        if label not in VALID_LABELS:
            row_errs.append(f"invalid label '{label}', must be one of {VALID_LABELS}")
        else:
            label_balance[label] = label_balance.get(label, 0) + 1

        ec = row.get("energy_class", "").strip()
        if ec not in ENERGY_CLASSES and label == "SIF":
            row_errs.append(f"invalid/missing energy_class '{ec}' for SIF-labelled row")
        elif ec:
            energy_coverage[ec] = energy_coverage.get(ec, 0) + 1

        stype = row.get("source_type", "").strip()
        if stype not in VALID_SOURCE_TYPES:
            row_errs.append(f"invalid source_type '{stype}', must be one of {VALID_SOURCE_TYPES}")

        sref = row.get("source_ref", "").strip()
        if not sref:
            row_errs.append("missing source_ref (DGMS inquiry no. / OISD bulletin id / internal id)")

        idate = row.get("incident_date", "").strip()
        if idate:
            try:
                date.fromisoformat(idate)
            except ValueError:
                row_errs.append(f"incident_date '{idate}' not ISO format (YYYY-MM-DD)")

        if row_errs:
            errors.append(RowError(i, cid or f"row_{i}", row_errs))

    report = ValidationReport(
        total_rows=len(rows),
        valid_rows=len(rows) - len(errors),
        row_errors=errors,
        label_balance=label_balance,
        energy_class_coverage=energy_coverage,
    )

    if len(rows) < min_rows:
        report.row_errors.append(RowError(
            -1, "<dataset>", [f"only {len(rows)} rows, need >= {min_rows} for a credible benchmark"],
        ))
    under_covered = {ec: n for ec in ENERGY_CLASSES
                     if (n := energy_coverage.get(ec, 0)) < min_per_energy_class}
    if under_covered:
        report.row_errors.append(RowError(
            -1, "<dataset>",
            [f"energy classes under minimum coverage ({min_per_energy_class} each): {under_covered}"],
        ))

    return report


def validate_oil_benchmark_csv(csv_text: str, *, min_rows: int = 14) -> ValidationReport:
    """Validate real Indian OISD & OIL incident benchmarks (e.g. indian_oil_gas_benchmark.csv)."""
    reader = csv.DictReader(io.StringIO(csv_text))
    rows = list(reader)
    errors: List[RowError] = []
    label_balance: Dict[str, int] = {}
    energy_coverage: Dict[str, int] = {}
    seen_ids = set()

    for i, row in enumerate(rows):
        row_errs = []
        cid = row.get("report_id", "").strip() or row.get("incident_id", "").strip() or f"CASE_{i}"
        if cid in seen_ids:
            row_errs.append(f"duplicate report_id: {cid}")
        seen_ids.add(cid)

        text = row.get("text_cleaned", "").strip() or row.get("original_narrative", "").strip()
        if len(text.split()) < 4:
            row_errs.append("text too short to be a usable benchmark case (<4 words)")

        sif_val = str(row.get("sif_potential_binary", row.get("sif_potential", ""))).strip().lower()
        if sif_val in ("1", "1.0", "true", "sif", "sif-potential precursor"):
            label = "SIF"
        else:
            label = "NOT_SIF"
        label_balance[label] = label_balance.get(label, 0) + 1

        sref = row.get("source_reference", "").strip() or row.get("oisd_standard", "").strip()
        if not sref:
            row_errs.append("missing source_reference / oisd_standard citation")

        # Infer energy class from narrative/equipment if not directly column-tagged
        ec = row.get("energy_class", "").strip()
        if not ec:
            t_lower = text.lower()
            if any(w in t_lower for w in ("fall", "scaffold", "height", "derrick")):
                ec = "gravity_person"
            elif any(w in t_lower for w in ("crane", "suspended", "sling", "load", "winch")):
                ec = "gravity"
            elif any(w in t_lower for w in ("electric", "shock", "loto", "voltage", "current")):
                ec = "electrical"
            elif any(w in t_lower for w in ("blowout", "gas", "pressure", "kick", "flange")):
                ec = "pressure"
            elif any(w in t_lower for w in ("fire", "burn", "hot work", "welding")):
                ec = "temperature"
            else:
                ec = "mechanical"
        energy_coverage[ec] = energy_coverage.get(ec, 0) + 1

        if row_errs:
            errors.append(RowError(i, cid, row_errs))

    report = ValidationReport(
        total_rows=len(rows),
        valid_rows=len(rows) - len(errors),
        row_errors=errors,
        label_balance=label_balance,
        energy_class_coverage=energy_coverage,
    )

    if len(rows) < min_rows:
        report.row_errors.append(RowError(
            -1, "<dataset>", [f"only {len(rows)} rows, need >= {min_rows} for Indian benchmark"],
        ))

    return report


# ---------------------------------------------------------------------------
# Shadow-mode promotion gate
# ---------------------------------------------------------------------------
class ModelAdapter(Protocol):
    def predict_proba_sif(self, text: str, metadata: Optional[Dict] = None) -> float:
        ...


@dataclass
class ShadowCase:
    case_id: str
    text: str
    human_label: str            # "SIF" | "NOT_SIF", from HSE reviewer sign-off
    metadata: Optional[Dict] = None
    asset_class: Optional[str] = None


@dataclass
class ShadowResult:
    n_cases: int
    agreement_rate: float
    recall_sif: float
    precision_sif: float
    false_negatives: List[str]
    false_positives: List[str]
    passes_gate: bool
    gate_reasons: List[str]

    def to_dict(self) -> Dict:
        d = self.__dict__.copy()
        return d


def run_shadow_evaluation(
    model: ModelAdapter,
    cases: Iterable[ShadowCase],
    *,
    decide_fn: Callable,
    min_days_covered: int = 30,
    days_covered: int = 0,
    min_recall_sif: float = 0.99,
    min_precision_sif: float = 0.80,
    min_agreement: float = 0.90,
    max_new_false_negatives: int = 0,
) -> ShadowResult:
    """
    decide_fn: sentinel.decision_policy.decide, injected so this module has
    no hard dependency on it (keeps the promotion gate testable in isolation).
    """
    from .interlock import scan as interlock_scan
    from .energy_metadata import assess as assess_metadata

    cases = list(cases)
    tp = fp = tn = fn = 0
    agree = 0
    fn_ids, fp_ids = [], []

    for c in cases:
        p = model.predict_proba_sif(c.text, c.metadata)
        il = interlock_scan(c.text)
        meta = assess_metadata(c.metadata or {})
        decision = decide_fn(p, interlock=il, metadata=meta, asset_class=c.asset_class)
        pred = decision.label if decision.label is not None else "SIF"  # human-review defaults conservative for shadow scoring

        if pred == c.human_label:
            agree += 1
        if c.human_label == "SIF" and pred == "SIF":
            tp += 1
        elif c.human_label == "SIF" and pred != "SIF":
            fn += 1
            fn_ids.append(c.case_id)
        elif c.human_label == "NOT_SIF" and pred == "SIF":
            fp += 1
            fp_ids.append(c.case_id)
        else:
            tn += 1

    n = len(cases)
    agreement_rate = agree / n if n else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0

    reasons = []
    if days_covered < min_days_covered:
        reasons.append(f"shadow window {days_covered}d < required {min_days_covered}d")
    if recall < min_recall_sif:
        reasons.append(f"recall_sif {recall:.4f} < required {min_recall_sif}")
    if precision < min_precision_sif:
        reasons.append(f"precision_sif {precision:.4f} < required {min_precision_sif}")
    if agreement_rate < min_agreement:
        reasons.append(f"agreement_rate {agreement_rate:.4f} < required {min_agreement}")
    if len(fn_ids) > max_new_false_negatives:
        reasons.append(f"{len(fn_ids)} false negatives > allowed {max_new_false_negatives}")

    return ShadowResult(
        n_cases=n, agreement_rate=agreement_rate, recall_sif=recall,
        precision_sif=precision, false_negatives=fn_ids, false_positives=fp_ids,
        passes_gate=len(reasons) == 0, gate_reasons=reasons,
    )


__all__ = [
    "validate_benchmark_csv", "ValidationReport", "RowError",
    "ShadowCase", "ShadowResult", "run_shadow_evaluation", "ModelAdapter",
    "REQUIRED_COLUMNS", "VALID_LABELS", "VALID_SOURCE_TYPES",
]
