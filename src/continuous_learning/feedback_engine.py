"""
Human-in-the-Loop Feedback Engine & Reward Attributor
=====================================================
Manages safety officer reviews, calculates RLHF-style reward metrics based on
process safety risk penalties, and prepares experience replay datasets.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from .audit_logger import AuditLogger

ALL_IOGP_RULES = [
    "rule_bypassing_safety_controls",
    "rule_confined_space",
    "rule_driving",
    "rule_energy_isolation",
    "rule_hot_work",
    "rule_line_of_fire",
    "rule_safe_mechanical_lifting",
    "rule_work_authorization",
    "rule_working_at_height",
]


@dataclass
class FeedbackRecord:
    report_id: str
    narrative: str
    predicted_sif: bool
    predicted_sif_prob: float
    predicted_rules: List[str]
    predicted_severity: float
    actual_sif: bool
    actual_rules: List[str]
    actual_severity: float
    reviewer_id: str
    reviewer_role: str = "HSE_OFFICER"
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reward_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeedbackEngine:
    """Manages HSE officer feedback, reward attribution, and replay memory."""

    def __init__(
        self,
        store_path: Optional[Path] = None,
        audit_logger: Optional[AuditLogger] = None,
    ):
        base_dir = Path(__file__).resolve().parents[2]
        self.store_path = store_path or (base_dir / "data" / "feedback" / "feedback_store.jsonl")
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_logger = audit_logger or AuditLogger()

    @staticmethod
    def calculate_reward(
        pred_sif: bool,
        act_sif: bool,
        pred_rules: List[str],
        act_rules: List[str],
        pred_sev: float,
        act_sev: float,
    ) -> float:
        """
        Compute industrial safety reward signal:
        - True Positive SIF: +1.5 (life-threatening precursor successfully flagged)
        - True Negative SIF: +1.0 (routine observation correctly classified)
        - False Positive SIF: -1.0 (false alarm requiring review time)
        - False Negative SIF: -5.0 (CRITICAL PENALTY for missed fatal risk)
        - Rule Matching: +0.5 per correct rule, -1.5 per missed critical rule
        - Severity Delta: penalty proportional to absolute error
        """
        reward = 0.0

        # SIF Precursor Reward / Penalty
        if pred_sif and act_sif:
            reward += 1.5  # Correct SIF catch
        elif not pred_sif and not act_sif:
            reward += 1.0  # Correct Non-SIF
        elif pred_sif and not act_sif:
            reward -= 1.0  # False alarm
        else:  # not pred_sif and act_sif
            reward -= 5.0  # DANGEROUS: Missed fatal hazard

        # IOGP Rules Matching
        pred_set = set(pred_rules)
        act_set = set(act_rules)
        true_pos_rules = pred_set.intersection(act_set)
        false_neg_rules = act_set.difference(pred_set)
        false_pos_rules = pred_set.difference(act_set)

        reward += len(true_pos_rules) * 0.5
        reward -= len(false_neg_rules) * 1.5  # Heavy penalty for missed life-saving rule
        reward -= len(false_pos_rules) * 0.3

        # Severity Calibration Penalty
        sev_error = abs(pred_sev - act_sev)
        reward -= float(np.clip(sev_error * 2.0, 0.0, 2.0))

        return round(float(reward), 3)

    def record_feedback(
        self,
        narrative: str,
        predicted_sif: bool,
        predicted_sif_prob: float,
        predicted_rules: List[str],
        predicted_severity: float,
        actual_sif: bool,
        actual_rules: List[str],
        actual_severity: float,
        reviewer_id: str,
        reviewer_role: str = "Senior Safety Officer",
        notes: str = "",
        report_id: Optional[str] = None,
    ) -> FeedbackRecord:
        """Record an HSE officer review and persist to feedback store and audit log."""
        if report_id is None:
            report_id = f"OIL-FB-{uuid.uuid4().hex[:8].upper()}"

        reward = self.calculate_reward(
            pred_sif=predicted_sif,
            act_sif=actual_sif,
            pred_rules=predicted_rules,
            act_rules=actual_rules,
            pred_sev=predicted_severity,
            act_sev=actual_severity,
        )

        record = FeedbackRecord(
            report_id=report_id,
            narrative=narrative,
            predicted_sif=predicted_sif,
            predicted_sif_prob=predicted_sif_prob,
            predicted_rules=predicted_rules,
            predicted_severity=predicted_severity,
            actual_sif=actual_sif,
            actual_rules=actual_rules,
            actual_severity=actual_severity,
            reviewer_id=reviewer_id,
            reviewer_role=reviewer_role,
            notes=notes,
            reward_score=reward,
        )

        # Save to feedback store
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

        # Log to regulatory audit trail
        self.audit_logger.log_feedback(
            report_id=report_id,
            reviewer_id=reviewer_id,
            reviewer_role=reviewer_role,
            model_prediction={
                "sif_predicted": predicted_sif,
                "sif_prob": predicted_sif_prob,
                "rules_predicted": predicted_rules,
                "severity_predicted": predicted_severity,
            },
            human_verification={
                "sif_actual": actual_sif,
                "rules_actual": actual_rules,
                "severity_actual": actual_severity,
            },
            reward_score=reward,
            notes=notes,
        )

        return record

    def load_all_feedback(self) -> List[FeedbackRecord]:
        """Load all recorded feedbacks from store."""
        if not self.store_path.exists():
            return []
        records = []
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line.strip())
                        records.append(FeedbackRecord(**data))
                    except Exception:
                        continue
        return records

    def build_replay_dataset(
        self,
        base_train_df: pd.DataFrame,
        sample_multiplier_new: int = 5,
        max_base_samples: int = 20000,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Create a stratified experience replay dataset combining verified field
        feedbacks (weighted heavily) with historical core anchors to prevent
        catastrophic forgetting.
        """
        feedback_list = self.load_all_feedback()
        if not feedback_list:
            # Return baseline subset
            sample_n = min(len(base_train_df), max_base_samples)
            subset = base_train_df.sample(n=sample_n, random_state=42).copy()
            weights = np.ones(len(subset), dtype=float)
            return subset, weights

        # Convert feedback to DataFrame format
        fb_rows = []
        for fb in feedback_list:
            row = {
                "text_cleaned": fb.narrative,
                "text_tokenized_no_stopwords": fb.narrative.lower(),
                "sif_potential_binary": 1 if fb.actual_sif else 0,
                "is_sif_precursor": 1 if fb.actual_sif else 0,
                "sif_confidence_score": float(fb.actual_severity),
                "severity_score": float(fb.actual_severity),
                "is_feedback_sample": 1,
            }
            # Assign rule binary flags
            for r in ALL_IOGP_RULES:
                # Check if rule matches normalized name
                is_tagged = any(r.replace("rule_", "").replace("_", " ") in act_r.lower().replace("_", " ") for act_r in fb.actual_rules)
                row[r] = 1 if is_tagged else 0
            fb_rows.append(row)

        fb_df = pd.DataFrame(fb_rows)

        # Baseline sample selection
        sample_n = min(len(base_train_df), max_base_samples)
        base_subset = base_train_df.sample(n=sample_n, random_state=42).copy()
        base_subset["is_feedback_sample"] = 0

        # Include Indian Upstream Disaster Cases as Core Safety Anchors
        base_dir = Path(__file__).resolve().parents[2]
        ind_csv = base_dir / "data" / "processed" / "indian_oil_gas_benchmark.csv"
        ind_gz = base_dir / "data" / "processed" / "indian_oil_gas_benchmark.csv.gz"
        anchor_dfs = [base_subset]
        if ind_csv.exists() or ind_gz.exists():
            ind_df = pd.read_csv(ind_csv if ind_csv.exists() else ind_gz)
            ind_df["is_feedback_sample"] = 1
            if "sif_potential_binary" in ind_df.columns:
                ind_df["is_sif_precursor"] = ind_df["sif_potential_binary"]
            if "sif_confidence_score" in ind_df.columns:
                ind_df["severity_score"] = ind_df["sif_confidence_score"]
            anchor_dfs.append(pd.concat([ind_df] * 10, ignore_index=True))

        # Oversample feedback rows so the model adapts quickly
        fb_oversampled = pd.concat([fb_df] * sample_multiplier_new, ignore_index=True)
        anchor_dfs.append(fb_oversampled)

        combined_df = pd.concat(anchor_dfs, ignore_index=True)

        # Sample weights: Higher weights on human verified feedback & critical safety anchors
        weights = np.where(combined_df["is_feedback_sample"] == 1, 3.0, 1.0)

        return combined_df, weights
