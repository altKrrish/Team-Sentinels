"""
Shadow Benchmarking Engine (Champion vs. Challenger)
====================================================
Runs newly trained candidate models in parallel shadow mode alongside the active
production champion model to benchmark latency, agreement, false positive
reduction, and recall improvement before triggering live promotion.
"""

import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from .audit_logger import AuditLogger


@dataclass
class ShadowBenchmarkReport:
    champion_version: str
    challenger_version: str
    samples_evaluated: int
    agreement_rate_pct: float
    champion_sif_recall_pct: Optional[float]
    challenger_sif_recall_pct: Optional[float]
    sif_recall_delta_pct: Optional[float]
    false_alarm_reduction_pct: Optional[float]
    latency_champion_ms: float
    latency_challenger_ms: float
    disagreements_count: int
    sample_disagreements: List[Dict[str, Any]] = field(default_factory=list)
    recommendation: str = "NEEDS_REVIEW"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ShadowBenchmarker:
    """Executes side-by-side Shadow Benchmarking of Champion and Challenger models."""

    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        self.audit_logger = audit_logger or AuditLogger()

    def benchmark_stream(
        self,
        champion_pipeline: Dict[str, Any],
        challenger_pipeline: Dict[str, Any],
        eval_df: pd.DataFrame,
        champion_version: str = "v1.0.0-champion",
        challenger_version: str = "v1.0.1-challenger",
        max_samples: int = 500,
    ) -> ShadowBenchmarkReport:
        """
        Run side-by-side evaluation on incident reports.
        """
        eval_subset = eval_df.head(max_samples).copy()
        n_samples = len(eval_subset)

        from sentinel import interlock as sentinel_il, decision_policy as sentinel_dp

        # 1. Benchmark Champion with Hardened Precedence
        t0 = time.perf_counter()
        X_champ = champion_pipeline["extractor"].transform(eval_subset)
        champ_raw_probs = champion_pipeline["sif_model"].predict_proba(X_champ)
        champ_sif_probs = champ_raw_probs[:, 1] if champ_raw_probs.shape[1] > 1 else np.zeros(champ_raw_probs.shape[0])
        champ_thresh = champion_pipeline.get("thresholds", {}).get("optimal_sif_threshold", 0.47)
        
        champ_preds_list = []
        for i in range(n_samples):
            txt = str(eval_subset.iloc[i].get("text_cleaned") or "")
            p = float(champ_sif_probs[i])
            il = sentinel_il.scan(txt)
            dec = sentinel_dp.decide(p, interlock=il, metadata=None)
            pred = 1 if dec.label == "SIF" else (1 if dec.route == sentinel_dp.Route.HUMAN_REVIEW else (1 if p >= champ_thresh else 0))
            champ_preds_list.append(pred)
        champ_preds = np.array(champ_preds_list)
        t_champ_total = (time.perf_counter() - t0) * 1000.0
        latency_champ = round(t_champ_total / max(n_samples, 1), 3)

        # 2. Benchmark Challenger with Hardened Precedence
        t0 = time.perf_counter()
        X_chal = challenger_pipeline["extractor"].transform(eval_subset)
        chal_raw_probs = challenger_pipeline["sif_model"].predict_proba(X_chal)
        chal_sif_probs = chal_raw_probs[:, 1] if chal_raw_probs.shape[1] > 1 else np.zeros(chal_raw_probs.shape[0])
        chal_thresh = challenger_pipeline.get("thresholds", {}).get("optimal_sif_threshold", 0.47)
        
        chal_preds_list = []
        for i in range(n_samples):
            txt = str(eval_subset.iloc[i].get("text_cleaned") or "")
            p = float(chal_sif_probs[i])
            il = sentinel_il.scan(txt)
            dec = sentinel_dp.decide(p, interlock=il, metadata=None)
            pred = 1 if dec.label == "SIF" else (1 if dec.route == sentinel_dp.Route.HUMAN_REVIEW else (1 if p >= chal_thresh else 0))
            chal_preds_list.append(pred)
        chal_preds = np.array(chal_preds_list)
        t_chal_total = (time.perf_counter() - t0) * 1000.0
        latency_chal = round(t_chal_total / max(n_samples, 1), 3)

        # 3. Agreement calculation
        agreements = (champ_preds == chal_preds).sum()
        agreement_rate = round((agreements / n_samples) * 100.0, 2)

        # 4. Disagreement inspection
        disagree_indices = np.where(champ_preds != chal_preds)[0]
        sample_disagreements = []
        for idx in disagree_indices[:5]:  # Capture top 5 examples
            sample_disagreements.append({
                "sample_index": int(idx),
                "text_snippet": str(eval_subset.iloc[idx].get("text_cleaned", ""))[:100] + "...",
                "champion_prediction": "SIF" if champ_preds[idx] == 1 else "NON-SIF",
                "champion_prob": round(float(champ_sif_probs[idx]), 3),
                "challenger_prediction": "SIF" if chal_preds[idx] == 1 else "NON-SIF",
                "challenger_prob": round(float(chal_sif_probs[idx]), 3),
            })

        # 5. Accuracy & Recall metrics if ground truth is present
        champ_rec = None
        chal_rec = None
        rec_delta = None
        false_alarm_reduc = None

        sif_col = "sif_potential_binary" if "sif_potential_binary" in eval_subset.columns else ("is_sif_precursor" if "is_sif_precursor" in eval_subset.columns else None)
        if sif_col is not None:
            y_true = eval_subset[sif_col].fillna(0).astype(int).values
            sif_pos = (y_true == 1)
            sif_neg = (y_true == 0)

            if sif_pos.sum() > 0:
                champ_rec = round(float((champ_preds[sif_pos] == 1).sum() / sif_pos.sum() * 100.0), 2)
                chal_rec = round(float((chal_preds[sif_pos] == 1).sum() / sif_pos.sum() * 100.0), 2)
                rec_delta = round(chal_rec - champ_rec, 2)

            if sif_neg.sum() > 0:
                champ_fp = (champ_preds[sif_neg] == 1).sum()
                chal_fp = (chal_preds[sif_neg] == 1).sum()
                fp_diff = champ_fp - chal_fp
                false_alarm_reduc = round(float((fp_diff / max(champ_fp, 1)) * 100.0), 2)

        # 6. Recommendation Logic
        if chal_rec is not None and chal_rec >= 98.0 and (rec_delta is None or rec_delta >= 0.0):
            recommendation = "RECOMMEND_PROMOTION: Superior or equal SIF recall with validated low latency."
        elif agreement_rate >= 95.0 and latency_chal <= latency_champ * 1.2:
            recommendation = "RECOMMEND_PROMOTION: High model stability and consensus with Champion."
        elif rec_delta is not None and rec_delta < 0.0:
            recommendation = "REJECT_PROMOTION: Challenger suffered recall regression on SIF precursors."
        else:
            recommendation = "NEEDS_REVIEW: Divergence observed; manual safety committee sign-off required."

        report = ShadowBenchmarkReport(
            champion_version=champion_version,
            challenger_version=challenger_version,
            samples_evaluated=n_samples,
            agreement_rate_pct=agreement_rate,
            champion_sif_recall_pct=champ_rec,
            challenger_sif_recall_pct=chal_rec,
            sif_recall_delta_pct=rec_delta,
            false_alarm_reduction_pct=false_alarm_reduc,
            latency_champion_ms=latency_champ,
            latency_challenger_ms=latency_chal,
            disagreements_count=len(disagree_indices),
            sample_disagreements=sample_disagreements,
            recommendation=recommendation,
        )

        # Log to audit trail
        self.audit_logger.log_shadow_benchmark(
            champion_version=champion_version,
            challenger_version=challenger_version,
            samples_evaluated=n_samples,
            agreement_rate_pct=agreement_rate,
            champion_sif_recall_pct=champ_rec or 0.0,
            challenger_sif_recall_pct=chal_rec or 0.0,
            latency_champion_ms=latency_champ,
            latency_challenger_ms=latency_chal,
            recommendation=recommendation,
        )

        return report
