"""
sentinel.explainability
========================
Feature Attribution & Explainability Engine for SIF classification.

Provides top-k contributing n-gram attribution using linear model coefficients:
    Contribution(t) = w_t * x_t

Because Estimators 1 (L2-Logistic Regression) and 3 (L1-Logistic Regression)
in the soft-voting ensemble are linear, we have direct access to learned
feature weights. This module extracts the most influential features driving
each prediction so that:

  1. Frontline HSE officers see *why* a report was flagged (not just p=0.78).
  2. Dashboard can highlight exact hazard phrases in the narrative.
  3. DGMS auditors can trace each decision to specific textual evidence.

Falls back gracefully when production model weights are unavailable (stub mode).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class FeatureAttribution:
    """Attribution for a single feature."""
    feature_name: str
    weight: float
    tfidf_value: float
    contribution: float      # weight * tfidf_value

    def to_dict(self) -> Dict:
        return {
            "feature": self.feature_name,
            "weight": round(self.weight, 6),
            "tfidf_value": round(self.tfidf_value, 6),
            "contribution": round(self.contribution, 6),
        }


@dataclass
class ExplanationResult:
    """Explanation for a single SIF classification prediction."""
    top_positive: List[FeatureAttribution] = field(default_factory=list)
    top_negative: List[FeatureAttribution] = field(default_factory=list)
    method: str = "linear_coefficient_attribution"
    available: bool = True

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "available": self.available,
            "top_positive_contributors": [a.to_dict() for a in self.top_positive],
            "top_negative_contributors": [a.to_dict() for a in self.top_negative],
        }


def explain(
    model: Any,
    feature_matrix: Any,
    *,
    top_k_positive: int = 5,
    top_k_negative: int = 3,
) -> ExplanationResult:
    """Extract top contributing features for a SIF classification prediction.

    Args:
        model: The production SIF classifier (VotingClassifier or similar).
               Must have .estimators_ attribute or .coef_ attribute.
        feature_matrix: The sparse/dense feature matrix (1 x D) for the report.
        top_k_positive: Number of top positive contributors to return.
        top_k_negative: Number of top negative contributors to return.

    Returns:
        ExplanationResult with ranked feature attributions.
    """
    try:
        # Try to extract coefficients from the model
        coef = _extract_coefficients(model)
        if coef is None:
            return ExplanationResult(available=False, method="model_weights_unavailable")

        # Get feature names if available
        feature_names = _extract_feature_names(model)

        # Compute contributions: w_t * x_t
        if hasattr(feature_matrix, 'toarray'):
            x = np.asarray(feature_matrix.toarray()).ravel()
        else:
            x = np.asarray(feature_matrix).ravel()

        # Ensure dimensions match
        n_features = min(len(coef), len(x))
        coef = coef[:n_features]
        x = x[:n_features]

        contributions = coef * x

        # Get indices of non-zero contributions
        nonzero_mask = x != 0
        nonzero_indices = np.where(nonzero_mask)[0]

        if len(nonzero_indices) == 0:
            return ExplanationResult(available=True)

        nonzero_contributions = contributions[nonzero_indices]

        # Top positive contributors (pushing toward SIF)
        pos_order = np.argsort(-nonzero_contributions)
        top_pos = []
        for rank_idx in pos_order[:top_k_positive]:
            feat_idx = nonzero_indices[rank_idx]
            c = contributions[feat_idx]
            if c <= 0:
                break
            name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
            top_pos.append(FeatureAttribution(
                feature_name=name,
                weight=float(coef[feat_idx]),
                tfidf_value=float(x[feat_idx]),
                contribution=float(c),
            ))

        # Top negative contributors (pushing away from SIF)
        neg_order = np.argsort(nonzero_contributions)
        top_neg = []
        for rank_idx in neg_order[:top_k_negative]:
            feat_idx = nonzero_indices[rank_idx]
            c = contributions[feat_idx]
            if c >= 0:
                break
            name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
            top_neg.append(FeatureAttribution(
                feature_name=name,
                weight=float(coef[feat_idx]),
                tfidf_value=float(x[feat_idx]),
                contribution=float(c),
            ))

        return ExplanationResult(
            top_positive=top_pos,
            top_negative=top_neg,
            available=True,
        )

    except Exception:
        return ExplanationResult(available=False, method="extraction_failed")


def _extract_coefficients(model: Any) -> Optional[np.ndarray]:
    """Extract linear coefficients from the model.

    Supports:
    - Direct model with .coef_ (LogisticRegression, SGDClassifier)
    - VotingClassifier with .estimators_ list
    - Wrapped production model with .sif_model attribute
    """
    # Case 1: Production pipeline model wrapper
    if hasattr(model, 'sif_model'):
        return _extract_coefficients(model.sif_model)

    # Case 2: VotingClassifier — use the first linear estimator (L2 LogReg)
    if hasattr(model, 'estimators_'):
        for est in model.estimators_:
            actual = est[1] if isinstance(est, tuple) else est
            if hasattr(actual, 'coef_'):
                return np.asarray(actual.coef_).ravel()
        return None

    # Case 3: Direct linear model
    if hasattr(model, 'coef_'):
        return np.asarray(model.coef_).ravel()

    return None


def _extract_feature_names(model: Any) -> List[str]:
    """Extract feature names from the model's feature extractor."""
    # Try production pipeline model
    if hasattr(model, 'extractor') and hasattr(model.extractor, 'get_feature_names_out'):
        try:
            return list(model.extractor.get_feature_names_out())
        except Exception:
            pass

    # Fallback: generate generic names
    coef = _extract_coefficients(model)
    if coef is not None:
        return [f"feature_{i}" for i in range(len(coef))]
    return []


__all__ = ["explain", "ExplanationResult", "FeatureAttribution"]
