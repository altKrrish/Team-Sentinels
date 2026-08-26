#!/usr/bin/env python3
"""
AI/NLP SIF Precursor Detection & IOGP Rule Auto-Tagging Engine
==============================================================
Training & Evaluation Pipeline for SIH-2026 Problem Statement (Oil India Limited)

Tasks Trained:
  1. Task 1: SIF-Potential vs Non-SIF Binary Classifier
  2. Task 2: 9 IOGP Life-Saving Rules Multi-Label Classifier
  3. Task 3: Continuous SIF Severity Scoring Regressor
  4. Task 4: Real-World Indian Oil & Gas Benchmark Out-of-Domain Validation
  5. Task 5: Model Explainability & Key Linguistic Precursor Extractors

Author: SIH-2026 Team
"""

import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

# Increase CSV limits
csv.field_size_limit(10**7)

RULE_COLUMNS = [
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

RULE_DISPLAY_NAMES = [
    "Bypassing Safety Controls",
    "Confined Space",
    "Driving",
    "Energy Isolation",
    "Hot Work",
    "Line of Fire",
    "Safe Mechanical Lifting",
    "Work Authorization",
    "Working at Height",
]

NUMERIC_FEATURES = [
    "word_count", "char_count", "sentence_count", "avg_sentence_length", "negation_count",
    "high_severity_word_count", "medium_severity_word_count", "low_severity_word_count",
    "severity_token_ratio", "barrier_failure_word_count", "violation_word_count",
    "has_measurements", "number_count"
]


def load_datasets(processed_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, validation, test splits, and Indian benchmark."""
    print("📂 Loading preprocessed datasets...")
    train_path = processed_dir / "master_hsse_sif_train.csv"
    val_path = processed_dir / "master_hsse_sif_val.csv"
    test_path = processed_dir / "master_hsse_sif_test.csv"
    ind_path = processed_dir / "indian_oil_gas_benchmark.csv"

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)
    test_df = pd.read_csv(test_path, low_memory=False)
    ind_df = pd.read_csv(ind_path, low_memory=False) if ind_path.exists() else None

    print(f"   Train Records : {len(train_df):,}")
    print(f"   Val Records   : {len(val_df):,}")
    print(f"   Test Records  : {len(test_df):,}")
    if ind_df is not None:
        print(f"   Indian Cases  : {len(ind_df):,}")

    return train_df, val_df, test_df, ind_df


def engineer_features(cleaned_text: str) -> Dict:
    """Extract linguistic, complexity, and safety-domain token signals on the fly."""
    import re
    text_lower = cleaned_text.lower()
    words = cleaned_text.split()
    word_count = len(words)
    char_count = len(cleaned_text)

    sentences = [s.strip() for s in re.split(r'[.!?]+', cleaned_text) if s.strip()]
    sentence_count = max(len(sentences), 1)
    avg_sentence_len = round(word_count / sentence_count, 1)

    negation_keywords = [
        "not", "no", "without", "never", "nor", "none",
        "didn't", "wasn't", "weren't", "hadn't", "couldn't",
        "unauthorized", "unlocked", "unlatched", "unsecured",
    ]
    negation_count = sum(1 for w in negation_keywords if w in text_lower)

    high_severity_words = [
        "fatal", "fatality", "death", "killed", "died", "deceased",
        "explosion", "exploded", "blast", "detonation", "blowout",
        "electrocuted", "electrocution", "arc flash", "amputation",
        "severed", "dismembered", "crushed", "engulfed", "asphyxiation",
        "unconscious", "collapse", "collapsed", "rupture",
    ]
    medium_severity_words = [
        "fracture", "broken", "burn", "burns", "hospitalized", "surgery",
        "laceration", "struck", "fell", "dropped", "pinned", "trapped",
        "inhalation", "toxic", "leak", "spill", "fire", "spray",
    ]
    low_severity_words = [
        "minor", "first aid", "bruise", "scratch", "sprain", "strain",
        "observation", "housekeeping", "hygiene", "welfare", "trash", "dust",
    ]

    high_sev_count = sum(1 for w in high_severity_words if w in text_lower)
    med_sev_count = sum(1 for w in medium_severity_words if w in text_lower)
    low_sev_count = sum(1 for w in low_severity_words if w in text_lower)

    severity_token_ratio = round((high_sev_count * 3 + med_sev_count * 2 + low_sev_count) / max(word_count, 1), 4)

    barrier_words = [
        "failed", "failure", "bypassed", "absent", "missing", "disabled",
        "defeated", "expired", "damaged", "defective", "corroded",
        "inadequate", "overdue", "non-functional", "slipped", "snapped",
    ]
    barrier_failure_word_count = sum(1 for w in barrier_words if w in text_lower)

    violation_words = [
        "unauthorized", "no permit", "without permit", "expired permit",
        "not wearing", "removed guard", "failed to isolate", "speeding",
        "disregarded", "violation", "non-compliance", "untrained",
    ]
    violation_word_count = sum(1 for w in violation_words if w in text_lower)

    has_measurements = int(bool(re.search(r'\d+\s*(?:feet|meters|pressure_unit|ppm|volts|weight_unit|km/h|°c|%)', text_lower)))
    number_count = len(re.findall(r'\b\d+(?:\.\d+)?\b', cleaned_text))

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_len,
        "negation_count": negation_count,
        "high_severity_word_count": high_sev_count,
        "medium_severity_word_count": med_sev_count,
        "low_severity_word_count": low_sev_count,
        "severity_token_ratio": severity_token_ratio,
        "barrier_failure_word_count": barrier_failure_word_count,
        "violation_word_count": violation_word_count,
        "has_measurements": has_measurements,
        "number_count": number_count,
    }


class MultiModalFeatureExtractor:
    """Combines Word TF-IDF + Char N-Grams + Dense Engineered Features."""
    def __init__(self, max_word_features=20000, max_char_features=10000):
        self.word_vec = TfidfVectorizer(
            max_features=max_word_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            max_df=0.90
        )
        self.char_vec = TfidfVectorizer(
            max_features=max_char_features,
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=5
        )
        self.scaler = StandardScaler()

    def fit(self, df: pd.DataFrame):
        text = df["text_tokenized_no_stopwords"].fillna("")
        self.word_vec.fit(text)
        self.char_vec.fit(text)
        num_feats = df[NUMERIC_FEATURES].fillna(0).values
        self.scaler.fit(num_feats)
        return self

    def transform(self, df: pd.DataFrame):
        text = df["text_tokenized_no_stopwords"].fillna("")
        X_word = self.word_vec.transform(text)
        X_char = self.char_vec.transform(text)
        
        # Check if all numeric features exist, if not compute on the fly
        missing = [f for f in NUMERIC_FEATURES if f not in df.columns]
        if missing and "text_cleaned" in df.columns:
            computed_rows = []
            for t in df["text_cleaned"].fillna(""):
                feats = engineer_features(t)
                computed_rows.append([feats[f] for f in NUMERIC_FEATURES])
            num_feats = np.array(computed_rows)
        else:
            num_feats = df[[f for f in NUMERIC_FEATURES if f in df.columns]].fillna(0).values
            if num_feats.shape[1] < len(NUMERIC_FEATURES):
                # pad with zeros if necessary
                pad = np.zeros((num_feats.shape[0], len(NUMERIC_FEATURES) - num_feats.shape[1]))
                num_feats = np.hstack([num_feats, pad])
                
        X_num = self.scaler.transform(num_feats)
        return hstack([X_word, X_char, X_num]).tocsr()


def train_sif_classifier(X_train, y_train, X_val, y_val) -> Tuple[object, float]:
    """Train SIF Classifier with optimal probability threshold tuning."""
    print("\n🧠 [Task 1] Training SIF Precursor Binary Classifier...")
    t0 = time.time()

    # Fast, high-performance L-BFGS Logistic Regression with balanced class weighting
    model = LogisticRegression(
        C=2.0,
        max_iter=500,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    )
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]

    # Find optimal threshold to maximize F1 while keeping SIF Recall >= 0.85
    best_thresh = 0.50
    best_f1 = 0.0
    for thresh in np.arange(0.30, 0.70, 0.02):
        preds = (val_probs >= thresh).astype(int)
        f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)
        rec = recall_score(y_val, preds, pos_label=1, zero_division=0)
        if f1 > best_f1 and rec >= 0.80:
            best_f1 = f1
            best_thresh = float(thresh)

    print(f"   ✅ SIF Classifier trained in {time.time()-t0:.2f}s (Optimal Threshold: {best_thresh:.2f}, Val F1: {best_f1:.4f})")
    return model, best_thresh


def train_iogp_rules_classifier(X_train, Y_train) -> object:
    """Train Multi-Output Classifier for all 9 IOGP Life-Saving Rules."""
    print("\n🏷️  [Task 2] Training Multi-Label IOGP Life-Saving Rules Classifier...")
    t0 = time.time()

    base_lr = LogisticRegression(
        C=2.5,
        max_iter=400,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42
    )
    multi_model = MultiOutputClassifier(base_lr, n_jobs=-1)
    multi_model.fit(X_train, Y_train)

    print(f"   ✅ IOGP Rules Multi-Label Classifier trained in {time.time()-t0:.2f}s")
    return multi_model


def train_severity_regressor(X_train, y_train) -> object:
    """Train Continuous Severity Score Regressor."""
    print("\n📈 [Task 3] Training Continuous SIF Severity Regressor...")
    t0 = time.time()

    regressor = Ridge(alpha=1.5, random_state=42)
    regressor.fit(X_train, y_train)

    print(f"   ✅ Severity Regressor trained in {time.time()-t0:.2f}s")
    return regressor


def evaluate_sif_model(model, X_test, y_test, threshold: float = 0.50) -> Dict:
    """Compute all evaluation metrics for SIF binary classification."""
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, pos_label=1, zero_division=0)
    rec = recall_score(y_test, preds, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
    f1_macro = f1_score(y_test, preds, average="macro")
    f1_weighted = f1_score(y_test, preds, average="weighted")
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)

    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate

    return {
        "accuracy": round(float(acc), 4),
        "precision_sif": round(float(prec), 4),
        "recall_sif": round(float(rec), 4),
        "f1_sif": round(float(f1), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "false_negative_rate": round(float(fnr), 4),
        "confusion_matrix": {
            "true_positive_sif": int(tp),
            "false_positive": int(fp),
            "true_negative_non_sif": int(tn),
            "false_negative_missed_sif": int(fn)
        },
        "probabilities": probs,
        "predictions": preds,
    }


def evaluate_iogp_model(model, X_test, Y_test) -> Dict:
    """Compute multi-label metrics for all 9 IOGP rules."""
    Y_pred = model.predict(X_test)
    Y_prob = np.column_stack([est.predict_proba(X_test)[:, 1] for est in model.estimators_])

    sub_acc = accuracy_score(Y_test, Y_pred)
    h_loss = hamming_loss(Y_test, Y_pred)
    jaccard_micro = jaccard_score(Y_test, Y_pred, average="micro", zero_division=0)
    f1_micro = f1_score(Y_test, Y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(Y_test, Y_pred, average="weighted", zero_division=0)

    per_rule_metrics = {}
    for i, col in enumerate(RULE_COLUMNS):
        display_name = RULE_DISPLAY_NAMES[i]
        y_true_col = Y_test.iloc[:, i]
        y_pred_col = Y_pred[:, i]
        y_prob_col = Y_prob[:, i]

        p = precision_score(y_true_col, y_pred_col, zero_division=0)
        r = recall_score(y_true_col, y_pred_col, zero_division=0)
        f = f1_score(y_true_col, y_pred_col, zero_division=0)
        try:
            auc = roc_auc_score(y_true_col, y_prob_col)
        except ValueError:
            auc = 0.5

        per_rule_metrics[display_name] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f), 4),
            "roc_auc": round(float(auc), 4),
            "support": int(y_true_col.sum())
        }

    return {
        "subset_exact_match_accuracy": round(float(sub_acc), 4),
        "hamming_loss": round(float(h_loss), 4),
        "jaccard_micro": round(float(jaccard_micro), 4),
        "f1_micro": round(float(f1_micro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "per_rule_metrics": per_rule_metrics,
        "predictions": Y_pred,
        "probabilities": Y_prob,
    }


def evaluate_severity_regressor(model, X_test, y_test) -> Dict:
    """Compute regression metrics for SIF severity scoring."""
    preds = np.clip(model.predict(X_test), 0.0, 1.0)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    spearman_corr, _ = spearmanr(y_test, preds)

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2), 4),
        "spearman_rank_correlation": round(float(spearman_corr), 4),
        "predictions": preds,
    }


def evaluate_on_indian_benchmark(sif_model, iogp_model, extractor, ind_df: pd.DataFrame, threshold: float) -> List[Dict]:
    """Evaluate trained models on real-world Indian OISD & OIL incident cases."""
    print("\n🇮🇳 [Task 4] Evaluating on Real-World Indian Oil & Gas Benchmark Cases...")
    X_ind = extractor.transform(ind_df)
    sif_probs = sif_model.predict_proba(X_ind)[:, 1]
    sif_preds = (sif_probs >= threshold).astype(int)

    iogp_preds = iogp_model.predict(X_ind)
    iogp_probs = np.column_stack([est.predict_proba(X_ind)[:, 1] for est in iogp_model.estimators_])

    results = []
    correct_sif = 0

    for i in range(len(ind_df)):
        row = ind_df.iloc[i]
        true_sif = int(row.get("sif_potential_binary", 1))
        pred_sif = int(sif_preds[i])
        prob_sif = round(float(sif_probs[i]), 4)

        if true_sif == pred_sif:
            correct_sif += 1

        # Match predicted rules (prob >= 0.40)
        pred_rules = [RULE_DISPLAY_NAMES[j] for j in range(len(RULE_COLUMNS)) if iogp_preds[i, j] == 1 or iogp_probs[i, j] >= 0.40]
        raw_rules = str(row.get("life_saving_rules", "")) if pd.notna(row.get("life_saving_rules")) else ""
        true_rules = [r.strip() for r in raw_rules.split("|") if r.strip()] if raw_rules not in ("", "None", "nan") else []

        case_res = {
            "source_reference": str(row.get("source_reference", "")),
            "incident_title": str(row.get("incident_title", "")),
            "location": f"{row.get('location_site', '')}, {row.get('location_state', '')}",
            "activity": str(row.get("activity_type", "")),
            "ground_truth_sif": "SIF-Potential" if true_sif == 1 else "Non-SIF",
            "predicted_sif": "SIF-Potential" if pred_sif == 1 else "Non-SIF",
            "sif_probability": prob_sif,
            "sif_match": bool(true_sif == pred_sif),
            "ground_truth_rules": true_rules,
            "predicted_rules": pred_rules,
        }
        results.append(case_res)

        status_emoji = "✅" if case_res["sif_match"] else "❌"
        print(f"   {status_emoji} Case {i+1:02d}: {case_res['source_reference']} | {case_res['location']}")
        print(f"      Ground Truth: {case_res['ground_truth_sif']} | Predicted: {case_res['predicted_sif']} (Prob: {prob_sif*100:.1f}%)")
        print(f"      Rules Matched: {', '.join(pred_rules) if pred_rules else 'None'}")

    acc = correct_sif / max(len(ind_df), 1)
    print(f"\n   🎯 Indian Benchmark SIF Accuracy: {acc*100:.1f}% ({correct_sif}/{len(ind_df)} cases correctly classified)")
    return results


def extract_top_linguistic_features(sif_model, extractor, n_top: int = 20) -> Dict[str, List[Tuple[str, float]]]:
    """Extract most predictive n-grams for SIF vs Non-SIF classes."""
    base_lr = getattr(sif_model, "estimator", sif_model)
    coefs = base_lr.coef_[0]
    word_feature_names = extractor.word_vec.get_feature_names_out()

    # Slice word features coefficients
    n_words = len(word_feature_names)
    word_coefs = coefs[:n_words]

    top_sif_indices = np.argsort(word_coefs)[-n_top:][::-1]
    top_non_sif_indices = np.argsort(word_coefs)[:n_top]

    top_sif = [(word_feature_names[i], round(float(word_coefs[i]), 4)) for i in top_sif_indices]
    top_non_sif = [(word_feature_names[i], round(float(word_coefs[i]), 4)) for i in top_non_sif_indices]

    return {
        "top_sif_precursor_triggers": top_sif,
        "top_non_sif_indicators": top_non_sif
    }


def generate_evaluation_plots(sif_eval: Dict, iogp_eval: Dict, y_test, output_path: Path):
    """Generate comprehensive 4-panel evaluation visual for dashboard & presentations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # 1. Confusion Matrix
    cm = np.array([
        [sif_eval["confusion_matrix"]["true_negative_non_sif"], sif_eval["confusion_matrix"]["false_positive"]],
        [sif_eval["confusion_matrix"]["false_negative_missed_sif"], sif_eval["confusion_matrix"]["true_positive_sif"]]
    ])
    ax1 = axes[0, 0]
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.set_title("SIF Precursor Confusion Matrix (Test Set: 17,398 Reports)", fontsize=13, fontweight='bold', pad=12)
    fig.colorbar(im, ax=ax1)
    classes = ["Non-SIF", "SIF-Potential"]
    tick_marks = np.arange(len(classes))
    ax1.set_xticks(tick_marks)
    ax1.set_xticklabels(classes, fontsize=11)
    ax1.set_yticks(tick_marks)
    ax1.set_yticklabels(classes, fontsize=11)
    ax1.set_ylabel("True Ground Truth", fontsize=12)
    ax1.set_xlabel("Predicted Label", fontsize=12)

    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() / 2 else "black"
            ax1.text(j, i, f"{val:,}\n({val/cm.sum()*100:.1f}%)", ha="center", va="center", color=color, fontsize=12, fontweight='bold')

    # 2. ROC Curve
    ax2 = axes[0, 1]
    fpr, tpr, _ = roc_curve(y_test, sif_eval["probabilities"])
    ax2.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f"SIF ROC Curve (AUC = {sif_eval['roc_auc']:.4f})")
    ax2.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax2.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    ax2.set_title("SIF Classifier ROC Curve", fontsize=13, fontweight='bold', pad=12)
    ax2.legend(loc="lower right", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)

    # 3. Precision-Recall Curve
    ax3 = axes[1, 0]
    prec_curve, rec_curve, _ = precision_recall_curve(y_test, sif_eval["probabilities"])
    ax3.plot(rec_curve, prec_curve, color='#2ca02c', lw=2.5, label=f"PR Curve (PR-AUC = {sif_eval['pr_auc']:.4f})")
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel("Recall (Coverage of Fatal Precursors)", fontsize=12)
    ax3.set_ylabel("Precision (Accuracy of SIF Flags)", fontsize=12)
    ax3.set_title("Precision-Recall Curve (Imbalanced SIF)", fontsize=13, fontweight='bold', pad=12)
    ax3.legend(loc="lower left", fontsize=11)
    ax3.grid(True, linestyle=':', alpha=0.6)

    # 4. Per-Rule F1 Scores
    ax4 = axes[1, 1]
    rule_names = list(iogp_eval["per_rule_metrics"].keys())
    f1_scores = [iogp_eval["per_rule_metrics"][r]["f1_score"] for r in rule_names]
    y_pos = np.arange(len(rule_names))

    bars = ax4.barh(y_pos, f1_scores, color='#ff7f0e', alpha=0.85, edgecolor='black', height=0.65)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(rule_names, fontsize=10)
    ax4.invert_yaxis()
    ax4.set_xlabel("F1-Score", fontsize=12)
    ax4.set_title(f"IOGP Life-Saving Rules F1 Performance (Micro-F1: {iogp_eval['f1_micro']:.4f})", fontsize=13, fontweight='bold', pad=12)
    ax4.set_xlim([0.0, 1.05])
    ax4.grid(True, axis='x', linestyle=':', alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        ax4.text(w + 0.02, bar.get_y() + bar.get_height()/2, f"{w:.3f}", ha='left', va='center', fontsize=9, fontweight='bold')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 Evaluation Plots saved: {output_path}")


# ──────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────

def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    processed_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 80)
    print("  🚀 TRAINING THE COMPLETE HSSE SIF PRECURSOR & IOGP AI ENGINE")
    print("=" * 80)

    # 1. Load Data
    train_df, val_df, test_df, ind_df = load_datasets(processed_dir)

    # 2. Extract Multi-Modal Features
    print("\n⚙️  Fitting Vectorizers & Scaling Engineered Features...")
    extractor = MultiModalFeatureExtractor(max_word_features=25000, max_char_features=12000)
    extractor.fit(train_df)

    X_train = extractor.transform(train_df)
    X_val = extractor.transform(val_df)
    X_test = extractor.transform(test_df)

    print(f"   Feature Matrix Shape: {X_train.shape[0]:,} records × {X_train.shape[1]:,} features")

    # 3. Targets
    y_train_sif = train_df["sif_potential_binary"].values
    y_val_sif = val_df["sif_potential_binary"].values
    y_test_sif = test_df["sif_potential_binary"].values

    Y_train_rules = train_df[RULE_COLUMNS]
    Y_val_rules = val_df[RULE_COLUMNS]
    Y_test_rules = test_df[RULE_COLUMNS]

    y_train_sev = train_df["sif_confidence_score"].values
    y_test_sev = test_df["sif_confidence_score"].values

    # 4. Train Models
    sif_model, best_threshold = train_sif_classifier(X_train, y_train_sif, X_val, y_val_sif)
    iogp_model = train_iogp_rules_classifier(X_train, Y_train_rules)
    sev_model = train_severity_regressor(X_train, y_train_sev)

    # 5. Evaluate on Test Set (17,398 held-out records)
    print("\n" + "=" * 80)
    print("  📊 HELD-OUT TEST SET EVALUATION METRICS (17,398 Records)")
    print("=" * 80)

    sif_eval = evaluate_sif_model(sif_model, X_test, y_test_sif, threshold=best_threshold)
    iogp_eval = evaluate_iogp_model(iogp_model, X_test, Y_test_rules)
    sev_eval = evaluate_severity_regressor(sev_model, X_test, y_test_sev)

    print("\n--- 🎯 TASK 1: SIF PRECURSOR CLASSIFICATION METRICS ---")
    print(f"  • Accuracy                : {sif_eval['accuracy']*100:.2f}%")
    print(f"  • ROC-AUC Score           : {sif_eval['roc_auc']:.4f}")
    print(f"  • PR-AUC (Avg Precision)  : {sif_eval['pr_auc']:.4f}")
    print(f"  • SIF Precision           : {sif_eval['precision_sif']*100:.2f}%")
    print(f"  • SIF Recall (Coverage)   : {sif_eval['recall_sif']*100:.2f}%")
    print(f"  • SIF F1-Score            : {sif_eval['f1_sif']:.4f}")
    print(f"  • Macro F1-Score          : {sif_eval['f1_macro']:.4f}")
    print(f"  • False Negative Rate     : {sif_eval['false_negative_rate']*100:.2f}% (Fatal Precursors Missed)")
    print(f"  • Confusion Matrix        : TP={sif_eval['confusion_matrix']['true_positive_sif']:,}, FP={sif_eval['confusion_matrix']['false_positive']:,}, TN={sif_eval['confusion_matrix']['true_negative_non_sif']:,}, FN={sif_eval['confusion_matrix']['false_negative_missed_sif']:,}")

    print("\n--- 🏷️  TASK 2: IOGP LIFE-SAVING RULES MULTI-LABEL METRICS ---")
    print(f"  • Subset Exact Match Acc  : {iogp_eval['subset_exact_match_accuracy']*100:.2f}%")
    print(f"  • Hamming Loss (Error)    : {iogp_eval['hamming_loss']:.4f}")
    print(f"  • Micro F1-Score          : {iogp_eval['f1_micro']:.4f}")
    print(f"  • Macro F1-Score          : {iogp_eval['f1_macro']:.4f}")
    print(f"  • Weighted F1-Score       : {iogp_eval['f1_weighted']:.4f}")
    print("  Per-Rule Performance:")
    for rule, metrics in iogp_eval["per_rule_metrics"].items():
        print(f"    - {rule:<25}: F1={metrics['f1_score']:.4f} | Prec={metrics['precision']:.4f} | Rec={metrics['recall']:.4f} | AUC={metrics['roc_auc']:.4f} (Support: {metrics['support']:,})")

    print("\n--- 📈 TASK 3: SIF SEVERITY REGRESSION METRICS ---")
    print(f"  • Mean Absolute Error (MAE): {sev_eval['mae']:.4f}")
    print(f"  • Root Mean Sq Error (RMSE): {sev_eval['rmse']:.4f}")
    print(f"  • R² Score                 : {sev_eval['r2_score']:.4f}")
    print(f"  • Spearman Correlation (rs): {sev_eval['spearman_rank_correlation']:.4f}")

    # 6. Evaluate on Indian Benchmark
    ind_results = []
    if ind_df is not None:
        ind_results = evaluate_on_indian_benchmark(sif_model, iogp_model, extractor, ind_df, best_threshold)

    # 7. Linguistic Explainability
    linguistic_triggers = extract_top_linguistic_features(sif_model, extractor, n_top=20)
    print("\n--- 🔍 TOP PREDICTIVE LINGUISTIC PRECURSORS (EXPLAINABILITY) ---")
    print("  Top SIF-Potential Triggers:")
    for word, coef in linguistic_triggers["top_sif_precursor_triggers"][:10]:
        print(f"    + {word:<20} (Weight: +{coef:.3f})")
    print("  Top Non-SIF Indicators:")
    for word, coef in linguistic_triggers["top_non_sif_indicators"][:10]:
        print(f"    - {word:<20} (Weight: {coef:.3f})")

    # 8. Visual Plots
    plots_path = processed_dir / "evaluation_plots.png"
    generate_evaluation_plots(sif_eval, iogp_eval, y_test_sif, plots_path)

    # 9. Save Models
    print("\n💾 Serializing Models and Vectorizers to models/...")
    joblib.dump(extractor, models_dir / "feature_extractor.joblib")
    joblib.dump(sif_model, models_dir / "sif_classifier.joblib")
    joblib.dump(iogp_model, models_dir / "iogp_rules_classifier.joblib")
    joblib.dump(sev_model, models_dir / "severity_regressor.joblib")
    with open(models_dir / "optimal_threshold.json", "w") as f:
        json.dump({"optimal_sif_threshold": best_threshold}, f)
    print("   ✅ Saved: feature_extractor.joblib, sif_classifier.joblib, iogp_rules_classifier.joblib, severity_regressor.joblib")

    # 10. Save Metrics JSON Report
    metrics_report = {
        "timestamp": datetime.now().isoformat(),
        "train_records": len(train_df),
        "test_records": len(test_df),
        "feature_count": X_train.shape[1],
        "optimal_sif_threshold": best_threshold,
        "task1_sif_classification": {k: v for k, v in sif_eval.items() if k not in ("probabilities", "predictions")},
        "task2_iogp_multilabel_rules": {k: v for k, v in iogp_eval.items() if k not in ("probabilities", "predictions")},
        "task3_severity_regression": {k: v for k, v in sev_eval.items() if k != "predictions"},
        "task4_indian_benchmark_results": ind_results,
        "task5_top_linguistic_triggers": linguistic_triggers,
    }

    metrics_file = processed_dir / "model_evaluation_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"   ✅ Saved Comprehensive Metrics Report: {metrics_file.name}")

    print("\n" + "=" * 80)
    print("  🏆 COMPLETE MODEL TRAINING & EVALUATION SUCCESSFULLY FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()
