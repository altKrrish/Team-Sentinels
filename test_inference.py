#!/usr/bin/env python3
"""
Quick Inference Tester for the Trained SIF Engine
==================================================
Usage:
  python test_inference.py
  python test_inference.py "During workover at DS-14, floorman was working under suspended casing without safety harness."
"""

import json
import re
import sys
import unicodedata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.models.train_sif_engine import MultiModalFeatureExtractor, engineer_features

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


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_text(text: str) -> str:
    tokens = re.findall(r'[a-z0-9_]+(?:-[a-z0-9_]+)*', text.lower())
    return " ".join(tokens)


def predict_report(narrative: str, models_dir: Path):
    extractor = joblib.load(models_dir / "feature_extractor.joblib")
    sif_model = joblib.load(models_dir / "sif_classifier.joblib")
    iogp_model = joblib.load(models_dir / "iogp_rules_classifier.joblib")
    sev_model = joblib.load(models_dir / "severity_regressor.joblib")

    with open(models_dir / "optimal_threshold.json") as f:
        threshold = json.load(f)["optimal_sif_threshold"]

    cleaned = clean_text(narrative)
    tokenized = tokenize_text(cleaned)
    df_sample = pd.DataFrame([{
        "text_cleaned": cleaned,
        "text_tokenized_no_stopwords": tokenized
    }])

    X = extractor.transform(df_sample)

    sif_prob = float(sif_model.predict_proba(X)[0, 1])
    is_sif = bool(sif_prob >= threshold)
    sev_score = float(np.clip(sev_model.predict(X)[0], 0.0, 1.0))

    rule_probs = [float(est.predict_proba(X)[0, 1]) for est in iogp_model.estimators_]
    tagged_rules = [
        (RULE_DISPLAY_NAMES[i], round(rule_probs[i] * 100, 1))
        for i in range(len(RULE_DISPLAY_NAMES))
        if rule_probs[i] >= 0.40
    ]
    tagged_rules.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 70)
    print("  📋 AI/NLP SIF PRECURSOR INFERENCE RESULT")
    print("=" * 70)
    print(f"  Input Narrative : {narrative}")
    print("─" * 70)
    status_str = "🚨 SIF-POTENTIAL PRECURSOR" if is_sif else "✅ NON-SIF OBSERVATION"
    print(f"  Classification  : {status_str}")
    print(f"  SIF Probability : {sif_prob*100:.2f}% (Threshold: {threshold*100:.0f}%)")
    print(f"  Severity Score  : {sev_score:.4f} / 1.0000")
    print("─" * 70)
    print("  Tagged IOGP Life-Saving Rules:")
    if tagged_rules:
        for rule, prob in tagged_rules:
            print(f"    • {rule:<30} (Confidence: {prob}%)")
    else:
        print("    • None (General observation)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    models_path = Path(__file__).resolve().parent / "models"
    if len(sys.argv) > 1:
        sample_text = " ".join(sys.argv[1:])
    else:
        sample_text = (
            "During tripping operations at Drilling Site DS-14 near Duliajan, "
            "the roughneck stood directly under a 2-ton suspended drill pipe rack "
            "while the hydraulic winch line showed visible fraying without LOTO isolation."
        )
    predict_report(sample_text, models_path)
