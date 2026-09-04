#!/usr/bin/env python3
"""
Hardened Inference Engine for SIF Precursor Detection & IOGP Life-Saving Rules
=============================================================================
Combines:
  1. Multilingual text normalization (Devanagari, Bengali, Assamese, Hinglish)
  2. Deterministic safety interlocks (negation-scoped, zero-tolerance tripwires)
  3. Industrial standards physical energy assessment (OSHA 1910.28, NFPA 70E, API RP 500)
  4. Asset-class aware threshold routing (tau = 0.40 high-energy vs 0.44 default)
  5. Calibrated soft-voting ML ensemble (45,013-dimensional MultiModal pipeline)
  6. Multi-label IOGP Life-Saving Rules cross-corroboration
  7. Continuous severity regression index
  8. Ingest-time non-blocking form guidance

Usage:
  python test_inference.py
  python test_inference.py "Roughneck working under 2-ton suspended drill pipe without LOTO."
  python test_inference.py --text "Gas leak near pump" --asset wellhead --pressure 120
  python test_inference.py --json "Electric shock from live switchgear."
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

# Add repo root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data.preprocess_pipeline import clean_text, tokenize_for_nlp
from sentinel import decision_policy, energy_metadata, form_guidance, interlock
from src.models.train_sif_engine import MultiModalFeatureExtractor, engineer_features

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

RULE_MAP = dict(zip(RULE_COLUMNS, RULE_DISPLAY_NAMES))

INTERLOCK_RULE_CORROBORATION = {
    "suspended_load": ["Safe Mechanical Lifting", "Line of Fire"],
    "live_contact": ["Energy Isolation", "Bypassing Safety Controls"],
    "arc_flash": ["Energy Isolation"],
    "fall_from_height": ["Working at Height"],
    "caught_between": ["Line of Fire"],
    "stored_mechanical": ["Energy Isolation", "Line of Fire"],
    "blowout": ["Energy Isolation", "Line of Fire"],
    "uncontrolled_release": ["Energy Isolation", "Line of Fire"],
    "fire_explosion": ["Hot Work"],
    "hot_work_uncontrolled": ["Hot Work", "Work Authorization"],
    "h2s_exposure": ["Confined Space", "Work Authorization"],
    "confined_space": ["Confined Space", "Work Authorization"],
    "vehicle_incident": ["Driving"],
}


def run_inference(
    narrative: str,
    models_dir: Path,
    asset_class: str = "drilling_rig",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    metadata = metadata or {}

    # 1. Ingest-time Form Guidance (non-blocking)
    guidance = form_guidance.evaluate(narrative)

    # 2. Deterministic Safety Interlock Scan
    il_result = interlock.scan(narrative)

    # 3. Industrial Energy Metadata Assessment
    meta_result = energy_metadata.assess(metadata)

    # 4. Multilingual text cleaning & tokenization
    cleaned = clean_text(narrative)
    tokenized = tokenize_for_nlp(cleaned, remove_stopwords=True)
    df_sample = pd.DataFrame([{
        "text_cleaned": cleaned,
        "text_tokenized_no_stopwords": tokenized
    }])

    # 5. ML Pipeline Prediction
    extractor = joblib.load(models_dir / "feature_extractor.joblib")
    sif_model = joblib.load(models_dir / "sif_classifier.joblib")
    iogp_model = joblib.load(models_dir / "iogp_rules_classifier.joblib")
    sev_model = joblib.load(models_dir / "severity_regressor.joblib")

    with open(models_dir / "optimal_threshold.json", encoding="utf-8") as f:
        threshold_data = json.load(f)

    rule_thresholds = threshold_data.get("rule_thresholds", {})

    X = extractor.transform(df_sample)

    sif_prob = float(sif_model.predict_proba(X)[0, 1])
    sev_score = float(np.clip(sev_model.predict(X)[0], 0.0, 1.0))

    # 6. Hardened Decision Policy
    decision = decision_policy.decide(
        sif_prob,
        interlock=il_result,
        metadata=meta_result,
        asset_class=asset_class,
    )

    # 7. IOGP Life-Saving Rules Tagging + Interlock Corroboration
    rule_probs = [float(est.predict_proba(X)[0, 1]) for est in iogp_model.estimators_]
    tagged_rules_dict: Dict[str, float] = {}

    # From ML model
    for i in range(len(RULE_DISPLAY_NAMES)):
        r_col = RULE_COLUMNS[i]
        r_name = RULE_DISPLAY_NAMES[i]
        r_thresh = rule_thresholds.get(r_col, 0.40)
        prob_pct = round(rule_probs[i] * 100, 1)
        if rule_probs[i] >= r_thresh:
            tagged_rules_dict[r_name] = prob_pct

    # Corroborate with interlock matches
    for m in il_result.matches:
        corroborated = INTERLOCK_RULE_CORROBORATION.get(m.canonical, [])
        for rule_name in corroborated:
            current = tagged_rules_dict.get(rule_name, 0.0)
            tagged_rules_dict[rule_name] = max(current, 95.0)

    tagged_rules = sorted(tagged_rules_dict.items(), key=lambda x: x[1], reverse=True)

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "narrative": narrative,
        "cleaned_text": cleaned,
        "asset_class": asset_class,
        "decision_label": decision.label,
        "decision_route": decision.route.value,
        "decision_reason": decision.reason,
        "tau_used": decision.tau_used,
        "sif_probability": sif_prob,
        "severity_score": sev_score,
        "interlock_fired": il_result.fired,
        "energy_classes_hit": sorted(list(il_result.energy_classes_hit)),
        "interlock_matches": [
            {
                "canonical": m.canonical,
                "tier": m.tier,
                "energy_class": m.energy_class,
                "matched_surface": m.matched_surface,
                "matched_span": m.matched_span_text,
                "negated": m.negated,
            }
            for m in il_result.matches
        ],
        "metadata_triggered": meta_result.any_triggered,
        "metadata_classes": sorted(list(meta_result.energy_classes)),
        "metadata_reasons": [
            f"{s.field_used} = {s.value} exceeds {s.threshold} ({s.standard_ref})"
            for s in meta_result.signals if s.triggered
        ],
        "tagged_iogp_rules": tagged_rules,
        "form_guidance": {
            "meets_minimum": guidance.meets_minimum,
            "needs_detail": guidance.needs_detail,
            "word_count": guidance.word_count,
            "missing_slots": guidance.missing_slots,
            "suggestions": guidance.prompts,
        },
        "latency_ms": latency_ms,
    }


def print_formatted_report(res: Dict[str, Any]):
    print("\n" + "=" * 76)
    print("  🛡️  SENTINEL HARDENED SIF PRECURSOR INFERENCE ENGINE (v2.1 OIL)")
    print("=" * 76)
    print(f"  Input Narrative : {res['narrative']}")
    if res['cleaned_text'] != res['narrative']:
        print(f"  Normalized Text : {res['cleaned_text']}")
    print(f"  Asset Class     : {res['asset_class']} (Calibrated tau: {res['tau_used']:.3f})")
    print("─" * 76)

    # Classification Banner
    if res['decision_label'] == "SIF":
        if res['interlock_fired']:
            banner = "🚨 SIF-POTENTIAL PRECURSOR [HARDENED INTERLOCK OVERRIDE]"
        elif res['metadata_triggered']:
            banner = "🚨 SIF-POTENTIAL PRECURSOR [STANDARDS METADATA BREACH]"
        else:
            banner = "🚨 SIF-POTENTIAL PRECURSOR [STATISTICAL CLASSIFIER]"
    else:
        if res['decision_route'] == "HUMAN_REVIEW":
            banner = "⚠️ ROUTED TO HUMAN SAFETY REVIEW [CONFIDENCE BAND]"
        else:
            banner = "✅ NON-SIF OBSERVATION [SAFE/ROUTINE]"

    print(f"  Final Decision  : {banner}")
    print(f"  Precedence Route: {res['decision_route'].upper()}")
    print(f"  Audit Reason    : {res['decision_reason']}")
    print(f"  Model SIF Prob  : {res['sif_probability']*100:.2f}% (Threshold: {res['tau_used']*100:.1f}%)")
    print(f"  Severity Score  : {res['severity_score']:.4f} / 1.0000")
    print("─" * 76)

    # Energy Classes & Interlock
    if res['energy_classes_hit'] or res['metadata_classes']:
        all_ec = sorted(list(set(res['energy_classes_hit'] + res['metadata_classes'])))
        print(f"  ⚡ Energy Sources : {', '.join(all_ec)}")
    if res['interlock_matches']:
        print("  🔍 Interlock Hits :")
        for m in res['interlock_matches']:
            print(f"     • [{m['tier']}] {m['canonical']} ({m['energy_class']}) -> '{m['matched_span']}'")
    if res['metadata_reasons']:
        print("  📏 Sourced Breaches:")
        for r in res['metadata_reasons']:
            print(f"     • {r}")

    print("─" * 76)
    print("  🏷️  Tagged IOGP Life-Saving Rules:")
    if res['tagged_iogp_rules']:
        for rule, prob in res['tagged_iogp_rules']:
            print(f"     • {rule:<32} (Confidence: {prob}%)")
    else:
        print("     • None (General observation / non-SIF)")

    # Form Guidance
    fg = res['form_guidance']
    if fg['needs_detail'] or fg['suggestions']:
        print("─" * 76)
        print("  📝 Form Guidance Alerts (Non-blocking):")
        if not fg['meets_minimum']:
            print(f"     • Sparse report warning: only {fg['word_count']} words (recommended: >= 8).")
        for s in fg['suggestions']:
            print(f"     • Prompt: {s}")

    print("─" * 76)
    print(f"  ⏱️  Inference Latency: {res['latency_ms']} ms")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Sentinel Hardened SIF Inference Engine")
    parser.add_argument("narrative", nargs="*", default=None, help="Incident report narrative")
    parser.add_argument("--text", type=str, default=None, help="Incident narrative text")
    parser.add_argument("--asset", type=str, default="drilling_rig", help="Asset class (e.g. drilling_rig, wellhead, ctf)")
    parser.add_argument("--height", type=float, default=None, help="Fall height in meters")
    parser.add_argument("--voltage", type=float, default=None, help="Voltage in volts")
    parser.add_argument("--pressure", type=float, default=None, help="Pressure in psi")
    parser.add_argument("--h2s", type=float, default=None, help="H2S gas in ppm")
    parser.add_argument("--weight", type=float, default=None, help="Suspended load in kg")
    parser.add_argument("--json-metadata", type=str, default=None, help="Raw metadata as JSON string")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--models-dir", type=str, default=None, help="Custom models directory path")

    args = parser.parse_args()

    # Determine text
    if args.text:
        narrative = args.text
    elif args.narrative:
        narrative = " ".join(args.narrative)
    else:
        narrative = (
            "During tripping operations at Drilling Site DS-14 near Duliajan, "
            "the roughneck stood directly under a 2-ton suspended drill pipe rack "
            "while the hydraulic winch line showed visible fraying without LOTO isolation."
        )

    # Determine metadata
    metadata = {}
    if args.json_metadata:
        try:
            metadata.update(json.loads(args.json_metadata))
        except Exception as e:
            print(f"Warning: Failed to parse --json-metadata: {e}", file=sys.stderr)
    if args.height is not None:
        metadata["working_height_m"] = args.height
    if args.voltage is not None:
        metadata["voltage_v"] = args.voltage
    if args.pressure is not None:
        metadata["pressure_psi"] = args.pressure
    if args.h2s is not None:
        metadata["h2s_ppm"] = args.h2s
    if args.weight is not None:
        metadata["suspended_weight_kg"] = args.weight

    models_path = Path(args.models_dir) if args.models_dir else (Path(__file__).resolve().parent / "models")

    res = run_inference(
        narrative=narrative,
        models_dir=models_path,
        asset_class=args.asset,
        metadata=metadata,
    )

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print_formatted_report(res)


if __name__ == "__main__":
    main()
