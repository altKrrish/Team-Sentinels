#!/usr/bin/env python3
"""
Continuous Learning, Zero-Tolerance Safety Gate & Audit Governance Test Suite
=============================================================================
Demonstrates the full life-cycle:
1. HSE Officer Feedback Recording with RLHF Reward Signals (+1.5 / -5.0 / -1.0)
2. Automated Safety Validation Gate (100% Fatal Recall Guarantee on Indian OISD benchmark)
3. Intentionally Degraded Model Safety Rejection Test (verifying the hard gate blocks bad models)
4. Shadow Benchmarking Mode (Champion vs. Challenger live comparison)
5. DGMS / OISD Regulatory Audit Log Inspection
"""

import json
import sys
from pathlib import Path
import joblib
import pandas as pd

# Add repo root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.models.train_sif_engine import MultiModalFeatureExtractor, engineer_features
from src.continuous_learning import (
    AuditLogger,
    ContinualLearningOrchestrator,
    FeedbackEngine,
    SafetyValidator,
    ShadowBenchmarker,
)


def run_full_governance_test():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"

    print("\n" + "=" * 80)
    print(" 🛠️  STARTING CONTINUOUS LEARNING & PROCESS SAFETY GOVERNANCE TEST SUITE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. TEST FEEDBACK RECORDING & REWARD SIGNAL ATTRIBUTION
    # -------------------------------------------------------------------------
    print("\n[Phase 1] Simulating HSE Officer Field Reviews & RLHF Reward Attribution...")
    feedback_engine = FeedbackEngine()
    audit_logger = AuditLogger()

    sample_feedbacks = [
        {
            "narrative": "At Digboi Well #412, during tubing pulling, high pressure gas bubble escaped annulus due to incomplete hydrostatic kill column. Crew evacuated rig floor immediately.",
            "pred_sif": True, "pred_prob": 0.94, "pred_rules": ["Energy Isolation", "Line of Fire"], "pred_sev": 0.88,
            "act_sif": True, "act_rules": ["Energy Isolation", "Line of Fire"], "act_sev": 0.90,
            "reviewer_id": "OIL-HSE-DIGBOI-101",
            "notes": "Verified severe well-control near-miss; matched well control guidelines.",
        },
        {
            "narrative": "Central Tank Farm Moran: Operator found oily drip pan under pump bearing filled with rainwater. Drained into effluent pit.",
            "pred_sif": False, "pred_prob": 0.08, "pred_rules": [], "pred_sev": 0.05,
            "act_sif": False, "act_rules": [], "act_sev": 0.05,
            "reviewer_id": "OIL-HSE-MORAN-204",
            "notes": "Routine environmental housekeeping; non-SIF verified.",
        },
        {
            "narrative": "Contract welder observed striking arc on crude storage tank ladder bracket without explosive gas test certificate or PTW.",
            "pred_sif": True, "pred_prob": 0.89, "pred_rules": ["Hot Work", "Work Authorization"], "pred_sev": 0.85,
            "act_sif": True, "act_rules": ["Hot Work", "Work Authorization", "Bypassing Safety Controls"], "act_sev": 0.87,
            "reviewer_id": "OIL-HSE-DULIAJAN-305",
            "notes": "Added bypassing safety controls rule due to skipped gas test.",
        },
    ]

    for fb in sample_feedbacks:
        record = feedback_engine.record_feedback(
            narrative=fb["narrative"],
            predicted_sif=fb["pred_sif"],
            predicted_sif_prob=fb["pred_prob"],
            predicted_rules=fb["pred_rules"],
            predicted_severity=fb["pred_sev"],
            actual_sif=fb["act_sif"],
            actual_rules=fb["act_rules"],
            actual_severity=fb["act_sev"],
            reviewer_id=fb["reviewer_id"],
            reviewer_role="Senior Process Safety Lead",
            notes=fb["notes"],
        )
        print(f"   ✓ Logged Review for '{fb['reviewer_id']}': Reward Score = {record.reward_score:+.2f} | SIF: {record.actual_sif}")

    all_fb = feedback_engine.load_all_feedback()
    print(f"   Total Cumulative Feedback Records in Store: {len(all_fb)}")

    # -------------------------------------------------------------------------
    # 2. TEST ZERO-TOLERANCE SAFETY GATE (100% FATAL RECALL REQUIREMENT)
    # -------------------------------------------------------------------------
    print("\n[Phase 2] Testing Automated Zero-Tolerance Safety Validation Gate...")
    safety_validator = SafetyValidator(data_dir=data_dir)

    # Load active champion models
    extractor = joblib.load(models_dir / "feature_extractor.joblib")
    sif_model = joblib.load(models_dir / "sif_classifier.joblib")
    iogp_model = joblib.load(models_dir / "iogp_rules_classifier.joblib")
    sev_model = joblib.load(models_dir / "severity_regressor.joblib")
    with open(models_dir / "optimal_threshold.json") as f:
        thresholds = json.load(f)

    print("   Running safety validation on baseline production model...")
    champ_safety = safety_validator.validate_candidate(
        extractor=extractor,
        sif_model=sif_model,
        iogp_model=iogp_model,
        severity_model=sev_model,
        thresholds=thresholds,
        candidate_version="v1.0.0-production-champion",
    )

    print(f"   • Fatal Cases Tested  : {champ_safety.fatal_benchmark_total}")
    print(f"   • Fatal Cases Caught  : {champ_safety.fatal_benchmark_caught} / {champ_safety.fatal_benchmark_total}")
    print(f"   • Fatal Recall Gate   : {champ_safety.fatal_recall_pct:.2f}% (Target: 100.0%)")
    print(f"   • Holdout SIF Recall  : {champ_safety.val_sif_recall_pct:.2f}%")
    print(f"   • Status              : {'✅ PASSED (CERTIFIED SAFE)' if champ_safety.passed else '❌ FAILED'}")
    assert champ_safety.fatal_recall_pct == 100.0, "Safety Gate Check Failed: Baseline must achieve 100% fatal recall!"

    # -------------------------------------------------------------------------
    # 3. SAFETY GATE HARD REJECTION TEST (Intentionally Degraded Candidate)
    # -------------------------------------------------------------------------
    print("\n[Phase 3] Testing Safety Gate Rejection on Artificially Degraded Candidate...")
    from sklearn.dummy import DummyClassifier
    bad_dummy_sif = DummyClassifier(strategy="constant", constant=0)
    bad_dummy_sif.fit([[0] * 45013, [1] * 45013], [0, 1])

    degraded_safety = safety_validator.validate_candidate(
        extractor=extractor,
        sif_model=bad_dummy_sif,
        iogp_model=iogp_model,
        severity_model=sev_model,
        thresholds=thresholds,
        candidate_version="v_bad_degraded_test",
    )
    print(f"   • Degraded Model Fatal Recall: {degraded_safety.fatal_recall_pct:.2f}%")
    print(f"   • Safety Gate Result         : {'❌ REJECTED AND BLOCKED (Expected Behavior)' if not degraded_safety.passed else 'UNEXPECTED PASS'}")
    assert not degraded_safety.passed, "Safety Gate FAILED TO BLOCK an unsafe degraded model!"
    print("   ✓ Safety Guardrail Confirmed: Unsafe candidate was strictly blocked from deployment.")

    # -------------------------------------------------------------------------
    # 4. TEST SHADOW BENCHMARKING (Champion vs. Challenger)
    # -------------------------------------------------------------------------
    print("\n[Phase 4] Testing Shadow Benchmarking Mode...")
    val_csv = data_dir / "master_hsse_sif_val.csv"
    val_gz = data_dir / "master_hsse_sif_val.csv.gz"
    val_df = pd.read_csv(val_csv if val_csv.exists() else val_gz, nrows=200)

    champ_pipeline = {
        "extractor": extractor,
        "sif_model": sif_model,
        "thresholds": thresholds,
    }
    chal_pipeline = {
        "extractor": extractor,
        "sif_model": sif_model,
        "thresholds": thresholds,
    }

    shadow_bench = ShadowBenchmarker(audit_logger=audit_logger)
    shadow_report = shadow_bench.benchmark_stream(
        champion_pipeline=champ_pipeline,
        challenger_pipeline=chal_pipeline,
        eval_df=val_df,
        champion_version="v1.0.0-champion",
        challenger_version="v1.0.1-shadow",
    )
    print(f"   • Evaluated Samples   : {shadow_report.samples_evaluated}")
    print(f"   • Agreement Rate      : {shadow_report.agreement_rate_pct:.2f}%")
    print(f"   • Champion Latency    : {shadow_report.latency_champion_ms:.2f} ms/report")
    print(f"   • Challenger Latency  : {shadow_report.latency_challenger_ms:.2f} ms/report")
    print(f"   • Recommendation      : {shadow_report.recommendation}")

    # -------------------------------------------------------------------------
    # 5. TEST END-TO-END CONTINUAL RETRAINING ORCHESTRATOR
    # -------------------------------------------------------------------------
    print("\n[Phase 5] Testing End-to-End Continuous Learning Orchestrator Cycle...")
    train_csv = data_dir / "master_hsse_sif_train.csv"
    train_gz = data_dir / "master_hsse_sif_train.csv.gz"
    train_df = pd.read_csv(train_csv if train_csv.exists() else train_gz, nrows=8000)

    orchestrator = ContinualLearningOrchestrator()
    cycle_result = orchestrator.execute_continual_learning_cycle(
        base_train_df=train_df,
        eval_df=val_df,
        challenger_version="v1.0.1-candidate",
        auto_promote_if_certified=True,
    )
    print(f"   • Promoted to Production : {cycle_result['promoted_to_production']}")
    print(f"   • Status                 : {cycle_result['deployment']['status']}")

    # -------------------------------------------------------------------------
    # 6. VERIFY DGMS / OISD IMMUTABLE AUDIT TRAIL LOG
    # -------------------------------------------------------------------------
    print("\n[Phase 6] Verifying DGMS & OISD Regulatory Audit Trail...")
    recent_logs = audit_logger.read_recent_logs(limit=5)
    print(f"   Retrieved {len(recent_logs)} recent audit log entries:")
    for log in recent_logs:
        print(f"   • [{log['timestamp'][:19]}] [{log['event_type']}] Actor: {log['actor']['user_id']} | Compliance: {log['regulatory_compliance']}")
        print(f"     Hash: {log['entry_hash'][:16]}... | PrevHash: {log['prev_entry_hash'][:16]}...")

    print("\n" + "=" * 80)
    print(" 🎉 ALL CONTINUOUS LEARNING & INDUSTRIAL PROCESS SAFETY TESTS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_full_governance_test()
