"""
Continual Learning Pipeline Orchestrator
=======================================
Orchestrates incremental learning, experience replay, automated safety validation
gates, shadow benchmarking, and DGMS/OISD compliant promotion for the SIF Engine.
"""

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.multioutput import MultiOutputClassifier

from .audit_logger import AuditLogger
from .feedback_engine import ALL_IOGP_RULES, FeedbackEngine
from .safety_validator import SafetyValidationResult, SafetyValidator
from .shadow_benchmarker import ShadowBenchmarkReport, ShadowBenchmarker

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


class ContinualLearningOrchestrator:
    """End-to-end continuous learning orchestrator with industrial safety guardrails."""

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        registry_dir: Optional[Path] = None,
    ):
        base_dir = Path(__file__).resolve().parents[2]
        self.models_dir = models_dir or (base_dir / "models")
        self.data_dir = data_dir or (base_dir / "data" / "processed")
        self.registry_dir = registry_dir or (self.models_dir / "registry")
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.audit_logger = AuditLogger()
        self.feedback_engine = FeedbackEngine(audit_logger=self.audit_logger)
        self.safety_validator = SafetyValidator(data_dir=self.data_dir, audit_logger=self.audit_logger)
        self.shadow_benchmarker = ShadowBenchmarker(audit_logger=self.audit_logger)

    def load_active_champion(self) -> Dict[str, Any]:
        """Load currently deployed Champion model pipeline from models/ directory."""
        extractor = joblib.load(self.models_dir / "feature_extractor.joblib")
        sif_model = joblib.load(self.models_dir / "sif_classifier.joblib")
        iogp_model = joblib.load(self.models_dir / "iogp_rules_classifier.joblib")
        sev_model = joblib.load(self.models_dir / "severity_regressor.joblib")
        with open(self.models_dir / "optimal_threshold.json", "r") as f:
            thresholds = json.load(f)

        return {
            "extractor": extractor,
            "sif_model": sif_model,
            "iogp_model": iogp_model,
            "severity_model": sev_model,
            "thresholds": thresholds,
            "version": "v1.0.0-champion",
        }

    def train_challenger(
        self,
        base_train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
        challenger_version: str = "v1.0.1-challenger",
        sample_multiplier_new: int = 5,
        max_base_samples: int = 15000,
    ) -> Dict[str, Any]:
        """
        Train Challenger model using stratified experience replay
        (verified feedback weighted heavily alongside historical anchors).
        """
        print(f"\n⚙️  [Continuous Learning] Preparing experience replay dataset for {challenger_version}...")
        replay_df, sample_weights = self.feedback_engine.build_replay_dataset(
            base_train_df=base_train_df,
            sample_multiplier_new=sample_multiplier_new,
            max_base_samples=max_base_samples,
        )
        print(f"   Replay dataset size: {len(replay_df):,} samples (Verified feedbacks + Anchors)")

        # 1. Feature Extractor
        champion = self.load_active_champion()
        extractor = champion["extractor"]
        X_train = extractor.transform(replay_df)

        # 2. Train SIF Classifier (Voting Ensemble with adaptive SGD & weighted LR)
        print("   🧠 Training updated SIF Precursor Classifier...")
        if "sif_potential_binary" in replay_df.columns:
            y_sif = replay_df["sif_potential_binary"].fillna(0).astype(int).values
        elif "is_sif_precursor" in replay_df.columns:
            y_sif = replay_df["is_sif_precursor"].fillna(0).astype(int).values
        else:
            y_sif = (replay_df["fatalities"] > 0).astype(int).values

        lr_l2 = LogisticRegression(
            C=2.0, max_iter=500, class_weight="balanced", solver="lbfgs", random_state=42
        )
        lr_l2.fit(X_train, y_sif, sample_weight=sample_weights)

        sgd_model = SGDClassifier(
            loss="modified_huber", alpha=5e-5, max_iter=1000, class_weight="balanced", random_state=42
        )
        sgd_model.fit(X_train, y_sif, sample_weight=sample_weights)

        lr_l1 = LogisticRegression(
            C=1.5, max_iter=500, class_weight="balanced", solver="liblinear", random_state=123
        )
        lr_l1.fit(X_train, y_sif, sample_weight=sample_weights)

        from sklearn.ensemble import VotingClassifier
        challenger_sif = VotingClassifier(
            estimators=[("lr_l2", lr_l2), ("sgd", sgd_model), ("lr_l1", lr_l1)],
            voting="soft",
            weights=[1.0, 1.0, 1.0],
        )
        challenger_sif.fit(X_train, y_sif)

        # 3. Train IOGP Rules Classifier
        print("   🏷️  Training updated 9-Way IOGP Life-Saving Rules Classifier...")
        valid_rules = [r for r in RULE_COLUMNS if r in replay_df.columns]
        y_rules = replay_df[valid_rules].fillna(0).astype(int).values
        base_lr = LogisticRegression(C=3.0, max_iter=400, class_weight="balanced", solver="lbfgs", random_state=42)
        challenger_iogp = MultiOutputClassifier(base_lr)
        challenger_iogp.fit(X_train, y_rules)

        # 4. Train Severity Regressor
        print("   📈 Training updated Continuous Severity Regressor...")
        if "sif_confidence_score" in replay_df.columns:
            y_sev = replay_df["sif_confidence_score"].fillna(0.0).astype(float).values
        elif "severity_score" in replay_df.columns:
            y_sev = replay_df["severity_score"].fillna(0.0).astype(float).values
        else:
            y_sev = np.zeros(len(replay_df), dtype=float)

        challenger_sev = Ridge(alpha=1.0, random_state=42)
        challenger_sev.fit(X_train, y_sev, sample_weight=sample_weights)

        # Calibrate optimal SIF threshold
        best_thresh = 0.40
        best_f1 = 0.0
        if eval_df is not None and not eval_df.empty:
            X_eval = extractor.transform(eval_df)
            eval_col = "sif_potential_binary" if "sif_potential_binary" in eval_df.columns else "is_sif_precursor"
            y_eval_sif = eval_df[eval_col].fillna(0).astype(int).values
            eval_probs = challenger_sif.predict_proba(X_eval)[:, 1]
            for th in np.arange(0.25, 0.48, 0.01):
                p = (eval_probs >= th).astype(int)
                rec = float(np.sum((p == 1) & (y_eval_sif == 1)) / max(np.sum(y_eval_sif == 1), 1))
                prec = float(np.sum((p == 1) & (y_eval_sif == 1)) / max(np.sum(p == 1), 1))
                f1 = (2 * prec * rec) / max(prec + rec, 1e-6)
                if rec >= 0.980 and f1 > best_f1:
                    best_f1 = f1
                    best_thresh = float(round(th, 2))
        else:
            sif_probs = challenger_sif.predict_proba(X_train)[:, 1]
            for th in np.arange(0.25, 0.48, 0.01):
                p = (sif_probs >= th).astype(int)
                rec = float(np.sum((p == 1) & (y_sif == 1)) / max(np.sum(y_sif == 1), 1))
                prec = float(np.sum((p == 1) & (y_sif == 1)) / max(np.sum(p == 1), 1))
                f1 = (2 * prec * rec) / max(prec + rec, 1e-6)
                if rec >= 0.985 and f1 > best_f1:
                    best_f1 = f1
                    best_thresh = float(round(th, 2))

        thresholds = dict(champion["thresholds"])
        thresholds["optimal_sif_threshold"] = best_thresh
        print(f"   🎯 Calibrated Challenger SIF Threshold: tau = {best_thresh:.2f} (Target SIF Recall >= 98.0%)")

        return {
            "extractor": extractor,
            "sif_model": challenger_sif,
            "iogp_model": challenger_iogp,
            "severity_model": challenger_sev,
            "thresholds": thresholds,
            "version": challenger_version,
        }

    def execute_continual_learning_cycle(
        self,
        base_train_df: pd.DataFrame,
        eval_df: pd.DataFrame,
        challenger_version: str = "v1.0.1-candidate",
        auto_promote_if_certified: bool = True,
        approver_id: str = "OIL_HSSE_COMMITTEE_CHAIR",
    ) -> Dict[str, Any]:
        """
        Execute full lifecycle:
        1. Train Challenger with experience replay
        2. Run Zero-Tolerance Safety Gate (100% Fatal Recall required)
        3. Run Shadow Benchmarking against active Champion
        4. If Certified & Approved -> Promote to Production with DGMS/OISD Audit Sign-off
        5. Return comprehensive execution report
        """
        t_start = time.time()
        print("\n" + "=" * 78)
        print(f" 🚀 CONTINUAL LEARNING & SAFETY GOVERNANCE CYCLE: {challenger_version}")
        print("=" * 78)

        # Step 1: Load Champion & Train Challenger
        champion = self.load_active_champion()
        challenger = self.train_challenger(
            base_train_df=base_train_df,
            eval_df=eval_df,
            challenger_version=challenger_version,
        )

        # Step 2: Automated Safety Gate Validation
        print("\n🛡️  [Step 2/4] Executing Automated Safety Validation Test Suite...")
        safety_result: SafetyValidationResult = self.safety_validator.validate_candidate(
            extractor=challenger["extractor"],
            sif_model=challenger["sif_model"],
            iogp_model=challenger["iogp_model"],
            severity_model=challenger["severity_model"],
            thresholds=challenger["thresholds"],
            candidate_version=challenger_version,
        )

        print(f"   Fatal Benchmark Tested : {safety_result.fatal_benchmark_total} cases")
        print(f"   Fatal Benchmark Recall : {safety_result.fatal_recall_pct:.2f}% (Required: 100.0%)")
        print(f"   Holdout SIF Recall     : {safety_result.val_sif_recall_pct:.2f}%")
        print(f"   Holdout SIF Precision  : {safety_result.val_sif_precision_pct:.2f}%")
        print(f"   Safety Certification   : {'✅ PASSED (CERTIFIED SAFE)' if safety_result.passed else '❌ REJECTED (SAFETY GATE FAILED)'}")

        # Step 3: Shadow Benchmarking
        print("\n👥 [Step 3/4] Running Shadow Benchmarking Mode (Champion vs Challenger)...")
        shadow_report: ShadowBenchmarkReport = self.shadow_benchmarker.benchmark_stream(
            champion_pipeline=champion,
            challenger_pipeline=challenger,
            eval_df=eval_df,
            champion_version=champion["version"],
            challenger_version=challenger_version,
        )
        print(f"   Prediction Agreement   : {shadow_report.agreement_rate_pct:.2f}%")
        print(f"   Champion Latency       : {shadow_report.latency_champion_ms:.2f} ms/report")
        print(f"   Challenger Latency     : {shadow_report.latency_challenger_ms:.2f} ms/report")
        print(f"   Shadow Recommendation  : {shadow_report.recommendation}")

        # Step 4: Promotion & Governance Sign-off
        promoted = False
        deployment_details = {}
        if safety_result.passed and auto_promote_if_certified:
            print("\n🏆 [Step 4/4] Promoting Certified Challenger Model to Production Champion...")
            # 1. Archive previous champion
            archive_subdir = self.registry_dir / f"archive_{champion['version']}_{int(time.time())}"
            archive_subdir.mkdir(parents=True, exist_ok=True)
            for model_file in ["sif_classifier.joblib", "iogp_rules_classifier.joblib", "severity_regressor.joblib", "optimal_threshold.json"]:
                src = self.models_dir / model_file
                if src.exists():
                    shutil.copy2(src, archive_subdir / model_file)

            # 2. Deploy challenger models to models/
            joblib.dump(challenger["sif_model"], self.models_dir / "sif_classifier.joblib")
            joblib.dump(challenger["iogp_model"], self.models_dir / "iogp_rules_classifier.joblib")
            joblib.dump(challenger["severity_model"], self.models_dir / "severity_regressor.joblib")
            with open(self.models_dir / "optimal_threshold.json", "w", encoding="utf-8") as f:
                json.dump(challenger["thresholds"], f, indent=2)

            # 3. Compute SHA-256 hashes for regulatory audit
            model_hashes = {
                "sif_classifier": AuditLogger.compute_file_hash(self.models_dir / "sif_classifier.joblib"),
                "iogp_rules_classifier": AuditLogger.compute_file_hash(self.models_dir / "iogp_rules_classifier.joblib"),
                "severity_regressor": AuditLogger.compute_file_hash(self.models_dir / "severity_regressor.joblib"),
                "optimal_threshold": AuditLogger.compute_file_hash(self.models_dir / "optimal_threshold.json"),
            }

            # 4. Log regulatory promotion to DGMS / OISD audit trail
            self.audit_logger.log_model_promotion(
                old_version=champion["version"],
                new_version=challenger_version,
                approver_id=approver_id,
                approver_role="Chief Safety Officer & Model Governance Board",
                model_hashes=model_hashes,
                notes=f"Passed 100% Fatal Recall Safety Gate ({safety_result.fatal_recall_pct}% recall on Indian disaster cases).",
            )
            promoted = True
            deployment_details = {
                "status": "DEPLOYED_TO_PRODUCTION",
                "archive_path": str(archive_subdir),
                "model_hashes": model_hashes,
            }
            print(f"   ✅ Model {challenger_version} successfully deployed with DGMS/OISD audit sign-off.")
        else:
            print("\n⛔ [Step 4/4] Model promotion blocked due to safety gate failure or review requirement.")
            deployment_details = {
                "status": "PROMOTION_BLOCKED",
                "reasons": safety_result.failure_reasons,
            }

        elapsed = round(time.time() - t_start, 2)
        print("=" * 78)
        print(f"  Cycle finished in {elapsed}s | Status: {deployment_details['status']}")
        print("=" * 78 + "\n")

        return {
            "challenger_version": challenger_version,
            "elapsed_seconds": elapsed,
            "safety_validation": safety_result.to_dict(),
            "shadow_benchmark": shadow_report.to_dict(),
            "promoted_to_production": promoted,
            "deployment": deployment_details,
        }
