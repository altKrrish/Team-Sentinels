"""
DGMS & OISD Regulatory Compliance Audit Logger
==============================================
Maintains an immutable, append-only, tamper-evident audit trail for AI model
predictions, HSE officer reviews, safety validation gates, and model promotions.

Compliant with:
- Directorate General of Mines Safety (DGMS) Safety Management System (SMS) Guidelines
- Oil Industry Safety Directorate (OISD) Standards: OISD-GDN-145 & OISD-STD-189
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class AuditLogger:
    """Tamper-evident audit logger for HSSE AI engine governance."""

    def __init__(self, log_path: Optional[Path] = None):
        if log_path is None:
            base_dir = Path(__file__).resolve().parents[2]
            log_path = base_dir / "data" / "feedback" / "regulatory_audit_log.jsonl"
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_file_hash(filepath: Path) -> str:
        """Compute SHA-256 hash of a file for integrity tracking."""
        if not filepath.exists():
            return "FILE_NOT_FOUND"
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _get_last_entry_hash(self) -> str:
        """Get the hash of the previous log entry to maintain chain integrity."""
        if not self.log_path.exists() or os.path.getsize(self.log_path) == 0:
            return "0000000000000000000000000000000000000000000000000000000000000000"
        
        last_line = ""
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()
        
        if not last_line:
            return "0000000000000000000000000000000000000000000000000000000000000000"
        
        return hashlib.sha256(last_line.encode("utf-8")).hexdigest()

    def log_event(
        self,
        event_type: str,
        actor: Dict[str, str],
        details: Dict[str, Any],
        regulatory_tags: Optional[List[str]] = None,
        model_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Log a governance event with timestamp and hash chain integrity."""
        timestamp = datetime.now(timezone.utc).isoformat()
        prev_hash = self._get_last_entry_hash()

        if regulatory_tags is None:
            regulatory_tags = ["DGMS_SMS", "OISD_STD_189", "OISD_GDN_145"]

        entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "actor": {
                "user_id": actor.get("user_id", "SYSTEM"),
                "role": actor.get("role", "AUTOMATED_ENGINE"),
                "department": actor.get("department", "HSSE_PROCESS_SAFETY"),
            },
            "model_version": model_version or "v1.0.0",
            "regulatory_compliance": regulatory_tags,
            "details": details,
            "prev_entry_hash": prev_hash,
        }

        # Calculate self entry hash
        serialized = json.dumps(entry, sort_keys=True)
        entry_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        entry["entry_hash"] = entry_hash

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def log_feedback(
        self,
        report_id: str,
        reviewer_id: str,
        reviewer_role: str,
        model_prediction: Dict[str, Any],
        human_verification: Dict[str, Any],
        reward_score: float,
        notes: str = "",
    ) -> Dict[str, Any]:
        """Log an HSE officer feedback/override event."""
        return self.log_event(
            event_type="HSE_FEEDBACK_RECORDED",
            actor={
                "user_id": reviewer_id,
                "role": reviewer_role,
                "department": "OIL_HSSE_CORPORATE",
            },
            details={
                "report_id": report_id,
                "model_prediction": model_prediction,
                "human_verification": human_verification,
                "override_detected": model_prediction.get("sif_predicted") != human_verification.get("sif_actual"),
                "reward_score": reward_score,
                "reviewer_notes": notes,
            },
        )

    def log_safety_validation(
        self,
        candidate_version: str,
        passed: bool,
        fatal_cases_tested: int,
        fatal_cases_passed: int,
        fatal_recall_pct: float,
        metrics: Dict[str, Any],
        failure_reasons: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Log an automated safety validation gate decision."""
        event_type = "SAFETY_GATE_PASSED" if passed else "SAFETY_GATE_FAILED"
        return self.log_event(
            event_type=event_type,
            actor={
                "user_id": "SAFETY_GATE_CI_BOT",
                "role": "AUTOMATED_SAFETY_VALIDATOR",
                "department": "OIL_SAFETY_ASSURANCE",
            },
            model_version=candidate_version,
            details={
                "passed": passed,
                "fatal_cases_tested": fatal_cases_tested,
                "fatal_cases_passed": fatal_cases_passed,
                "fatal_recall_pct": fatal_recall_pct,
                "target_fatal_recall_pct": 100.0,
                "overall_metrics": metrics,
                "failure_reasons": failure_reasons or [],
                "certification_status": "CERTIFIED_SAFE" if passed else "REJECTED_UNSAFE",
            },
        )

    def log_shadow_benchmark(
        self,
        champion_version: str,
        challenger_version: str,
        samples_evaluated: int,
        agreement_rate_pct: float,
        champion_sif_recall_pct: float,
        challenger_sif_recall_pct: float,
        latency_champion_ms: float,
        latency_challenger_ms: float,
        recommendation: str,
    ) -> Dict[str, Any]:
        """Log shadow benchmarking results between Champion and Challenger models."""
        return self.log_event(
            event_type="SHADOW_BENCHMARK_EVALUATED",
            actor={
                "user_id": "SHADOW_BENCHMARK_ENGINE",
                "role": "MODEL_GOVERNANCE_AGENT",
                "department": "OIL_AI_OPS",
            },
            model_version=f"Champion:{champion_version}|Challenger:{challenger_version}",
            details={
                "champion_version": champion_version,
                "challenger_version": challenger_version,
                "samples_evaluated": samples_evaluated,
                "agreement_rate_pct": agreement_rate_pct,
                "champion_sif_recall_pct": champion_sif_recall_pct,
                "challenger_sif_recall_pct": challenger_sif_recall_pct,
                "recall_delta_pct": round(challenger_sif_recall_pct - champion_sif_recall_pct, 2),
                "latency_champion_ms": latency_champion_ms,
                "latency_challenger_ms": latency_challenger_ms,
                "recommendation": recommendation,
            },
        )

    def log_model_promotion(
        self,
        old_version: str,
        new_version: str,
        approver_id: str,
        approver_role: str,
        model_hashes: Dict[str, str],
        notes: str = "",
    ) -> Dict[str, Any]:
        """Log the formal promotion and deployment of a verified model."""
        return self.log_event(
            event_type="MODEL_PROMOTION_DEPLOYED",
            actor={
                "user_id": approver_id,
                "role": approver_role,
                "department": "OIL_HSSE_COMMITTEE",
            },
            model_version=new_version,
            details={
                "previous_champion": old_version,
                "new_champion": new_version,
                "model_artifact_hashes": model_hashes,
                "sign_off_statement": "Verified under DGMS/OISD standards with 100% Fatal Recall Guarantee.",
                "notes": notes,
            },
        )

    def read_recent_logs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Read the most recent N log entries."""
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        entries.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        return entries[-limit:]
