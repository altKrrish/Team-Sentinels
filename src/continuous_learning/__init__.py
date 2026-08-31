"""
Continuous Learning & Industrial Process Safety Governance Package
===================================================================
Oil India Limited (OIL) HSSE AI/NLP Engine

Modules:
- audit_logger: DGMS & OISD-compliant regulatory audit trail
- feedback_engine: Human-in-the-loop feedback store and reward attribution
- safety_validator: Zero-tolerance 100% fatal recall safety gate
- shadow_benchmarker: Champion vs. Challenger shadow-mode comparison
- continual_trainer: End-to-end continuous learning orchestrator
"""

from .audit_logger import AuditLogger
from .feedback_engine import FeedbackEngine, FeedbackRecord
from .safety_validator import SafetyValidator, SafetyValidationResult
from .shadow_benchmarker import ShadowBenchmarker, ShadowBenchmarkReport
from .continual_trainer import ContinualLearningOrchestrator

__all__ = [
    "AuditLogger",
    "FeedbackEngine",
    "FeedbackRecord",
    "SafetyValidator",
    "SafetyValidationResult",
    "ShadowBenchmarker",
    "ShadowBenchmarkReport",
    "ContinualLearningOrchestrator",
]
