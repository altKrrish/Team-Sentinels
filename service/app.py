"""
service.app
============
FastAPI wrapper around the hardened SIF inference pipeline (the
"Containerized Deployment" item). This wraps test_inference.py's model
artifact behind a stable HTTP contract; it does not reimplement the model.

Endpoints:
  POST /v1/classify        -> full pipeline: guidance + interlock + metadata + decision
  POST /v1/guidance/check  -> form-guidance-only, for the client to call as-you-type
  GET  /healthz            -> liveness
  GET  /readyz              -> readiness (model artifact loaded)

Run: uvicorn service.app:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from sentinel import decision_policy, energy_metadata, form_guidance, interlock

logger = logging.getLogger("sentinel.service")

app = FastAPI(
    title="Sentinel SIF Classification Service",
    version="1.0.0",
    description="Hardened SIF triage: deterministic interlock + metadata cross-reference "
                "+ calibrated model probability, behind one HTTP contract.",
)

_MODEL = None  # populated by load_model() at startup; kept swappable for shadow deploys


class ModelNotLoadedError(RuntimeError):
    pass


class ProductionPipelineModel:
    """Production SIF model ensemble wrapper."""
    def __init__(self, models_dir: Path):
        import sys
        import joblib
        import json
        import src.models.train_sif_engine as train_sif_engine
        if not hasattr(sys.modules["__main__"], "MultiModalFeatureExtractor"):
            setattr(sys.modules["__main__"], "MultiModalFeatureExtractor", train_sif_engine.MultiModalFeatureExtractor)
        if not hasattr(sys.modules["__main__"], "engineer_features"):
            setattr(sys.modules["__main__"], "engineer_features", train_sif_engine.engineer_features)

        self.models_dir = Path(models_dir)
        self.extractor = joblib.load(self.models_dir / "feature_extractor.joblib")
        self.sif_model = joblib.load(self.models_dir / "sif_classifier.joblib")
        self.iogp_model = joblib.load(self.models_dir / "iogp_rules_classifier.joblib")
        self.sev_model = joblib.load(self.models_dir / "severity_regressor.joblib")
        with open(self.models_dir / "optimal_threshold.json", encoding="utf-8") as f:
            self.threshold_data = json.load(f)

    def predict_proba_sif(self, text: str, metadata: Optional[Dict] = None) -> float:
        import pandas as pd
        from data.preprocess_pipeline import clean_text, tokenize_for_nlp
        cleaned = clean_text(text)
        tokenized = tokenize_for_nlp(cleaned, remove_stopwords=True)
        df_sample = pd.DataFrame([{"text_cleaned": cleaned, "text_tokenized_no_stopwords": tokenized}])
        X = self.extractor.transform(df_sample)
        return float(self.sif_model.predict_proba(X)[0, 1])

    def predict_extra(self, text: str) -> Dict[str, Any]:
        import numpy as np
        import pandas as pd
        from data.preprocess_pipeline import clean_text, tokenize_for_nlp
        cleaned = clean_text(text)
        tokenized = tokenize_for_nlp(cleaned, remove_stopwords=True)
        df_sample = pd.DataFrame([{"text_cleaned": cleaned, "text_tokenized_no_stopwords": tokenized}])
        X = self.extractor.transform(df_sample)
        sev_score = float(np.clip(self.sev_model.predict(X)[0], 0.0, 1.0))
        rule_thresholds = self.threshold_data.get("rule_thresholds", {})
        rule_cols = [
            "rule_bypassing_safety_controls", "rule_confined_space", "rule_driving",
            "rule_energy_isolation", "rule_hot_work", "rule_line_of_fire",
            "rule_safe_mechanical_lifting", "rule_work_authorization", "rule_working_at_height"
        ]
        rule_names = [
            "Bypassing Safety Controls", "Confined Space", "Driving",
            "Energy Isolation", "Hot Work", "Line of Fire",
            "Safe Mechanical Lifting", "Work Authorization", "Working at Height"
        ]
        probs = [float(est.predict_proba(X)[0, 1]) for est in self.iogp_model.estimators_]
        tagged = []
        for i, name in enumerate(rule_names):
            th = rule_thresholds.get(rule_cols[i], 0.40)
            if probs[i] >= th:
                tagged.append({"rule": name, "confidence_pct": round(probs[i] * 100, 1)})
        return {"severity_score": sev_score, "tagged_rules": tagged}


def load_model(path: Optional[str] = None):
    """Load the serialized ensemble from models/ or path."""
    global _MODEL
    from pathlib import Path
    candidate_paths = []
    if path:
        candidate_paths.append(Path(path))
    candidate_paths.extend([
        Path("models"),
        Path(__file__).resolve().parent.parent / "models",
        Path(__file__).resolve().parent / "models",
    ])

    for cp in candidate_paths:
        if cp.exists() and (cp / "feature_extractor.joblib").exists():
            try:
                _MODEL = ProductionPipelineModel(cp)
                logger.info("Loaded production model ensemble from %s", cp)
                return _MODEL
            except Exception as e:
                logger.warning("Failed to load production ensemble from %s: %s", cp, e)

    if path and Path(path).is_file():
        import joblib
        _MODEL = joblib.load(path)
        return _MODEL

    if _MODEL is None:
        _MODEL = _StubModel()
        logger.warning("No valid model path found; using deterministic stub model.")
    return _MODEL


class _StubModel:
    """Deterministic placeholder for unit tests and fallback."""
    def predict_proba_sif(self, text: str, metadata: Optional[Dict] = None) -> float:
        wc = len((text or "").split())
        return min(0.9, 0.2 + 0.02 * wc)


class ClassifyRequest(BaseModel):
    report_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    asset_class: Optional[str] = None
    metadata: Optional[Dict] = None

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class ClassifyResponse(BaseModel):
    report_id: str
    label: Optional[str]
    route: str
    probability: float
    tau_used: float
    reason: str
    interlock: Dict
    metadata_assessment: Dict
    guidance: Dict
    severity_score: Optional[float] = None
    tagged_iogp_rules: Optional[list] = None
    latency_ms: float


class GuidanceRequest(BaseModel):
    text: str
    min_words: int = 8
    relevant_slots: Optional[list] = None


@app.on_event("startup")
def _startup():
    import os
    load_model(os.environ.get("MODEL_PATH"))


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    ready = not isinstance(_MODEL, _StubModel)
    return JSONResponse(
        status_code=200 if ready else 503,
        content={"status": "ready" if ready else "stub_model_only"},
    )


@app.post("/v1/guidance/check")
def check_guidance(req: GuidanceRequest):
    result = form_guidance.evaluate(
        req.text, min_words=req.min_words, relevant_slots=req.relevant_slots,
    )
    return result.to_dict()


@app.post("/v1/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    if _MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    t0 = time.perf_counter()
    try:
        guidance = form_guidance.evaluate(req.text)
        il_result = interlock.scan(req.text)
        meta_result = energy_metadata.assess(req.metadata or {})
        proba = _MODEL.predict_proba_sif(req.text, req.metadata)
        decision = decision_policy.decide(
            proba, interlock=il_result, metadata=meta_result,
            asset_class=req.asset_class,
        )
        sev_score = None
        tagged_rules = None
        if hasattr(_MODEL, "predict_extra"):
            extra = _MODEL.predict_extra(req.text)
            sev_score = extra.get("severity_score")
            tagged_rules = extra.get("tagged_rules")
    except Exception:
        logger.exception("classification failed for report_id=%s", req.report_id)
        raise HTTPException(status_code=500, detail="internal classification error")

    latency_ms = (time.perf_counter() - t0) * 1000.0
    if decision.route == decision_policy.Route.HUMAN_REVIEW:
        logger.info("report_id=%s routed to HUMAN_REVIEW: %s", req.report_id, decision.reason)

    return ClassifyResponse(
        report_id=req.report_id,
        label=decision.label,
        route=decision.route.value,
        probability=decision.probability,
        tau_used=decision.tau_used,
        reason=decision.reason,
        interlock=il_result.to_dict(),
        metadata_assessment=meta_result.to_dict(),
        guidance=guidance.to_dict(),
        severity_score=sev_score,
        tagged_iogp_rules=tagged_rules,
        latency_ms=latency_ms,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled exception on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})
