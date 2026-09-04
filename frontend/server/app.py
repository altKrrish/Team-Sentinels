"""
The CloseCall API.

    python -m uvicorn app:api --reload --port 8000

Three endpoints, matching `src/lib/api.js` exactly:

    GET  /reports                  the scored report stream
    POST /classify {"text": ...}   score one pasted narrative
    POST /reports/{id}/review      record a reviewer's confirm / override

Plus two for the demo:

    GET  /health                   is the model loaded, and what was it trained on
    GET  /metrics                  the held-out numbers from artifacts/metrics.json

How the data works
------------------
The engine is loaded once at startup from `artifacts/engine.joblib`. The report
stream is generated once too, from a seed range disjoint from training, then
scored by the model - so every safety field the dashboard shows (SIF verdict,
confidence, rule tags, hazard energy, barrier state, severity, evidence spans,
feature pushes) is model output. Only the record metadata - id, date, site, who
raised it, workflow status - is fixture.

Reviews are the one piece of real state. They are held in memory, keyed by report
id, and merged over the stream on read. A restart clears them; that is fine for a
prototype and is the one thing to replace with a table when this goes anywhere
real.

This is a PROTOTYPE. The engine is trained on generated narratives because OIL's
report text is internal and unpublished. `GET /health` says so, out loud, so
nothing downstream can quietly imply otherwise. To train on real reports, drop a
labelled CSV at `data/reports.csv` and re-run `train.py` - see
`closecall/dataset.py`.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from closecall.serve import END_DATE, build_stream, classification

ARTIFACTS = Path(__file__).parent / "artifacts"
ENGINE_PATH = ARTIFACTS / "engine.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"

#: How many reports the dashboard gets. 150 is enough for the monthly trend to
#: have a shape and for per-site density to mean something.
STREAM_SIZE = 150

#: Local Vite development origins. The LAN range is needed when the frontend
#: is opened from another device or via the host machine's network address.
ALLOWED_ORIGINS = [
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]
ALLOWED_ORIGIN_REGEX = r"http://(?:192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+):5174"

api = FastAPI(
    title="CloseCall",
    description="SIF-precursor screening for unsafe-act, unsafe-condition and near-miss reports.",
    version="0.1.0-prototype",
)
api.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


class _State:
    """Everything loaded once, at import, and then read-only."""

    def __init__(self) -> None:
        if not ENGINE_PATH.exists():
            raise RuntimeError(
                f"no model at {ENGINE_PATH}. Run `python train.py` first."
            )
        self.engine = joblib.load(ENGINE_PATH)
        self.reports = build_stream(self.engine, n=STREAM_SIZE)
        self.by_id = {r["id"]: r for r in self.reports}
        self.metrics = (
            json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            if METRICS_PATH.exists()
            else {}
        )
        #: report id -> the reviewer's decision, overriding the seeded one
        self.reviews: dict[str, dict] = {}


state = _State()


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class ReviewRequest(BaseModel):
    #: ``in-progress`` records that an officer has taken the report without
    #: ruling on it. It is stored rather than treated as a clear, because it
    #: carries the reviewer's name - that is the whole point of the state.
    state: Literal["pending", "in-progress", "confirmed", "overridden"]
    note: str | None = Field(default=None, max_length=2000)
    by: str | None = Field(default=None, max_length=120)


@api.get("/health")
def health() -> dict:
    """Is the model up, and what is it? Read this before believing any number."""
    provenance = state.metrics.get("provenance", {})
    return {
        "ok": True,
        "model": "loaded",
        "status": provenance.get("status", "PROTOTYPE"),
        "trainedOn": provenance.get("trainedOn", "unknown"),
        "warning": provenance.get("warning"),
        "reports": len(state.reports),
        "reviewsRecorded": len(state.reviews),
        "windowEnd": END_DATE.isoformat(),
    }


@api.get("/metrics")
def metrics() -> dict:
    """The held-out numbers, verbatim from the last training run."""
    if not state.metrics:
        raise HTTPException(404, "no metrics.json - run `python train.py`")
    return state.metrics


@api.get("/reports")
def reports() -> list[dict]:
    """The scored stream, with any reviewer decisions merged over it."""
    if not state.reviews:
        return state.reports
    return [
        {**r, "review": state.reviews[r["id"]]} if r["id"] in state.reviews else r
        for r in state.reports
    ]


@api.post("/classify")
def classify(body: ClassifyRequest) -> dict:
    """Score one narrative.

    Runs the same `classification()` the stream does, so a pasted note and a
    stored report can never be scored differently.
    """
    text = body.text.strip()
    if not text:
        raise HTTPException(422, "text is empty")

    matrix = state.engine.transform([text])
    pred = state.engine.predict(matrix)[0]
    return classification(state.engine, text, pred, row=matrix[0])


@api.post("/reports/{report_id}/review")
def review(report_id: str, body: ReviewRequest) -> dict:
    """Record a decision, or the fact that someone has taken the report.

    Returns the stored `Review`, which is what `submitReview` in `api.js` reads.
    The model's own verdict is never altered - the reviewer's decision sits
    alongside it, so disagreement stays visible instead of being overwritten.

    ``pending`` is the only state that clears: it means "nobody has this", so
    there is nothing to attribute. Every other state, ``in-progress`` included,
    is stored with the reviewer's name.
    """
    if report_id not in state.by_id:
        raise HTTPException(404, f"no report {report_id}")

    saved = {
        "state": body.state,
        "by": body.by or "You - HSE reviewer",
        "at": END_DATE.isoformat(),
        "note": body.note,
    }
    if body.state == "pending":
        state.reviews.pop(report_id, None)
        return {"state": "pending", "by": None, "at": None, "note": None}

    state.reviews[report_id] = saved
    return saved
