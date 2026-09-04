"""
Turning model output into the JSON the dashboard reads.

Two entry points:

`classification` - one narrative's verdict in `src/lib/contract.js` shape. Used by
    both `POST /classify` and the report stream, so a pasted note and a stored
    report can never be scored or shaped differently.

`build_stream` - the report stream behind `GET /reports`. Narratives are generated
    from a seed range **disjoint from the training corpus** and then scored by the
    trained model. Every safety field a report carries is a prediction; only the
    record metadata (id, date, site, who raised it, workflow status) is fixture.
    That is the honest arrangement for a demo: the numbers on the dashboard are
    the model's, not a designer's.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

from . import corpus
from .explain import evidence_spans, feature_contributions
from .model import SifEngine, Prediction
from .normalize import expansions_in

#: The stream ends here so the dashboard's 12-month window is stable across runs
#: and screenshots. No `date.today()` anywhere - a demo that shifts every day is
#: impossible to talk about.
END_DATE = date(2026, 8, 31)
WINDOW_DAYS = 365

#: Offset well clear of the training seed so no served report is a training row.
STREAM_SEED = 26165 + 900_001


def classification(engine: SifEngine, text: str, pred: Prediction, row=None) -> dict:
    """One `Classification` object, exactly as `src/lib/contract.js` declares it."""
    if row is None:
        row = engine.transform([text])

    tags = [{"id": rule, "confidence": round(conf, 3)} for rule, conf in pred.rules]

    return {
        "sifPotential": pred.sif,
        "sifConfidence": round(pred.sif_confidence, 3),
        "lsrTags": tags,
        # Mirrors kept so single-rule table columns and filters stay simple.
        "lsr": tags[0],
        "lsrSecondary": tags[1] if len(tags) > 1 else None,
        "precursors": {
            "hazardEnergy": pred.hazard_energy,
            "barrierFailure": pred.barrier_failure,
        },
        "severityScore": round(pred.severity_score, 2),
        "severityActual": pred.severity_actual,
        "severityPotential": pred.severity_potential,
        "evidence": evidence_spans(engine, text, row=row),
        "features": feature_contributions(engine, text, row=row),
        "normalized": expansions_in(text),
    }


def _status(rng: random.Random, sif: bool, age_days: int) -> str:
    """Workflow state. Recent and flagged skews open, which is what makes the
    "flagged and still open" tile on the dashboard mean anything."""
    if age_days > 240:
        return rng.choices(("closed", "in-progress"), weights=(88, 12))[0]
    if age_days > 90:
        return rng.choices(("closed", "in-progress", "open"), weights=(62, 24, 14))[0]
    if sif:
        return rng.choices(("open", "in-progress", "closed"), weights=(46, 34, 20))[0]
    return rng.choices(("open", "in-progress", "closed"), weights=(28, 30, 42))[0]


#: Share of undecided flags an officer has already picked up. Enough that the
#: state is visible in the demo without anyone clicking, not so many that the
#: queue looks like it is being worked faster than the numbers claim.
IN_PROGRESS_SHARE = 0.38


def _review(rng: random.Random, sif: bool, age_days: int) -> dict:
    """Human-in-the-loop state.

    Only flagged reports get reviewed at all - that is the point of the screening
    layer. Older flags are more likely to have been decided.

    Every flagged report consumes the same number of draws whichever state it
    lands in, so adjusting the odds below re-labels reviews without reshuffling
    the report types and statuses drawn after them.
    """
    if not sif:
        return {"state": "pending", "by": None, "at": None, "note": None}

    roll = rng.random()
    verdict = rng.choices(("confirmed", "overridden"), weights=(80, 20))[0]
    reviewer = f"{rng.choice(corpus.REPORTERS)} - HSE"
    at = (END_DATE - timedelta(days=max(0, age_days - rng.randint(1, 12)))).isoformat()

    decided_odds = 0.85 if age_days > 120 else 0.55 if age_days > 45 else 0.25
    if roll <= decided_odds:
        # Real reviewers disagree with the model a meaningful fraction of the time.
        note = (
            "Verified against the site log; action assigned to the area authority."
            if verdict == "confirmed"
            else "Reviewed - hazard was already controlled by a compensating measure."
        )
        return {"state": verdict, "by": reviewer, "at": at, "note": note}

    # Nobody has ruled. A share of these have still been claimed - an officer holds
    # the file and has not decided, which is the honest state for anything needing
    # a site walk-down first. It counts as backlog either way.
    if (roll - decided_odds) / (1.0 - decided_odds) >= IN_PROGRESS_SHARE:
        return {"state": "pending", "by": None, "at": None, "note": None}
    return {
        "state": "in-progress",
        "by": reviewer,
        "at": at,
        "note": "Picked up for review - awaiting a site walk-down before a decision.",
    }


def build_stream(engine: SifEngine, n: int = 150, seed: int = STREAM_SEED) -> list[dict]:
    """Generate `n` unseen narratives and score them all in one batched pass."""
    rng = random.Random(seed)

    drafts = []
    for _ in range(n):
        frame = rng.choice(corpus.FRAMES)
        energy = corpus._draw_energy(rng, frame)
        potential = corpus._draw_potential(rng, frame)
        barrier = corpus._draw_barrier(rng)
        outcome = corpus._draw_outcome(rng, potential, barrier)
        text, facts = corpus._compose(rng, frame, energy, potential, barrier, outcome)
        drafts.append((frame, text, facts))

    texts = [d[1] for d in drafts]
    matrix = engine.transform(texts)
    preds = engine.predict(matrix)

    # Dates spread across the window, then sorted so ids run in date order.
    offsets = sorted(rng.randint(0, WINDOW_DAYS - 1) for _ in range(n))

    reports = []
    for i, ((frame, text, facts), pred, offset) in enumerate(zip(drafts, preds, offsets)):
        reported_at = END_DATE - timedelta(days=WINDOW_DAYS - 1 - offset)
        age_days = (END_DATE - reported_at).days

        body = classification(engine, text, pred, row=matrix[i])
        body.update(
            {
                "id": f"HSSE-{reported_at.year}-{i + 1:04d}",
                "reportedAt": reported_at.isoformat(),
                "type": rng.choice(frame.report_types),
                "site": facts["site"],
                "asset": facts["asset"],
                "department": frame.department,
                "activity": frame.activity,
                "text": text,
                "reportedBy": f"{rng.choice(corpus.REPORTERS)} - {rng.choice(corpus.REPORTER_ROLES)}",
                "status": _status(rng, pred.sif, age_days),
                "review": _review(rng, pred.sif, age_days),
            }
        )
        reports.append(body)

    reports.sort(key=lambda r: r["reportedAt"])
    return reports
