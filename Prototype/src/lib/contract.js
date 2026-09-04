/**
 * CloseCall model <-> UI data contract.
 *
 * This file is the single source of truth for the boundary between the
 * SIF classification model and this dashboard. The UI reads nothing that
 * isn't declared here.
 *
 * Endpoints the UI expects (see src/lib/api.js):
 *   GET  {VITE_API_BASE}/reports          -> Report[]
 *   POST {VITE_API_BASE}/classify         -> { text: string } => Classification
 *   POST {VITE_API_BASE}/reports/:id/review -> { state, note } => Review
 *
 * @typedef {'UA'|'UC'|'near-miss'|'incident'} ReportType
 *   UA = unsafe act, UC = unsafe condition.
 *
 * @typedef {'gravity'|'pressure'|'electrical'|'thermal'|'mechanical'|'chemical'|'motion'} HazardEnergy
 *   The hazardous energy in play. Per the DEKRA / EEI precursor model, SIF
 *   potential requires a high-energy source PLUS a failed barrier.
 *
 * @typedef {'absent'|'failed'|'bypassed'|'inadequate'|'not-verified'} BarrierFailure
 *   How the control protecting against that energy broke down.
 *
 * @typedef {'open'|'in-progress'|'closed'} ReportStatus
 *
 * @typedef {Object} LsrTag
 * @property {string} id          One of the 9 IOGP Life-Saving Rule ids in lsr.js.
 * @property {number} confidence  0..1 per-rule probability from the One-vs-Rest
 *                                head. Independent per rule — these do NOT sum
 *                                to 1, because a report can breach several rules.
 *
 * @typedef {Object} Evidence
 * @property {string} span    Verbatim substring of `text` that drove the score.
 *                            Must appear in `text` for highlighting to work.
 * @property {number} weight  0..1 contribution. Drives highlight intensity.
 *
 * @typedef {Object} Precursors
 * @property {HazardEnergy} hazardEnergy
 * @property {BarrierFailure} barrierFailure
 *
 * @typedef {Object} FeatureScore
 *   One of the engineered feature families in model.js contributing to the SIF
 *   score. Named groups, not raw TF-IDF weights — a 40k-dim sparse vector is not
 *   an explanation an HSE officer can act on.
 * @property {string} group        FEATURE_GROUPS id.
 * @property {number} value        0..1 normalised activation of that group.
 * @property {number} contribution Signed push on the SIF logit, -1..1.
 *
 * @typedef {Object} TermExpansion
 *   A normalisation the preprocessor applied before vectorising.
 * @property {string} from  As written by the observer, e.g. "LOTO".
 * @property {string} to    Standardised form, e.g. "Lockout Tagout".
 *
 * @typedef {'pending'|'in-progress'|'confirmed'|'overridden'} ReviewState
 *   `pending` - screened, nobody has opened it. `in-progress` - a reviewer has
 *   picked it up and has not ruled yet. The middle state matters operationally:
 *   it stops a second officer duplicating the work, while still counting as an
 *   open decision. Only `confirmed` and `overridden` are rulings.
 *
 * @typedef {Object} Review
 *   The human-in-the-loop decision. The model screens; an HSE professional
 *   verifies. Absent/pending means nobody has looked at it yet.
 * @property {ReviewState} state
 * @property {string|null} by
 * @property {string|null} at    ISO date.
 * @property {string|null} note
 *
 * @typedef {Object} Classification
 *   What POST /classify returns — the model's verdict with no record metadata.
 *   Three heads, per the pipeline in model.js.
 * @property {boolean} sifPotential      (a) SIF-potential vs non-SIF-potential.
 *                                       Binary logistic-regression head.
 * @property {number}  sifConfidence     0..1 model score for SIF potential.
 *                                       `sifPotential` is this score >= 0.5.
 * @property {LsrTag[]} lsrTags          (b) EVERY rule above threshold, sorted
 *                                       by confidence desc. Multi-label: a hot
 *                                       work job on an expired permit breaches
 *                                       two rules and must show both.
 * @property {LsrTag}  lsr               Convenience mirror of lsrTags[0] — the
 *                                       primary rule. Kept so single-rule
 *                                       columns and filters stay simple.
 * @property {LsrTag|null} lsrSecondary   Mirror of lsrTags[1], or null.
 * @property {Precursors} precursors     (c) Pattern fields for the dashboard.
 * @property {number} severityScore      0..10 CONTINUOUS score from the ridge
 *                                       regression head. This is what the
 *                                       priority queue sorts on.
 * @property {number} severityActual     1..5 band — what DID happen.
 * @property {number} severityPotential  1..5 band — what COULD have happened.
 *                                       The whole point: potential >> actual.
 * @property {Evidence[]} evidence       Explainability spans.
 * @property {FeatureScore[]} features   Engineered-feature contributions.
 * @property {TermExpansion[]} normalized Terms the preprocessor standardised.
 *
 * @typedef {Classification & {
 *   id: string,
 *   reportedAt: string,
 *   type: ReportType,
 *   site: string,
 *   asset: string,
 *   department: string,
 *   activity: string,
 *   text: string,
 *   reportedBy: string,
 *   status: ReportStatus,
 *   review: Review,
 * }} Report
 */

/** Canonical enum values, exported so filters and legends stay in sync. */
export const REPORT_TYPES = ["UA", "UC", "near-miss", "incident"]

export const REPORT_TYPE_LABEL = {
  UA: "Unsafe act",
  UC: "Unsafe condition",
  "near-miss": "Near miss",
  incident: "Incident",
}

export const HAZARD_ENERGIES = [
  "gravity",
  "pressure",
  "electrical",
  "thermal",
  "mechanical",
  "chemical",
  "motion",
]

export const BARRIER_FAILURES = [
  "absent",
  "failed",
  "bypassed",
  "inadequate",
  "not-verified",
]

export const BARRIER_LABEL = {
  absent: "Barrier absent",
  failed: "Barrier failed",
  bypassed: "Barrier bypassed",
  inadequate: "Barrier inadequate",
  "not-verified": "Barrier not verified",
}

export const SEVERITY_LABEL = {
  1: "Negligible",
  2: "Minor",
  3: "Moderate",
  4: "Major",
  5: "Fatal / life-altering",
}

/**
 * The multi-label head emits a probability per rule. Only rules at or above this
 * are shown as tags — below it the model is guessing, and a wrong rule sends the
 * HSE team to the wrong standing committee.
 */
export const LSR_TAG_THRESHOLD = 0.35

export const REVIEW_LABEL = {
  pending: "Awaiting review",
  "in-progress": "Review in progress",
  confirmed: "Verified by HSE",
  overridden: "Overridden by HSE",
}

/**
 * Workflow order. Used to sort the HSE-decision column, because alphabetical
 * order puts `pending` last and buries the untouched backlog at the bottom of
 * the table — the opposite of what someone sorting that column wants.
 */
export const REVIEW_ORDER = ["pending", "in-progress", "confirmed", "overridden"]

/**
 * States in which the flag has been screened but nobody has ruled. Counted
 * together wherever the backlog is reported, because claiming a report is not
 * the same as closing it out and the queue number must not drop when a reviewer
 * merely opens one.
 */
export const REVIEW_UNDECIDED = ["pending", "in-progress"]

/**
 * Industry benchmark band from the problem statement: leading operators find
 * ~20-25% of reports carry genuine fatal potential. The dashboard meters OIL's
 * flagged share against this band.
 */
export const SIF_BENCHMARK = { low: 0.2, high: 0.25 }
