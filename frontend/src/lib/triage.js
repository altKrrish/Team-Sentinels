/**
 * Screening and prioritisation. Pure functions over Report[].
 *
 * The model supplies three outputs — SIF potential, rule tags, a severity score.
 * Turning those into a work queue is this layer's job, and it is deliberately
 * NOT a black box: the tier is decided by an explicit safety rule an HSE auditor
 * can read and challenge, and only the ordering *within* a tier uses the model's
 * continuous scores.
 */
import { REVIEW_UNDECIDED, SIF_BENCHMARK } from "./contract.js"

const BROKEN_BARRIER = ["absent", "failed", "bypassed"]

export const PRIORITY_TIERS = [
  {
    id: "P1",
    label: "Immediate",
    tone: "critical",
    blurb: "Review before the next shift",
    rule: "Flagged SIF-potential · fatal potential severity · barrier absent, failed or bypassed · not yet closed out.",
  },
  {
    id: "P2",
    label: "Priority",
    tone: "serious",
    blurb: "Review this week",
    rule: "Flagged SIF-potential and still open, but missing one of the Immediate conditions.",
  },
  {
    id: "P3",
    label: "Routine",
    tone: "good",
    blurb: "Normal close-out",
    rule: "Not flagged as SIF-potential, or flagged and already closed out.",
  },
]

export const TIER_BY_ID = Object.fromEntries(PRIORITY_TIERS.map((t) => [t.id, t]))

/** The rule-based tier. Explicit, auditable, no model threshold involved. */
export function tierOf(r) {
  if (!r.sifPotential) return "P3"
  if (r.status === "closed") return "P3"
  if (r.severityPotential >= 5 && BROKEN_BARRIER.includes(r.precursors.barrierFailure)) return "P1"
  return "P2"
}

/**
 * Ordering *inside* a tier. Weighted toward the model's own confidence, then the
 * continuous severity score, with a nudge for work nobody has started.
 */
export function priorityScore(r) {
  const openness = r.status === "open" ? 1 : r.status === "in-progress" ? 0.5 : 0
  const sev = (r.severityScore ?? r.severityPotential * 2) / 10
  return 0.55 * r.sifConfidence + 0.35 * sev + 0.1 * openness
}

/** The work queue: tier first, then score. What the HSE team opens each morning. */
export function triageQueue(reports) {
  const order = { P1: 0, P2: 1, P3: 2 }
  return reports
    .map((r) => ({ ...r, tier: tierOf(r), score: priorityScore(r) }))
    .sort((a, b) => order[a.tier] - order[b.tier] || b.score - a.score)
}

export function tierCounts(reports) {
  const acc = { P1: 0, P2: 0, P3: 0 }
  for (const r of reports) acc[tierOf(r)] += 1
  return acc
}

/**
 * Nested subsets of one measure, widest first — the "instead of reading all
 * 1,000 with equal priority" story as a shape.
 */
export function screeningFunnel(reports) {
  const total = reports.length
  const flagged = reports.filter((r) => r.sifPotential)
  const t = tierCounts(reports)
  const stages = [
    { id: "all", label: "Reports received", count: total, note: "Free-text UA / UC / near-miss / incident" },
    { id: "flagged", label: "Screened SIF-potential", count: flagged.length, note: "Binary classifier at threshold 0.50" },
    { id: "open", label: "Flagged and still open", count: flagged.filter((r) => r.status !== "closed").length, note: "Not yet closed out" },
    { id: "p1", label: "Immediate attention", count: t.P1, note: "Fatal potential with a broken barrier" },
  ]
  return { stages, max: Math.max(1, total) }
}

/**
 * Manual vs assisted screening effort.
 *
 * The engine does not read reports faster than a person — it decides which ones
 * a person must read closely. So: everything in P1/P2 still gets a full review,
 * and P3 gets a sampled audit instead of a full read. `sampleRate` is the audit
 * fraction; both assumptions are stated on screen rather than buried here.
 */
export function workload(reports, { minutesPerReport = 4, sampleRate = 0.1 } = {}) {
  const t = tierCounts(reports)
  const total = reports.length
  const fullReads = t.P1 + t.P2
  const sampled = Math.ceil(t.P3 * sampleRate)

  const manualMin = total * minutesPerReport
  const assistedMin = (fullReads + sampled) * minutesPerReport

  return {
    total,
    fullReads,
    sampled,
    skipped: t.P3 - sampled,
    manualHours: manualMin / 60,
    assistedHours: assistedMin / 60,
    savedHours: Math.max(0, (manualMin - assistedMin) / 60),
    savedShare: manualMin ? Math.max(0, (manualMin - assistedMin) / manualMin) : 0,
    minutesPerReport,
    sampleRate,
  }
}

/* ---------------- human-in-the-loop ---------------- */

export function reviewStats(reports) {
  const flagged = reports.filter((r) => r.sifPotential)
  const state = (r) => r.review?.state ?? "pending"
  const pending = flagged.filter((r) => state(r) === "pending")
  const inProgress = flagged.filter((r) => state(r) === "in-progress")
  const confirmed = flagged.filter((r) => state(r) === "confirmed")
  const overridden = flagged.filter((r) => state(r) === "overridden")
  const decided = confirmed.length + overridden.length

  /* Claiming a report is not ruling on it. `undecided` - not `pending` - is what
     the backlog tiles read, so a reviewer opening a P1 cannot make the queue
     look shorter than it is. */
  const undecided = flagged.filter((r) => REVIEW_UNDECIDED.includes(state(r)))

  return {
    flagged: flagged.length,
    pending: pending.length,
    inProgress: inProgress.length,
    undecided: undecided.length,
    confirmed: confirmed.length,
    overridden: overridden.length,
    decided,
    /** Of the verdicts HSE has ruled on, how many they agreed with. */
    agreement: decided ? confirmed.length / decided : null,
    p1Undecided: undecided.filter((r) => tierOf(r) === "P1").length,
  }
}

/* ---------------- distributions the new screens need ---------------- */

/**
 * Severity-score histogram in whole-point bins, split SIF / non-SIF. Grouped
 * bars, categorical slots 1–2 — the same hue-per-entity as the trend line.
 */
export function severityScoreBins(reports) {
  const bins = Array.from({ length: 10 }, (_, i) => ({
    bin: i,
    label: `${i}–${i + 1}`,
    sif: 0,
    nonSif: 0,
  }))
  for (const r of reports) {
    const s = r.severityScore ?? 0
    const i = Math.min(9, Math.max(0, Math.floor(s)))
    if (r.sifPotential) bins[i].sif += 1
    else bins[i].nonSif += 1
  }
  return bins
}

/**
 * Multi-label rule counts: a report with three tags contributes to all three,
 * so these sum above the report count. That is the point of a multi-label head
 * and the totals are labelled accordingly wherever this is shown.
 */
export function lsrTagDistribution(reports) {
  const acc = new Map()
  let tagged = 0
  let tags = 0
  for (const r of reports) {
    const list = r.lsrTags?.length ? r.lsrTags : r.lsr ? [r.lsr] : []
    if (list.length) tagged += 1
    for (const t of list) {
      tags += 1
      if (!acc.has(t.id)) acc.set(t.id, { id: t.id, total: 0, sif: 0, confSum: 0 })
      const row = acc.get(t.id)
      row.total += 1
      row.confSum += t.confidence
      if (r.sifPotential) row.sif += 1
    }
  }
  const rows = [...acc.values()]
    .map((r) => ({ ...r, meanConfidence: r.total ? r.confSum / r.total : 0 }))
    .sort((a, b) => b.total - a.total)
  return {
    rows,
    tags,
    tagged,
    /** Rule breaches per tagged report — the multi-label multiplier. */
    tagsPerReport: tagged ? tags / tagged : 0,
  }
}

/** Reports breaching more than one rule — the case single-label triage misses. */
export function multiRuleReports(reports) {
  return reports.filter((r) => (r.lsrTags?.length ?? 0) > 1)
}

export { SIF_BENCHMARK }
