/**
 * Pure derivations over Report[]. No fetching, no React, no side effects —
 * every panel on every screen reads its numbers from here so the whole app
 * always agrees with itself.
 */
import { LSR, lsrName, lsrShort } from "./lsr.js"
import { BARRIER_FAILURES, BARRIER_LABEL, HAZARD_ENERGIES, REPORT_TYPES, REPORT_TYPE_LABEL } from "./contract.js"
import { tierOf } from "./triage.js"
import { monthKey, monthLabel } from "./format.js"

/** A site needs at least this many reports before its density is ranked. */
export const MIN_REPORTS_FOR_DENSITY = 8

export const EMPTY_FILTERS = {
  months: 12, // trailing window, relative to the newest month in the data
  site: "all",
  lsr: "all",
  type: "all",
  priority: "all",
  sifOnly: false,
  q: "",
}

/**
 * Every rule the multi-label head tagged, strongest first. Falls back to the
 * single-rule mirror so a payload from an older model version still renders.
 * @returns {import('./contract.js').LsrTag[]}
 */
export function rulesOf(r) {
  if (r.lsrTags?.length) return r.lsrTags
  return r.lsr && r.lsr.id !== "unmapped" ? [r.lsr] : []
}

/* ---------------- month axis ---------------- */

/** Every month between the oldest and newest report, gaps included. */
export function monthAxis(reports) {
  if (!reports.length) return []
  const keys = reports.map((r) => monthKey(r.reportedAt))
  const min = keys.reduce((a, b) => (a < b ? a : b))
  const max = keys.reduce((a, b) => (a > b ? a : b))
  const out = []
  let [y, m] = min.split("-").map(Number)
  const [maxY, maxM] = max.split("-").map(Number)
  while (y < maxY || (y === maxY && m <= maxM)) {
    out.push(`${y}-${String(m).padStart(2, "0")}`)
    m += 1
    if (m > 12) {
      m = 1
      y += 1
    }
  }
  return out
}

/** The trailing `months` slice of a month axis. */
export function windowOf(axis, months) {
  if (!months || months >= axis.length) return axis
  return axis.slice(axis.length - months)
}

/* ---------------- filtering ---------------- */

export function applyFilters(reports, filters, axis) {
  const keep = new Set(windowOf(axis ?? monthAxis(reports), filters.months))
  const q = filters.q.trim().toLowerCase()

  return reports.filter((r) => {
    if (!keep.has(monthKey(r.reportedAt))) return false
    if (filters.site !== "all" && r.site !== filters.site) return false
    // multi-label: a report matches the rule filter if ANY of its tags do
    if (filters.lsr !== "all") {
      const tags = rulesOf(r)
      const hit = filters.lsr === "unmapped" ? tags.length === 0 : tags.some((t) => t.id === filters.lsr)
      if (!hit) return false
    }
    if (filters.type !== "all" && r.type !== filters.type) return false
    if (filters.priority && filters.priority !== "all" && tierOf(r) !== filters.priority) return false
    if (filters.sifOnly && !r.sifPotential) return false
    if (q) {
      const hay = `${r.text} ${r.asset} ${r.activity} ${r.id} ${r.site}`.toLowerCase()
      if (!hay.includes(q)) return false
    }
    return true
  })
}

/* ---------------- headline numbers ---------------- */

export function kpis(reports) {
  const total = reports.length
  const sif = reports.filter((r) => r.sifPotential)
  const share = total ? sif.length / total : 0
  const openSif = sif.filter((r) => r.status !== "closed").length

  const byLsr = new Map()
  for (const r of sif) for (const t of rulesOf(r)) byLsr.set(t.id, (byLsr.get(t.id) ?? 0) + 1)
  const top = [...byLsr.entries()].sort((a, b) => b[1] - a[1])[0]

  const gaps = sif.map((r) => r.severityPotential - r.severityActual)
  const avgGap = gaps.length ? gaps.reduce((a, b) => a + b, 0) / gaps.length : 0

  return {
    total,
    sifCount: sif.length,
    sifShare: share,
    openSif,
    topLsrId: top?.[0] ?? null,
    topLsrCount: top?.[1] ?? 0,
    avgGap,
  }
}

/* ---------------- trend ---------------- */

export function monthlyTrend(reports, keys) {
  const acc = new Map(keys.map((k) => [k, { sif: 0, nonSif: 0 }]))
  for (const r of reports) {
    const bucket = acc.get(monthKey(r.reportedAt))
    if (!bucket) continue
    if (r.sifPotential) bucket.sif += 1
    else bucket.nonSif += 1
  }
  return keys.map((k) => {
    const b = acc.get(k)
    const total = b.sif + b.nonSif
    return {
      key: k,
      label: monthLabel(k),
      sif: b.sif,
      nonSif: b.nonSif,
      total,
      share: total ? b.sif / total : 0,
    }
  })
}

/* ---------------- rankings ---------------- */

/**
 * SIF-precursor density per site = flagged ÷ total. Sites under
 * MIN_REPORTS_FOR_DENSITY are returned but marked `lowSample` and sorted
 * below the ranked ones — a site with 2 reports must not top the list.
 */
export function siteDensity(reports, min = MIN_REPORTS_FOR_DENSITY) {
  const acc = new Map()
  for (const r of reports) {
    if (!acc.has(r.site)) acc.set(r.site, { site: r.site, total: 0, sif: 0 })
    const s = acc.get(r.site)
    s.total += 1
    if (r.sifPotential) s.sif += 1
  }
  return [...acc.values()]
    .map((s) => ({ ...s, density: s.total ? s.sif / s.total : 0, lowSample: s.total < min }))
    .sort((a, b) => {
      if (a.lowSample !== b.lowSample) return a.lowSample ? 1 : -1
      return b.density - a.density || b.sif - a.sif
    })
}

/**
 * Rule breaches across all 9 rules. **Multi-label**: a report tagged with both
 * Hot Work and Work Authorisation counts under both, so `total` summed across
 * rows exceeds the report count. Every rule is always present so the chart
 * doesn't reshuffle its rows under a filter.
 */
export function lsrDistribution(reports) {
  const acc = new Map(LSR.map((r) => [r.id, { id: r.id, total: 0, sif: 0 }]))
  acc.set("unmapped", { id: "unmapped", total: 0, sif: 0 })
  for (const r of reports) {
    const tags = rulesOf(r)
    if (!tags.length) {
      const row = acc.get("unmapped")
      row.total += 1
      if (r.sifPotential) row.sif += 1
      continue
    }
    for (const t of tags) {
      const row = acc.get(t.id) ?? acc.get("unmapped")
      row.total += 1
      if (r.sifPotential) row.sif += 1
    }
  }
  return [...acc.values()]
    .map((row) => ({ ...row, name: lsrName(row.id), short: lsrShort(row.id) }))
    .sort((a, b) => b.sif - a.sif || b.total - a.total)
}

export function barrierBreakdown(reports) {
  const acc = new Map(
    BARRIER_FAILURES.map((b) => [b, { id: b, label: BARRIER_LABEL[b], total: 0, sif: 0 }]),
  )
  for (const r of reports) {
    const row = acc.get(r.precursors.barrierFailure)
    if (!row) continue
    row.total += 1
    if (r.sifPotential) row.sif += 1
  }
  return [...acc.values()].sort((a, b) => b.sif - a.sif || b.total - a.total)
}

export function energyBreakdown(reports) {
  const acc = new Map(HAZARD_ENERGIES.map((e) => [e, { id: e, total: 0, sif: 0 }]))
  for (const r of reports) {
    const row = acc.get(r.precursors.hazardEnergy)
    if (!row) continue
    row.total += 1
    if (r.sifPotential) row.sif += 1
  }
  return [...acc.values()].sort((a, b) => b.sif - a.sif || b.total - a.total)
}

/** Which report categories carry fatal potential — UA/UC vs near-miss vs incident. */
export function typeBreakdown(reports) {
  const acc = new Map(
    REPORT_TYPES.map((t) => [t, { id: t, label: REPORT_TYPE_LABEL[t], total: 0, sif: 0 }]),
  )
  for (const r of reports) {
    const row = acc.get(r.type)
    if (!row) continue
    row.total += 1
    if (r.sifPotential) row.sif += 1
  }
  return [...acc.values()].sort((a, b) => b.sif - a.sif || b.total - a.total)
}

export function activityRanking(reports, min = MIN_REPORTS_FOR_DENSITY) {
  const acc = new Map()
  for (const r of reports) {
    if (!acc.has(r.activity)) acc.set(r.activity, { activity: r.activity, total: 0, sif: 0 })
    const a = acc.get(r.activity)
    a.total += 1
    if (r.sifPotential) a.sif += 1
  }
  return [...acc.values()]
    .map((a) => ({ ...a, density: a.total ? a.sif / a.total : 0, lowSample: a.total < min }))
    .sort((a, b) => b.sif - a.sif || b.density - a.density)
}

/* ---------------- matrices ---------------- */

/**
 * Site × Life-Saving Rule concentration of SIF-flagged reports.
 * Columns are limited to rules that actually appear, so the grid stays legible.
 */
export function siteLsrMatrix(reports) {
  const sifReports = reports.filter((r) => r.sifPotential)
  const sites = [...new Set(reports.map((r) => r.site))].sort()

  const colTotals = new Map()
  for (const r of sifReports) colTotals.set(r.lsr.id, (colTotals.get(r.lsr.id) ?? 0) + 1)
  const cols = [...colTotals.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([id]) => ({ id, short: lsrShort(id), name: lsrName(id) }))

  const cells = new Map()
  for (const r of sifReports) {
    const k = `${r.site}|${r.lsr.id}`
    cells.set(k, (cells.get(k) ?? 0) + 1)
  }
  const max = Math.max(1, ...cells.values())

  const rows = sites.map((site) => ({
    site,
    total: sifReports.filter((r) => r.site === site).length,
    cells: cols.map((c) => ({
      lsr: c.id,
      count: cells.get(`${site}|${c.id}`) ?? 0,
    })),
  }))

  return { rows, cols, max }
}

/** 5×5 actual-vs-potential severity grid. The SIF story lives above the diagonal. */
export function severityMatrix(reports) {
  const counts = new Map()
  for (const r of reports) {
    const k = `${r.severityActual}|${r.severityPotential}`
    counts.set(k, (counts.get(k) ?? 0) + 1)
  }
  const cells = []
  for (let p = 5; p >= 1; p--) {
    for (let a = 1; a <= 5; a++) {
      cells.push({ actual: a, potential: p, count: counts.get(`${a}|${p}`) ?? 0 })
    }
  }
  return { cells, max: Math.max(1, ...counts.values()) }
}

/** Repeat precursor signatures: activity + barrier failure, ranked by SIF count. */
export function precursorPatterns(reports, limit = 8) {
  const acc = new Map()
  for (const r of reports) {
    const k = `${r.activity}|${r.precursors.barrierFailure}|${r.precursors.hazardEnergy}`
    if (!acc.has(k)) {
      acc.set(k, {
        key: k,
        activity: r.activity,
        barrier: r.precursors.barrierFailure,
        energy: r.precursors.hazardEnergy,
        total: 0,
        sif: 0,
        sites: new Set(),
        lastSeen: r.reportedAt,
      })
    }
    const p = acc.get(k)
    p.total += 1
    if (r.sifPotential) p.sif += 1
    p.sites.add(r.site)
    if (r.reportedAt > p.lastSeen) p.lastSeen = r.reportedAt
  }
  return [...acc.values()]
    .map((p) => ({ ...p, siteCount: p.sites.size, sites: [...p.sites].sort() }))
    .filter((p) => p.total > 1)
    .sort((a, b) => b.sif - a.sif || b.total - a.total)
    .slice(0, limit)
}
