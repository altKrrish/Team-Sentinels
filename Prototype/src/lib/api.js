/**
 * The only place this app touches the network.
 *
 * Swap the mock fixtures for your SIF model by setting, in `.env.local`:
 *   VITE_USE_MOCK=false
 *   VITE_API_BASE=https://your-model-host
 *
 * Contract lives in ./contract.js. If a live fetch fails, the UI falls back to
 * the demo fixtures and reports `source: 'fallback'` so the header can say so
 * out loud — a dashboard that silently shows sample data is worse than an error.
 */
import { MOCK_REPORTS, mockClassify } from "./mock/reports.js"

const env = import.meta.env ?? {}
const USE_MOCK = String(env.VITE_USE_MOCK ?? "true") !== "false"
const BASE = (env.VITE_API_BASE ?? "").replace(/\/$/, "")

const wait = (ms) => new Promise((res) => setTimeout(res, ms))

export const IS_MOCK = USE_MOCK

/**
 * @returns {Promise<{reports: import('./contract.js').Report[],
 *                    source: 'demo'|'live'|'fallback', error?: string}>}
 */
export async function fetchReports() {
  if (USE_MOCK) {
    await wait(180) // let the loading state actually render
    return { reports: MOCK_REPORTS, source: "demo" }
  }
  try {
    const res = await fetch(`${BASE}/reports`, { headers: { Accept: "application/json" } })
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`)
    const data = await res.json()
    const reports = Array.isArray(data) ? data : data?.reports
    if (!Array.isArray(reports)) throw new Error("Expected an array of reports")
    return { reports, source: "live" }
  } catch (err) {
    return {
      reports: MOCK_REPORTS,
      source: "fallback",
      error: err instanceof Error ? err.message : String(err),
    }
  }
}

/**
 * @param {string} text free-text observation
 * @returns {Promise<{result: import('./contract.js').Classification,
 *                    source: 'demo'|'live'|'fallback', error?: string}>}
 */
export async function classifyText(text) {
  if (USE_MOCK) {
    await wait(650) // visible "classifying…" state
    return { result: mockClassify(text), source: "demo" }
  }
  try {
    const res = await fetch(`${BASE}/classify`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ text }),
    })
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`)
    const result = await res.json()
    if (typeof result?.sifPotential !== "boolean") {
      throw new Error("Response is missing sifPotential")
    }
    return { result, source: "live" }
  } catch (err) {
    return {
      result: mockClassify(text),
      source: "fallback",
      error: err instanceof Error ? err.message : String(err),
    }
  }
}

/**
 * The human-in-the-loop write. An HSE professional confirms or overrides the
 * model's verdict; the engine never closes a report on its own.
 *
 * On mock (or on a failed live POST) the review is returned unchanged so the UI
 * still records the decision locally — the reviewer's click is never silently
 * dropped, and the header keeps saying which mode we're in.
 *
 * @param {string} id
 * @param {{state: import('./contract.js').ReviewState, note?: string|null, by?: string}} decision
 * @returns {Promise<{review: import('./contract.js').Review, source: 'demo'|'live'|'fallback', error?: string}>}
 */
export async function submitReview(id, { state, note = null, by = "You · HSE reviewer" }) {
  const review = { state, by, at: new Date().toISOString().slice(0, 10), note }

  if (USE_MOCK) {
    await wait(220)
    return { review, source: "demo" }
  }
  try {
    const res = await fetch(`${BASE}/reports/${encodeURIComponent(id)}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ state, note }),
    })
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`)
    const saved = await res.json()
    return { review: saved?.state ? saved : review, source: "live" }
  } catch (err) {
    return {
      review,
      source: "fallback",
      error: err instanceof Error ? err.message : String(err),
    }
  }
}
