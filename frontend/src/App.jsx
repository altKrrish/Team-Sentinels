import { useCallback, useEffect, useMemo, useState } from "react"
import { fetchReports, submitReview } from "./lib/api.js"
import { applyFilters, EMPTY_FILTERS, monthAxis } from "./lib/aggregate.js"
import Shell from "./components/Shell.jsx"
import FilterBar from "./components/FilterBar.jsx"
import Dashboard from "./components/Dashboard.jsx"
import Triage from "./components/Triage.jsx"
import Reports from "./components/Reports.jsx"
import Patterns from "./components/Patterns.jsx"

const THEME_KEY = "closecall.theme"

function LoadingReports() {
  return (
    <div
      role="status"
      aria-label="Loading CloseCall dashboard"
      className="animate-pulse p-4"
      style={{ animationDuration: "1.8s" }}
    >
      <span className="sr-only">Loading CloseCall dashboard</span>

      <div className="card flex flex-wrap gap-2 p-3">
        {["w-[88px]", "w-[92px]", "w-[108px]", "w-[116px]", "w-[180px]"].map((width, index) => (
          <div
            key={index}
            className={`h-8 rounded-md ${width}`}
            style={{ background: "var(--surface-2)" }}
          />
        ))}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {[1, 2, 3, 4].map((tile) => (
          <div key={tile} className="card h-[112px] p-3.5">
            <div className="h-3 w-24 rounded" style={{ background: "var(--surface-2)" }} />
            <div className="mt-4 h-7 w-16 rounded" style={{ background: "var(--surface-2)" }} />
            <div className="mt-3 h-3 w-32 rounded" style={{ background: "var(--surface-2)" }} />
          </div>
        ))}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
        {[1, 2].map((chart) => (
          <div key={chart} className="card h-[310px] p-4">
            <div className="h-4 w-40 rounded" style={{ background: "var(--surface-2)" }} />
            <div className="mt-1 h-3 w-56 rounded" style={{ background: "var(--surface-2)" }} />
            <div className="mt-7 flex h-[220px] items-end gap-3">
              {[42, 68, 54, 82, 61, 94, 73, 48, 76, 58, 88, 64].map((height, index) => (
                <div
                  key={index}
                  className="min-w-0 flex-1 rounded-t"
                  style={{ height: `${height}%`, background: "var(--surface-2)" }}
                />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function App() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem(THEME_KEY) ?? "light",
  )
  const [screen, setScreen] = useState("dashboard")
  const [filters, setFilters] = useState(EMPTY_FILTERS)
  const [data, setData] = useState({ reports: [], source: "demo" })
  const [loading, setLoading] = useState(true)

  /* Human-in-the-loop decisions made in this session, keyed by report id. Held
     here rather than inside a screen so a decision recorded from the queue is
     visible on the reports table and in every count. */
  const [reviews, setReviews] = useState({})
  const [reviewing, setReviewing] = useState(null)
  const [activeId, setActiveId] = useState(null)

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme)
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  useEffect(() => {
    let alive = true
    fetchReports().then((res) => {
      if (!alive) return
      setData(res)
      setLoading(false)
    })
    return () => {
      alive = false
    }
  }, [])

  const reports = useMemo(() => {
    const ids = Object.keys(reviews)
    if (!ids.length) return data.reports
    return data.reports.map((r) => (reviews[r.id] ? { ...r, review: reviews[r.id] } : r))
  }, [data.reports, reviews])

  const axis = useMemo(() => monthAxis(reports), [reports])
  const filtered = useMemo(() => applyFilters(reports, filters, axis), [reports, filters, axis])

  const sites = useMemo(() => [...new Set(reports.map((r) => r.site))].sort(), [reports])

  const active = useMemo(
    () => (activeId ? (filtered.find((r) => r.id === activeId) ?? null) : null),
    [filtered, activeId],
  )

  const onActive = useCallback((report) => setActiveId(report?.id ?? null), [])

  const onReview = useCallback(async (id, state) => {
    setReviewing(id)
    const res = await submitReview(id, { state })
    setReviews((m) => ({ ...m, [id]: res.review }))
    setReviewing(null)
  }, [])

  const onImportReports = useCallback((imported) => {
    setData((current) => ({ ...current, reports: [...imported, ...current.reports] }))
  }, [])

  const rowProps = { active, onActive, onReview, reviewing, onImportReports }

  return (
    <Shell
      screen={screen}
      onScreen={setScreen}
      source={data.source}
      error={data.error}
      theme={theme}
      onTheme={setTheme}
    >
      {loading ? (
        <LoadingReports />
      ) : (
        <>
          <FilterBar
            filters={filters}
            onChange={setFilters}
            sites={sites}
            shown={filtered.length}
            total={reports.length}
          />

          {screen === "dashboard" && (
            <Dashboard reports={filtered} axis={axis} filters={filters} />
          )}
          {screen === "triage" && <Triage reports={filtered} {...rowProps} />}
          {screen === "reports" && <Reports reports={filtered} {...rowProps} />}
          {screen === "patterns" && <Patterns reports={filtered} />}
        </>
      )}
    </Shell>
  )
}
