import { useId, useState } from "react"
import { ChartColumnBig, Table2 } from "lucide-react"
import { Card } from "./index.jsx"

/**
 * Every chart lives in one of these. It supplies the title, the legend
 * (identity is never color-alone), and a table view so no value is
 * tooltip-only. Filters never live inside a chart card.
 */
export default function ChartFrame({
  title,
  hint,
  legend,
  footnote,
  table,
  height,
  children,
}) {
  const [view, setView] = useState("chart")
  const id = useId()
  const showToggle = Boolean(table)
  const definite = Boolean(height) && view === "chart"

  return (
    <Card className="flex min-w-0 flex-col p-4">
      <div className="mb-1 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-[13.5px] leading-tight font-semibold tracking-[-0.005em]">{title}</h3>
          {hint && (
            <p className="mt-1 text-[12px] leading-snug text-[var(--text-secondary)]">{hint}</p>
          )}
        </div>

        {showToggle && (
          <div
            role="tablist"
            aria-label={`${title} view`}
            className="flex shrink-0 overflow-hidden rounded-lg"
            style={{ border: "1px solid var(--border-strong)" }}
          >
            {[
              { key: "chart", Icon: ChartColumnBig, label: "Chart" },
              { key: "table", Icon: Table2, label: "Table" },
            ].map(({ key, Icon, label }) => (
              <button
                key={key}
                role="tab"
                aria-selected={view === key}
                aria-controls={`${id}-${key}`}
                onClick={() => setView(key)}
                title={`${label} view`}
                className="flex h-[26px] items-center gap-1 px-2 text-[11.5px] font-medium transition-colors"
                style={{
                  background: view === key ? "var(--surface-2)" : "transparent",
                  color: view === key ? "var(--text-primary)" : "var(--text-muted)",
                }}
              >
                <Icon size={12.5} strokeWidth={2} aria-hidden />
                <span className="hidden sm:inline">{label}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {legend?.length > 0 && (
        <ul className="mt-2 mb-1 flex flex-wrap items-center gap-x-4 gap-y-1">
          {legend.map((l) => (
            <li
              key={l.label}
              className="flex items-center gap-1.5 text-[11.5px] text-[var(--text-secondary)]"
            >
              <span
                aria-hidden
                className="inline-block shrink-0"
                style={{
                  width: l.shape === "line" ? 14 : 9,
                  height: l.shape === "line" ? 2 : 9,
                  borderRadius: l.shape === "line" ? 2 : 2,
                  background: l.color,
                }}
              />
              {l.label}
            </li>
          ))}
        </ul>
      )}

      {/*
        A chart panel needs a *definite* height for ResponsiveContainer's
        height="100%" to resolve. `flex-1` would set flex-basis:0% and, inside an
        auto-height column card, collapse the panel to zero, so an explicit
        height opts out of flexing entirely.
      */}
      <div
        id={`${id}-${view}`}
        role="tabpanel"
        className={definite ? "mt-2 min-w-0" : "mt-2 min-w-0 flex-1"}
        style={definite ? { height, flex: "0 0 auto" } : undefined}
      >
        {view === "chart" ? children : table}
      </div>

      {footnote && (
        <p className="mt-3 text-[11.5px] leading-snug text-[var(--text-muted)]">{footnote}</p>
      )}
    </Card>
  )
}

/** Compact table used as the table-view of a chart. */
export function MiniTable({ head, rows, align = [] }) {
  return (
    <div className="max-h-[280px] overflow-auto">
      <table className="w-full border-collapse text-[12px]">
        <thead>
          <tr>
            {head.map((h, i) => (
              <th
                key={h}
                scope="col"
                className="sticky top-0 z-1 px-2 py-1.5 font-medium whitespace-nowrap"
                style={{
                  background: "var(--surface-2)",
                  color: "var(--text-secondary)",
                  textAlign: align[i] === "right" ? "right" : "left",
                  borderBottom: "1px solid var(--border)",
                }}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri}>
              {r.map((c, ci) => (
                <td
                  key={ci}
                  className={align[ci] === "right" ? "tnum px-2 py-1.5" : "px-2 py-1.5"}
                  style={{
                    textAlign: align[ci] === "right" ? "right" : "left",
                    borderBottom: "1px solid var(--border)",
                    color: ci === 0 ? "var(--text-primary)" : "var(--text-secondary)",
                  }}
                >
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
