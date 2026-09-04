import { CircleCheck, Info, TriangleAlert } from "lucide-react"
import { SEVERITY_LABEL } from "../../lib/contract.js"

/** Tone follows the POTENTIAL severity — the axis the SIF model exists to expose. */
function toneFor(potential) {
  if (potential >= 5) return { key: "critical", color: "var(--status-critical)", label: "Fatal potential" }
  if (potential === 4) return { key: "serious", color: "var(--status-serious)", label: "Major potential" }
  if (potential === 3) return { key: "warning", color: "var(--status-warning)", label: "Moderate potential" }
  return { key: "good", color: "var(--success-text)", label: "Low potential" }
}

const LEGEND = [
  { label: "Fatal potential (5)", color: "var(--status-critical)", Icon: TriangleAlert },
  { label: "Major (4)", color: "var(--status-serious)", Icon: TriangleAlert },
  { label: "Moderate (3)", color: "var(--status-warning)", Icon: Info },
  { label: "Minor / negligible (1–2)", color: "var(--success-text)", Icon: CircleCheck },
]

/**
 * 5x5 actual vs potential severity. Everything well above the diagonal is a
 * report whose outcome was mild but whose potential was not — the whole point
 * of separating SIF precursors from ordinary severity triage.
 */
export default function RiskMatrix({ matrix }) {
  const { cells, max } = matrix
  const potentials = [5, 4, 3, 2, 1]

  return (
    <div className="min-w-0">
      <div className="flex gap-2">
        {/* y axis title */}
        <div
          className="flex shrink-0 items-center justify-center text-[10.5px] tracking-[0.04em] text-[var(--text-muted)]"
          style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}
        >
          POTENTIAL SEVERITY
        </div>

        <div className="min-w-0 flex-1">
          {potentials.map((p) => {
            const tone = toneFor(p)
            return (
              <div key={p} className="mb-[2px] flex items-stretch gap-[2px]">
                <div
                  className="tnum flex w-[18px] shrink-0 items-center justify-end pr-1 text-[11px] font-medium text-[var(--text-secondary)]"
                  title={SEVERITY_LABEL[p]}
                >
                  {p}
                </div>
                {[1, 2, 3, 4, 5].map((a) => {
                  const cell = cells.find((c) => c.actual === a && c.potential === p)
                  const count = cell?.count ?? 0
                  const t = count > 0 ? 0.12 + (count / max) * 0.3 : 0
                  return (
                    <div
                      key={a}
                      title={`${count} report${count === 1 ? "" : "s"} — actual ${SEVERITY_LABEL[a]}, potential ${SEVERITY_LABEL[p]}`}
                      className="tnum grid h-9 flex-1 place-items-center rounded-[4px] text-[12px] font-semibold"
                      style={{
                        background:
                          count > 0
                            ? `color-mix(in srgb, ${tone.color} ${(t * 100).toFixed(0)}%, var(--surface-1))`
                            : "var(--surface-2)",
                        border:
                          count > 0
                            ? `1px solid color-mix(in srgb, ${tone.color} 34%, transparent)`
                            : "1px solid var(--border)",
                        color: count > 0 ? "var(--text-primary)" : "var(--text-muted)",
                      }}
                    >
                      {count || "·"}
                    </div>
                  )
                })}
              </div>
            )
          })}

          {/* x axis */}
          <div className="mt-1 flex gap-[2px]">
            <div className="w-[18px] shrink-0" />
            {[1, 2, 3, 4, 5].map((a) => (
              <div
                key={a}
                className="tnum flex-1 text-center text-[11px] text-[var(--text-secondary)]"
                title={SEVERITY_LABEL[a]}
              >
                {a}
              </div>
            ))}
          </div>
          <div className="mt-1 pl-[20px] text-center text-[10.5px] tracking-[0.04em] text-[var(--text-muted)]">
            ACTUAL SEVERITY
          </div>
        </div>
      </div>

      <ul className="mt-3 flex flex-wrap gap-x-3.5 gap-y-1.5">
        {LEGEND.map(({ label, color, Icon }) => (
          <li
            key={label}
            className="flex items-center gap-1.5 text-[11px] text-[var(--text-secondary)]"
          >
            <Icon size={11.5} strokeWidth={2.2} aria-hidden style={{ color }} />
            {label}
          </li>
        ))}
      </ul>
    </div>
  )
}
