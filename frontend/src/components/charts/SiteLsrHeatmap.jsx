import { lsrShort } from "../../lib/lsr.js"
import { LsrIcon } from "../kit/icons.jsx"

const STEPS = [
  "var(--seq-100)",
  "var(--seq-200)",
  "var(--seq-300)",
  "var(--seq-400)",
  "var(--seq-500)",
  "var(--seq-600)",
  "var(--seq-700)",
]

/** magnitude -> one hue, light to dark. Zero stays on the plain surface. */
function stepFor(count, max) {
  if (count <= 0) return null
  const t = count / max
  const i = Math.min(STEPS.length - 1, Math.max(0, Math.round(t * (STEPS.length - 1))))
  return { bg: STEPS[i], strong: t >= 0.55 }
}

/**
 * Site x Life-Saving Rule concentration of SIF-flagged reports.
 * Sequential ramp + an explicit scale legend; every cell shows its number, so
 * nothing is encoded in color alone.
 */
export default function SiteLsrHeatmap({ matrix }) {
  const { rows, cols, max } = matrix
  if (!cols.length) {
    return (
      <p className="py-8 text-center text-[12.5px] text-[var(--text-secondary)]">
        No SIF-flagged reports in this selection.
      </p>
    )
  }

  const template = `104px repeat(${cols.length}, minmax(44px, 1fr)) 46px`

  return (
    <div className="min-w-0 overflow-x-auto">
      <div style={{ minWidth: 104 + cols.length * 46 + 46 }}>
        {/* header */}
        <div className="grid gap-[2px]" style={{ gridTemplateColumns: template }}>
          <div />
          {cols.map((c) => (
            <div
              key={c.id}
              title={c.name}
              className="flex flex-col items-center gap-0.5 pb-1.5 text-center"
            >
              <LsrIcon id={c.id} size={13} className="text-[var(--text-secondary)]" />
              <span className="text-[10px] leading-[1.15] text-[var(--text-muted)]">{c.short}</span>
            </div>
          ))}
          <div className="pb-1.5 text-right text-[10px] text-[var(--text-muted)]">All</div>
        </div>

        {/* body */}
        <div className="flex flex-col gap-[2px]">
          {rows.map((r) => (
            <div key={r.site} className="grid gap-[2px]" style={{ gridTemplateColumns: template }}>
              <div className="flex items-center pr-2 text-[12px] font-medium">{r.site}</div>
              {r.cells.map((cell) => {
                const s = stepFor(cell.count, max)
                return (
                  <div
                    key={cell.lsr}
                    title={`${r.site} · ${lsrShort(cell.lsr)} — ${cell.count} SIF-flagged`}
                    className="tnum grid h-8 place-items-center rounded-[4px] text-[12px] font-semibold"
                    style={{
                      background: s ? s.bg : "var(--surface-2)",
                      color: s
                        ? s.strong
                          ? "var(--surface-1)"
                          : "var(--text-primary)"
                        : "var(--text-muted)",
                    }}
                  >
                    {cell.count || "·"}
                  </div>
                )
              })}
              <div className="tnum grid h-8 place-items-center text-[12px] font-semibold text-[var(--text-secondary)]">
                {r.total}
              </div>
            </div>
          ))}
        </div>

        {/* scale legend */}
        <div className="mt-3 flex items-center gap-2">
          <span className="text-[11px] text-[var(--text-muted)]">0</span>
          <div className="flex gap-[2px]">
            {STEPS.map((s) => (
              <span
                key={s}
                aria-hidden
                className="block h-[9px] w-[18px] rounded-[2px]"
                style={{ background: s }}
              />
            ))}
          </div>
          <span className="tnum text-[11px] text-[var(--text-muted)]">{max} flagged</span>
        </div>
      </div>
    </div>
  )
}
