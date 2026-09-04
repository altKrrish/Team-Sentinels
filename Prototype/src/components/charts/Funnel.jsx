/**
 * Nested subsets of ONE measure, widest first — the screening story as a shape.
 *
 * Plain HTML, not recharts: the marks are four left-anchored bars, and a bar is
 * a div. The stage order is fixed by definition (each stage is a subset of the
 * one above), so the sequential ramp encodes screening depth on a stable ordinal
 * scale — it is not color-by-rank and a filter cannot repaint it.
 */
const RAMP = ["var(--seq-300)", "var(--seq-450)", "var(--seq-600)", "var(--seq-700)"]

export default function Funnel({ stages, max }) {
  return (
    <div className="flex flex-col gap-2">
      {stages.map((s, i) => {
        const w = max > 0 ? Math.max(0.012, s.count / max) : 0
        const share = stages[0]?.count ? s.count / stages[0].count : 0
        return (
          <div key={s.id}>
            <div className="flex items-baseline justify-between gap-3">
              <span className="min-w-0 truncate text-[12.5px] font-medium">{s.label}</span>
              <span className="tnum shrink-0 text-[12px] text-[var(--text-secondary)]">
                {s.count}
                <span className="text-[var(--text-muted)]">
                  {" · "}
                  {(share * 100).toFixed(share < 0.1 ? 1 : 0)}% of intake
                </span>
              </span>
            </div>
            {/* track + 4px rounded data end, anchored left */}
            <div
              className="mt-1 h-[18px] w-full overflow-hidden rounded-[4px]"
              style={{ background: "var(--surface-2)" }}
              role="img"
              aria-label={`${s.label}: ${s.count} reports, ${(share * 100).toFixed(0)} percent of intake`}
            >
              <div
                className="h-full rounded-[4px]"
                style={{
                  width: `${w * 100}%`,
                  background: RAMP[Math.min(i, RAMP.length - 1)],
                  transition: "width 0.35s cubic-bezier(0.2,0.8,0.2,1)",
                }}
              />
            </div>
            {s.note && (
              <p className="mt-1 text-[11.5px] leading-snug text-[var(--text-muted)]">{s.note}</p>
            )}
          </div>
        )
      })}
    </div>
  )
}

/** Scale legend for the ramp — a sequential encode always ships one. */
export function FunnelScale() {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[11px] text-[var(--text-muted)]">Screening depth</span>
      <span aria-hidden className="flex gap-[2px]">
        {RAMP.map((c, i) => (
          <span key={i} className="block h-[8px] w-[18px] rounded-[2px]" style={{ background: c }} />
        ))}
      </span>
      <span className="text-[11px] text-[var(--text-muted)]">narrower / more urgent</span>
    </div>
  )
}
