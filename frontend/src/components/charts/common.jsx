/** Shared chart chrome. Colors are CSS vars, so a theme flip repaints itself. */

export const AXIS = {
  stroke: "var(--axis)",
  tickLine: false,
  tick: { fontSize: 11.5 },
}

export function ChartTooltip({ active, payload, label, rows }) {
  if (!active || !payload?.length) return null
  const lines = rows ? rows(payload[0].payload, payload) : payload.map((p) => ({
    label: p.name,
    value: p.value,
    color: p.color ?? p.fill,
  }))

  return (
    <div
      className="pointer-events-none rounded-lg px-2.5 py-2 text-[12px]"
      style={{
        background: "var(--surface-1)",
        border: "1px solid var(--border-strong)",
        boxShadow: "var(--shadow-pop)",
        color: "var(--text-primary)",
        minWidth: 132,
      }}
    >
      {label != null && (
        <div className="mb-1 text-[11.5px] font-medium text-[var(--text-secondary)]">{label}</div>
      )}
      <ul className="flex flex-col gap-1">
        {lines.map((l) => (
          <li key={l.label} className="flex items-center justify-between gap-3 whitespace-nowrap">
            <span className="flex items-center gap-1.5 text-[var(--text-secondary)]">
              {l.color && (
                <span
                  aria-hidden
                  className="inline-block size-[8px] rounded-[2px]"
                  style={{ background: l.color }}
                />
              )}
              {l.label}
            </span>
            <span className="tnum font-medium">{l.value}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/**
 * Selective direct labels: only the final point of a series gets a number.
 * Ink, not series color — the line it sits against carries the identity.
 */
export function endLabel(total) {
  return function EndLabel({ x, y, value, index }) {
    if (index !== total - 1) return null
    return (
      <text
        x={x + 8}
        y={y}
        dy={4}
        fontSize={11.5}
        fontWeight={600}
        fill="var(--text-primary)"
        style={{ fontVariantNumeric: "tabular-nums" }}
      >
        {value}
      </text>
    )
  }
}
