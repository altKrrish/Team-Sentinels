import { RotateCcw, Search, TriangleAlert } from "lucide-react"
import { REPORT_TYPES, REPORT_TYPE_LABEL } from "../lib/contract.js"
import { LSR } from "../lib/lsr.js"
import { PRIORITY_TIERS } from "../lib/triage.js"
import { EMPTY_FILTERS } from "../lib/aggregate.js"
import { num } from "../lib/format.js"

const RANGES = [
  { months: 3, label: "3M" },
  { months: 6, label: "6M" },
  { months: 12, label: "12M" },
]

function Select({ label, value, onChange, options }) {
  return (
    <label className="flex items-center gap-1.5">
      <span className="text-[11.5px] text-[var(--text-muted)]">{label}</span>
      <select className="ctl max-w-[168px]" value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  )
}

/**
 * One filter row, above everything it scopes. Never inside a chart card —
 * every panel on the screen reads this same slice.
 */
export default function FilterBar({ filters, onChange, sites, shown, total }) {
  const set = (patch) => onChange({ ...filters, ...patch })
  const dirty = JSON.stringify(filters) !== JSON.stringify(EMPTY_FILTERS)

  return (
    <div
      className="sticky top-0 z-20 flex flex-wrap items-center gap-x-3 gap-y-2 px-4 py-2.5"
      style={{
        background: "color-mix(in srgb, var(--page) 88%, transparent)",
        backdropFilter: "blur(8px)",
        borderBottom: "1px solid var(--border)",
      }}
    >
      {/* time range */}
      <div
        role="group"
        aria-label="Time range"
        className="flex overflow-hidden rounded-lg"
        style={{ border: "1px solid var(--border-strong)" }}
      >
        {RANGES.map((r) => {
          const active = filters.months === r.months
          return (
            <button
              key={r.months}
              onClick={() => set({ months: r.months })}
              aria-pressed={active}
              className="tnum h-[30px] px-2.5 text-[12px] font-medium transition-colors"
              style={{
                background: active ? "var(--surface-2)" : "transparent",
                color: active ? "var(--text-primary)" : "var(--text-muted)",
              }}
            >
              {r.label}
            </button>
          )
        })}
      </div>

      <Select
        label="Site"
        value={filters.site}
        onChange={(site) => set({ site })}
        options={[
          { value: "all", label: "All sites" },
          ...sites.map((s) => ({ value: s, label: s })),
        ]}
      />

      <Select
        label="Rule"
        value={filters.lsr}
        onChange={(lsr) => set({ lsr })}
        options={[
          { value: "all", label: "All rules" },
          ...LSR.map((r) => ({ value: r.id, label: r.name })),
          { value: "unmapped", label: "Not rule-mapped" },
        ]}
      />

      <Select
        label="Type"
        value={filters.type}
        onChange={(type) => set({ type })}
        options={[
          { value: "all", label: "All types" },
          ...REPORT_TYPES.map((t) => ({ value: t, label: REPORT_TYPE_LABEL[t] })),
        ]}
      />

      <Select
        label="Priority"
        value={filters.priority ?? "all"}
        onChange={(priority) => set({ priority })}
        options={[
          { value: "all", label: "All priorities" },
          ...PRIORITY_TIERS.map((t) => ({ value: t.id, label: `${t.id} · ${t.label}` })),
        ]}
      />

      <button
        onClick={() => set({ sifOnly: !filters.sifOnly })}
        aria-pressed={filters.sifOnly}
        className="flex h-[30px] items-center gap-1.5 rounded-lg px-2.5 text-[12px] font-medium transition-colors"
        style={{
          border: `1px solid ${
            filters.sifOnly
              ? "color-mix(in srgb, var(--status-critical) 42%, transparent)"
              : "var(--border-strong)"
          }`,
          background: filters.sifOnly
            ? "color-mix(in srgb, var(--status-critical) 11%, transparent)"
            : "transparent",
          color: filters.sifOnly ? "var(--status-critical)" : "var(--text-secondary)",
        }}
      >
        <TriangleAlert size={12.5} strokeWidth={2.2} aria-hidden />
        SIF only
      </button>

      <label className="relative flex items-center">
        <Search
          size={13}
          strokeWidth={2}
          aria-hidden
          className="pointer-events-none absolute left-2.5 text-[var(--text-muted)]"
        />
        <input
          className="ctl w-[168px] pl-[28px]"
          type="search"
          placeholder="Search text, asset, id…"
          aria-label="Search reports"
          value={filters.q}
          onChange={(e) => set({ q: e.target.value })}
        />
      </label>

      <div className="ml-auto flex items-center gap-2">
        <span className="tnum text-[12px] text-[var(--text-secondary)]">
          {num(shown)} of {num(total)} reports
        </span>
        {dirty && (
          <button
            onClick={() => onChange(EMPTY_FILTERS)}
            className="flex h-[30px] items-center gap-1.5 rounded-lg px-2.5 text-[12px] font-medium transition-colors hover:bg-[var(--surface-2)]"
            style={{ border: "1px solid var(--border-strong)", color: "var(--text-secondary)" }}
          >
            <RotateCcw size={12.5} strokeWidth={2} aria-hidden />
            Reset
          </button>
        )}
      </div>
    </div>
  )
}
