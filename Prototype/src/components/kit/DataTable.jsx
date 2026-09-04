import { useMemo, useState } from "react"
import { ChevronDown, ChevronUp } from "lucide-react"

/**
 * Sortable table. Columns:
 *   { key, label, align?: 'right', width?, render?(row), sortValue?(row), sortable? }
 */
export default function DataTable({
  columns,
  rows,
  getRowKey,
  onRowClick,
  initialSort,
  maxHeight = "calc(100vh - 300px)",
  empty,
}) {
  const [sort, setSort] = useState(initialSort ?? { key: null, dir: "desc" })

  const sorted = useMemo(() => {
    if (!sort.key) return rows
    const col = columns.find((c) => c.key === sort.key)
    if (!col) return rows
    const val = col.sortValue ?? ((r) => r[col.key])
    return [...rows].sort((a, b) => {
      const av = val(a)
      const bv = val(b)
      const cmp =
        typeof av === "number" && typeof bv === "number"
          ? av - bv
          : String(av).localeCompare(String(bv))
      return sort.dir === "asc" ? cmp : -cmp
    })
  }, [rows, sort, columns])

  const toggle = (key) =>
    setSort((s) =>
      s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: "desc" },
    )

  if (!rows.length && empty) return empty

  return (
    <div className="overflow-auto" style={{ maxHeight }}>
      <table className="w-full border-collapse text-[12.5px]">
        <thead>
          <tr>
            {columns.map((c) => {
              const active = sort.key === c.key
              const sortable = c.sortable !== false
              return (
                <th
                  key={c.key}
                  scope="col"
                  style={{
                    width: c.width,
                    textAlign: c.align === "right" ? "right" : "left",
                    background: "var(--surface-2)",
                    borderBottom: "1px solid var(--border-strong)",
                  }}
                  className="sticky top-0 z-1 p-0 font-medium"
                  aria-sort={active ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}
                >
                  {sortable ? (
                    <button
                      onClick={() => toggle(c.key)}
                      className="flex h-8 w-full items-center gap-1 px-2.5 text-[11.5px] font-medium tracking-[0.01em] transition-colors hover:text-[var(--text-primary)]"
                      style={{
                        color: active ? "var(--text-primary)" : "var(--text-secondary)",
                        justifyContent: c.align === "right" ? "flex-end" : "flex-start",
                      }}
                    >
                      {c.label}
                      {active ? (
                        sort.dir === "asc" ? (
                          <ChevronUp size={12} strokeWidth={2.4} aria-hidden />
                        ) : (
                          <ChevronDown size={12} strokeWidth={2.4} aria-hidden />
                        )
                      ) : (
                        <ChevronDown
                          size={12}
                          strokeWidth={2.4}
                          aria-hidden
                          style={{ opacity: 0.22 }}
                        />
                      )}
                    </button>
                  ) : (
                    <span
                      className="flex h-8 items-center px-2.5 text-[11.5px] tracking-[0.01em]"
                      style={{
                        color: "var(--text-secondary)",
                        justifyContent: c.align === "right" ? "flex-end" : "flex-start",
                      }}
                    >
                      {c.label}
                    </span>
                  )}
                </th>
              )
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr
              key={getRowKey(row)}
              tabIndex={onRowClick ? 0 : undefined}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              onKeyDown={
                onRowClick
                  ? (e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault()
                        onRowClick(row)
                      }
                    }
                  : undefined
              }
              className={
                onRowClick
                  ? "cursor-pointer transition-colors hover:bg-[var(--surface-2)] focus-visible:bg-[var(--surface-2)]"
                  : undefined
              }
            >
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={c.align === "right" ? "num px-2.5 py-2 align-top" : "px-2.5 py-2 align-top"}
                  style={{
                    textAlign: c.align === "right" ? "right" : "left",
                    borderBottom: "1px solid var(--border)",
                  }}
                >
                  {c.render ? c.render(row) : row[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
