export const pct = (n, digits = 0) =>
  `${(n * 100).toFixed(digits)}%`

export const num = (n) => new Intl.NumberFormat("en-IN").format(n)

const DATE_FMT = new Intl.DateTimeFormat("en-IN", {
  day: "2-digit",
  month: "short",
  year: "numeric",
})

export const fmtDate = (iso) => DATE_FMT.format(new Date(iso))

export const monthKey = (iso) => iso.slice(0, 7) // YYYY-MM

const MONTH_FMT = new Intl.DateTimeFormat("en-IN", { month: "short" })

/** "2026-03" -> "Mar" (with year appended each January for a readable axis) */
export const monthLabel = (key) => {
  const [y, m] = key.split("-")
  const d = new Date(Number(y), Number(m) - 1, 1)
  const label = MONTH_FMT.format(d)
  return m === "01" ? `${label} ${y.slice(2)}` : label
}

/** Truncate on a word boundary so table cells don't cut mid-word. */
export const truncate = (s, max = 90) => {
  if (s.length <= max) return s
  const cut = s.slice(0, max)
  const sp = cut.lastIndexOf(" ")
  return `${cut.slice(0, sp > max * 0.6 ? sp : max)}…`
}

export const titleCase = (s) =>
  s.replace(/(^|[\s-])([a-z])/g, (_, p, c) => p + c.toUpperCase())
