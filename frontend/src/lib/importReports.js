const TEXT_KEYS = ["text", "observation", "description", "narrative", "report"]

function valueFor(row, keys, fallback = "") {
  const key = Object.keys(row ?? {}).find((rowKey) => {
    const candidate = keys.some((keyName) => rowKey.toLowerCase() === keyName.toLowerCase())
    return candidate && String(row[rowKey]).trim()
  })
  return key ? String(row[key]).trim() : fallback
}

function parseCsv(text) {
  const rows = []
  let row = []
  let cell = ""
  let quoted = false

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i]
    const next = text[i + 1]
    if (char === '"' && quoted && next === '"') {
      cell += '"'
      i += 1
    } else if (char === '"') {
      quoted = !quoted
    } else if (char === "," && !quoted) {
      row.push(cell.trim())
      cell = ""
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") i += 1
      row.push(cell.trim())
      if (row.some(Boolean)) rows.push(row)
      row = []
      cell = ""
    } else {
      cell += char
    }
  }
  row.push(cell.trim())
  if (row.some(Boolean)) rows.push(row)
  if (rows.length < 2) return []

  const headers = rows[0].map((header, index) => header || `column${index + 1}`)
  return rows.slice(1).map((values) =>
    Object.fromEntries(headers.map((header, index) => [header.trim().toLowerCase(), values[index] ?? ""])),
  )
}

function normalizeRows(value) {
  if (Array.isArray(value)) return value
  if (value && Array.isArray(value.reports)) return value.reports
  if (value && typeof value === "object") return [value]
  return []
}

export async function readReportFile(file) {
  const text = await file.text()
  const extension = file.name.split(".").pop()?.toLowerCase()

  if (extension === "json") {
    let parsed
    try {
      parsed = JSON.parse(text)
    } catch {
      throw new Error(`${file.name} is not valid JSON`)
    }
    return normalizeRows(parsed).map((row) => ({ ...row, sourceName: file.name }))
  }

  if (extension === "csv") {
    return parseCsv(text).map((row) => ({ ...row, sourceName: file.name }))
  }

  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => ({ text: line, sourceName: file.name }))
}

export function observationFrom(row) {
  return valueFor(row, TEXT_KEYS)
}

export function reportFromClassification(row, classification, index) {
  const today = new Date().toISOString().slice(0, 10)
  const type = valueFor(row, ["type", "reporttype"], "near-miss")
  const allowedTypes = new Set(["UA", "UC", "near-miss", "incident"])

  return {
    ...classification,
    id: `IMP-${today.replaceAll("-", "")}-${Date.now().toString(36)}-${String(index + 1).padStart(3, "0")}`,
    reportedAt: valueFor(row, ["reportedat", "date", "reported_date"], today),
    type: allowedTypes.has(type) ? type : "near-miss",
    site: valueFor(row, ["site", "location"], "Imported report"),
    asset: valueFor(row, ["asset", "equipment"], "Not specified"),
    department: valueFor(row, ["department", "function"], "Not specified"),
    activity: valueFor(row, ["activity", "task"], "Imported observation"),
    text: observationFrom(row),
    reportedBy: valueFor(row, ["reportedby", "reporter", "author"], "Imported · pending attribution"),
    status: "open",
    review: { state: "pending", by: null, at: null, note: null },
  }
}
