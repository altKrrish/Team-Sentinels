import { useState } from "react"
import { Check, FileUp, Info, ListChecks, Upload } from "lucide-react"
import { classifyText, IS_MOCK } from "../lib/api.js"
import { observationFrom, readReportFile, reportFromClassification } from "../lib/importReports.js"
import { Card, Spinner, SifBadge } from "./kit/index.jsx"
import ReportDrawer from "./ReportDrawer.jsx"

export default function Analyze({ onImportReports }) {
  const [busy, setBusy] = useState(false)
  const [rows, setRows] = useState([])
  const [pending, setPending] = useState([])
  const [importError, setImportError] = useState("")
  const [selected, setSelected] = useState(null)

  async function importFiles(files) {
    if (!files.length || busy) return
    setImportError("")
    setPending([])
    setRows([])
    setSelected(null)
    setBusy(true)
    try {
      const parsed = (await Promise.all([...files].map(readReportFile))).flat()
      const usable = parsed.filter((row) => observationFrom(row))
      if (!usable.length) throw new Error("No observation text was found in the selected files")
      setRows(usable)
      const results = []
      for (const [index, row] of usable.entries()) {
        const response = await classifyText(observationFrom(row))
        results.push({ row, result: response.result, source: response.source, index })
        setPending([...results])
      }
    } catch (error) {
      setImportError(error instanceof Error ? error.message : String(error))
    } finally {
      setBusy(false)
    }
  }

  function pushReports() {
    onImportReports(pending.map(({ row, result }, index) => reportFromClassification(row, result, index)))
    setPending([])
    setRows([])
  }

  return (
    <div className="flex flex-col gap-4 p-4">
      <Card className="p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="flex items-center gap-2 text-[14px] font-semibold">
              <FileUp size={15} style={{ color: "var(--series-1)" }} aria-hidden />
              Import report batch
            </h2>
            <p className="mt-1 max-w-[70ch] text-[12.5px] leading-snug text-[var(--text-secondary)]">
              Upload CSV, JSON or TXT files. Reports are segregated, scored independently, and held here until you push them into Reports. Select any scored report to inspect the model evidence.
            </p>
          </div>
          <label
            className="flex h-[34px] cursor-pointer items-center gap-2 rounded-lg px-3.5 text-[13px] font-medium"
            style={{ background: "var(--series-1)", color: "#fff" }}
          >
            <Upload size={14} aria-hidden />
            Choose files
            <input
              type="file"
              multiple
              accept=".csv,.json,.txt,text/csv,application/json,text/plain"
              className="sr-only"
              onChange={(event) => importFiles(event.target.files)}
            />
          </label>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11.5px] text-[var(--text-muted)]">
          <span>CSV: observation, site, asset, date</span>
          <span>JSON: array of report objects</span>
          <span>TXT: one report per line</span>
        </div>
        {importError && <p className="mt-3 text-[12px] text-[var(--status-critical)]">{importError}</p>}
        {IS_MOCK && (
          <p className="mt-3 flex items-start gap-1.5 text-[11.5px] leading-snug text-[var(--text-muted)]">
            <Info size={12.5} strokeWidth={2} className="mt-[1px] shrink-0" aria-hidden />
            Scored by the bundled demo shim. Set <code className="tnum">VITE_USE_MOCK=false</code> to use the live model.
          </p>
        )}
        {rows.length > 0 && (
          <div className="mt-4 overflow-hidden rounded-lg" style={{ border: "1px solid var(--border)" }}>
            <div className="flex items-center justify-between gap-3 px-3 py-2" style={{ background: "var(--surface-2)" }}>
              <span className="flex items-center gap-2 text-[12px] font-medium">
                <ListChecks size={14} aria-hidden />
                {pending.length} of {rows.length} reports classified
              </span>
              {pending.length === rows.length && !busy && (
                <button
                  onClick={pushReports}
                  className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[12px] font-medium"
                  style={{ background: "var(--success-text)", color: "#fff" }}
                >
                  <Check size={13} aria-hidden /> Push to Reports
                </button>
              )}
            </div>
            <div className="max-h-[250px] overflow-y-auto">
              {pending.map(({ row, result }, index) => (
                <button
                  key={`${observationFrom(row)}-${index}`}
                  onClick={() => setSelected(reportFromClassification(row, result, index))}
                  className="flex w-full items-center gap-3 px-3 py-2 text-left transition-colors hover:bg-[var(--surface-2)]"
                  style={{ borderTop: "1px solid var(--border)" }}
                >
                  <SifBadge sif={result.sifPotential} confidence={result.sifConfidence} />
                  <span className="min-w-0 flex-1 truncate text-[12px] text-[var(--text-secondary)]">{observationFrom(row)}</span>
                  <span className="shrink-0 text-[11px] text-[var(--text-muted)]">View report</span>
                </button>
              ))}
              {busy && <p className="px-3 py-2 text-[12px] text-[var(--text-muted)]"><Spinner /> Scoring next report…</p>}
            </div>
          </div>
        )}
      </Card>
      <ReportDrawer report={selected} onClose={() => setSelected(null)} />
    </div>
  )
}
