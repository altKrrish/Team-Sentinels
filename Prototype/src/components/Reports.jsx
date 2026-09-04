import { FileText } from "lucide-react"
import { REPORT_TYPE_LABEL } from "../lib/contract.js"
import { lsrName, lsrShort } from "../lib/lsr.js"
import { rulesOf } from "../lib/aggregate.js"
import { tierOf } from "../lib/triage.js"
import { fmtDate, truncate } from "../lib/format.js"
import { Badge, Empty, ReviewBadge, SifBadge, TierBadge } from "./kit/index.jsx"
import DataTable from "./kit/DataTable.jsx"
import { LsrIcon } from "./kit/icons.jsx"
import ReportDrawer, { STATUS_TONE } from "./ReportDrawer.jsx"
import Analyze from "./Analyze.jsx"

const TIER_ORDER = { P1: 0, P2: 1, P3: 2 }

/** Every rule the multi-label head tagged, with the extras folded into a chip. */
export function RuleCell({ report }) {
  const tags = rulesOf(report)
  if (!tags.length) {
    return <span className="text-[var(--text-muted)]">Not rule-mapped</span>
  }
  return (
    <span className="flex items-center gap-1.5 whitespace-nowrap">
      <span className="text-[var(--text-secondary)]">
        <LsrIcon id={tags[0].id} size={12.5} />
      </span>
      {lsrShort(tags[0].id)}
      {tags.length > 1 && (
        <span
          className="tnum rounded-full px-1.5 py-[1px] text-[10.5px] font-medium"
          style={{
            background: "var(--surface-2)",
            border: "1px solid var(--border)",
            color: "var(--text-secondary)",
          }}
          title={tags.map((t) => `${lsrName(t.id)} ${(t.confidence * 100).toFixed(0)}%`).join(" · ")}
        >
          +{tags.length - 1}
        </span>
      )}
    </span>
  )
}

export const REPORT_COLUMNS = [
  {
    key: "tier",
    label: "Priority",
    width: 116,
    sortValue: (r) => TIER_ORDER[tierOf(r)],
    render: (r) => <TierBadge tier={tierOf(r)} />,
  },
  {
    key: "reportedAt",
    label: "Date",
    width: 92,
    render: (r) => <span className="tnum whitespace-nowrap">{fmtDate(r.reportedAt)}</span>,
  },
  { key: "site", label: "Site", width: 92 },
  {
    key: "type",
    label: "Type",
    width: 108,
    render: (r) => (
      <span className="whitespace-nowrap text-[var(--text-secondary)]">
        {REPORT_TYPE_LABEL[r.type]}
      </span>
    ),
  },
  {
    key: "lsr",
    label: "Life-Saving Rules",
    width: 158,
    sortValue: (r) => lsrShort(rulesOf(r)[0]?.id ?? "unmapped"),
    render: (r) => <RuleCell report={r} />,
  },
  {
    key: "text",
    label: "Observation",
    render: (r) => (
      <span className="block max-w-[380px] leading-snug text-[var(--text-secondary)]">
        {truncate(r.text, 96)}
      </span>
    ),
  },
  {
    key: "severityScore",
    label: "Severity",
    align: "right",
    width: 84,
    sortValue: (r) => r.severityScore ?? r.severityPotential * 2,
    render: (r) => (
      <span
        className="tnum font-semibold"
        style={{
          color:
            r.severityPotential >= 5
              ? "var(--status-critical)"
              : r.severityPotential === 4
                ? "var(--status-serious)"
                : "var(--text-secondary)",
        }}
        title={`regression score ${(r.severityScore ?? 0).toFixed(2)} of 10 · potential band ${r.severityPotential} of 5`}
      >
        {(r.severityScore ?? 0).toFixed(1)}
      </span>
    ),
  },
  {
    key: "sifConfidence",
    label: "Verdict · review",
    width: 158,
    sortValue: (r) => r.sifConfidence,
    render: (r) => (
      <span className="flex flex-col items-start gap-1">
        <SifBadge sif={r.sifPotential} confidence={r.sifConfidence} />
        {r.sifPotential && <ReviewBadge review={r.review} />}
      </span>
    ),
  },
  {
    key: "status",
    label: "Status",
    width: 100,
    render: (r) => <Badge tone={STATUS_TONE[r.status]}>{r.status}</Badge>,
  },
]

export default function Reports({ reports, active, onActive, onReview, reviewing, onImportReports }) {
  return (
    <div>
      <Analyze onImportReports={onImportReports} />
      <div className="px-4">
        <div className="card overflow-hidden">
          <DataTable
            columns={REPORT_COLUMNS}
            rows={reports}
            getRowKey={(r) => r.id}
            onRowClick={onActive}
            initialSort={{ key: "reportedAt", dir: "desc" }}
            empty={
              <Empty
                icon={FileText}
                title="No reports match these filters"
                hint="Widen the time range, clear the site or rule filter, or turn off “SIF only”."
              />
            }
          />
        </div>
        <p className="mt-2 px-1 text-[11.5px] text-[var(--text-muted)]">
          Sort by Priority to work the queue, or by Severity to rank inside a band. Select a row for
          the model’s verdict, its rule mapping, the features it fired on and the text spans it scored.
        </p>
      </div>

      <ReportDrawer
        report={active}
        onClose={() => onActive(null)}
        onReview={onReview}
        busy={reviewing === active?.id}
      />
    </div>
  )
}
