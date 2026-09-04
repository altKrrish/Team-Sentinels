import { CircleCheck, Hourglass, PenLine } from "lucide-react"
import { REPORT_TYPE_LABEL, REVIEW_LABEL } from "../lib/contract.js"
import { tierOf, TIER_BY_ID } from "../lib/triage.js"
import { fmtDate } from "../lib/format.js"
import { Badge, ReviewBadge, Spinner, TierBadge } from "./kit/index.jsx"
import Drawer from "./kit/Drawer.jsx"
import VerdictPanel from "./Verdict.jsx"

const STATUS_TONE = { open: "serious", "in-progress": "warning", closed: "good" }

/**
 * The decisions an HSE officer can record.
 *
 * The two rulings come first: they are what closes a flag out. `in-progress`
 * sits last because it is not a ruling - the officer has taken the report but
 * has not decided, which is the honest state for anything needing a site visit.
 * It is styled in ink rather than a status colour because it makes no claim
 * about the hazard, only about who is holding the file. At a narrow drawer width
 * the row wraps and it lands on its own line, which is the right grouping.
 */
const REVIEW_ACTIONS = [
  {
    state: "confirmed",
    label: "Confirm SIF potential",
    icon: CircleCheck,
    color: "var(--success-text)",
  },
  {
    state: "overridden",
    label: "Override — not a precursor",
    icon: PenLine,
    color: "var(--series-1)",
  },
  {
    state: "in-progress",
    label: "Mark in progress",
    icon: Hourglass,
    color: "var(--text-secondary)",
  },
]

function ReviewButton({ action, active, busy, onClick }) {
  const Icon = action.icon
  return (
    <button
      onClick={onClick}
      disabled={busy}
      aria-pressed={active}
      className="flex h-8 items-center gap-1.5 rounded-lg px-3 text-[12.5px] font-medium transition-colors disabled:opacity-50"
      style={{
        border: `1px solid color-mix(in srgb, ${action.color} ${active ? 55 : 32}%, transparent)`,
        background: active ? `color-mix(in srgb, ${action.color} 13%, transparent)` : "transparent",
        color: action.color,
      }}
    >
      <Icon size={13} strokeWidth={2.2} aria-hidden />
      {action.label}
    </button>
  )
}

function MetaRow({ label, value }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-[5px]">
      <span className="text-[11.5px] text-[var(--text-muted)]">{label}</span>
      <span className="text-right text-[12.5px]">{value}</span>
    </div>
  )
}

/**
 * One report, fully explained, plus the human decision.
 *
 * The review buttons are the entire point of the "human in the loop" claim: the
 * engine has ranked this report, and nothing happens to it until an HSE
 * professional takes it, agrees or disagrees on the record.
 */
export default function ReportDrawer({ report, onClose, onReview, busy = false }) {
  const tier = report ? tierOf(report) : null
  const state = report?.review?.state ?? "pending"

  return (
    <Drawer
      open={Boolean(report)}
      onClose={onClose}
      title={report ? `${report.id} · ${report.site}` : ""}
      subtitle={
        report ? `${REPORT_TYPE_LABEL[report.type]} · ${fmtDate(report.reportedAt)}` : ""
      }
      footer={
        report && onReview && report.sifPotential ? (
          <div className="flex flex-col gap-2">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-[11.5px] text-[var(--text-secondary)]">
                {state === "pending"
                  ? "The model has screened this report. An HSE professional decides."
                  : `${REVIEW_LABEL[state]}${report.review?.by ? ` — ${report.review.by}` : ""}` +
                    (state === "in-progress" ? " · still counted as awaiting a decision" : "")}
              </span>
              {busy && <Spinner />}
            </div>
            <div className="flex flex-wrap gap-2">
              {REVIEW_ACTIONS.map((action) => (
                <ReviewButton
                  key={action.state}
                  action={action}
                  active={state === action.state}
                  busy={busy}
                  onClick={() => onReview(report.id, action.state)}
                />
              ))}
            </div>
          </div>
        ) : null
      }
    >
      {report && (
        <div className="flex flex-col gap-5">
          <div className="flex flex-wrap items-center gap-1.5">
            <TierBadge tier={tier} />
            <ReviewBadge review={report.review} sif={report.sifPotential} />
          </div>
          {tier && (
            <p className="-mt-3 text-[11.5px] leading-snug text-[var(--text-muted)]">
              {TIER_BY_ID[tier].rule}
            </p>
          )}

          <div
            className="rounded-lg px-3 py-1.5"
            style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
          >
            <MetaRow label="Asset" value={report.asset} />
            <MetaRow label="Department" value={report.department} />
            <MetaRow label="Activity" value={report.activity} />
            <MetaRow label="Reported by" value={report.reportedBy} />
            <MetaRow
              label="Status"
              value={<Badge tone={STATUS_TONE[report.status]}>{report.status}</Badge>}
            />
          </div>

          <VerdictPanel item={report} />
        </div>
      )}
    </Drawer>
  )
}

export { STATUS_TONE }
