import { useMemo, useState } from "react"
import {
  CircleCheck,
  Clock,
  Filter,
  Hourglass,
  ListChecks,
  PenLine,
  ShieldAlert,
  Timer,
  TriangleAlert,
} from "lucide-react"
import {
  lsrTagDistribution,
  multiRuleReports,
  reviewStats,
  screeningFunnel,
  tierCounts,
  triageQueue,
  workload,
  PRIORITY_TIERS,
} from "../lib/triage.js"
import { rulesOf } from "../lib/aggregate.js"
import { REVIEW_ORDER } from "../lib/contract.js"
import { lsrName, lsrShort } from "../lib/lsr.js"
import { fmtDate, num, pct, truncate } from "../lib/format.js"
import { Badge, Bar, Card, Empty, ReviewBadge, SectionHead, StatTile, TierBadge } from "./kit/index.jsx"
import ChartFrame, { MiniTable } from "./kit/ChartFrame.jsx"
import DataTable from "./kit/DataTable.jsx"
import RankedBar from "./charts/RankedBar.jsx"
import Funnel, { FunnelScale } from "./charts/Funnel.jsx"
import ReportDrawer from "./ReportDrawer.jsx"
import { RuleCell } from "./Reports.jsx"

const QUEUE_LIMIT = 40

/** One number with a hairline rule under it — used inside the review panel. */
function Stat({ label, value, sub, icon: Icon, color = "var(--text-primary)" }) {
  return (
    <div className="min-w-0">
      <div className="flex items-center gap-1.5 text-[11.5px] text-[var(--text-secondary)]">
        {Icon && <Icon size={12.5} strokeWidth={2} aria-hidden style={{ color }} />}
        <span className="truncate">{label}</span>
      </div>
      <div className="tnum mt-0.5 text-[20px] leading-none font-semibold tracking-[-0.02em]">
        {value}
      </div>
      {sub && <div className="mt-1 text-[11px] text-[var(--text-muted)]">{sub}</div>}
    </div>
  )
}

export default function Triage({ reports, active, onActive, onReview, reviewing }) {
  const [showAll, setShowAll] = useState(false)

  const tiers = useMemo(() => tierCounts(reports), [reports])
  const funnel = useMemo(() => screeningFunnel(reports), [reports])
  const load = useMemo(() => workload(reports), [reports])
  const review = useMemo(() => reviewStats(reports), [reports])
  const queue = useMemo(() => triageQueue(reports), [reports])
  const tagDist = useMemo(() => lsrTagDistribution(reports), [reports])
  const multiRule = useMemo(() => multiRuleReports(reports), [reports])

  const tagRows = useMemo(
    () =>
      tagDist.rows.map((r) => ({
        ...r,
        short: lsrShort(r.id),
        name: lsrName(r.id),
      })),
    [tagDist.rows],
  )

  const shown = showAll ? queue : queue.slice(0, QUEUE_LIMIT)
  const loadMax = Math.max(1, load.manualHours)

  const columns = [
    {
      key: "tier",
      label: "Priority",
      width: 116,
      sortValue: (r) => ({ P1: 0, P2: 1, P3: 2 })[r.tier],
      render: (r) => <TierBadge tier={r.tier} />,
    },
    {
      key: "score",
      label: "Rank score",
      align: "right",
      width: 96,
      render: (r) => (
        <span
          className="tnum"
          title="0.55 × model confidence + 0.35 × severity score + 0.10 × not-yet-started"
        >
          {r.score.toFixed(3)}
        </span>
      ),
    },
    {
      key: "reportedAt",
      label: "Date",
      width: 92,
      render: (r) => <span className="tnum whitespace-nowrap">{fmtDate(r.reportedAt)}</span>,
    },
    { key: "site", label: "Site", width: 92 },
    {
      key: "lsr",
      label: "Rules breached",
      width: 158,
      sortValue: (r) => rulesOf(r).length,
      render: (r) => <RuleCell report={r} />,
    },
    {
      key: "text",
      label: "Observation",
      render: (r) => (
        <span className="block max-w-[360px] leading-snug text-[var(--text-secondary)]">
          {truncate(r.text, 88)}
        </span>
      ),
    },
    {
      key: "severityScore",
      label: "Severity",
      align: "right",
      width: 84,
      sortValue: (r) => r.severityScore ?? 0,
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
        >
          {(r.severityScore ?? 0).toFixed(1)}
        </span>
      ),
    },
    {
      key: "review",
      label: "HSE decision",
      width: 152,
      sortValue: (r) => REVIEW_ORDER.indexOf(r.review?.state ?? "pending"),
      render: (r) => <ReviewBadge review={r.review} sif={r.sifPotential} />,
    },
  ]

  return (
    <div className="flex flex-col gap-4 p-4">
      <SectionHead
        title="Screening queue"
        hint="Every report is screened before anyone reads it, then ranked. The tier comes from a written safety rule; only the order inside a tier uses the model’s scores."
        right={
          review.p1Undecided > 0 ? (
            <Badge tone="critical">
              {review.p1Undecided} immediate {review.p1Undecided === 1 ? "report" : "reports"}{" "}
              awaiting a decision
            </Badge>
          ) : (
            <Badge tone="good">No immediate reports awaiting a decision</Badge>
          )
        }
      />

      {/* tier counts + the review backlog */}
      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <StatTile
          label="P1 · Immediate"
          value={num(tiers.P1)}
          sub="Fatal potential with a broken barrier, still open"
          tone="critical"
          icon={TriangleAlert}
          footer="Review before the next shift"
        />
        <StatTile
          label="P2 · Priority"
          value={num(tiers.P2)}
          sub="Flagged SIF-potential and still open"
          tone="serious"
          icon={ShieldAlert}
          footer="Review this week"
        />
        <StatTile
          label="P3 · Routine"
          value={num(tiers.P3)}
          sub="Not flagged, or flagged and closed out"
          tone="good"
          icon={CircleCheck}
          footer="Normal close-out"
        />
        <StatTile
          label="Awaiting HSE decision"
          value={num(review.undecided)}
          sub={`of ${num(review.flagged)} flagged · ${num(review.inProgress)} already picked up`}
          tone={review.undecided > 0 ? "warning" : "good"}
          icon={Clock}
          footer="The engine screens; a person decides"
        />
      </div>

      {/* the screening story + the effort it saves */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <ChartFrame
          title="What screening removes from the reading pile"
          hint="Nested subsets of the same reports — each stage is a subset of the one above it."
          footnote="Without screening, all of these reports arrive at the same priority. The engine does not close any of them; it decides reading order."
          table={
            <MiniTable
              head={["Stage", "Reports", "% of intake", "Definition"]}
              align={["left", "right", "right", "left"]}
              rows={funnel.stages.map((s) => [
                s.label,
                s.count,
                pct(funnel.stages[0].count ? s.count / funnel.stages[0].count : 0, 1),
                s.note,
              ])}
            />
          }
        >
          <div>
            <Funnel stages={funnel.stages} max={funnel.max} />
            <div className="mt-3">
              <FunnelScale />
            </div>
          </div>
        </ChartFrame>

        <Card className="flex flex-col gap-3 p-4">
          <div>
            <h3 className="text-[13.5px] font-semibold tracking-[-0.005em]">
              Screening effort, manual vs assisted
            </h3>
            <p className="mt-1 text-[12px] leading-snug text-[var(--text-secondary)]">
              Same measure, two ways of working — hours of HSE reading time for the reports in scope.
            </p>
          </div>

          <div className="flex flex-col gap-3">
            {[
              {
                key: "manual",
                label: "Read every report equally",
                hours: load.manualHours,
                note: `${num(load.total)} full reads`,
              },
              {
                key: "assisted",
                label: "Read the queue, audit the rest",
                hours: load.assistedHours,
                note: `${num(load.fullReads)} full reads + ${num(load.sampled)} sampled`,
              },
            ].map((row) => (
              <div key={row.key}>
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-[12.5px]">{row.label}</span>
                  <span className="tnum text-[12.5px] font-semibold">
                    {row.hours.toFixed(1)}
                    <span className="font-normal text-[var(--text-muted)]"> h</span>
                  </span>
                </div>
                <div className="mt-1">
                  <Bar value={row.hours} max={loadMax} height={14} />
                </div>
                <p className="mt-1 text-[11px] text-[var(--text-muted)]">{row.note}</p>
              </div>
            ))}
          </div>

          <div
            className="mt-auto rounded-lg px-3 py-2.5"
            style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
          >
            <div className="flex items-center gap-2">
              <Timer size={13} strokeWidth={2} aria-hidden className="text-[var(--text-secondary)]" />
              <span className="text-[12px] text-[var(--text-secondary)]">
                Reading time avoided
              </span>
              <span className="tnum ml-auto text-[17px] leading-none font-semibold tracking-[-0.01em]">
                {load.savedHours.toFixed(1)} h
                <span className="text-[12px] font-normal text-[var(--text-muted)]">
                  {" · "}
                  {pct(load.savedShare, 0)}
                </span>
              </span>
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-[var(--text-muted)]">
              Assumes {load.minutesPerReport} minutes to read and judge one report, and that Routine
              reports get a {pct(load.sampleRate, 0)} audit sample instead of a full read —{" "}
              {num(load.skipped)} reports are not read closely. Change either assumption and the
              saving changes; the engine’s contribution is the ranking, not faster reading.
            </p>
          </div>
        </Card>
      </div>

      {/* human in the loop */}
      <Card className="p-4">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h3 className="text-[13.5px] font-semibold tracking-[-0.005em]">
              Human-in-the-loop decisions
            </h3>
            <p className="mt-1 max-w-[70ch] text-[12px] leading-snug text-[var(--text-secondary)]">
              The engine does not replace the HSE professional. It screens and ranks; an officer
              picks a flag up, then confirms or overrides it, and the override rate is how the next
              training round learns.
            </p>
          </div>
          {review.agreement != null && (
            <Badge tone={review.agreement >= 0.75 ? "good" : "warning"}>
              {pct(review.agreement, 0)} agreement with the model
            </Badge>
          )}
        </div>

        {/* The four states partition the flagged set exactly, so these add up to
            "Flagged for review". Two of them are open work, two are rulings. */}
        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
          <Stat
            label="Flagged for review"
            value={num(review.flagged)}
            sub="entered the queue"
            icon={ListChecks}
          />
          <Stat
            label="Not yet opened"
            value={num(review.pending)}
            sub="nobody has picked these up"
            icon={Clock}
            color="var(--status-warning)"
          />
          <Stat
            label="Review in progress"
            value={num(review.inProgress)}
            sub="an officer holds these"
            icon={Hourglass}
          />
          <Stat
            label="Confirmed by HSE"
            value={num(review.confirmed)}
            sub="model agreed with"
            icon={CircleCheck}
            color="var(--success-text)"
          />
          <Stat
            label="Overridden by HSE"
            value={num(review.overridden)}
            sub="judged not a precursor"
            icon={PenLine}
            color="var(--series-1)"
          />
        </div>

        <p className="mt-3 text-[11.5px] leading-snug text-[var(--text-muted)]">
          Not yet opened and Review in progress both count as awaiting a decision —{" "}
          {num(review.undecided)} reports, {num(review.p1Undecided)} of them Immediate. Claiming a
          report does not shorten the backlog; only a confirm or an override does.
        </p>

        {review.decided > 0 && (
          <div className="mt-4">
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-[12px] text-[var(--text-secondary)]">
                Of {num(review.decided)} decided flags, HSE confirmed {num(review.confirmed)}
              </span>
              <span className="tnum text-[12.5px] font-semibold">{pct(review.agreement, 1)}</span>
            </div>
            <div className="mt-1.5">
              <Bar value={review.confirmed} max={review.decided} height={10} />
            </div>
          </div>
        )}
      </Card>

      {/* multi-label rule tags */}
      <ChartFrame
        title="Rules breached across the reports in scope"
        hint={`Multi-label: one report can breach several rules at once. ${num(tagDist.tags)} rule breaches across ${num(tagDist.tagged)} tagged reports — ${tagDist.tagsPerReport.toFixed(2)} per report.`}
        height={Math.max(200, tagRows.length * 26 + 24)}
        legend={[{ label: "Reports carrying this rule tag", color: "var(--series-1)" }]}
        footnote={`These bars deliberately sum above the report count — ${num(multiRule.length)} reports carry more than one tag. A single-label model would have to pick one of those rules and drop the rest.`}
        table={
          <MiniTable
            head={["Life-Saving Rule", "Tagged", "Of which flagged", "Mean confidence"]}
            align={["left", "right", "right", "right"]}
            rows={tagRows.map((r) => [r.name, r.total, r.sif, r.meanConfidence.toFixed(2)])}
          />
        }
      >
        {tagRows.length ? (
          <RankedBar
            data={tagRows}
            categoryKey="short"
            valueKey="total"
            categoryWidth={112}
            barSize={11}
            tooltipRows={(row) => [
              { label: "Tagged reports", value: row.total, color: "var(--series-1)" },
              { label: "Of which SIF-flagged", value: row.sif },
              { label: "Mean confidence", value: row.meanConfidence.toFixed(2) },
              { label: "Rule", value: row.name },
            ]}
          />
        ) : (
          <Empty icon={Filter} title="No rule tags in scope" hint="Widen the filters above." />
        )}
      </ChartFrame>

      {/* the queue itself */}
      <div>
        <div className="mb-2 flex flex-wrap items-end justify-between gap-3 px-1">
          <div>
            <h3 className="text-[13.5px] font-semibold tracking-[-0.005em]">
              Work this queue top-down
            </h3>
            <p className="mt-0.5 text-[12px] text-[var(--text-secondary)]">
              Tier first, then rank score. Select a row to read the narrative, see why it scored, and
              record a decision.
            </p>
          </div>
          <span className="tnum text-[12px] text-[var(--text-secondary)]">
            showing {num(shown.length)} of {num(queue.length)}
          </span>
        </div>

        <div className="card overflow-hidden">
          <DataTable
            columns={columns}
            rows={shown}
            getRowKey={(r) => r.id}
            onRowClick={onActive}
            maxHeight="calc(100vh - 260px)"
            empty={
              <Empty
                icon={Filter}
                title="Nothing in the queue"
                hint="No reports match the filters above."
              />
            }
          />
        </div>

        {queue.length > QUEUE_LIMIT && (
          <button
            onClick={() => setShowAll((v) => !v)}
            className="mt-2 h-8 rounded-lg px-3 text-[12px] font-medium transition-colors hover:bg-[var(--surface-2)]"
            style={{ border: "1px solid var(--border-strong)", color: "var(--text-secondary)" }}
          >
            {showAll ? `Show the first ${QUEUE_LIMIT} only` : `Show all ${num(queue.length)} reports`}
          </button>
        )}
      </div>

      {/* the tier rules, written out */}
      <Card className="p-4">
        <h3 className="text-[13.5px] font-semibold tracking-[-0.005em]">
          How a tier is decided
        </h3>
        <p className="mt-1 max-w-[80ch] text-[12px] leading-snug text-[var(--text-secondary)]">
          These are rules, not model outputs — an HSE auditor can read them and challenge them. The
          model contributes the SIF flag, the rule tags and the severity score that those rules read.
        </p>
        <ul className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-3">
          {PRIORITY_TIERS.map((t) => (
            <li
              key={t.id}
              className="rounded-lg p-3"
              style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
            >
              <div className="flex items-center gap-2">
                <TierBadge tier={t.id} />
                <span className="tnum text-[11.5px] text-[var(--text-muted)]">
                  {num(tiers[t.id])} reports
                </span>
              </div>
              <p className="mt-2 text-[12px] font-medium">{t.blurb}</p>
              <p className="mt-1 text-[11.5px] leading-snug text-[var(--text-muted)]">{t.rule}</p>
            </li>
          ))}
        </ul>
      </Card>

      <ReportDrawer
        report={active}
        onClose={() => onActive(null)}
        onReview={onReview}
        busy={reviewing === active?.id}
      />
    </div>
  )
}
