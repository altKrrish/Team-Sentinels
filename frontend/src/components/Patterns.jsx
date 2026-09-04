import { useMemo } from "react"
import { Radar } from "lucide-react"
import {
  activityRanking,
  barrierBreakdown,
  energyBreakdown,
  precursorPatterns,
  siteLsrMatrix,
  MIN_REPORTS_FOR_DENSITY,
} from "../lib/aggregate.js"
import { BARRIER_LABEL } from "../lib/contract.js"
import { fmtDate, pct, titleCase } from "../lib/format.js"
import { Badge, Card, Empty } from "./kit/index.jsx"
import ChartFrame, { MiniTable } from "./kit/ChartFrame.jsx"
import { HAZARD_ICON } from "./kit/icons.jsx"
import RankedBar from "./charts/RankedBar.jsx"
import SiteLsrHeatmap from "./charts/SiteLsrHeatmap.jsx"

export default function Patterns({ reports }) {
  const matrix = useMemo(() => siteLsrMatrix(reports), [reports])
  const activities = useMemo(() => activityRanking(reports), [reports])
  const barriers = useMemo(() => barrierBreakdown(reports).filter((b) => b.total > 0), [reports])
  const energies = useMemo(() => energyBreakdown(reports).filter((e) => e.total > 0), [reports])
  const repeats = useMemo(() => precursorPatterns(reports), [reports])

  const activityRows = useMemo(
    () => [...activities].sort((a, b) => {
      if (a.lowSample !== b.lowSample) return a.lowSample ? 1 : -1
      return b.density - a.density
    }).slice(0, 10),
    [activities],
  )

  if (!reports.length) {
    return (
      <div className="p-4">
        <Card>
          <Empty
            icon={Radar}
            title="Nothing to correlate"
            hint="No reports match the current filters, so there are no precursor patterns to surface."
          />
        </Card>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4 p-4">
      <ChartFrame
        title="Where fatal potential concentrates — site × Life-Saving Rule"
        hint="SIF-flagged reports only. Read across a row for a site's exposure profile, down a column for a rule that is failing everywhere."
        footnote="One hue, light to dark, with the scale shown. Every cell carries its count, so nothing is encoded in color alone."
        table={
          <MiniTable
            head={["Site", ...matrix.cols.map((c) => c.short), "All"]}
            align={["left", ...matrix.cols.map(() => "right"), "right"]}
            rows={matrix.rows.map((r) => [r.site, ...r.cells.map((c) => c.count), r.total])}
          />
        }
      >
        <SiteLsrHeatmap matrix={matrix} />
      </ChartFrame>

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <ChartFrame
          title="Activities ranked by SIF-precursor density"
          hint="Flagged ÷ all reports for that activity. Top 10 in scope."
          height={Math.max(200, activityRows.length * 28 + 20)}
          legend={[
            { label: "SIF-precursor density", color: "var(--series-1)" },
            {
              label: `Under ${MIN_REPORTS_FOR_DENSITY} reports — not ranked`,
              color: "var(--series-mute)",
            },
          ]}
          footnote={`Activities with fewer than ${MIN_REPORTS_FOR_DENSITY} reports in scope are gray and sorted last.`}
          table={
            <MiniTable
              head={["Activity", "Flagged", "Total", "Density"]}
              align={["left", "right", "right", "right"]}
              rows={activities.map((a) => [
                a.lowSample ? `${a.activity} (low sample)` : a.activity,
                a.sif,
                a.total,
                pct(a.density, 1),
              ])}
            />
          }
        >
          <RankedBar
            data={activityRows}
            categoryKey="activity"
            valueKey="density"
            valueFormat={(v) => pct(v, 0)}
            categoryWidth={168}
            barSize={11}
            domain={[0, Math.max(0.4, ...activityRows.map((a) => a.density)) * 1.12]}
            tooltipRows={(row) => [
              { label: "Density", value: pct(row.density, 1), color: "var(--series-1)" },
              { label: "Flagged", value: row.sif },
              { label: "All reports", value: row.total },
              ...(row.lowSample ? [{ label: "Note", value: "low sample" }] : []),
            ]}
          />
        </ChartFrame>

        <div className="flex flex-col gap-4">
          <ChartFrame
            title="Barrier failure mode of flagged reports"
            hint="How the control protecting against the hazard energy broke down."
            height={Math.max(140, barriers.length * 28 + 20)}
            legend={[{ label: "SIF-flagged reports", color: "var(--series-1)" }]}
            table={
              <MiniTable
                head={["Barrier state", "Flagged", "All reports"]}
                align={["left", "right", "right"]}
                rows={barriers.map((b) => [b.label, b.sif, b.total])}
              />
            }
          >
            <RankedBar
              data={barriers.map((b) => ({ ...b, label: b.label.replace("Barrier ", "") }))}
              categoryKey="label"
              valueKey="sif"
              categoryWidth={104}
              barSize={11}
              tooltipRows={(row) => [
                { label: "Flagged", value: row.sif, color: "var(--series-1)" },
                { label: "All reports", value: row.total },
              ]}
            />
          </ChartFrame>

          <ChartFrame
            title="Hazard energy of flagged reports"
            hint="The energy that would have done the harm."
            height={Math.max(140, energies.length * 26 + 20)}
            legend={[{ label: "SIF-flagged reports", color: "var(--series-1)" }]}
            table={
              <MiniTable
                head={["Hazard energy", "Flagged", "All reports"]}
                align={["left", "right", "right"]}
                rows={energies.map((e) => [titleCase(e.id), e.sif, e.total])}
              />
            }
          >
            <RankedBar
              data={energies.map((e) => ({ ...e, label: titleCase(e.id) }))}
              categoryKey="label"
              valueKey="sif"
              categoryWidth={92}
              barSize={11}
              tooltipRows={(row) => [
                { label: "Flagged", value: row.sif, color: "var(--series-1)" },
                { label: "All reports", value: row.total },
              ]}
            />
          </ChartFrame>
        </div>
      </div>

      {/* recurring signatures */}
      <Card className="p-4">
        <h3 className="text-[13.5px] font-semibold">Recurring precursor signatures</h3>
        <p className="mt-1 text-[12.5px] text-[var(--text-secondary)]">
          The same activity failing the same way, more than once. These are the repeat patterns worth
          a standing intervention rather than a case-by-case close-out.
        </p>

        {repeats.length === 0 ? (
          <p className="mt-4 text-[12.5px] text-[var(--text-secondary)]">
            No signature repeats in this selection.
          </p>
        ) : (
          <ul className="mt-3 flex flex-col gap-2">
            {repeats.map((p) => {
              const EnergyIcon = HAZARD_ICON[p.energy]
              return (
                <li
                  key={p.key}
                  className="flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-lg px-3 py-2"
                  style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
                >
                  <span className="text-[12.5px] font-medium">{p.activity}</span>
                  <Badge icon={EnergyIcon}>{p.energy} energy</Badge>
                  <Badge
                    tone={
                      ["absent", "failed", "bypassed"].includes(p.barrier) ? "critical" : "warning"
                    }
                  >
                    {BARRIER_LABEL[p.barrier]}
                  </Badge>
                  <span className="tnum ml-auto text-[11.5px] text-[var(--text-secondary)]">
                    {p.sif} flagged of {p.total} · {p.siteCount} site
                    {p.siteCount === 1 ? "" : "s"} · last {fmtDate(p.lastSeen)}
                  </span>
                </li>
              )
            })}
          </ul>
        )}
      </Card>
    </div>
  )
}
