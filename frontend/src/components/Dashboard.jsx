import { useMemo } from "react"
import { FileText, ShieldAlert, TriangleAlert } from "lucide-react"
import {
  kpis,
  lsrDistribution,
  monthlyTrend,
  severityMatrix,
  siteDensity,
  typeBreakdown,
  windowOf,
  MIN_REPORTS_FOR_DENSITY,
} from "../lib/aggregate.js"
import { screeningFunnel, severityScoreBins, tierCounts } from "../lib/triage.js"
import { SIF_BENCHMARK } from "../lib/contract.js"
import { lsrName } from "../lib/lsr.js"
import { num, pct } from "../lib/format.js"
import { Card, Meter, StatTile } from "./kit/index.jsx"
import ChartFrame, { MiniTable } from "./kit/ChartFrame.jsx"
import { LsrIcon } from "./kit/icons.jsx"
import TrendLine from "./charts/TrendLine.jsx"
import RankedBar from "./charts/RankedBar.jsx"
import RiskMatrix from "./charts/RiskMatrix.jsx"
import Funnel, { FunnelScale } from "./charts/Funnel.jsx"
import Histogram from "./charts/Histogram.jsx"

export default function Dashboard({ reports, axis, filters }) {
  const k = useMemo(() => kpis(reports), [reports])
  const trend = useMemo(
    () => monthlyTrend(reports, windowOf(axis, filters.months)),
    [reports, axis, filters.months],
  )
  const density = useMemo(() => siteDensity(reports), [reports])
  const rules = useMemo(() => lsrDistribution(reports).filter((r) => r.total > 0), [reports])
  const types = useMemo(() => typeBreakdown(reports).filter((t) => t.total > 0), [reports])
  const matrix = useMemo(() => severityMatrix(reports), [reports])
  const tiers = useMemo(() => tierCounts(reports), [reports])
  const funnel = useMemo(() => screeningFunnel(reports), [reports])
  const bins = useMemo(() => severityScoreBins(reports), [reports])

  const maxDensity = Math.max(0.35, ...density.map((d) => d.density))

  return (
    <div className="flex flex-col gap-4 p-4">
      {/* KPI row — single numbers are tiles, not charts */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-5">
        <StatTile
          label="Reports in scope"
          value={num(k.total)}
          sub={`${filters.months}-month window`}
          icon={FileText}
        />
        <StatTile
          label="SIF-potential flagged"
          value={num(k.sifCount)}
          sub={`${pct(k.sifShare, 1)} of reports in scope`}
          tone="critical"
          icon={TriangleAlert}
        />
        <StatTile
          label="Immediate attention · P1"
          value={num(tiers.P1)}
          sub="fatal potential, barrier broken, still open"
          tone={tiers.P1 > 0 ? "critical" : "good"}
          icon={TriangleAlert}
          footer="Work these before the next shift"
        />
        <StatTile
          label="Flagged and still open"
          value={num(k.openSif)}
          sub="not yet closed out"
          tone={k.openSif > 0 ? "serious" : "good"}
          icon={ShieldAlert}
        />
        <StatTile
          label="Top rule at risk"
          value={k.topLsrId ? lsrName(k.topLsrId) : "—"}
          sub={k.topLsrId ? `${k.topLsrCount} flagged reports` : "nothing flagged"}
        />
      </div>

      {/* trend + benchmark meter */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <div className="xl:col-span-2">
          <ChartFrame
            title="SIF-potential vs other reports by month"
            hint="One shared count axis. The flagged line is what severity-based triage tends to miss."
            height={236}
            legend={[
              { label: "SIF-potential", color: "var(--series-2)", shape: "line" },
              { label: "Other reports", color: "var(--series-1)", shape: "line" },
            ]}
            table={
              <MiniTable
                head={["Month", "SIF-potential", "Other", "Total", "Flagged share"]}
                align={["left", "right", "right", "right", "right"]}
                rows={trend.map((t) => [
                  t.label,
                  t.sif,
                  t.nonSif,
                  t.total,
                  pct(t.share, 0),
                ])}
              />
            }
          >
            <TrendLine data={trend} />
          </ChartFrame>
        </div>

        <Card className="flex flex-col justify-center p-4">
          <Meter
            value={k.sifShare}
            band={SIF_BENCHMARK}
            label="Share of reports flagged SIF-potential"
            caption="Leading operators find 20–25% of reports carry genuine fatal potential. A share far below the band usually means precursors are being closed out as ordinary observations, not that the fields are safer."
          />
        </Card>
      </div>

      {/* screening + the continuous severity score */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-3">
        <ChartFrame
          title="Screening narrows the reading pile"
          hint="Nested subsets of the same reports — each stage is a subset of the one above."
          footnote="Full queue, tier rules and the effort this saves are on the Triage screen."
          table={
            <MiniTable
              head={["Stage", "Reports", "% of intake"]}
              align={["left", "right", "right"]}
              rows={funnel.stages.map((s) => [
                s.label,
                s.count,
                pct(funnel.stages[0].count ? s.count / funnel.stages[0].count : 0, 1),
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

        <div className="xl:col-span-2">
          <ChartFrame
            title="Severity score distribution"
            hint="The regression head's continuous 0–10 output. Grouped bars, one shared count axis — the question is where each population sits, not what the bins total."
            height={236}
            legend={[
              { label: "SIF-potential", color: "var(--series-2)" },
              { label: "Other reports", color: "var(--series-1)" },
            ]}
            footnote="Unflagged reports sit low and flagged ones high — in this demo the two barely meet, because the fixture derives the score from the same severity bands the flag reads. On live narratives the tails overlap, which is why the flag stays a separate classifier rather than a threshold on this score. The score's own job is ranking inside a band: two reports that are both “potential 5” still get an order."
            table={
              <MiniTable
                head={["Score band", "SIF-potential", "Other", "Total"]}
                align={["left", "right", "right", "right"]}
                rows={bins
                  .filter((b) => b.sif + b.nonSif > 0)
                  .map((b) => [b.label, b.sif, b.nonSif, b.sif + b.nonSif])}
              />
            }
          >
            <Histogram data={bins} />
          </ChartFrame>
        </div>
      </div>

      {/* rankings */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <ChartFrame
          title="Sites ranked by SIF-precursor density"
          hint="Flagged reports ÷ all reports at that site."
          height={Math.max(160, density.length * 30 + 20)}
          legend={[
            { label: "SIF-precursor density", color: "var(--series-1)" },
            { label: `Under ${MIN_REPORTS_FOR_DENSITY} reports — not ranked`, color: "var(--series-mute)" },
          ]}
          footnote={`Sites with fewer than ${MIN_REPORTS_FOR_DENSITY} reports in scope are shown in gray and sorted last — two reports at 100% is not a ranking.${
            filters.sifOnly ? " The “SIF only” filter is on, so every site reads 100% by construction." : ""
          }`}
          table={
            <MiniTable
              head={["Site", "Flagged", "Total", "Density"]}
              align={["left", "right", "right", "right"]}
              rows={density.map((d) => [
                d.lowSample ? `${d.site} (low sample)` : d.site,
                d.sif,
                d.total,
                pct(d.density, 1),
              ])}
            />
          }
        >
          <RankedBar
            data={density}
            categoryKey="site"
            valueKey="density"
            domain={[0, maxDensity * 1.12]}
            valueFormat={(v) => pct(v, 0)}
            categoryWidth={96}
            tooltipRows={(row) => [
              { label: "Density", value: pct(row.density, 1), color: "var(--series-1)" },
              { label: "Flagged", value: row.sif },
              { label: "All reports", value: row.total },
              ...(row.lowSample ? [{ label: "Note", value: "low sample" }] : []),
            ]}
          />
        </ChartFrame>

        <ChartFrame
          title="Flagged reports by Life-Saving Rule"
          hint="Multi-label: a report breaching three rules is counted under all three. Bars count SIF-flagged reports only; totals are in the table view."
          height={Math.max(200, rules.length * 26 + 24)}
          legend={[{ label: "SIF-flagged reports", color: "var(--series-1)" }]}
          footnote="These bars sum above the flagged-report count by design — the one-vs-rest head tags every rule that clears 0.35, so nothing is dropped just because another rule scored higher."
          table={
            <MiniTable
              head={["Life-Saving Rule", "Flagged", "All reports", "Share flagged"]}
              align={["left", "right", "right", "right"]}
              rows={rules.map((r) => [
                r.name,
                r.sif,
                r.total,
                pct(r.total ? r.sif / r.total : 0, 0),
              ])}
            />
          }
        >
          <RankedBar
            data={rules}
            categoryKey="short"
            valueKey="sif"
            categoryWidth={112}
            barSize={11}
            tooltipRows={(row) => [
              { label: "Flagged", value: row.sif, color: "var(--series-1)" },
              { label: "All reports", value: row.total },
              { label: "Rule", value: row.name },
            ]}
          />
        </ChartFrame>
      </div>

      {/* severity contrast + report types */}
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
        <ChartFrame
          title="Actual vs potential severity"
          hint="Cells above the diagonal are reports whose outcome was mild but whose potential was not."
          footnote="Tone follows potential severity; the tint deepens with the number of reports in the cell."
          table={
            <MiniTable
              head={["Actual", "Potential", "Reports"]}
              align={["right", "right", "right"]}
              rows={matrix.cells
                .filter((c) => c.count > 0)
                .sort((a, b) => b.potential - a.potential || a.actual - b.actual)
                .map((c) => [c.actual, c.potential, c.count])}
            />
          }
        >
          <RiskMatrix matrix={matrix} />
        </ChartFrame>

        <ChartFrame
          title="Where fatal potential is hiding"
          hint="Flagged reports by record type. Unsafe acts and near-misses are ordinary paperwork until one of them is a precursor."
          height={Math.max(150, types.length * 34 + 20)}
          legend={[{ label: "SIF-flagged reports", color: "var(--series-1)" }]}
          table={
            <MiniTable
              head={["Report type", "Flagged", "All reports", "Share flagged"]}
              align={["left", "right", "right", "right"]}
              rows={types.map((t) => [
                t.label,
                t.sif,
                t.total,
                pct(t.total ? t.sif / t.total : 0, 0),
              ])}
            />
          }
        >
          <RankedBar
            data={types}
            categoryKey="label"
            valueKey="sif"
            categoryWidth={112}
            tooltipRows={(row) => [
              { label: "Flagged", value: row.sif, color: "var(--series-1)" },
              { label: "All reports", value: row.total },
              { label: "Share flagged", value: pct(row.total ? row.sif / row.total : 0, 0) },
            ]}
          />
        </ChartFrame>
      </div>

      {/* rule reference strip */}
      <Card className="p-4">
        <h3 className="text-[13.5px] font-semibold">IOGP Life-Saving Rules in this dataset</h3>
        <ul className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3">
          {rules.map((r) => (
            <li key={r.id} className="flex items-start gap-2">
              <span className="mt-[2px] text-[var(--text-secondary)]">
                <LsrIcon id={r.id} size={13.5} />
              </span>
              <span className="min-w-0 text-[12px]">
                <span className="font-medium">{r.name}</span>
                <span className="tnum text-[var(--text-muted)]">
                  {" — "}
                  {r.sif} flagged of {r.total}
                </span>
              </span>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  )
}
