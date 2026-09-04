import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { AXIS, ChartTooltip, endLabel } from "./common.jsx"

/**
 * SIF-flagged vs other reports per month. Two series, one shared count axis —
 * never a second y-scale.
 */
export default function TrendLine({ data }) {
  const n = data.length
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 8, right: 30, bottom: 4, left: -18 }}>
        <CartesianGrid vertical={false} stroke="var(--grid)" />
        <XAxis dataKey="label" {...AXIS} interval="preserveStartEnd" minTickGap={8} />
        <YAxis {...AXIS} allowDecimals={false} width={44} />
        <Tooltip
          cursor={{ stroke: "var(--axis)", strokeWidth: 1 }}
          content={
            <ChartTooltip
              rows={(row) => [
                { label: "SIF-potential", value: row.sif, color: "var(--series-2)" },
                { label: "Other reports", value: row.nonSif, color: "var(--series-1)" },
                { label: "Flagged share", value: `${(row.share * 100).toFixed(0)}%` },
              ]}
            />
          }
        />
        <Line
          type="monotone"
          dataKey="nonSif"
          name="Other reports"
          stroke="var(--series-1)"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4.5, strokeWidth: 2, stroke: "var(--surface-1)" }}
          label={endLabel(n)}
          isAnimationActive={false}
        />
        <Line
          type="monotone"
          dataKey="sif"
          name="SIF-potential"
          stroke="var(--series-2)"
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4.5, strokeWidth: 2, stroke: "var(--surface-1)" }}
          label={endLabel(n)}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
