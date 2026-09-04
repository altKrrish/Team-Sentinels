import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { AXIS, ChartTooltip } from "./common.jsx"

/**
 * Distribution of the ridge head's continuous severity score, split by verdict.
 *
 * Grouped (not stacked) — the question is "where does each population sit", not
 * "what is the total per bin". Two categorical slots, and the hues match the
 * trend line so SIF-potential is the same orange everywhere in the app.
 */
export default function Histogram({ data }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 6, right: 8, bottom: 2, left: -18 }} barGap={2}>
        <CartesianGrid vertical={false} />
        <XAxis dataKey="label" {...AXIS} interval={0} />
        <YAxis {...AXIS} axisLine={false} allowDecimals={false} />
        <Tooltip
          cursor={{ fill: "color-mix(in srgb, var(--text-primary) 5%, transparent)" }}
          content={
            <ChartTooltip
              rows={(row) => [
                { label: "SIF-potential", value: row.sif, color: "var(--series-2)" },
                { label: "Other reports", value: row.nonSif, color: "var(--series-1)" },
                { label: "Score band", value: row.label },
              ]}
            />
          }
        />
        {/* 2px surface gap between adjacent fills comes from barGap */}
        <Bar
          dataKey="nonSif"
          name="Other reports"
          fill="var(--series-1)"
          barSize={11}
          radius={[4, 4, 0, 0]}
          isAnimationActive={false}
        />
        <Bar
          dataKey="sif"
          name="SIF-potential"
          fill="var(--series-2)"
          barSize={11}
          radius={[4, 4, 0, 0]}
          isAnimationActive={false}
        />
      </BarChart>
    </ResponsiveContainer>
  )
}
