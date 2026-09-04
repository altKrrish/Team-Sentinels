import { Bar, BarChart, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { AXIS, ChartTooltip } from "./common.jsx"

/**
 * Ranked horizontal bars. Sites and rules are nominal categories, so there is
 * exactly ONE bar color — no value ramp, no color-by-rank. Rows below the
 * minimum sample size are drawn in the de-emphasis gray instead.
 */
export default function RankedBar({
  data,
  categoryKey,
  valueKey,
  valueFormat = (v) => v,
  tooltipRows,
  categoryWidth = 108,
  barSize = 13,
  domain,
}) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 2, right: 52, bottom: 2, left: 0 }}
        barCategoryGap={6}
      >
        <XAxis type="number" hide domain={domain ?? [0, "dataMax"]} />
        <YAxis
          type="category"
          dataKey={categoryKey}
          width={categoryWidth}
          {...AXIS}
          axisLine={false}
        />
        <Tooltip
          cursor={{ fill: "color-mix(in srgb, var(--text-primary) 5%, transparent)" }}
          content={<ChartTooltip rows={tooltipRows} />}
        />
        <Bar dataKey={valueKey} barSize={barSize} radius={[0, 4, 4, 0]} isAnimationActive={false}>
          {data.map((d) => (
            <Cell
              key={d[categoryKey]}
              fill={d.lowSample ? "var(--series-mute)" : "var(--series-1)"}
            />
          ))}
          <LabelList
            dataKey={valueKey}
            position="right"
            offset={7}
            formatter={valueFormat}
            style={{
              fill: "var(--text-primary)",
              fontSize: 11.5,
              fontWeight: 600,
              fontVariantNumeric: "tabular-nums",
            }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
