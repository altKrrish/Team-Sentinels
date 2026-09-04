import { CircleCheck, Clock, Hourglass, Info, PenLine, ShieldCheck, TriangleAlert } from "lucide-react"
import { REVIEW_LABEL } from "../../lib/contract.js"
import { TIER_BY_ID } from "../../lib/triage.js"

/* ---------------- surfaces ---------------- */

export function Card({ className = "", children, ...rest }) {
  return (
    <div className={`card ${className}`} {...rest}>
      {children}
    </div>
  )
}

export function SectionHead({ title, hint, right }) {
  return (
    <div className="mb-3 flex items-end justify-between gap-4">
      <div>
        <h2 className="text-[15px] font-semibold tracking-[-0.01em]">{title}</h2>
        {hint && (
          <p className="mt-0.5 text-[12.5px] leading-snug text-[var(--text-secondary)]">{hint}</p>
        )}
      </div>
      {right}
    </div>
  )
}

/* ---------------- badges ---------------- */

const TONES = {
  critical: { color: "var(--status-critical)", Icon: TriangleAlert },
  serious: { color: "var(--status-serious)", Icon: TriangleAlert },
  warning: { color: "var(--status-warning)", Icon: Info },
  good: { color: "var(--success-text)", Icon: CircleCheck },
  info: { color: "var(--series-1)", Icon: Info },
  neutral: { color: "var(--text-secondary)", Icon: null },
}

/** Status is never color alone — every toned badge carries an icon and a word. */
export function Badge({ tone = "neutral", icon, children, title }) {
  const t = TONES[tone] ?? TONES.neutral
  const Icon = icon ?? t.Icon
  const isStatus = tone !== "neutral"
  return (
    <span
      title={title}
      className="inline-flex shrink-0 items-center gap-1.5 rounded-full px-2 py-[3px] text-[11.5px] font-medium whitespace-nowrap"
      style={{
        color: isStatus ? t.color : "var(--text-secondary)",
        background: isStatus ? `color-mix(in srgb, ${t.color} 13%, transparent)` : "var(--surface-2)",
        border: `1px solid ${isStatus ? `color-mix(in srgb, ${t.color} 30%, transparent)` : "var(--border)"}`,
      }}
    >
      {Icon && <Icon size={12} strokeWidth={2.2} aria-hidden />}
      {children}
    </span>
  )
}

export function SifBadge({ sif, confidence }) {
  return sif ? (
    <Badge tone="critical" title={confidence != null ? `score ${confidence}` : undefined}>
      SIF potential
    </Badge>
  ) : (
    <Badge tone="good" icon={ShieldCheck} title={confidence != null ? `score ${confidence}` : undefined}>
      No SIF potential
    </Badge>
  )
}

/** Screening priority. The tier comes from a written rule, not a model cut-off. */
export function TierBadge({ tier, withLabel = true }) {
  const t = TIER_BY_ID[tier]
  if (!t) return null
  return (
    <Badge tone={t.tone} title={t.rule}>
      {tier}
      {withLabel ? ` · ${t.label}` : ""}
    </Badge>
  )
}

/**
 * `pending` is amber because an unclaimed flag needs someone; `in-progress` goes
 * quiet on purpose - once an officer owns it the badge should stop shouting,
 * without pretending a decision has been made.
 */
const REVIEW_TONE = {
  pending: "warning",
  "in-progress": "neutral",
  confirmed: "good",
  overridden: "info",
}
const REVIEW_ICON = {
  pending: Clock,
  "in-progress": Hourglass,
  confirmed: CircleCheck,
  overridden: PenLine,
}

/** Who has actually ruled on the verdict. The model screens; a person decides. */
export function ReviewBadge({ review, sif = true }) {
  if (!sif) return <Badge>Not queued for review</Badge>
  const state = review?.state ?? "pending"
  return (
    <Badge tone={REVIEW_TONE[state]} icon={REVIEW_ICON[state]} title={review?.by ?? undefined}>
      {REVIEW_LABEL[state]}
    </Badge>
  )
}

/* ---------------- stat tiles ---------------- */

/** A single headline number is a tile, not a chart. */
export function StatTile({ label, value, sub, tone, icon: Icon, footer }) {
  const color = tone ? (TONES[tone]?.color ?? "var(--text-primary)") : "var(--text-primary)"
  return (
    <Card className="flex min-w-0 flex-col gap-1 p-3.5">
      <div className="flex items-center gap-1.5 text-[11.5px] font-medium tracking-[0.02em] text-[var(--text-secondary)]">
        {Icon && <Icon size={13} strokeWidth={2} aria-hidden style={{ color }} />}
        <span className="truncate">{label}</span>
      </div>
      <div className="tnum text-[26px] leading-[1.1] font-semibold tracking-[-0.02em]" style={{ color }}>
        {value}
      </div>
      {sub && <div className="text-[12px] leading-snug text-[var(--text-secondary)]">{sub}</div>}
      {footer && <div className="mt-1 text-[11.5px] text-[var(--text-muted)]">{footer}</div>}
    </Card>
  )
}

/* ---------------- meter ---------------- */

/**
 * A ratio against a limit — one hue on a track, with the reference band drawn
 * in place. Not a two-slice pie.
 */
export function Meter({ value, band, label, caption }) {
  const clamp = Math.max(0, Math.min(1, value))
  const scaleMax = Math.max(0.4, Math.ceil((Math.max(clamp, band.high) + 0.08) * 10) / 10)
  const x = (v) => `${(v / scaleMax) * 100}%`

  const verdict =
    clamp < band.low ? { text: "below the benchmark band", tone: "warning" }
    : clamp > band.high ? { text: "above the benchmark band", tone: "serious" }
    : { text: "within the benchmark band", tone: "good" }

  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <span className="text-[12.5px] text-[var(--text-secondary)]">{label}</span>
        <span className="tnum text-[19px] font-semibold tracking-[-0.01em]">
          {(clamp * 100).toFixed(1)}%
        </span>
      </div>

      <div
        className="relative mt-2 h-[26px] w-full overflow-hidden rounded-md"
        style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
        role="img"
        aria-label={`${label}: ${(clamp * 100).toFixed(1)} percent, benchmark ${band.low * 100} to ${band.high * 100} percent`}
      >
        {/* reference band */}
        <div
          className="absolute inset-y-0"
          style={{
            left: x(band.low),
            width: `calc(${x(band.high)} - ${x(band.low)})`,
            background: "color-mix(in srgb, var(--text-primary) 8%, transparent)",
            borderLeft: "1px dashed var(--axis)",
            borderRight: "1px dashed var(--axis)",
          }}
        />
        {/* value fill — 4px rounded data end, anchored to the baseline */}
        <div
          className="absolute inset-y-[3px] left-[3px] rounded-[4px]"
          style={{
            width: `calc(${x(clamp)} - 6px)`,
            background: "var(--series-1)",
            transition: "width 0.35s cubic-bezier(0.2,0.8,0.2,1)",
          }}
        />
      </div>

      <div className="mt-1.5 flex items-center justify-between gap-2">
        <span className="tnum text-[11px] text-[var(--text-muted)]">
          benchmark {band.low * 100}–{band.high * 100}%
        </span>
        <Badge tone={verdict.tone}>{verdict.text}</Badge>
      </div>
      {caption && (
        <p className="mt-2 text-[12px] leading-snug text-[var(--text-secondary)]">{caption}</p>
      )}
    </div>
  )
}

/* ---------------- misc ---------------- */

export function Empty({ icon: Icon, title, hint }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 px-6 py-12 text-center">
      {Icon && <Icon size={22} strokeWidth={1.8} className="text-[var(--text-muted)]" aria-hidden />}
      <p className="text-[13.5px] font-medium">{title}</p>
      {hint && <p className="max-w-[46ch] text-[12.5px] text-[var(--text-secondary)]">{hint}</p>}
    </div>
  )
}

export function Spinner({ size = 14 }) {
  return (
    <span
      aria-hidden
      className="inline-block animate-spin rounded-full align-[-2px]"
      style={{
        width: size,
        height: size,
        border: "2px solid var(--border-strong)",
        borderTopColor: "var(--series-1)",
      }}
    />
  )
}

export function Bar({ value, max, color = "var(--series-1)", height = 8 }) {
  const w = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0
  return (
    <span
      className="block w-full overflow-hidden rounded-[4px]"
      style={{ height, background: "var(--surface-2)" }}
    >
      <span
        className="block h-full rounded-[4px]"
        style={{ width: `${w * 100}%`, background: color }}
      />
    </span>
  )
}
