import { BARRIER_LABEL, SEVERITY_LABEL } from "../lib/contract.js"
import { lsrName, LSR_BY_ID } from "../lib/lsr.js"
import { FEATURE_GROUP_BY_ID } from "../lib/model.js"
import { rulesOf } from "../lib/aggregate.js"
import { Badge, SifBadge } from "./kit/index.jsx"
import { HAZARD_ICON, LsrIcon } from "./kit/icons.jsx"

/* ---------- evidence highlighting ---------- */

/** Marks the model's evidence spans inside the raw text. Underline + tint, so
 *  the highlight is not carried by color alone. */
export function HighlightedText({ text, evidence = [] }) {
  const ranges = []
  for (const { span, weight } of evidence) {
    const i = text.indexOf(span)
    if (i === -1) continue
    ranges.push({ start: i, end: i + span.length, weight })
  }
  ranges.sort((a, b) => a.start - b.start)

  const merged = []
  for (const r of ranges) {
    const last = merged[merged.length - 1]
    if (last && r.start <= last.end) {
      last.end = Math.max(last.end, r.end)
      last.weight = Math.max(last.weight, r.weight)
    } else merged.push({ ...r })
  }

  const out = []
  let cursor = 0
  merged.forEach((r, i) => {
    if (r.start > cursor) out.push(<span key={`t${i}`}>{text.slice(cursor, r.start)}</span>)
    const pct = 12 + r.weight * 20
    out.push(
      <mark
        key={`m${i}`}
        title={`evidence weight ${r.weight.toFixed(2)}`}
        style={{
          background: `color-mix(in srgb, var(--series-2) ${pct.toFixed(0)}%, transparent)`,
          borderBottom: "1.5px solid color-mix(in srgb, var(--series-2) 65%, transparent)",
          color: "var(--text-primary)",
          borderRadius: 2,
          padding: "0 1px",
        }}
      >
        {text.slice(r.start, r.end)}
      </mark>,
    )
    cursor = r.end
  })
  if (cursor < text.length) out.push(<span key="tail">{text.slice(cursor)}</span>)
  return <>{out}</>
}

/* ---------- small pieces ---------- */

function ConfidenceRow({ label, value, icon }) {
  return (
    <div className="flex items-center gap-2">
      <span className="flex min-w-0 flex-1 items-center gap-1.5 text-[12.5px]">
        {icon}
        <span className="truncate">{label}</span>
      </span>
      <span
        aria-hidden
        className="h-[6px] w-[68px] shrink-0 overflow-hidden rounded-full"
        style={{ background: "var(--surface-2)" }}
      >
        <span
          className="block h-full rounded-full"
          style={{ width: `${value * 100}%`, background: "var(--series-1)" }}
        />
      </span>
      <span className="tnum w-[34px] shrink-0 text-right text-[11.5px] text-[var(--text-secondary)]">
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  )
}

function SeverityScale({ value, tone }) {
  return (
    <span className="flex gap-[2px]" aria-hidden>
      {[1, 2, 3, 4, 5].map((i) => (
        <span
          key={i}
          className="block h-[7px] w-[13px] rounded-[2px]"
          style={{ background: i <= value ? tone : "var(--surface-2)" }}
        />
      ))}
    </span>
  )
}

function Field({ label, children }) {
  return (
    <div>
      <div className="mb-1 text-[10.5px] font-medium tracking-[0.045em] text-[var(--text-muted)]">
        {label}
      </div>
      {children}
    </div>
  )
}

/**
 * Signed contributions of the six engineered feature families.
 *
 * A diverging encode — two poles and a neutral zero line — because the sign is
 * the whole point: some groups argue the report IS a precursor and some argue it
 * is not. The warm pole reuses the same hue the app gives SIF-potential
 * everywhere else, so "orange means fatal potential" holds across every screen.
 */
function FeatureBars({ features }) {
  const rows = [...features].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
  const max = Math.max(0.25, ...rows.map((f) => Math.abs(f.contribution)))

  return (
    <div className="flex flex-col gap-1.5">
      {rows.map((f) => {
        const g = FEATURE_GROUP_BY_ID[f.group]
        const up = f.contribution >= 0
        const w = (Math.abs(f.contribution) / max) * 50
        return (
          <div key={f.group} className="grid grid-cols-[104px_1fr_38px] items-center gap-2">
            <span
              className="truncate text-[12px] text-[var(--text-secondary)]"
              title={g ? `${g.label} — ${g.blurb}` : f.group}
            >
              {g?.label ?? f.group}
            </span>
            <span
              className="relative block h-[9px] w-full rounded-[2px]"
              style={{ background: "var(--surface-2)" }}
              aria-hidden
            >
              {/* neutral zero line */}
              <span
                className="absolute inset-y-[-2px] left-1/2 w-px"
                style={{ background: "var(--axis)" }}
              />
              <span
                className="absolute inset-y-0 rounded-[2px]"
                style={{
                  left: up ? "50%" : `${50 - w}%`,
                  width: `${w}%`,
                  background: up ? "var(--series-2)" : "var(--series-1)",
                }}
              />
            </span>
            <span className="tnum text-right text-[11px] text-[var(--text-muted)]">
              {up ? "+" : "−"}
              {Math.abs(f.contribution).toFixed(2)}
            </span>
          </div>
        )
      })}
      <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1">
        <span className="flex items-center gap-1.5 text-[11px] text-[var(--text-secondary)]">
          <span
            aria-hidden
            className="inline-block size-[9px] rounded-[2px]"
            style={{ background: "var(--series-2)" }}
          />
          pushes toward SIF potential
        </span>
        <span className="flex items-center gap-1.5 text-[11px] text-[var(--text-secondary)]">
          <span
            aria-hidden
            className="inline-block size-[9px] rounded-[2px]"
            style={{ background: "var(--series-1)" }}
          />
          pushes away
        </span>
      </div>
    </div>
  )
}

/* ---------- the panel ---------- */

/**
 * One classification, fully explained. Shared by the Reports drawer and the
 * Analyze screen so a verdict always looks the same wherever it appears.
 */
export default function VerdictPanel({ item, showText = true }) {
  const gap = item.severityPotential - item.severityActual
  const EnergyIcon = HAZARD_ICON[item.precursors.hazardEnergy]
  const barrierBroken = ["absent", "failed", "bypassed"].includes(item.precursors.barrierFailure)
  const tags = rulesOf(item)

  return (
    <div className="flex flex-col gap-5">
      {/* verdict */}
      <div
        className="rounded-lg p-3"
        style={{
          background: item.sifPotential
            ? "color-mix(in srgb, var(--status-critical) 7%, var(--surface-1))"
            : "var(--surface-2)",
          border: `1px solid ${
            item.sifPotential
              ? "color-mix(in srgb, var(--status-critical) 28%, transparent)"
              : "var(--border)"
          }`,
        }}
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <SifBadge sif={item.sifPotential} confidence={item.sifConfidence} />
          <span className="tnum text-[12px] text-[var(--text-secondary)]">
            model score {item.sifConfidence.toFixed(2)}
          </span>
        </div>
        <p className="mt-2 text-[12.5px] leading-relaxed text-[var(--text-secondary)]">
          {item.sifPotential ? (
            <>
              High hazard energy with a{" "}
              <strong className="font-semibold text-[var(--text-primary)]">
                {BARRIER_LABEL[item.precursors.barrierFailure].toLowerCase()}
              </strong>
              . Treat as a fatality precursor regardless of what actually happened.
            </>
          ) : (
            <>
              No fatal-energy / barrier-failure combination detected. Route through normal
              observation close-out.
            </>
          )}
        </p>
      </div>

      {/* severity contrast — the SIF insight */}
      <Field label="SEVERITY: ACTUAL vs POTENTIAL">
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-3">
            <span className="w-[62px] shrink-0 text-[12px] text-[var(--text-secondary)]">Actual</span>
            <SeverityScale value={item.severityActual} tone="var(--text-muted)" />
            <span className="text-[12px]">{SEVERITY_LABEL[item.severityActual]}</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="w-[62px] shrink-0 text-[12px] text-[var(--text-secondary)]">
              Potential
            </span>
            <SeverityScale
              value={item.severityPotential}
              tone={item.severityPotential >= 4 ? "var(--status-critical)" : "var(--status-warning)"}
            />
            <span className="text-[12px] font-medium">
              {SEVERITY_LABEL[item.severityPotential]}
            </span>
          </div>
          {gap >= 2 && (
            <p className="text-[12px] text-[var(--text-secondary)]">
              Escalation gap of{" "}
              <strong className="tnum font-semibold text-[var(--text-primary)]">{gap} levels</strong>{" "}
              — a low-severity record with fatal potential. Severity-based triage would have missed
              this.
            </p>
          )}

          {item.severityScore != null && (
            <div
              className="mt-1 flex items-center gap-3 rounded-lg px-3 py-2"
              style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
            >
              <span className="min-w-0 flex-1 text-[12px] text-[var(--text-secondary)]">
                Continuous severity score
                <span className="block text-[11px] text-[var(--text-muted)]">
                  Regression output, 0–10. Breaks ties inside a band so the queue can rank two
                  reports that are both “potential 5”.
                </span>
              </span>
              <span className="tnum shrink-0 text-[19px] leading-none font-semibold tracking-[-0.01em]">
                {item.severityScore.toFixed(2)}
                <span className="text-[12px] font-normal text-[var(--text-muted)]">/10</span>
              </span>
            </div>
          )}
        </div>
      </Field>

      {/* LSR mapping — multi-label: every rule the report breached */}
      <Field label={tags.length > 1 ? `LIFE-SAVING RULES BREACHED · ${tags.length}` : "LIFE-SAVING RULE MAPPING"}>
        {tags.length === 0 ? (
          <p className="text-[12.5px] text-[var(--text-secondary)]">
            No rule cleared the 0.35 tagging threshold. Routed as a general observation — see the
            limitations note on the Model screen.
          </p>
        ) : (
          <>
            <div className="flex flex-col gap-2">
              {tags.map((t) => (
                <ConfidenceRow
                  key={t.id}
                  label={lsrName(t.id)}
                  value={t.confidence}
                  icon={<LsrIcon id={t.id} />}
                />
              ))}
            </div>
            {LSR_BY_ID[tags[0].id] && (
              <p className="mt-1.5 text-[11.5px] text-[var(--text-muted)]">
                “{LSR_BY_ID[tags[0].id].rule}”
              </p>
            )}
            {tags.length > 1 && (
              <p className="mt-1.5 text-[11.5px] leading-snug text-[var(--text-muted)]">
                Independent per-rule probabilities from the one-vs-rest head — they do not sum to
                100%, because this report breaches {tags.length} rules at once.
              </p>
            )}
          </>
        )}
      </Field>

      {/* precursors */}
      <Field label="PRECURSORS">
        <div className="flex flex-wrap gap-1.5">
          <Badge icon={EnergyIcon}>{item.precursors.hazardEnergy} energy</Badge>
          <Badge tone={barrierBroken ? "critical" : "warning"}>
            {BARRIER_LABEL[item.precursors.barrierFailure]}
          </Badge>
          {item.activity && <Badge>{item.activity}</Badge>}
        </div>
      </Field>

      {/* engineered features — the explainable half of the feature matrix */}
      {item.features?.length > 0 && (
        <Field label="WHY · ENGINEERED SAFETY FEATURES">
          <FeatureBars features={item.features} />
          <p className="mt-2 text-[11.5px] leading-snug text-[var(--text-muted)]">
            The 16 domain features are reported as six named families. The tens of thousands of
            TF-IDF and character-n-gram weights behind them are not an explanation an HSE officer
            can act on; these are.
          </p>
        </Field>
      )}

      {/* preprocessing made visible */}
      {item.normalized?.length > 0 && (
        <Field label="TERMS STANDARDISED BEFORE SCORING">
          <div className="flex flex-wrap gap-1.5">
            {item.normalized.map((n) => (
              <span
                key={n.from}
                className="inline-flex items-center gap-1 rounded-full px-2 py-[3px] text-[11.5px] whitespace-nowrap"
                style={{ background: "var(--surface-2)", border: "1px solid var(--border)" }}
              >
                <span className="font-semibold">{n.from}</span>
                <span className="text-[var(--text-muted)]">→</span>
                <span className="text-[var(--text-secondary)]">{n.to}</span>
              </span>
            ))}
          </div>
          <p className="mt-1.5 text-[11.5px] leading-snug text-[var(--text-muted)]">
            Shorthand is expanded to one canonical form during preprocessing, so “LOTO”, “L.O.T.O.”
            and “lockout tagout” all vectorise as the same term.
          </p>
        </Field>
      )}

      {/* evidence */}
      {showText && item.text && (
        <Field label="REPORT TEXT · MODEL EVIDENCE HIGHLIGHTED">
          <p className="text-[13px] leading-relaxed">
            <HighlightedText text={item.text} evidence={item.evidence} />
          </p>
        </Field>
      )}

      {item.evidence?.length > 0 && (
        <Field label="EVIDENCE WEIGHTS">
          <ul className="flex flex-col gap-1.5">
            {[...item.evidence]
              .sort((a, b) => b.weight - a.weight)
              .map((e) => (
                <li key={e.span} className="flex items-center gap-2">
                  <span className="min-w-0 flex-1 truncate text-[12px] text-[var(--text-secondary)]">
                    “{e.span}”
                  </span>
                  <span
                    aria-hidden
                    className="h-[6px] w-[54px] shrink-0 overflow-hidden rounded-full"
                    style={{ background: "var(--surface-2)" }}
                  >
                    <span
                      className="block h-full rounded-full"
                      style={{ width: `${e.weight * 100}%`, background: "var(--series-2)" }}
                    />
                  </span>
                  <span className="tnum w-[30px] shrink-0 text-right text-[11px] text-[var(--text-muted)]">
                    {e.weight.toFixed(2)}
                  </span>
                </li>
              ))}
          </ul>
        </Field>
      )}
    </div>
  )
}
