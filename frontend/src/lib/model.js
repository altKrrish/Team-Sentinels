/**
 * The engine's shared vocabulary: the six families of engineered domain features
 * and the industry-term glossary used during preprocessing.
 *
 * This file is the FRONTEND MIRROR of `server/closecall/features.py` and
 * `server/closecall/normalize.py`. The backend is authoritative — it is what
 * actually vectorises text and scores it. The copies here exist so the UI can
 * name a feature group and show which shorthand was expanded without a round
 * trip. If you add a feature or a glossary term, change the Python first and
 * then mirror the id/label here.
 */

/* ---------------- the 16 engineered features ---------------- */

/** Six families, sixteen features. Grouped because a group is explainable. */
export const FEATURE_GROUPS = [
  {
    id: "severity",
    label: "Severity indicators",
    blurb: "Outcome and harm language, ranked by seriousness.",
    features: [
      { name: "sev_lexicon_hits", detail: "count of graded harm terms (bruise → fatality)" },
      { name: "max_sev_term_rank", detail: "rank of the most severe term present" },
      { name: "injury_outcome_flag", detail: "an injury actually occurred" },
    ],
  },
  {
    id: "barrier",
    label: "Barrier failures",
    blurb: "Whether the control protecting against the energy held.",
    features: [
      { name: "barrier_absent_cue", detail: "the control was never there" },
      { name: "barrier_bypass_cue", detail: "inhibited, overridden, defeated, unclipped" },
      { name: "barrier_verify_gap", detail: "assumed safe but never verified" },
    ],
  },
  {
    id: "violation",
    label: "Rule violations",
    blurb: "Permit, PPE and procedural non-compliance.",
    features: [
      { name: "permit_violation_cue", detail: "expired, missing or mismatched permit" },
      { name: "ppe_violation_cue", detail: "required protection not worn" },
      { name: "procedure_deviation_cue", detail: "work done outside the written method" },
    ],
  },
  {
    id: "negation",
    label: "Negation handling",
    blurb: "“No gas test” must not read like “gas test”.",
    features: [
      { name: "negation_count", detail: "negation markers in the narrative" },
      { name: "negated_control_scope", detail: "a negation attached to a named control" },
    ],
  },
  {
    id: "measurement",
    label: "Measurements",
    blurb: "Numbers with units — the energy magnitude.",
    features: [
      { name: "pressure_qty", detail: "kg/cm², bar, psi values" },
      { name: "height_qty", detail: "working height in metres" },
      { name: "gas_conc_qty", detail: "%LEL, ppm H₂S, % oxygen" },
    ],
  },
  {
    id: "temporal",
    label: "Temporal patterns",
    blurb: "Overdue intervals and handover windows.",
    features: [
      { name: "overdue_interval", detail: "inspection or test past due, in days" },
      { name: "shift_handover_cue", detail: "night shift, meal break, crew change" },
    ],
  },
]

export const FEATURE_GROUP_BY_ID = Object.fromEntries(FEATURE_GROUPS.map((g) => [g.id, g]))

export const FEATURE_COUNT = FEATURE_GROUPS.reduce((n, g) => n + g.features.length, 0)

/* ---------------- term normalisation ---------------- */

/** Expanded before vectorising so shorthand and longhand share one token. */
export const GLOSSARY = [
  { from: "LOTO", to: "Lockout Tagout" },
  { from: "PTW", to: "Permit to Work" },
  { from: "BOP", to: "Blowout Preventer" },
  { from: "JSA", to: "Job Safety Analysis" },
  { from: "LEL", to: "Lower Explosive Limit" },
  { from: "H2S", to: "Hydrogen Sulphide" },
  { from: "PPE", to: "Personal Protective Equipment" },
  { from: "MCC", to: "Motor Control Centre" },
  { from: "GCS", to: "Gas Collecting Station" },
  { from: "OCS", to: "Oil Collecting Station" },
  { from: "GGS", to: "Group Gathering Station" },
  { from: "ROW", to: "Right of Way" },
  { from: "CSE", to: "Confined Space Entry" },
  { from: "WAH", to: "Work at Height" },
  { from: "SIMOPS", to: "Simultaneous Operations" },
  { from: "TBT", to: "Toolbox Talk" },
  { from: "MOC", to: "Management of Change" },
  { from: "DCS", to: "Distributed Control System" },
]

/** Longest-first so "H2S" inside a longer key can't be matched early. */
const GLOSSARY_SORTED = [...GLOSSARY].sort((a, b) => b.from.length - a.from.length)

/**
 * Which glossary terms appear in a narrative. This is the *visible* half of
 * preprocessing — the UI shows it so the demo can prove normalisation happened.
 * @returns {import('./contract.js').TermExpansion[]}
 */
export function expansionsIn(text) {
  const out = []
  for (const g of GLOSSARY_SORTED) {
    const re = new RegExp(`(^|[^A-Za-z0-9])${g.from.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}([^A-Za-z0-9]|$)`, "i")
    if (re.test(text)) out.push({ from: g.from, to: g.to })
  }
  return out
}
