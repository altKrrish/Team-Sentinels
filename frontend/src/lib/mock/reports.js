/**
 * Demo fixtures in exactly the `Report` shape from ../contract.js.
 *
 * These stand in for the output of the SIF classification model until
 * VITE_USE_MOCK=false points the UI at the real endpoint. Nothing here
 * classifies anything — every field is pre-labelled sample data.
 *
 * Deterministic: a fixed seed, a fixed month anchor, no Date.now(). The demo
 * looks identical on every reload and on every machine.
 */
import { LSR_TAG_THRESHOLD } from "../contract.js"
import { expansionsIn } from "../model.js"

/* ---------- deterministic PRNG (mulberry32) ---------- */
function mulberry32(seed) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const rand = mulberry32(26165)
const pick = (arr) => arr[Math.floor(rand() * arr.length)]
const between = (lo, hi) => lo + rand() * (hi - lo)
const round2 = (n) => Math.round(n * 100) / 100

function weightedPick(items, weightOf) {
  const total = items.reduce((s, it) => s + weightOf(it), 0)
  let r = rand() * total
  for (const it of items) {
    r -= weightOf(it)
    if (r <= 0) return it
  }
  return items[items.length - 1]
}

const clamp = (n, lo, hi) => Math.max(lo, Math.min(hi, n))

/* ---------- engineered-feature detectors ----------
   Stand-ins for the 16 features in ../model.js. Crude regexes, but they read the
   ACTUAL narrative, so a feature bar on screen always corresponds to something
   in the text rather than to a random number. Shared by the fixtures and the
   classify shim so both explain a report the same way.
   ------------------------------------------------------------------------ */

const countMatches = (text, re) => (text.match(re) ?? []).length

const DETECT = {
  negation: (text) =>
    countMatches(text, /\b(no|not|never|without|neither|nobody|failed to|omitted)\b/gi),
  measurement: (text) =>
    countMatches(
      text,
      /\d+(?:\.\d+)?\s*(?:%|ppm|kg\/cm2|bar|psi|kv|m\b|mm\b|t\b|ohm|hrs|kmph|lel)/gi,
    ),
  temporal: (text) =>
    countMatches(
      text,
      /\b(overdue|expired|past (?:its|the)|previous (?:night|month|day|shift|overhaul)|night shift|meal break|crew change|earlier outage|not been signed|three months|days past)\b/gi,
    ),
  violation: (text) =>
    countMatches(
      text,
      /\b(permit|authoris|certificate|goggles|helmet|ppe|procedure|method statement|lift plan|register|tag)\b/gi,
    ),
  severityTerms: (text) =>
    countMatches(
      text,
      /\b(fatal|death|hospitalis|stitch|fractur|crush|amput|burn|asphyxi|injur|dizziness|bruise|first aid)\b/gi,
    ),
}

/**
 * Six named feature-group activations for one narrative.
 * @returns {import('../contract.js').FeatureScore[]}
 */
function buildFeatures({ text, barrier, sevA, sevP }) {
  const broken = ["absent", "failed", "bypassed"].includes(barrier)
  const unverified = barrier === "not-verified"

  const raw = {
    severity: clamp(DETECT.severityTerms(text) * 0.3 + (sevP - 1) / 4 * 0.7, 0, 1),
    barrier: clamp(broken ? 0.72 + (sevP - 1) / 4 * 0.28 : unverified ? 0.46 : 0.3, 0, 1),
    violation: clamp(DETECT.violation(text) * 0.22, 0, 1),
    negation: clamp(DETECT.negation(text) * 0.22, 0, 1),
    measurement: clamp(DETECT.measurement(text) * 0.26, 0, 1),
    temporal: clamp(DETECT.temporal(text) * 0.34, 0, 1),
  }

  /* Signed push on the SIF logit. A broken barrier and a negated control push up;
     an intact/merely-unverified barrier pulls the score down, which is why some
     groups show a negative contribution on routine observations. */
  const WEIGHT = {
    severity: 0.9,
    barrier: 1.15,
    violation: 0.6,
    negation: 0.75,
    measurement: 0.55,
    temporal: 0.5,
  }
  const CENTRE = 0.34 // group activation above this pushes toward SIF

  return Object.entries(raw).map(([group, value]) => ({
    group,
    value: round2(value),
    contribution: round2(clamp((value - CENTRE) * WEIGHT[group], -1, 1)),
  }))
}

/** Continuous ridge-style severity on 0–10, driven by both severity bands. */
function severityScoreFor(sevP, sevA) {
  const base = sevP * 1.7 + sevA * 0.35
  return round2(clamp(base + between(-0.45, 0.45), 0.2, 10))
}

const HSE_REVIEWERS = [
  "P. Baruah · HSE Officer",
  "D. Borah · Fire & Safety",
  "S. Bhuyan · Head HSSE",
  "A. Mahanta · HSE Lead",
]

/* ---------- OIL operating context ---------- */

/** kind: rig | workover | well | gcs | ocs | plant | pipeline | workshop | substation | transport */
const ASSETS = [
  { site: "Duliajan", label: "Rig ITD-9", kind: "rig", dept: "Drilling" },
  { site: "Duliajan", label: "LPG Plant Duliajan", kind: "plant", dept: "Gas Processing" },
  { site: "Duliajan", label: "Central Workshop", kind: "workshop", dept: "Mechanical" },
  { site: "Duliajan", label: "GCS Duliajan", kind: "gcs", dept: "Production" },
  { site: "Duliajan", label: "OCS Duliajan", kind: "ocs", dept: "Production" },
  { site: "Duliajan", label: "Field Transport Duliajan", kind: "transport", dept: "Logistics" },

  { site: "Moran", label: "Rig ITD-4", kind: "rig", dept: "Drilling" },
  { site: "Moran", label: "Workover Rig WR-7", kind: "workover", dept: "Well Services" },
  { site: "Moran", label: "OCS Moran", kind: "ocs", dept: "Production" },
  { site: "Moran", label: "GGS Moran-3", kind: "gcs", dept: "Production" },
  { site: "Moran", label: "Pipeline ROW KM-77", kind: "pipeline", dept: "Pipelines" },
  { site: "Moran", label: "Field Transport Moran", kind: "transport", dept: "Logistics" },

  { site: "Naharkatiya", label: "Well NHK-234", kind: "well", dept: "Well Services" },
  { site: "Naharkatiya", label: "GCS Naharkatiya", kind: "gcs", dept: "Production" },
  { site: "Naharkatiya", label: "Workover Rig WR-2", kind: "workover", dept: "Well Services" },
  { site: "Naharkatiya", label: "Pipeline ROW KM-118", kind: "pipeline", dept: "Pipelines" },
  { site: "Naharkatiya", label: "Substation NHK-2", kind: "substation", dept: "Electrical & Inst." },

  { site: "Baghjan", label: "Rig ITD-12", kind: "rig", dept: "Drilling" },
  { site: "Baghjan", label: "Well BGN-5", kind: "well", dept: "Well Services" },
  { site: "Baghjan", label: "GGS Baghjan-2", kind: "gcs", dept: "Production" },
  { site: "Baghjan", label: "Pipeline ROW KM-42", kind: "pipeline", dept: "Pipelines" },
  { site: "Baghjan", label: "Field Transport Baghjan", kind: "transport", dept: "Logistics" },

  { site: "Kumchai", label: "Workover Rig WR-11", kind: "workover", dept: "Well Services" },
  { site: "Kumchai", label: "Well KUM-3", kind: "well", dept: "Well Services" },
  { site: "Kumchai", label: "GGS Kumchai", kind: "gcs", dept: "Production" },
  { site: "Kumchai", label: "Field Transport Kumchai", kind: "transport", dept: "Logistics" },

  /* Small satellite: few reports, high flagged share. Exists so the density
     ranking's minimum-report guard has something real to catch. */
  { site: "Jaisalmer", label: "Rig JSM-2", kind: "rig", dept: "Drilling" },
  { site: "Jaisalmer", label: "Well Tanot-4", kind: "well", dept: "Well Services" },
]

const MAIN_ASSETS = ASSETS.filter((a) => a.site !== "Jaisalmer")
const SAT_ASSETS = ASSETS.filter((a) => a.site === "Jaisalmer")

/** How much reporting volume each site generates. */
const SITE_WEIGHT = { Duliajan: 30, Moran: 26, Naharkatiya: 24, Baghjan: 20, Kumchai: 14 }
/** Where fatal potential actually concentrates — drives the density ranking. */
const SITE_SIF_WEIGHT = { Duliajan: 15, Moran: 26, Naharkatiya: 17, Baghjan: 33, Kumchai: 23 }

const REPORTERS = [
  "R. Gogoi · Shift Supervisor",
  "P. Baruah · HSE Officer",
  "A. Saikia · Drilling Engineer",
  "M. Rahman · Rig Mechanic",
  "S. Dutta · Production Chemist",
  "K. Phukan · Safety Steward",
  "B. Tamuli · Electrical Foreman",
  "N. Hazarika · Area Operator",
  "J. Kalita · Pipeline Inspector",
  "T. Chetia · Contractor Supervisor",
  "D. Borah · Fire & Safety",
  "L. Nath · Instrument Technician",
  "V. Sharma · Lifting Supervisor",
  "H. Das · Transport Coordinator",
]

/* ---------- scenario templates ----------
   Each is one recurring real-world pattern. `evidence` phrases are verbatim
   substrings of `text`, so the drawer's highlighting always lands.
   ------------------------------------------------------------------------ */

/** SIF-potential: high hazard energy present AND a barrier absent/failed/bypassed. */
const SIF_SCENARIOS = [
  {
    activity: "Hot work / welding",
    type: "near-miss",
    kinds: ["gcs", "ocs", "plant", "workshop"],
    lsr: "hot-work",
    lsr2: "work-authorisation",
    energy: "thermal",
    barrier: "bypassed",
    sevA: 1,
    sevP: 5,
    text: (a) =>
      `Welding on a produced-water line at ${a.label} resumed after the meal break without a fresh gas test. The hot work permit had expired at 1200 hrs and the standby fire watch had left the location. A portable meter later read 8% LEL at the weld joint.`,
    evidence: [
      ["without a fresh gas test", 0.94],
      ["hot work permit had expired", 0.87],
      ["standby fire watch had left the location", 0.71],
      ["8% LEL at the weld joint", 0.63],
    ],
  },
  {
    activity: "Pump / rotating equipment maintenance",
    type: "near-miss",
    kinds: ["gcs", "ocs", "plant"],
    lsr: "energy-isolation",
    lsr2: "bypassing-safety-controls",
    energy: "mechanical",
    barrier: "absent",
    sevA: 1,
    sevP: 5,
    text: (a) =>
      `A fitter opened the mechanical seal housing of the transfer pump at ${a.label} while the motor was still racked in. No lock-out tag was fitted at the MCC and the isolation was not verified by try-out. The pump auto-started on level control with his hand inside the coupling guard.`,
    evidence: [
      ["No lock-out tag was fitted at the MCC", 0.96],
      ["isolation was not verified by try-out", 0.9],
      ["auto-started on level control", 0.78],
      ["hand inside the coupling guard", 0.74],
    ],
  },
  {
    activity: "Vessel entry / cleaning",
    type: "incident",
    kinds: ["gcs", "ocs", "plant"],
    lsr: "confined-space",
    lsr2: "work-authorisation",
    energy: "chemical",
    barrier: "absent",
    sevA: 2,
    sevP: 5,
    text: (a) =>
      `Two contract workers entered the test separator at ${a.label} to remove sludge wearing only cloth masks. There was no entry permit, no continuous gas monitoring and no attendant posted at the manway. Oxygen at the vessel bottom measured 16.4% when the area officer arrived and one man reported dizziness.`,
    evidence: [
      ["no continuous gas monitoring", 0.95],
      ["There was no entry permit", 0.91],
      ["no attendant posted at the manway", 0.86],
      ["Oxygen at the vessel bottom measured 16.4%", 0.8],
    ],
  },
  {
    activity: "Pressure testing",
    type: "near-miss",
    kinds: ["well", "pipeline", "gcs"],
    lsr: "line-of-fire",
    lsr2: "bypassing-safety-controls",
    energy: "pressure",
    barrier: "absent",
    sevA: 1,
    sevP: 5,
    text: (a) =>
      `During hydrotest of the flowline at ${a.label} three crew members stood in line of fire of the 2 inch test hose at 210 kg/cm2. No whip check was installed on the hammer union and the barricade tape had been removed to bring in a tool box.`,
    evidence: [
      ["stood in line of fire", 0.92],
      ["No whip check was installed", 0.9],
      ["barricade tape had been removed", 0.76],
    ],
  },
  {
    activity: "Derrick / monkey board work",
    type: "UA",
    kinds: ["rig"],
    lsr: "working-at-height",
    lsr2: null,
    energy: "gravity",
    barrier: "bypassed",
    sevA: 1,
    sevP: 5,
    text: (a) =>
      `A derrickman at ${a.label} was observed working at the monkey board with his fall arrest lanyard unclipped while changing pipe fingers. The secondary retention line was frayed at the thimble and the drops register had not been signed for the shift.`,
    evidence: [
      ["fall arrest lanyard unclipped", 0.95],
      ["secondary retention line was frayed", 0.79],
      ["drops register had not been signed", 0.6],
    ],
  },
  {
    activity: "Crane / mechanical lifting",
    type: "UA",
    kinds: ["rig", "workover"],
    lsr: "safe-mechanical-lifting",
    lsr2: "line-of-fire",
    energy: "gravity",
    barrier: "inadequate",
    sevA: 1,
    sevP: 5,
    text: (a) =>
      `The 25 T crane at ${a.label} lifted a 3.5 T BOP spool over the live gas header because the approved lift plan route was blocked by a parked trailer. One leg of the wire rope sling showed six broken wires and no tag line was used to control the load.`,
    evidence: [
      ["six broken wires", 0.91],
      ["over the live gas header", 0.85],
      ["no tag line was used", 0.79],
      ["approved lift plan route was blocked", 0.7],
    ],
  },
  {
    activity: "Gas detection / alarm maintenance",
    type: "UC",
    kinds: ["gcs", "plant"],
    lsr: "bypassing-safety-controls",
    lsr2: "work-authorisation",
    energy: "chemical",
    barrier: "bypassed",
    sevA: 1,
    sevP: 5,
    text: (a) =>
      `Gas detection covering the compressor shed at ${a.label} was found inhibited in the DCS since the previous night shift so that a nuisance alarm would stop. The override was not recorded in the impairment register and this area has a history of H2S at 12 ppm.`,
    evidence: [
      ["found inhibited in the DCS", 0.96],
      ["override was not recorded in the impairment register", 0.88],
      ["history of H2S at 12 ppm", 0.72],
    ],
  },
  {
    activity: "Crew transport",
    type: "UA",
    kinds: ["transport"],
    lsr: "driving",
    lsr2: null,
    energy: "motion",
    barrier: "inadequate",
    sevA: 1,
    sevP: 5,
    text: (a) =>
      `The crew change bus operated by ${a.label} overtook a loaded tanker on a blind curve at 2130 hrs in light rain. Front tyres were below the tread limit and four of the twenty passengers were standing in the aisle without seats.`,
    evidence: [
      ["overtook a loaded tanker on a blind curve", 0.93],
      ["below the tread limit", 0.81],
      ["four of the twenty passengers were standing", 0.7],
    ],
  },
  {
    activity: "Permit management",
    type: "near-miss",
    kinds: ["gcs", "ocs", "plant"],
    lsr: "work-authorisation",
    lsr2: "energy-isolation",
    energy: "pressure",
    barrier: "failed",
    sevA: 1,
    sevP: 5,
    text: (a) =>
      `Two separate work permits were issued against the same isolation certificate at ${a.label} — a valve replacement and an instrument calibration. Neither team knew about the other and the isolation was broken while the second team was still on the line.`,
    evidence: [
      ["isolation was broken while the second team was still on the line", 0.95],
      ["same isolation certificate", 0.9],
      ["Neither team knew about the other", 0.84],
    ],
  },
  {
    activity: "HT electrical maintenance",
    type: "near-miss",
    kinds: ["substation", "plant", "gcs"],
    lsr: "energy-isolation",
    lsr2: "bypassing-safety-controls",
    energy: "electrical",
    barrier: "bypassed",
    sevA: 1,
    sevP: 5,
    text: (a) =>
      `An electrician opened the 11 kV incoming panel at ${a.label} to replace a CT while the bus was still charged from the alternate feeder. The panel interlock had been defeated during an earlier outage and no live-line detection was carried out before opening.`,
    evidence: [
      ["while the bus was still charged", 0.96],
      ["panel interlock had been defeated", 0.91],
      ["no live-line detection was carried out", 0.86],
    ],
  },
  {
    activity: "Hot work / grinding",
    type: "near-miss",
    kinds: ["plant"],
    lsr: "hot-work",
    lsr2: null,
    energy: "thermal",
    barrier: "failed",
    sevA: 1,
    sevP: 5,
    text: (a) =>
      `Grinding was in progress within 8 m of the LPG bullet at ${a.label} while its water draw-off was left open and unattended. Sparks were observed reaching the bund wall and the fire water ring main isolation valve was found closed at the pump house.`,
    evidence: [
      ["fire water ring main isolation valve was found closed", 0.94],
      ["water draw-off was left open and unattended", 0.9],
      ["Sparks were observed reaching the bund wall", 0.85],
    ],
  },
  {
    activity: "Excavation / ROW work",
    type: "incident",
    kinds: ["pipeline"],
    lsr: "line-of-fire",
    lsr2: "work-authorisation",
    energy: "pressure",
    barrier: "absent",
    sevA: 2,
    sevP: 5,
    text: (a) =>
      `A contractor excavator working on ${a.label} struck an unmarked 8 inch gas line at 1.2 m depth. The line had been depressurised the previous day for pigging; otherwise a release would have occurred beside two workers standing in the trench.`,
    evidence: [
      ["struck an unmarked 8 inch gas line", 0.95],
      ["a release would have occurred beside two workers standing in the trench", 0.9],
      ["depressurised the previous day", 0.55],
    ],
  },
  {
    activity: "Cellar pit / sump entry",
    type: "UA",
    kinds: ["workover", "rig", "well"],
    lsr: "confined-space",
    lsr2: null,
    energy: "chemical",
    barrier: "absent",
    sevA: 1,
    sevP: 4,
    text: (a) =>
      `A rigger climbed into the cellar pit at ${a.label} to retrieve a dropped hammer while the mud system was circulating. There was no rescue plan and no tripod at the opening, and the pit held 400 mm of oily water with H2S odour reported by the crew.`,
    evidence: [
      ["no rescue plan and no tripod at the opening", 0.9],
      ["H2S odour reported by the crew", 0.84],
      ["while the mud system was circulating", 0.62],
    ],
  },
  {
    activity: "Scaffolding",
    type: "UC",
    kinds: ["plant", "gcs"],
    lsr: "working-at-height",
    lsr2: null,
    energy: "gravity",
    barrier: "inadequate",
    sevA: 1,
    sevP: 5,
    text: (a) =>
      `The scaffold erected against the absorber column at ${a.label} for insulation work had no toe boards and one missing ledger brace. The green tag from the previous month was still displayed after modification and three men were working from the top lift 9 m above paved ground.`,
    evidence: [
      ["green tag from the previous month was still displayed after modification", 0.91],
      ["no toe boards and one missing ledger brace", 0.86],
      ["9 m above paved ground", 0.72],
    ],
  },
  {
    activity: "Tripping / pipe handling",
    type: "near-miss",
    kinds: ["rig", "workover"],
    lsr: "line-of-fire",
    lsr2: "safe-mechanical-lifting",
    energy: "gravity",
    barrier: "absent",
    sevA: 1,
    sevP: 4,
    text: (a) =>
      `A 4 kg pipe wiper fell from the derrick board at ${a.label} and landed 1.5 m from the driller inside the working area. Tool tethering was not used at height and the dropped-object inspection was overdue by three months.`,
    evidence: [
      ["Tool tethering was not used at height", 0.9],
      ["landed 1.5 m from the driller", 0.85],
      ["dropped-object inspection was overdue by three months", 0.8],
    ],
  },
  {
    activity: "BOP / well control testing",
    type: "incident",
    kinds: ["rig", "workover"],
    lsr: "line-of-fire",
    lsr2: "energy-isolation",
    energy: "pressure",
    barrier: "failed",
    sevA: 3,
    sevP: 5,
    text: (a) =>
      `A high-pressure grease line on the BOP at ${a.label} failed during function testing and struck a technician on the forearm, requiring four stitches. The fitting had been reused after the previous overhaul and was not torque checked, and no exclusion zone was maintained during pressurisation.`,
    evidence: [
      ["fitting had been reused after the previous overhaul and was not torque checked", 0.93],
      ["no exclusion zone was maintained during pressurisation", 0.88],
      ["struck a technician on the forearm", 0.8],
    ],
  },
  {
    activity: "Tripping / pipe handling",
    type: "incident",
    kinds: ["rig", "workover"],
    lsr: "line-of-fire",
    lsr2: null,
    energy: "mechanical",
    barrier: "absent",
    sevA: 4,
    sevP: 5,
    text: (a) =>
      `A rigger's finger was crushed between the tong latch and the tool joint at ${a.label} while making up a connection; he was hospitalised for two days. The hands-free tool was available on site but not used and the driller could not see the floor hand from the console.`,
    evidence: [
      ["hands-free tool was available on site but not used", 0.94],
      ["crushed between the tong latch and the tool joint", 0.89],
      ["driller could not see the floor hand from the console", 0.78],
    ],
  },
  {
    activity: "Well intervention",
    type: "near-miss",
    kinds: ["well", "workover"],
    lsr: "energy-isolation",
    lsr2: "line-of-fire",
    energy: "pressure",
    barrier: "not-verified",
    sevA: 1,
    sevP: 5,
    text: (a) =>
      `The wellhead at ${a.label} was opened for wireline rig-up on the assumption that the well was dead, without a shut-in pressure check. Casing pressure was later found at 34 kg/cm2 and only one barrier was in place during rig-up.`,
    evidence: [
      ["without a shut-in pressure check", 0.94],
      ["only one barrier was in place during rig-up", 0.92],
      ["Casing pressure was later found at 34 kg/cm2", 0.83],
    ],
  },
]

/** Non-SIF: real findings worth closing, but no fatal energy / intact barriers. */
const BASE_SCENARIOS = [
  {
    activity: "Housekeeping",
    type: "UC",
    kinds: ["gcs", "ocs", "plant", "workshop", "rig", "workover"],
    lsr: null,
    energy: "gravity",
    barrier: "inadequate",
    sevA: 1,
    sevP: 2,
    text: (a) =>
      `Empty drums and scrap tubing were stacked along the walkway near the manifold at ${a.label}, reducing the escape route width to about 600 mm. The material was cleared the same shift.`,
    evidence: [["reducing the escape route width to about 600 mm", 0.5]],
  },
  {
    activity: "PPE compliance",
    type: "UA",
    kinds: ["workshop", "gcs", "ocs", "rig", "workover", "plant"],
    lsr: null,
    energy: "mechanical",
    barrier: "absent",
    sevA: 1,
    sevP: 2,
    text: (a) =>
      `A helper was chipping paint at ${a.label} without safety goggles, using only spectacles. Correct eye protection was issued on the spot and a toolbox talk was held.`,
    evidence: [["without safety goggles", 0.58]],
  },
  {
    activity: "Hot work / welding",
    type: "UC",
    kinds: ["workshop", "plant", "gcs"],
    lsr: "hot-work",
    energy: "thermal",
    barrier: "inadequate",
    sevA: 1,
    sevP: 2,
    lsrConf: [0.6, 0.75],
    text: (a) =>
      `Welding return cable insulation was found damaged in the fabrication bay at ${a.label}. Work was stopped and the cable replaced before hot work resumed under a valid permit.`,
    evidence: [["return cable insulation was found damaged", 0.55]],
  },
  {
    activity: "Crew transport",
    type: "UA",
    kinds: ["transport"],
    lsr: "driving",
    energy: "motion",
    barrier: "inadequate",
    sevA: 1,
    sevP: 2,
    lsrConf: [0.65, 0.8],
    text: (a) =>
      `A light vehicle from ${a.label} was reversed in the parking bay without a spotter; no contact occurred. The journey management form was available and recorded speed stayed under 20 kmph.`,
    evidence: [["reversed in the parking bay without a spotter", 0.52]],
  },
  {
    activity: "Access / ladders",
    type: "UA",
    kinds: ["workshop", "gcs", "ocs", "plant"],
    lsr: "working-at-height",
    energy: "gravity",
    barrier: "inadequate",
    sevA: 1,
    sevP: 2,
    lsrConf: [0.62, 0.78],
    text: (a) =>
      `A step ladder in the store at ${a.label} was used at full extension on an uneven floor to reach a shelf 1.4 m high, with nobody footing the ladder.`,
    evidence: [["used at full extension on an uneven floor", 0.56]],
  },
  {
    activity: "Crane / mechanical lifting",
    type: "UC",
    kinds: ["rig", "workover", "workshop"],
    lsr: "safe-mechanical-lifting",
    energy: "gravity",
    barrier: "not-verified",
    sevA: 1,
    sevP: 2,
    lsrConf: [0.68, 0.82],
    text: (a) =>
      `Colour code tags were missing on two round slings in the rigging loft at ${a.label}. Both slings were withdrawn from service during the inspection and no lift was in progress.`,
    evidence: [["Colour code tags were missing on two round slings", 0.5]],
  },
  {
    activity: "Permit management",
    type: "UC",
    kinds: ["gcs", "ocs", "plant", "well"],
    lsr: "work-authorisation",
    energy: "mechanical",
    barrier: "not-verified",
    sevA: 1,
    sevP: 2,
    lsrConf: [0.62, 0.78],
    text: (a) =>
      `The permit copy was not displayed at the job site at ${a.label}, although a valid permit existed in the control room register and the job scope matched.`,
    evidence: [["permit copy was not displayed at the job site", 0.54]],
  },
  {
    activity: "Vessel entry / cleaning",
    type: "UC",
    kinds: ["gcs", "ocs", "plant"],
    lsr: "confined-space",
    energy: "chemical",
    barrier: "not-verified",
    sevA: 1,
    sevP: 1,
    lsrConf: [0.55, 0.7],
    text: (a) =>
      `The confined space register at ${a.label} had two closed entries without the closing signature. No entry was in progress at the time of the audit and both attendants confirmed exit.`,
    evidence: [["two closed entries without the closing signature", 0.45]],
  },
  {
    activity: "Pump / rotating equipment maintenance",
    type: "UC",
    kinds: ["gcs", "ocs", "plant", "workshop"],
    lsr: "energy-isolation",
    energy: "electrical",
    barrier: "inadequate",
    sevA: 1,
    sevP: 2,
    lsrConf: [0.58, 0.72],
    text: (a) =>
      `Two lock-out padlocks in the electrical store at ${a.label} were found without unique keys. They were removed from service the same day and replaced from the spare set.`,
    evidence: [["found without unique keys", 0.48]],
  },
  {
    activity: "Housekeeping",
    type: "UA",
    kinds: ["workshop", "rig", "workover", "gcs"],
    lsr: "line-of-fire",
    energy: "gravity",
    barrier: "not-verified",
    sevA: 1,
    sevP: 2,
    lsrConf: [0.6, 0.75],
    text: (a) =>
      `An operator walked under a stationary suspended chain block hook at ${a.label}. No load was attached and the area was not an active lift zone.`,
    evidence: [["walked under a stationary suspended chain block hook", 0.52]],
  },
  {
    activity: "Gas detection / alarm maintenance",
    type: "UC",
    kinds: ["gcs", "plant", "ocs"],
    lsr: "bypassing-safety-controls",
    energy: "chemical",
    barrier: "inadequate",
    sevA: 1,
    sevP: 2,
    lsrConf: [0.55, 0.7],
    text: (a) =>
      `A local alarm horn at ${a.label} was found with its volume dial turned to minimum during the shift round and was restored immediately. Logic and field devices tested healthy.`,
    evidence: [["volume dial turned to minimum", 0.5]],
  },
  {
    activity: "Gas detection / alarm maintenance",
    type: "UC",
    kinds: ["gcs", "plant", "ocs", "rig", "workover"],
    lsr: null,
    energy: "chemical",
    barrier: "not-verified",
    sevA: 1,
    sevP: 2,
    text: (a) =>
      `A portable gas detector at ${a.label} was 11 days past its bump test due date. It was withdrawn from use and replaced from the calibrated spare pool.`,
    evidence: [["11 days past its bump test due date", 0.47]],
  },
  {
    activity: "Well intervention",
    type: "UC",
    kinds: ["workover", "well"],
    lsr: "energy-isolation",
    energy: "pressure",
    barrier: "failed",
    sevA: 1,
    sevP: 3,
    lsrConf: [0.5, 0.66],
    text: (a) =>
      `A hydraulic hose on the workover unit at ${a.label} was weeping oil at the swivel end. Pressure was bled down and the hose was replaced within the shift under a fresh permit.`,
    evidence: [["weeping oil at the swivel end", 0.5]],
  },
  {
    activity: "Housekeeping",
    type: "incident",
    kinds: ["rig", "workover", "gcs"],
    lsr: null,
    energy: "gravity",
    barrier: "inadequate",
    sevA: 2,
    sevP: 3,
    text: (a) =>
      `A contract helper slipped on a greasy floor plate near the shale shaker at ${a.label} and sustained a bruised elbow. First aid was given at the site dispensary and he returned to work.`,
    evidence: [["slipped on a greasy floor plate", 0.55]],
  },
  {
    activity: "Steam / thermal systems",
    type: "UC",
    kinds: ["gcs", "ocs", "plant"],
    lsr: "line-of-fire",
    energy: "thermal",
    barrier: "failed",
    sevA: 1,
    sevP: 3,
    lsrConf: [0.48, 0.64],
    text: (a) =>
      `A steam tracing line at ${a.label} was leaking beside the walkway and the lagging was hot to touch. The area was cordoned and the leak scheduled for the next shutdown window.`,
    evidence: [["leaking beside the walkway", 0.5], ["lagging was hot to touch", 0.58]],
  },
  {
    activity: "Tripping / pipe handling",
    type: "UC",
    kinds: ["rig", "workover", "workshop"],
    lsr: "line-of-fire",
    energy: "gravity",
    barrier: "absent",
    sevA: 1,
    sevP: 3,
    lsrConf: [0.66, 0.8],
    text: (a) =>
      `A 5 kg spanner was left on the pipe rack 3 m above grade at ${a.label} after the shift and was found during the housekeeping round with nobody working below.`,
    evidence: [["left on the pipe rack 3 m above grade", 0.68]],
  },
  {
    activity: "Fire protection",
    type: "UC",
    kinds: ["gcs", "ocs", "plant", "workshop", "transport"],
    lsr: null,
    energy: "thermal",
    barrier: "inadequate",
    sevA: 1,
    sevP: 2,
    text: (a) =>
      `A wooden pallet stacked against the fire hydrant at ${a.label} obstructed access by about half a metre. It was removed and the hydrant access line was repainted.`,
    evidence: [["obstructed access by about half a metre", 0.46]],
  },
  {
    activity: "Chemical handling",
    type: "UC",
    kinds: ["rig", "workover", "plant", "gcs"],
    lsr: null,
    energy: "chemical",
    barrier: "absent",
    sevA: 1,
    sevP: 3,
    text: (a) =>
      `An unlabelled chemical drum was found in the mud chemical store at ${a.label}. Contents were identified as caustic soda solution, re-labelled and moved to the bunded area.`,
    evidence: [["An unlabelled chemical drum", 0.6], ["identified as caustic soda solution", 0.5]],
  },
  {
    activity: "HT electrical maintenance",
    type: "UC",
    kinds: ["substation", "gcs", "ocs", "plant"],
    lsr: "energy-isolation",
    energy: "electrical",
    barrier: "inadequate",
    sevA: 1,
    sevP: 3,
    lsrConf: [0.5, 0.65],
    text: (a) =>
      `Earth pit resistance at ${a.label} was recorded as 6.2 ohm against the 5 ohm limit in the monthly check. The pit was watered and re-tested at 4.1 ohm the following day.`,
    evidence: [["recorded as 6.2 ohm against the 5 ohm limit", 0.55]],
  },
  {
    activity: "Crew transport",
    type: "UA",
    kinds: ["transport"],
    lsr: "driving",
    energy: "motion",
    barrier: "absent",
    sevA: 1,
    sevP: 3,
    lsrConf: [0.6, 0.75],
    text: (a) =>
      `A tanker driver at ${a.label} did not chock the wheels during unloading. The vehicle was on level ground with the engine off and the driver was briefed before the next trip.`,
    evidence: [["did not chock the wheels during unloading", 0.62]],
  },
  {
    activity: "Pump / rotating equipment maintenance",
    type: "UC",
    kinds: ["gcs", "ocs", "plant"],
    lsr: "line-of-fire",
    energy: "mechanical",
    barrier: "inadequate",
    sevA: 1,
    sevP: 3,
    lsrConf: [0.46, 0.62],
    text: (a) =>
      `The belt drive guard on the water injection pump at ${a.label} was loose on two bolts. The machine was stopped and the guard refitted before restart.`,
    evidence: [["guard on the water injection pump", 0.5], ["loose on two bolts", 0.55]],
  },
  {
    activity: "Excavation / ROW work",
    type: "UC",
    kinds: ["pipeline"],
    lsr: "work-authorisation",
    energy: "gravity",
    barrier: "inadequate",
    sevA: 1,
    sevP: 3,
    lsrConf: [0.55, 0.7],
    text: (a) =>
      `A 1.1 m deep inspection trench on ${a.label} was left without edge protection or warning signs overnight. Barricades and reflective tape were installed the next morning.`,
    evidence: [["left without edge protection or warning signs overnight", 0.6]],
  },
  {
    activity: "Access / ladders",
    type: "UC",
    kinds: ["gcs", "ocs", "plant", "rig"],
    lsr: "working-at-height",
    energy: "gravity",
    barrier: "failed",
    sevA: 1,
    sevP: 3,
    lsrConf: [0.6, 0.75],
    text: (a) =>
      `A handrail section on the second-level walkway at ${a.label} was corroded and moved under hand pressure. The span was barricaded and a work order raised for replacement.`,
    evidence: [["corroded and moved under hand pressure", 0.62]],
  },
  {
    activity: "Housekeeping",
    type: "UC",
    kinds: ["gcs", "ocs", "plant", "workshop", "substation"],
    lsr: null,
    energy: "electrical",
    barrier: "inadequate",
    sevA: 1,
    sevP: 2,
    text: (a) =>
      `Illumination at the pump house at ${a.label} measured below the required level after two fittings failed. Temporary lighting was arranged for the night shift.`,
    evidence: [["measured below the required level", 0.44]],
  },
]

/* ---------- month / volume schedule ----------
   Anchored to a fixed month so the fixture never shifts under the demo. */
const END_MONTH = { year: 2026, month: 8 } // Aug 2026, inclusive
const MONTH_VOLUME = [9, 11, 10, 13, 12, 10, 14, 11, 13, 12, 12, 13] // = 140
const MONTH_SIF = [2, 3, 2, 4, 3, 2, 4, 2, 3, 2, 2, 2] //             = 31 (22.1%)

function monthsBack(count) {
  const out = []
  for (let i = count - 1; i >= 0; i--) {
    let m = END_MONTH.month - i
    let y = END_MONTH.year
    while (m <= 0) {
      m += 12
      y -= 1
    }
    out.push({ y, m })
  }
  return out
}

const MONTHS = monthsBack(12)

/* ---------- assembly ---------- */

let seq = 0

function buildReport({ scenario, asset, y, m, sif }) {
  const day = 1 + Math.floor(rand() * 27)
  const iso = `${y}-${String(m).padStart(2, "0")}-${String(day).padStart(2, "0")}`
  const text = scenario.text(asset)

  const [lo, hi] = scenario.lsrConf ?? (sif ? [0.78, 0.97] : [0.5, 0.7])
  const lsr = scenario.lsr
    ? { id: scenario.lsr, confidence: round2(between(lo, hi)) }
    : { id: "unmapped", confidence: round2(between(0.22, 0.44)) }
  const lsrSecondary =
    scenario.lsr2 && rand() > 0.25
      ? { id: scenario.lsr2, confidence: round2(between(0.42, 0.68)) }
      : null

  /* Multi-label head: every rule that cleared the threshold, strongest first.
     An unmapped primary means nothing cleared it, so the list is empty rather
     than carrying a tag the model isn't confident about. */
  const lsrTags = [lsr, lsrSecondary]
    .filter((t) => t && t.id !== "unmapped" && t.confidence >= LSR_TAG_THRESHOLD)
    .sort((a, b) => b.confidence - a.confidence)

  // age in months from the anchor -> older reports are mostly closed
  const age = (END_MONTH.year - y) * 12 + (END_MONTH.month - m)
  const r = rand()
  const status =
    age >= 5 ? (r > 0.08 ? "closed" : "in-progress")
    : age >= 2 ? (r > 0.55 ? "closed" : r > 0.2 ? "in-progress" : "open")
    : r > 0.62 ? "closed" : r > 0.25 ? "in-progress" : "open"

  /* Human-in-the-loop: only flagged reports go to an HSE officer, and the older
     the report the likelier someone has already ruled on it. An override is HSE
     disagreeing with the model — the disagreement rate is worth showing.

     A share of the undecided flags are `in-progress`: an officer holds the file
     and has not ruled. Mirrors _review() in server/closecall/serve.py so demo
     mode and the live backend tell the same story. */
  const decideRoll = rand()
  const decideOdds = !sif ? 0 : age >= 3 ? 0.88 : age >= 1 ? 0.55 : 0.22
  const claimRoll = rand()
  const reviewer = pick(HSE_REVIEWERS)
  const review =
    decideRoll < decideOdds
      ? {
          state: rand() < 0.84 ? "confirmed" : "overridden",
          by: reviewer,
          at: iso,
          note: null,
        }
      : sif && claimRoll < 0.38
        ? {
            state: "in-progress",
            by: reviewer,
            at: iso,
            note: "Picked up for review — awaiting a site walk-down before a decision.",
          }
        : { state: "pending", by: null, at: null, note: null }

  seq += 1

  return {
    id: `HSSE-${y}-${String(seq).padStart(4, "0")}`,
    reportedAt: iso,
    type: scenario.type,
    site: asset.site,
    asset: asset.label,
    department: asset.dept,
    activity: scenario.activity,
    text,
    reportedBy: pick(REPORTERS),
    sifPotential: sif,
    sifConfidence: round2(sif ? between(0.63, 0.97) : between(0.03, 0.41)),
    lsr,
    lsrSecondary,
    lsrTags,
    precursors: {
      hazardEnergy: scenario.energy,
      barrierFailure: scenario.barrier,
    },
    severityScore: severityScoreFor(scenario.sevP, scenario.sevA),
    severityActual: scenario.sevA,
    severityPotential: scenario.sevP,
    // keep only spans that really occur in the text — highlighting depends on it
    evidence: scenario.evidence
      .filter(([span]) => text.includes(span))
      .map(([span, weight]) => ({ span, weight })),
    features: buildFeatures({
      text,
      barrier: scenario.barrier,
      sevA: scenario.sevA,
      sevP: scenario.sevP,
    }),
    normalized: expansionsIn(text),
    status,
    review,
  }
}

function assetFor(scenario, sif, pool = MAIN_ASSETS) {
  const candidates = pool.filter((a) => scenario.kinds.includes(a.kind))
  const list = candidates.length ? candidates : pool
  const weights = sif ? SITE_SIF_WEIGHT : SITE_WEIGHT
  return weightedPick(list, (a) => weights[a.site] ?? 10)
}

function generate() {
  const out = []

  MONTHS.forEach(({ y, m }, i) => {
    const total = MONTH_VOLUME[i]
    const sifCount = MONTH_SIF[i]

    for (let k = 0; k < total; k++) {
      const sif = k < sifCount
      const scenario = pick(sif ? SIF_SCENARIOS : BASE_SCENARIOS)
      out.push({ scenario, asset: assetFor(scenario, sif), y, m, sif })
    }
  })

  /* Satellite site: 5 reports, 2 flagged (40% density). High enough to top the
     ranking on raw ratio — the dashboard's minimum-report guard holds it back. */
  const satPlan = [
    { mi: 4, sif: true },
    { mi: 6, sif: false },
    { mi: 8, sif: true },
    { mi: 9, sif: false },
    { mi: 11, sif: false },
  ]
  satPlan.forEach(({ mi, sif }) => {
    const { y, m } = MONTHS[mi]
    const scenario = pick(sif ? SIF_SCENARIOS : BASE_SCENARIOS)
    out.push({ scenario, asset: assetFor(scenario, sif, SAT_ASSETS), y, m, sif })
  })

  return out
    .map(buildReport)
    .sort((a, b) => (a.reportedAt < b.reportedAt ? 1 : a.reportedAt > b.reportedAt ? -1 : 0))
}

/** @type {import('../contract.js').Report[]} */
export const MOCK_REPORTS = generate()

export const SITES = [...new Set(ASSETS.map((a) => a.site))]
export const ACTIVITIES = [
  ...new Set([...SIF_SCENARIOS, ...BASE_SCENARIOS].map((s) => s.activity)),
].sort()

/**
 * Stand-in for POST /classify so the Analyze screen works without a backend.
 * Deliberately dumb keyword matching — it is a demo shim, NOT the model. It
 * returns the same Classification shape the real endpoint must return.
 */
export function mockClassify(text) {
  const t = text.toLowerCase()

  const RULES = [
    { id: "confined-space", energy: "chemical", words: ["confined space", "vessel", "separator", "manway", "tank entry", "cellar pit", "sump", "oxygen"] },
    { id: "hot-work", energy: "thermal", words: ["hot work", "weld", "grinding", "spark", "cutting torch", "lel", "flammable"] },
    { id: "energy-isolation", energy: "electrical", words: ["isolation", "lock-out", "lockout", "loto", "de-energis", "energis", "11 kv", "breaker", "racked in"] },
    { id: "working-at-height", energy: "gravity", words: ["height", "scaffold", "harness", "lanyard", "monkey board", "ladder", "fall arrest", "derrick"] },
    { id: "line-of-fire", energy: "pressure", words: ["line of fire", "hydrotest", "whip", "pressuris", "dropped object", "trench", "struck"] },
    { id: "safe-mechanical-lifting", energy: "gravity", words: ["crane", "lift", "sling", "tag line", "suspended load", "hoist", "rigging"] },
    { id: "bypassing-safety-controls", energy: "chemical", words: ["bypass", "inhibit", "override", "defeat", "disable", "alarm off", "impair"] },
    { id: "driving", energy: "motion", words: ["driv", "vehicle", "bus", "tanker", "overtak", "speed", "seat belt", "tyre"] },
    { id: "work-authorisation", energy: "pressure", words: ["permit", "authoris", "work order", "certificate", "jsa"] },
  ]

  const scored = RULES.map((r) => ({
    ...r,
    hits: r.words.filter((w) => t.includes(w)).length,
  }))
    .filter((r) => r.hits > 0)
    .sort((a, b) => b.hits - a.hits)

  const BARRIERS = [
    { id: "bypassed", words: ["bypass", "inhibit", "override", "defeat", "disable", "unclipped", "removed"] },
    { id: "absent", words: ["no ", "not used", "without", "none", "missing", "absent"] },
    { id: "failed", words: ["fail", "broke", "burst", "leak", "gave way", "damaged", "expired"] },
    { id: "inadequate", words: ["inadequate", "insufficient", "loose", "worn", "frayed", "below", "overdue"] },
    { id: "not-verified", words: ["not verified", "assum", "not checked", "unconfirmed", "not tested"] },
  ]
  const barrier =
    BARRIERS.find((b) => b.words.some((w) => t.includes(w)))?.id ?? "not-verified"

  const HIGH_ENERGY = [
    "h2s", "lel", "kg/cm2", "kv", "pressuris", "flammable", "gas", "height",
    "suspended", "confined", "live", "charged", "excavat", "crane", "tanker",
  ]
  const energyHits = HIGH_ENERGY.filter((w) => t.includes(w)).length
  const barrierBroken = ["bypassed", "absent", "failed"].includes(barrier)

  const raw = 0.16 + energyHits * 0.13 + (barrierBroken ? 0.28 : 0.05) + (scored[0]?.hits ?? 0) * 0.04
  const score = Math.max(0.03, Math.min(0.97, raw))
  const sif = score >= 0.5

  const evidence = []
  const collect = (words, weight) => {
    for (const w of words) {
      const i = t.indexOf(w)
      if (i === -1) continue
      // widen to whole words in the ORIGINAL casing so highlighting matches
      let s = i
      while (s > 0 && /[^\s.,;]/.test(text[s - 1])) s--
      let e = i + w.length
      while (e < text.length && /[^\s.,;]/.test(text[e])) e++
      const span = text.slice(s, e)
      if (span.length > 2 && !evidence.some((x) => x.span === span)) {
        evidence.push({ span, weight })
      }
      if (evidence.length >= 5) return
    }
  }
  collect(BARRIERS.find((b) => b.id === barrier)?.words ?? [], 0.86)
  collect(HIGH_ENERGY, 0.72)
  collect(scored[0]?.words ?? [], 0.6)

  const sevA = /injur|hospitalis|stitch|fractur|burn|crush/.test(t) ? 3 : 1
  const sevP = sif ? (energyHits >= 3 ? 5 : 4) : energyHits >= 1 ? 3 : 2

  /* Multi-label: every rule the text supports, not just the strongest. Third and
     beyond are kept when the evidence is there, exactly as the OvR head would. */
  const lsrTags = scored
    .slice(0, 3)
    .map((r, i) => ({
      id: r.id,
      confidence: round2(Math.min(i === 0 ? 0.95 : 0.72, (i === 0 ? 0.52 : 0.3) + r.hits * (i === 0 ? 0.11 : 0.09))),
    }))
    .filter((tag) => tag.confidence >= LSR_TAG_THRESHOLD)
    .sort((a, b) => b.confidence - a.confidence)

  return {
    sifPotential: sif,
    sifConfidence: round2(score),
    lsr: lsrTags[0] ?? { id: "unmapped", confidence: 0.28 },
    lsrSecondary: lsrTags[1] ?? null,
    lsrTags,
    precursors: {
      hazardEnergy: scored[0]?.energy ?? "mechanical",
      barrierFailure: barrier,
    },
    // deterministic for a given paste — no rand() here, unlike the fixtures
    severityScore: round2(clamp(sevP * 1.7 + sevA * 0.35 + (score - 0.5) * 0.6, 0.2, 10)),
    severityActual: sevA,
    severityPotential: sevP,
    evidence,
    features: buildFeatures({ text, barrier, sevA, sevP }),
    normalized: expansionsIn(text),
  }
}
