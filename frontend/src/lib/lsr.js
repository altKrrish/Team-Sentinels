/**
 * The 9 IOGP Life-Saving Rules (2018 revision).
 * Report.lsr.id must be one of these ids.
 */
export const LSR = [
  {
    id: "energy-isolation",
    name: "Energy Isolation",
    short: "Energy Isolation",
    rule: "Verify isolation and zero energy before work begins.",
  },
  {
    id: "hot-work",
    name: "Hot Work",
    short: "Hot Work",
    rule: "Control flammables and ignition sources.",
  },
  {
    id: "confined-space",
    name: "Confined Space",
    short: "Confined Space",
    rule: "Obtain authorisation before entering a confined space.",
  },
  {
    id: "line-of-fire",
    name: "Line of Fire",
    short: "Line of Fire",
    rule: "Keep yourself and others out of the line of fire.",
  },
  {
    id: "working-at-height",
    name: "Working at Height",
    short: "At Height",
    rule: "Protect yourself against a fall when working at height.",
  },
  {
    id: "safe-mechanical-lifting",
    name: "Safe Mechanical Lifting",
    short: "Mech. Lifting",
    rule: "Plan lifting operations and control the area.",
  },
  {
    id: "bypassing-safety-controls",
    name: "Bypassing Safety Controls",
    short: "Bypassing Controls",
    rule: "Obtain authorisation before overriding or disabling safety controls.",
  },
  {
    id: "work-authorisation",
    name: "Work Authorisation",
    short: "Work Auth.",
    rule: "Work with a valid permit when required.",
  },
  {
    id: "driving",
    name: "Driving",
    short: "Driving",
    rule: "Follow safe driving rules.",
  },
]

export const LSR_BY_ID = Object.fromEntries(LSR.map((r) => [r.id, r]))

export const lsrName = (id) => LSR_BY_ID[id]?.name ?? "Unmapped"
export const lsrShort = (id) => LSR_BY_ID[id]?.short ?? "Unmapped"
