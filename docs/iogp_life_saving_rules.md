# IOGP Life-Saving Rules — Labeling Taxonomy

Source: IOGP Report 459 (2018 revision), https://www.iogp.org/workstreams/safety/safety/life-savingrules/

No public labeled dataset exists for mapping free text to these 9 rules — this is a genuine gap.
Approach: weak-label with keyword/phrase triggers below, use as bootstrap labels for a multi-label
classifier (one sigmoid output per rule; a single report can trigger multiple rules).

| Rule | Hazard focus | Example narrative triggers |
|---|---|---|
| 1. Bypassing Safety Controls | Interlock overrides, safety device tampering | "bypassed", "interlock", "bridged", "safety valve isolated", "override" |
| 2. Confined Space | Toxic atmospheres, oxygen deficiency | "vessel entry", "tank", "pit", "gas test", "attendant missing", "O2 levels", "LEL" |
| 3. Driving | Vehicle crash, rollover, speed | "speeding", "overturning", "seatbelt", "mobile phone while driving" |
| 4. Energy Isolation | Stored hydraulic/electrical/pneumatic/chemical energy | "LOTO", "lockout tagout", "residual pressure", "zero energy state", "live circuit" |
| 5. Hot Work | Ignition sources near hydrocarbons | "welding", "grinding", "sparks", "gas check", "hot work permit", "fire watch" |
| 6. Line of Fire | Moving machinery, dropped objects | "dropped object", "struck by", "swing zone", "high pressure line", "unsecured load" |
| 7. Safe Mechanical Lifting | Suspended loads, crane operations | "crane", "suspended load", "rigging", "slings", "tag line", "lift plan" |
| 8. Work Authorisation | Uncontrolled high-risk task execution | "no PTW", "permit expired", "unauthorized work", "JSA missing", "scope change" |
| 9. Working at Height | Gravitational potential energy (>1.8m) | "scaffolding", "harness not tied off", "ladder", "handrail missing", "fall arrest" |

Verify trigger list against real OIL report language once sample data is available — these are
starting points from IOGP's official rule descriptions, not tuned to OIL's actual vocabulary yet.
