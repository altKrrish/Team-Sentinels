"""
Domain Templates for OIL HSSE Synthetic Data Generation
=========================================================
Contains realistic oil & gas upstream safety report narratives,
IOGP Life-Saving Rules definitions, location hierarchies, equipment
taxonomies, and barrier failure modes used to generate the dataset.

Author: SIH-2026 Team
"""

# ──────────────────────────────────────────────────────────────
# IOGP LIFE-SAVING RULES (9 Rules)
# ──────────────────────────────────────────────────────────────
LIFE_SAVING_RULES = {
    "Energy Isolation": {
        "keywords": [
            "loto", "lockout", "tagout", "lock out", "tag out",
            "isolation", "de-energize", "de-energise", "energized",
            "live equipment", "stored energy", "residual energy",
            "isolation point", "isolation certificate", "electrical isolation",
            "zero energy", "energy source", "try procedure",
            "isolation valve", "blind flange", "blinding",
        ],
        "description": "Verify isolation and zero energy before work begins",
    },
    "Confined Space": {
        "keywords": [
            "confined space", "vessel entry", "tank entry", "manhole",
            "enclosed space", "oxygen deficient", "oxygen enriched",
            "toxic atmosphere", "gas test", "entry permit", "standby person",
            "rescue plan", "ventilation", "purging", "atmospheric monitoring",
            "entry watch", "confined space permit", "breathing apparatus",
            "scba", "forced ventilation", "inert atmosphere",
        ],
        "description": "Obtain authorization before entering a confined space",
    },
    "Driving": {
        "keywords": [
            "driving", "vehicle", "speeding", "seatbelt", "seat belt",
            "journey management", "fatigue driving", "mobile phone driving",
            "road safety", "defensive driving", "vehicle inspection",
            "reversing", "collision", "rollover", "transport",
            "convoy", "overtaking", "speed limit", "distracted driving",
            "driver fitness", "heavy vehicle", "tanker",
        ],
        "description": "Follow safe driving rules",
    },
    "Hot Work": {
        "keywords": [
            "hot work", "welding", "cutting", "grinding", "brazing",
            "flame", "spark", "ignition source", "flammable atmosphere",
            "fire watch", "hot work permit", "gas free", "lel",
            "lower explosive limit", "flash point", "combustible",
            "oxy-acetylene", "arc welding", "fire blanket", "fire extinguisher",
            "pyrophoric", "auto-ignition",
        ],
        "description": "Control flammables and ignition sources",
    },
    "Line of Fire": {
        "keywords": [
            "line of fire", "struck by", "caught between", "pinch point",
            "crush", "falling object", "dropped object", "suspended load",
            "pressurized", "pressure release", "stored energy release",
            "ricochet", "projectile", "ejection", "whip",
            "swing radius", "exclusion zone", "barricade",
            "impact", "trajectory", "recoil",
        ],
        "description": "Keep yourself and others out of the line of fire",
    },
    "Safe Mechanical Lifting": {
        "keywords": [
            "crane", "lifting", "rigging", "sling", "shackle",
            "load chart", "swl", "safe working load", "lift plan",
            "banksman", "signal person", "outrigger", "boom",
            "winch", "hoist", "overhead crane", "forklift",
            "man-basket", "chain block", "come-along",
            "load test", "certification", "inspection color code",
        ],
        "description": "Plan lifting operations and control the area",
    },
    "Work Authorization": {
        "keywords": [
            "permit to work", "ptw", "work permit", "authorization",
            "risk assessment", "jsa", "job safety analysis", "toolbox talk",
            "method statement", "simultaneous operations", "simops",
            "permit conditions", "permit validity", "permit holder",
            "area authority", "performing authority", "cold work permit",
            "general work permit", "critical task", "task risk assessment",
        ],
        "description": "Work with a valid permit when required",
    },
    "Working at Height": {
        "keywords": [
            "height", "scaffold", "scaffolding", "ladder", "harness",
            "fall protection", "fall arrest", "guardrail", "toe board",
            "safety net", "anchorage point", "lanyard", "lifeline",
            "elevated work", "roof work", "platform", "aerial work",
            "cherry picker", "mewp", "mobile elevating work platform",
            "edge protection", "hole cover", "open hole",
        ],
        "description": "Protect yourself against a fall when working at height",
    },
    "Bypassing Safety Controls": {
        "keywords": [
            "bypass", "override", "interlock", "safety device",
            "alarm disabled", "safety valve", "pressure relief",
            "inhibit", "defeat", "safety system", "sis",
            "emergency shutdown", "esd", "fire and gas",
            "safety critical element", "sce", "maintenance override",
            "process safety", "safety instrumented system",
            "trip", "protective device",
        ],
        "description": "Do not bypass or disable safety controls",
    },
}

# ──────────────────────────────────────────────────────────────
# OIL INDIA LIMITED — LOCATION HIERARCHY
# ──────────────────────────────────────────────────────────────
OIL_LOCATIONS = {
    "Duliajan": {
        "state": "Assam",
        "type": "Oilfield",
        "areas": [
            "Drilling Site DS-14", "Drilling Site DS-22", "Well Pad WP-07",
            "Well Pad WP-12", "Central Tank Farm", "Gas Collecting Station GCS-3",
            "EPS Naharkatiya", "Group Gathering Station GGS-II",
            "Crude Oil Terminal", "Pipeline RoW Sector-4",
            "Workover Rig WR-05", "Water Injection Plant WIP-2",
        ],
    },
    "Moran": {
        "state": "Assam",
        "type": "Oilfield",
        "areas": [
            "Drilling Site DS-06", "Well Pad WP-03", "Well Pad WP-09",
            "EPS Moran", "Gas Compressor Station", "Crude Dispatch Terminal",
            "Pipeline RoW Sector-2", "Workover Rig WR-08",
            "Water Treatment Plant", "Tank Battery TB-4",
        ],
    },
    "Digboi": {
        "state": "Assam",
        "type": "Refinery & Oilfield",
        "areas": [
            "Digboi Refinery CDU", "Refinery Tank Farm", "Effluent Treatment Plant",
            "Drilling Site DS-31", "Well Pad WP-15", "Old Well Area",
            "Refinery Boiler House", "Product Dispatch Area",
            "LPG Bottling Plant", "Pipeline Terminal",
        ],
    },
    "Kumchai": {
        "state": "Arunachal Pradesh",
        "type": "Oilfield",
        "areas": [
            "Drilling Site DS-41", "Drilling Site DS-42",
            "Well Pad WP-20", "EPS Kumchai", "Camp Base",
            "Access Road Construction", "Pipeline RoW Hilly Sector",
            "Helipad Area", "Material Storage Yard",
        ],
    },
    "Tengakhat": {
        "state": "Assam",
        "type": "Oilfield",
        "areas": [
            "Drilling Site DS-18", "Well Pad WP-11",
            "Group Gathering Station GGS-IV", "Pipeline RoW Sector-6",
            "Workover Rig WR-12", "Chemical Storage Area",
            "Produced Water Treatment Facility",
        ],
    },
    "Jorhat": {
        "state": "Assam",
        "type": "Pipeline Installation",
        "areas": [
            "Pump Station PS-2", "Pipeline Section KM-120 to KM-145",
            "Valve Station VS-3", "Cathodic Protection Station",
            "Pig Launcher/Receiver Station", "Crude Oil Terminal Jorhat",
        ],
    },
    "Kakinada": {
        "state": "Andhra Pradesh",
        "type": "Offshore & Onshore Terminal",
        "areas": [
            "Onshore Terminal", "Jetty Area", "Gas Processing Plant",
            "Control Room", "Pipe Yard", "Logistics Base",
            "Marine Operations", "Helideck",
        ],
    },
    "Rajasthan Block": {
        "state": "Rajasthan",
        "type": "Exploration Block",
        "areas": [
            "Exploration Rig ER-01", "Seismic Survey Camp",
            "Base Camp RJ", "Mud Logging Unit", "Shot Point Area",
            "Temporary Access Road", "Water Well Drilling Site",
        ],
    },
}

# ──────────────────────────────────────────────────────────────
# REPORT TYPES & OBSERVATION CATEGORIES
# ──────────────────────────────────────────────────────────────
REPORT_TYPES = [
    "Unsafe Act",
    "Unsafe Condition",
    "Near Miss",
    "Incident",
    "Hazard Observation",
    "Positive Observation",
]

OBSERVATION_CATEGORIES = [
    "Housekeeping", "PPE Non-Compliance", "Procedural Violation",
    "Equipment Defect", "Environmental Hazard", "Electrical Hazard",
    "Fire Hazard", "Chemical Exposure", "Ergonomic Issue",
    "Excavation Hazard", "Process Safety Event", "Transportation Hazard",
    "Crane & Lifting", "Scaffolding Deficiency", "Confined Space Issue",
    "Working at Height Issue", "Slip/Trip/Fall Hazard",
    "Pressure System Issue", "Radiation Hazard", "Dropped Object Potential",
    "H2S Exposure Risk", "Permit Violation", "Fatigue / Fitness",
    "Positive Safe Behavior",
]

# ──────────────────────────────────────────────────────────────
# ACTIVITY TYPES (Oil & Gas Upstream)
# ──────────────────────────────────────────────────────────────
ACTIVITY_TYPES = [
    "Drilling Operations", "Well Completion", "Workover Operations",
    "Production Operations", "Pipeline Laying", "Pipeline Maintenance",
    "Hot Tapping", "Pigging Operations", "Tank Cleaning",
    "Vessel Inspection", "Turnaround / Shutdown", "Well Testing",
    "Rig Move", "Cementing", "Mud Circulation",
    "Casing Running", "Perforation", "Stimulation / Fracturing",
    "Artificial Lift Installation", "ESP Installation",
    "Electrical Maintenance", "Instrument Maintenance",
    "Mechanical Maintenance", "Civil Construction",
    "Scaffolding Erection", "Scaffolding Dismantling",
    "Crane Operations", "Heavy Lift", "Material Handling",
    "Welding & Fabrication", "Painting & Coating",
    "Insulation Work", "Excavation / Trenching",
    "Road Construction", "Vehicle Operations", "Logistics & Transport",
    "Chemical Injection", "Water Injection", "Gas Compression",
    "Flare Operations", "Tank Gauging", "Loading / Unloading",
    "Sampling Operations", "Lab Testing", "HSE Audit / Inspection",
    "Emergency Drill", "Routine Patrol",
]

# ──────────────────────────────────────────────────────────────
# EQUIPMENT TAXONOMY
# ──────────────────────────────────────────────────────────────
EQUIPMENT_TYPES = [
    "Drilling Rig", "Workover Rig", "Mud Pump", "Drawworks",
    "Top Drive", "Rotary Table", "BOP (Blowout Preventer)",
    "Choke Manifold", "Shale Shaker", "Centrifuge",
    "Christmas Tree", "Wellhead Assembly", "Flowline",
    "Pipeline (Buried)", "Pipeline (Above Ground)", "Pig Launcher",
    "Pig Receiver", "Storage Tank", "Separator (2-Phase)",
    "Separator (3-Phase)", "Heater Treater", "Gas Scrubber",
    "Compressor", "Pump (Centrifugal)", "Pump (Reciprocating)",
    "Heat Exchanger", "Pressure Vessel", "Flare Stack",
    "Boiler", "Generator", "Transformer", "Switchgear",
    "Motor Control Center", "VFD (Variable Frequency Drive)",
    "Crane (Mobile)", "Crane (Pedestal)", "Forklift",
    "Cherry Picker / MEWP", "Scaffold", "Ladder (Fixed)",
    "Ladder (Portable)", "Safety Shower / Eyewash",
    "Fire Extinguisher", "Fire Water Pump", "Gas Detector (Fixed)",
    "Gas Detector (Portable)", "SCBA Set", "Winch",
    "Chain Block", "Wire Rope Sling", "Synthetic Sling",
    "Shackle", "Spreader Bar", "Man-Basket",
    "Tanker Truck", "Crew Bus", "Pickup Truck",
    "Ambulance", "Light Vehicle", "Heavy Equipment (Excavator)",
    "Bulldozer", "Backhoe Loader",
]

# ──────────────────────────────────────────────────────────────
# BARRIER TYPES & FAILURE MODES
# ──────────────────────────────────────────────────────────────
BARRIER_TYPES = [
    "Administrative Control", "Engineering Control", "PPE",
    "Permit to Work System", "Isolation Procedure", "Gas Detection System",
    "Fire Detection System", "Emergency Shutdown System (ESD)",
    "Pressure Relief Device", "Interlock / Safety Instrumented System",
    "Guardrail / Edge Protection", "Fall Arrest System",
    "Ventilation System", "Barricade / Exclusion Zone",
    "Signage & Warning", "Training & Competency",
    "Supervision", "Buddy System / Standby Person",
    "Vehicle Safety Device (Seatbelt, GPS, Speed Limiter)",
    "Lifting Equipment Certification",
]

BARRIER_STATUSES = [
    "Intact – Functioning as designed",
    "Degraded – Partially effective",
    "Failed – Not functioning",
    "Absent – Not installed / Not in place",
    "Bypassed – Intentionally defeated",
    "Not Applicable",
]

# ──────────────────────────────────────────────────────────────
# REPORTER DESIGNATIONS
# ──────────────────────────────────────────────────────────────
REPORTER_DESIGNATIONS = [
    "Drilling Engineer", "Production Engineer", "HSE Officer",
    "HSE Manager", "Field Operator", "Control Room Operator",
    "Maintenance Technician", "Electrical Technician",
    "Instrument Technician", "Crane Operator", "Rigger",
    "Scaffolder", "Welder", "Fitter", "Helper / Roustabout",
    "Supervisor (Operations)", "Supervisor (Maintenance)",
    "Supervisor (Construction)", "Contractor Safety Officer",
    "Contractor Foreman", "Security Guard", "Driver",
    "Geologist", "Mud Engineer", "Cementing Engineer",
    "Pipeline Engineer", "Project Engineer", "Area Manager",
    "Rig Manager / Tool Pusher", "Company Man / Drilling Superintendent",
]

# ──────────────────────────────────────────────────────────────
# SHIFT & WEATHER
# ──────────────────────────────────────────────────────────────
SHIFTS = ["Day Shift (06:00–18:00)", "Night Shift (18:00–06:00)"]

WEATHER_CONDITIONS = [
    "Clear", "Partly Cloudy", "Overcast", "Light Rain",
    "Heavy Rain", "Thunderstorm", "Fog / Low Visibility",
    "Hot & Humid (>40°C)", "Cold (<10°C)", "Windy (>30 km/h)",
    "Dust Storm", "Normal",
]

# ──────────────────────────────────────────────────────────────
# IMMEDIATE ACTIONS
# ──────────────────────────────────────────────────────────────
IMMEDIATE_ACTIONS = [
    "Work stopped immediately and area secured",
    "Personnel evacuated from hazard zone",
    "Verbal warning issued to involved personnel",
    "Defective equipment tagged out and removed from service",
    "Temporary barricade erected around hazard",
    "First aid administered on site",
    "Emergency response team activated",
    "Gas test conducted and area declared safe",
    "Spill containment measures deployed",
    "Fire watch posted at the location",
    "PPE provided and compliance enforced",
    "Toolbox talk conducted on the spot",
    "Supervisor notified and corrective guidance given",
    "Permit to work suspended pending review",
    "Isolation re-verified and secured",
    "Load set down and lifting operation suspended",
    "Vehicle taken out of service for inspection",
    "Scaffold tagged as unsafe — access blocked",
    "Temporary ventilation arranged for confined space",
    "No immediate action — observation logged for trending",
]

# ──────────────────────────────────────────────────────────────
# CORRECTIVE ACTIONS
# ──────────────────────────────────────────────────────────────
CORRECTIVE_ACTIONS = [
    "Conduct refresher training on relevant Life-Saving Rule for all site personnel",
    "Revise Job Safety Analysis (JSA) to include identified hazard",
    "Install permanent engineering control (guardrail / barricade / ventilation)",
    "Replace defective equipment and update inspection register",
    "Implement additional supervisory checks for high-risk activities",
    "Update permit-to-work procedure to address identified gap",
    "Install additional signage and hazard warnings at location",
    "Conduct root cause analysis (RCA) and share learnings across sites",
    "Schedule equipment for certified third-party inspection",
    "Review and update emergency response plan for the area",
    "Enforce strict PPE compliance through daily audits",
    "Add the scenario to HSSE induction training material",
    "Install / repair gas detection system in the affected area",
    "Conduct behavioral safety observation campaign",
    "Implement engineering modification to eliminate hazard at source",
    "Review contractor safety management plan and performance",
    "Add interlock / safety instrumented function for the process",
    "Increase frequency of planned maintenance for critical equipment",
    "Conduct management safety walk-through at the location",
    "Share safety alert / lessons learned bulletin across the organization",
]


# ──────────────────────────────────────────────────────────────
# SIF-POTENTIAL NARRATIVE TEMPLATES
# ──────────────────────────────────────────────────────────────
# Each template is a tuple:
#   (narrative_template, life_saving_rules, severity_rationale)
# Placeholders: {location}, {area}, {equipment}, {activity}, {personnel}

SIF_TEMPLATES = [
    # ── Energy Isolation ──
    (
        "During {activity} at {area}, {location}, a maintenance technician was found working on {equipment} without verifying energy isolation. The LOTO locks were not applied and the isolation certificate had expired two days prior. The equipment had residual hydraulic pressure of approximately 150 psi. When the technician loosened a flange bolt, a small amount of hydraulic fluid sprayed out, narrowly missing his face. Work was stopped immediately.",
        ["Energy Isolation", "Work Authorization"],
        "Potential for high-pressure fluid injection injury or fatal hydraulic release",
    ),
    (
        "Electrician observed working inside the MCC panel at {area}, {location} without applying LOTO. The panel was found energized at 415V. The electrician stated he was 'just checking' and did not think isolation was necessary for a visual inspection. The PTW did not cover electrical work. Supervisor was not present at the worksite.",
        ["Energy Isolation", "Work Authorization", "Bypassing Safety Controls"],
        "Electrocution hazard — direct contact with 415V energized panel",
    ),
    (
        "During well workover operations at {area}, {location}, the crew began disconnecting the flowline from the wellhead without confirming that the wing valve was fully closed and the pressure had been bled down. A residual pressure of 85 psi was later measured. The {equipment} was not isolated per the approved procedure. The area authority had not signed off on the isolation certificate.",
        ["Energy Isolation", "Work Authorization"],
        "Uncontrolled hydrocarbon release potential from pressurized wellhead",
    ),
    (
        "Night shift crew at {area}, {location} found that the isolation valves on the {equipment} were not locked in the closed position despite the ongoing maintenance activity. Tags were present but no physical locks. A passing operator nearly opened the valve to restore production, not realizing that personnel were working downstream. The near miss was caught by the control room operator monitoring the SCADA system.",
        ["Energy Isolation", "Bypassing Safety Controls"],
        "Potential catastrophic release of process fluid onto maintenance crew",
    ),

    # ── Confined Space ──
    (
        "During tank cleaning operations at {area}, {location}, a contract worker entered the crude oil storage tank without a valid confined space entry permit. No atmospheric testing was conducted prior to entry. The standby person had left the manhole to fetch tools. Upon rescue, the worker was found unconscious at the bottom of the tank due to oxygen-deficient atmosphere (O2 measured at 16.5%). He was revived after emergency oxygen administration.",
        ["Confined Space", "Work Authorization"],
        "Oxygen deficiency causing loss of consciousness — potential fatality",
    ),
    (
        "Vessel entry for internal inspection of {equipment} at {area}, {location} was being conducted. The gas test showed 0% LEL and 20.8% O2 initially, but continuous monitoring was not maintained. After 45 minutes, the portable gas detector alarmed for H2S at 12 ppm inside the vessel. The two technicians inside were evacuated immediately. Investigation revealed that a connected line had not been blinded, allowing sour gas ingress from an adjacent process unit.",
        ["Confined Space", "Energy Isolation", "Bypassing Safety Controls"],
        "H2S exposure in confined space — immediate danger to life",
    ),
    (
        "Contract workers were cleaning the inside of a separator vessel at {area}, {location}. The entry permit had expired 2 hours prior and was not renewed. The forced ventilation fan was found switched off. Three workers were inside without SCBA or escape sets. The manhole was partially obstructed by cleaning hoses, which would have impeded emergency egress. No rescue tripod was rigged at the entry point.",
        ["Confined Space", "Work Authorization"],
        "Multiple barrier failures in confined space — potential multi-fatality",
    ),

    # ── Working at Height ──
    (
        "A painter was observed working at approximately 12 meters height on the flare stack structure at {area}, {location}, without wearing a fall arrest harness. The scaffold platform he was standing on was missing the mid-rail and toe board on one side. Upon being stopped, the worker stated his harness was 'uncomfortable' and that he had removed it. The scaffold had a green tag but the last inspection was 45 days ago, exceeding the 7-day re-inspection requirement.",
        ["Working at Height", "Bypassing Safety Controls"],
        "Fall from 12m height — high probability of fatality",
    ),
    (
        "During scaffolding erection at {area}, {location}, a scaffolder was observed climbing the scaffold structure without clipping his lanyard to any anchorage point. He was at approximately 8 meters height. The scaffold was incomplete (still under construction) and no guardrails were in place. Another worker below was not wearing a hard hat and was within the drop zone without any overhead protection. The lift plan for scaffold materials had not been prepared.",
        ["Working at Height", "Line of Fire", "Safe Mechanical Lifting"],
        "Unprotected fall from height and dropped object hazard",
    ),
    (
        "A technician used a damaged portable ladder to access the top of a horizontal pressure vessel at {area}, {location}. The ladder had a cracked rung (3rd from top) and was not secured at the top. The technician was carrying tools in both hands while climbing, leaving no points of contact on the ladder. The vessel top was approximately 5 meters above grade. No fall protection was used. A proper access platform exists 20 meters away but the technician chose the ladder to 'save time.'",
        ["Working at Height"],
        "Fall from height due to damaged equipment and unsafe climbing practice",
    ),

    # ── Hot Work ──
    (
        "Welding was being performed on a crude oil flowline at {area}, {location} for a leak repair. The hot work permit indicated gas-free conditions, but the last gas test was conducted 3 hours before welding started. No fire watch was posted. Oily rags and paint thinners were found within 5 meters of the welding point. During welding, sparks landed on an oily patch and a small fire ignited. The fire was controlled with a portable extinguisher but the welder sustained minor burns to his forearms.",
        ["Hot Work", "Work Authorization"],
        "Flash fire from welding near flammable materials — burn injuries",
    ),
    (
        "Cutting operations were observed at {area}, {location} using oxy-acetylene equipment on a decommissioned pipeline. The pipeline had not been gas-freed or purged — it was assumed to be empty because it was 'decommissioned 6 months ago.' No gas test was performed. The cutting crew had no fire blanket, fire watch, or fire extinguisher at the worksite. The hot work permit was signed but the isolation section was left blank.",
        ["Hot Work", "Energy Isolation", "Work Authorization"],
        "Explosion risk from cutting on non-gas-freed pipeline",
    ),
    (
        "Grinding operation at {area}, {location} near the {equipment} generated sparks that traveled approximately 8 meters and landed near an open drain containing oily water. The LEL reading at the drain was 15% (above the 10% action level). No spark containment measures (fire blanket, welding curtain) were in place. The hot work permit did not identify the open drain as a nearby hazard. The fire watch had been reassigned to another job.",
        ["Hot Work", "Work Authorization"],
        "Potential flash fire / explosion from sparks reaching flammable atmosphere",
    ),

    # ── Line of Fire ──
    (
        "During pressure testing of {equipment} at {area}, {location}, two workers were standing directly in front of a blanked flange at 250 psi test pressure. The exclusion zone had not been established. A blind flange gasket failed during the test, ejecting the gasket and releasing high-pressure water, striking one worker on the chest. He was knocked backwards and sustained bruising. Had this been a hydrocarbon test, the consequences would have been severe.",
        ["Line of Fire", "Work Authorization"],
        "High-pressure ejection — potential fatal projectile injury",
    ),
    (
        "A rigger was standing directly under a 2-ton load being lifted by the mobile crane at {area}, {location}. The crane operator could not see the rigger from the cab. The banksman was on the phone and not providing signals. The load swung due to wind gust and nearly struck the rigger. Other personnel in the area were not wearing hard hats. The lift plan specified a 10-meter exclusion zone which was not enforced.",
        ["Line of Fire", "Safe Mechanical Lifting"],
        "Struck-by / crush hazard from swinging 2-ton suspended load",
    ),
    (
        "During drilling operations at {area}, {location}, a roughneck was positioning pipe on the pipe rack when the pipe slipped from the pipe handler and rolled off the rack. The pipe (5-inch, ~400 kg) rolled towards two workers who were resting within 3 meters of the rack — an area that should have been barricaded. One worker jumped clear; the other was struck on the lower leg and sustained a fracture.",
        ["Line of Fire"],
        "Struck-by rolling pipe — fracture injury, potential fatality",
    ),

    # ── Safe Mechanical Lifting ──
    (
        "Mobile crane lifting a 5-ton BOP stack at {area}, {location} was observed operating at 95% of its rated capacity without a critical lift plan. The outriggers were not fully extended on the right side due to uneven ground. No load chart verification was done for the radius. The slings used were 3-ton SWL for a 5-ton load. Mid-lift, the crane's overload alarm activated. The operator continued the lift against procedure. The load was eventually set down safely after the HSE officer intervened.",
        ["Safe Mechanical Lifting", "Bypassing Safety Controls"],
        "Crane overload — potential catastrophic crane failure and dropped load",
    ),
    (
        "A chain block was being used to lift a pump motor (~800 kg) at {area}, {location}. The chain block was attached to a structural beam that had not been assessed for the load. The beam was visibly corroded. No lift plan was in place. The rigging crew used a single-leg sling instead of the required two-leg sling, and the sling did not have a current color-code inspection tag. The rigger was standing under the load while guiding it into position.",
        ["Safe Mechanical Lifting", "Line of Fire"],
        "Structural failure risk and dropped load onto personnel below",
    ),

    # ── Driving ──
    (
        "A company pickup truck was observed traveling at 95 km/h in a 40 km/h zone within the {area}, {location} premises. The vehicle's GPS tracker confirmed sustained speeding for 3.2 km. The driver was not wearing a seatbelt. A fatigue monitoring camera inside the cabin was covered with tape. The driver had completed 14 hours of driving that day against the 10-hour maximum policy. Journey management plan was not filed for this trip.",
        ["Driving", "Bypassing Safety Controls"],
        "High-speed vehicle incident potential — driver fatigue and policy violations",
    ),
    (
        "A loaded crude oil tanker was reversing without a banksman at the {area} loading bay, {location}. The rear-view camera was non-functional. Two pedestrian workers were in the blind spot behind the tanker. The driver did not sound the horn. A near collision occurred when the tanker reversed to within 30 cm of one worker before another worker shouted an alert. The tanker's reverse alarm was also found to be non-functional.",
        ["Driving", "Line of Fire"],
        "Near-fatal reversing incident — pedestrian in vehicle blind spot",
    ),

    # ── Work Authorization ──
    (
        "Excavation work was proceeding at {area}, {location} without a valid excavation permit. The trench was 1.8 meters deep and 1.2 meters wide with no shoring or trench box. Underground utility drawings had not been reviewed, and a buried 6-inch gas pipeline was located only 0.5 meters from the trench wall. Two workers were inside the unprotected trench. A cable strike had occurred 3 months prior at the same location during similar unauthorized excavation.",
        ["Work Authorization", "Line of Fire"],
        "Trench collapse and gas pipeline strike potential — multiple fatality scenario",
    ),
    (
        "Simultaneous operations (SIMOPS) were underway at {area}, {location} — hot work welding on a process line while a confined space entry was happening on an adjacent vessel. Neither permit referenced the other activity. The risk assessment did not consider the interaction between the two concurrent tasks. Welding sparks were observed falling near the confined space entry manhole. The PTW coordinator was unavailable at site.",
        ["Work Authorization", "Hot Work", "Confined Space"],
        "Unmanaged SIMOPS — welding sparks entering occupied confined space",
    ),

    # ── Bypassing Safety Controls ──
    (
        "The high-pressure shutdown (HPSD) on {equipment} at {area}, {location} was found in a bypassed state using a jumper wire. The bypass had been in place for approximately 3 weeks. No bypass log entry existed, and no management of change (MOC) had been raised. The process safety valve downstream was also found to be isolated for maintenance. This meant there were no active overpressure protection barriers on the system. Operating pressure was within 10% of the design pressure.",
        ["Bypassing Safety Controls", "Energy Isolation"],
        "Complete loss of overpressure protection — catastrophic vessel rupture scenario",
    ),
    (
        "A fire and gas detector in the {equipment} area at {area}, {location} was found inhibited in the control room. The inhibit had been active for 8 days with no documented justification. The area contained a running gas compressor. When checked, a small gas leak was found at a compressor flange — estimated at 0.5 kg/hr of natural gas. Without the active gas detector, the leak would not have triggered an alarm or shutdown. The leak was repaired immediately.",
        ["Bypassing Safety Controls"],
        "Undetected gas accumulation risk — potential explosion",
    ),

    # ── H2S / Process Safety specific scenarios ──
    (
        "During well testing operations at {area}, {location}, an unexpected H2S concentration of 85 ppm was detected downwind of the well. The well was known to produce sweet crude (non-sour). No H2S contingency plan was in place. Wind socks were not installed at the site. Personnel did not have escape-grade SCBA sets. The well test crew of 8 persons evacuated 200 meters upwind. Two workers reported headaches and dizziness. The medical emergency response vehicle was 45 minutes away.",
        ["Work Authorization", "Bypassing Safety Controls"],
        "H2S exposure above IDLH — immediate fatality potential for 8 persons",
    ),
    (
        "A process safety incident occurred at {area}, {location} when the {equipment} experienced an overpressure event. The pressure safety valve (PSV) failed to lift at its set pressure. Investigation revealed the PSV had not been tested in 3 years (annual testing required). The control room high-pressure alarm had been acknowledged and silenced repeatedly by the operator without taking corrective action. Pressure reached 115% of MAWP before the operator manually activated the emergency blowdown.",
        ["Bypassing Safety Controls", "Energy Isolation"],
        "Pressure vessel integrity threat — potential rupture and fatality",
    ),

    # ── Multi-rule complex scenarios ──
    (
        "A contract crew was performing maintenance on a gas compressor at {area}, {location}. The following violations were observed simultaneously: (1) The compressor suction valve isolation was done but not locked — tags only. (2) A welder was grinding on the compressor skid without a hot work permit. (3) The area gas detector showed a reading of 8% LEL, which was ignored. (4) One worker was standing on an overturned bucket to reach the compressor top at ~2.5m height instead of using the provided platform. (5) The toolbox talk attendance sheet showed only 3 of 7 crew members had attended. When confronted, the contract supervisor stated they were 'behind schedule' and needed to 'get the job done.'",
        ["Energy Isolation", "Hot Work", "Working at Height", "Bypassing Safety Controls", "Work Authorization"],
        "Multiple simultaneous barrier failures — gas explosion and fall potential",
    ),
    (
        "Nightshift incident at {area}, {location}: A crane was lifting a 3.5-ton heat exchanger bundle during removal for maintenance. The lift was being conducted under poor lighting conditions (only vehicle headlights). The crane operator's certification had expired 2 months ago. The rigging plan called for 2x 5-ton slings but only 1x 3-ton sling was used. During the lift, the bundle tilted at 30 degrees. Two workers ran under the suspended load to attach a tag line. The load was set down successfully, but the sling was found to have 3 broken strands near the thimble. The crane's LMI (load moment indicator) was showing an error code and had not been reset.",
        ["Safe Mechanical Lifting", "Line of Fire", "Bypassing Safety Controls", "Working at Height"],
        "Catastrophic lift failure risk — suspended load over personnel",
    ),
]


# ──────────────────────────────────────────────────────────────
# NON-SIF-POTENTIAL NARRATIVE TEMPLATES
# ──────────────────────────────────────────────────────────────
NON_SIF_TEMPLATES = [
    # ── Minor PPE / Housekeeping ──
    (
        "Worker observed at {area}, {location} not wearing safety glasses while walking through the production area. When approached, the worker immediately put on his glasses from his pocket. He stated he had removed them briefly as they were fogging up. No hazardous activity was taking place in the immediate vicinity. The area is designated as a mandatory safety glasses zone.",
        [],
        "Low severity — PPE non-compliance in low-hazard walking area",
    ),
    (
        "General housekeeping observation at {area}, {location}: Several empty chemical drums (cleaned and purged) were stored in the walkway near the warehouse, partially blocking the pedestrian path. The drums were not stacked properly and one had rolled to the edge of the pathway. The materials coordinator was notified and drums were relocated within 30 minutes.",
        [],
        "Minor housekeeping — no immediate injury risk",
    ),
    (
        "During routine patrol at {area}, {location}, an oil stain approximately 1m x 1m was observed on the concrete floor near the pump house entrance. The oil appeared to be a slow drip from a leaking gasket on a low-pressure lube oil line. Absorbent pads were placed immediately. A maintenance work order was raised for gasket replacement.",
        [],
        "Minor spill — slip hazard, no toxic / flammable risk",
    ),
    (
        "A worker at {area}, {location} was observed carrying a coffee cup into the workshop area where eating and drinking are prohibited. The worker was reminded of the policy and disposed of the cup in the canteen. No chemicals were being used in the workshop at the time.",
        [],
        "Administrative policy violation — no injury potential",
    ),
    (
        "Toolbox talk observation at {area}, {location}: The morning toolbox talk was conducted but the duration was only 3 minutes instead of the recommended 10-15 minutes. The supervisor covered the day's task but did not discuss specific hazards or the relevant Life-Saving Rules. Workers did not ask any questions. This was noted as a quality-of-engagement issue rather than a safety incident.",
        ["Work Authorization"],
        "Low quality safety communication — no immediate hazard",
    ),
    (
        "Fire extinguisher at {area}, {location} (Unit FE-047) was found with an expired inspection tag. The last inspection date shown was 5 months ago (monthly inspection required). The extinguisher appeared to be in good physical condition with the pressure gauge in the green zone. The fire safety coordinator was notified for immediate re-inspection.",
        [],
        "Inspection overdue — equipment appears functional",
    ),
    (
        "A 'No Smoking' sign at the entrance to {area}, {location} was found damaged (faded and partially torn). The sign was still legible but in poor condition. A replacement sign was requested through the maintenance work order system.",
        [],
        "Degraded signage — low-severity administrative finding",
    ),
    (
        "The hand wash station at {area}, {location} was found without soap. The paper towel dispenser was also empty. Workers reported this has been an ongoing issue for the past week. The camp boss was notified for immediate restocking. This is a hygiene and welfare observation.",
        [],
        "Welfare / hygiene issue — no safety consequence",
    ),
    (
        "During vehicle inspection at {area}, {location}, a company pickup truck (vehicle number OIL-{veh_num}) was found with a slightly cracked windshield (small chip approximately 2 cm, passenger side, not in driver's line of sight). The vehicle was otherwise in good condition with all safety features functional. The chip was marked for monitoring and a windshield replacement was scheduled.",
        ["Driving"],
        "Minor vehicle defect — not impacting safe operation",
    ),
    (
        "A visitor at {area}, {location} was observed without a visitor ID badge displayed. The visitor had signed in at the gate and had a valid visitor pass but had put it in his pocket. Security escorted the visitor back to the gate to display the badge properly. No security breach occurred.",
        [],
        "Minor administrative non-compliance",
    ),
    (
        "Office ergonomic observation at {area}, {location}: A desk workstation in the engineering office was set up with the monitor at an incorrect height (too low), causing the engineer to lean forward consistently. An ergonomic assessment was recommended. No musculoskeletal complaint has been reported yet.",
        [],
        "Ergonomic issue — long-term health, no acute injury risk",
    ),
    (
        "A spill kit at {area}, {location} was found with depleted absorbent pads — only 2 pads remaining out of the standard 10. The spill kit inventory sheet showed it was last used 3 weeks ago but was not restocked. The environmental officer was notified to replenish the kit.",
        [],
        "Spill response readiness gap — no active spill",
    ),
    (
        "Noise monitoring at {area}, {location} near the {equipment} showed readings of 88 dBA (TWA). The area is designated as a mandatory hearing protection zone (>85 dBA). All workers in the area were wearing ear plugs. One worker was observed wearing ear plugs that were visibly dirty. He was provided with a fresh pair and reminded about proper ear plug hygiene.",
        [],
        "Minor PPE hygiene issue — hearing protection in use",
    ),
    (
        "A safety shower at {area}, {location} was tested during the weekly inspection. Water flow was adequate but the water was discolored (rusty) for the first 5 seconds before clearing. This indicates pipe corrosion. A work order was raised for pipe inspection and flushing. The shower is functional and would provide adequate emergency decontamination.",
        [],
        "Equipment maintenance finding — shower is functional",
    ),
    (
        "During HSE audit at {area}, {location}, it was noted that the emergency evacuation map posted in the control room was from 2022 and did not reflect the new emergency assembly point (changed in 2024). An updated map was printed and posted within 2 hours of the finding.",
        [],
        "Outdated emergency information — administrative gap",
    ),
    (
        "A worker at {area}, {location} was observed using a wrench to hammer a stuck bolt instead of using a proper hammer. The bolt was on a non-pressurized utility water pipe at ground level. The worker was advised to use the correct tool. The supervisor issued a reminder about proper tool usage during the next toolbox talk.",
        [],
        "Improper tool use — low-risk scenario",
    ),
    (
        "Positive observation: The crane crew at {area}, {location} was observed conducting an exemplary pre-lift briefing. They reviewed the lift plan, checked all rigging equipment certifications, established the exclusion zone with physical barricades, and had a dedicated banksman with clear communication. The HSE officer commended the crew and documented this as a positive behavior observation.",
        ["Safe Mechanical Lifting"],
        "Positive observation — safe work practice",
    ),
    (
        "A dripping tap was reported in the washroom at the accommodation camp, {area}, {location}. The drip rate was approximately 1 drop per 2 seconds. A plumber was called and the tap washer was replaced the same day. This is a water conservation and welfare observation.",
        [],
        "Welfare / utility maintenance — no safety impact",
    ),
    (
        "Observation at {area}, {location}: The first aid kit in the workshop was checked and found to be complete except for medium-sized disposable gloves (box empty). All other supplies including bandages, antiseptic, eye wash, and burn cream were present and within expiry dates. Gloves were restocked from the HSE stores the same day.",
        [],
        "Minor first aid readiness gap — kit substantially complete",
    ),
    (
        "Weather monitoring observation at {area}, {location}: Wind speed was recorded at 28 km/h (approaching the 30 km/h threshold for crane operations). The crane operator proactively suspended lifting operations and waited for conditions to improve. Operations resumed after wind speed dropped to 20 km/h. This is documented as a positive proactive decision.",
        ["Safe Mechanical Lifting"],
        "Positive observation — proactive weather-based decision",
    ),
]


# ──────────────────────────────────────────────────────────────
# PERSONNEL NAME POOL (Indian names — OIL context)
# ──────────────────────────────────────────────────────────────
PERSONNEL_FIRST_NAMES = [
    "Rajesh", "Amit", "Sunil", "Pranjal", "Bhaskar",
    "Deepak", "Manish", "Ranjan", "Vikram", "Anil",
    "Sanjay", "Prakash", "Rupam", "Mrinal", "Hemant",
    "Kamal", "Dilip", "Ganesh", "Manoj", "Rohit",
    "Ajay", "Bibhuti", "Chinmoy", "Debojit", "Gaurav",
    "Hiren", "Jayanta", "Kishor", "Lakshman", "Naren",
    "Partha", "Rituraj", "Sarat", "Tapan", "Utpal",
]

PERSONNEL_LAST_NAMES = [
    "Borah", "Gogoi", "Hazarika", "Barua", "Saikia",
    "Das", "Kalita", "Choudhury", "Nath", "Sharma",
    "Dutta", "Phukan", "Mahanta", "Bhuyan", "Rajkhowa",
    "Medhi", "Talukdar", "Kakati", "Tamuli", "Handique",
    "Konwar", "Bora", "Neog", "Sarma", "Goswami",
]
