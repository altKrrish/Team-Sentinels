"""
The labelled training corpus.

PROVENANCE - READ THIS BEFORE QUOTING ANY METRIC
-----------------------------------------------
This corpus is **synthesised**, not scraped. OIL's HSSE reports are internal and
the public OSHA/OISD narratives are not redistributable here, so the training set
is generated from a compositional model of how safety reports are actually
written: a scenario frame supplies the activity and the hazard, a barrier clause
supplies the state of the control, and optional measurement / temporal / outcome
clauses fill in the rest. Roughly 40 frames x thousands of phrasing combinations.

What that does and does not buy you:

* It IS a real machine-learning problem. Every head sees only the *text*. The
  labels are a function of latent generative facts (hazard energy, barrier
  state, outcome) that reach the model exclusively through natural language,
  expressed with a different wording each time. Nothing leaks a label directly.
  Frames deliberately overlap in vocabulary - "confined space" appears both in a
  fatal-potential entry with no gas test and in a harmless register audit - so
  keyword spotting alone cannot solve it.
* Deliberate ~6% label noise on the SIF head mirrors the real disagreement rate
  between two HSE reviewers reading the same report.
* It is NOT evidence about performance on OIL's live reports. Held-out scores
  here measure whether the pipeline learns the pattern it was shown. Replace
  `synthetic_corpus()` with real labelled narratives and re-run `train.py` -
  nothing else in the codebase changes.

The SIF definition encoded below is the DEKRA / EEI precursor model as cited in
the problem statement: SIF potential requires a **high-energy hazard** AND a
**barrier that was absent, failed, bypassed, inadequate or unverified**. High
energy with an intact barrier is deliberately labelled NOT a precursor.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# barrier states
# ---------------------------------------------------------------------------

#: How much a barrier state adds to the latent risk. "intact" adds nothing -
#: that is the whole point of a barrier.
BARRIER_WEIGHT: dict[str, float] = {
    "intact": 0.0,
    "not-verified": 0.7,
    "inadequate": 0.9,
    "absent": 1.3,
    "failed": 1.3,
    "bypassed": 1.6,
}

#: Latent risk at or above this is a SIF precursor. Calibrated so that
#: potential 5 + intact barrier (5.0) is NOT a precursor while
#: potential 4 + absent barrier (5.3) is, and so that the resulting positive
#: share lands in the 20-25% band the literature reports.
SIF_THRESHOLD = 5.6

#: Spread of the reviewer-judgement term added to latent risk. Without it the
#: risk score is quantised (integer potential + one of six barrier weights), so
#: the decision boundary would be a small lookup table and the threshold would
#: not be tunable. With it, reports near the line are genuinely borderline.
JUDGEMENT_SD = 0.45

#: Probability a SIF label is flipped, standing in for outright reviewer
#: disagreement on top of the borderline cases above.
LABEL_NOISE = 0.05

#: Rule-tagging disagreement. Two HSE officers reading the same report almost
#: always agree on the PRIMARY rule and regularly disagree about secondary ones,
#: so the noise is applied only to secondary tags.
RULE_DROP_NOISE = 0.26
RULE_ADD_NOISE = 0.13

#: Rules that get confused for one another in practice, used to add a plausible
#: wrong tag rather than a random one.
_CONFUSABLE: dict[str, tuple[str, ...]] = {
    "energy-isolation": ("bypassing-safety-controls", "work-authorisation"),
    "hot-work": ("work-authorisation", "confined-space"),
    "confined-space": ("work-authorisation", "bypassing-safety-controls"),
    "line-of-fire": ("energy-isolation", "safe-mechanical-lifting"),
    "working-at-height": ("line-of-fire", "safe-mechanical-lifting"),
    "safe-mechanical-lifting": ("line-of-fire", "working-at-height"),
    "bypassing-safety-controls": ("energy-isolation", "work-authorisation"),
    "work-authorisation": ("energy-isolation", "hot-work"),
    "driving": ("work-authorisation",),
}

# ---------------------------------------------------------------------------
# clause pools
# ---------------------------------------------------------------------------

_OPENERS = (
    "During a routine walkthrough at {place}, {subject} was observed {gerund}.",
    "At {place}, {subject} was found {gerund}.",
    "{subject} was seen {gerund} at {place}.",
    "While {activity_low} was in progress at {place}, {subject} was noticed {gerund}.",
    "Observation raised at {place}: {subject} {past}.",
    "During {activity_low} at {place}, it was found that {subject} {past}.",
    "{place} - {subject} {past} during {activity_low}.",
    "A near miss was reported at {place} where {subject} {past}.",
)

#: Generic hazard phrasings shared by every frame carrying the same energy.
#: Roughly a third of narratives use one of these instead of the frame's own
#: wording, so the text does NOT uniquely identify which frame produced it.
#: Without this the corpus is trivially separable and every head scores 1.000 -
#: which measures the generator, not the model.
_GENERIC_HAZARD: dict[str, tuple[str, ...]] = {
    "gravity": (
        "working above grade with nothing between him and the drop",
        "moving about at height on an incomplete working platform",
        "positioned under a load that was being handled overhead",
    ),
    "pressure": (
        "opening up equipment that had not been proved depressurised",
        "working on a system that was still holding pressure",
        "standing in the discharge path of a pressurised connection",
    ),
    "electrical": (
        "working on equipment that had not been proved dead",
        "opening a panel that was still connected to its supply",
    ),
    "thermal": (
        "generating sparks close to a hydrocarbon source",
        "carrying out hot work in an area that had not been made safe",
    ),
    "mechanical": (
        "working on machinery that could start without warning",
        "reaching into equipment while it was still capable of movement",
    ),
    "chemical": (
        "entering an atmosphere that had not been proved fit to breathe",
        "working where a toxic or oxygen-deficient atmosphere was possible",
    ),
    "motion": (
        "driving on a field road in a way that left no room to stop",
        "moving a vehicle through an area shared with people on foot",
    ),
}

_BARRIER_CLAUSES: dict[str, tuple[str, ...]] = {
    "intact": (
        "The {control} was in place and had been verified by the area authority.",
        "A valid {control} was available at the worksite and was checked before work started.",
        "The {control} was confirmed in position and signed off by the shift in-charge.",
        "{control} was correctly applied and independently witnessed.",
        "The {control} was found in order, and the crew had stopped work to confirm it.",
        "Records showed the {control} had been tested the same shift and was effective.",
    ),
    "absent": (
        "No {control} was in place at the time.",
        "There was no {control} at the location.",
        "The {control} was not provided.",
        "Work had started without any {control}.",
        "The {control} was absent and nobody had raised it.",
        "No {control} could be produced when asked.",
        "The {control} was missing entirely.",
        "Not a single {control} was available at the worksite.",
    ),
    "failed": (
        "The {control} had failed and was passing.",
        "The {control} was defective and did not hold.",
        "The {control} gave way under load.",
        "The {control} was found damaged and non-functional.",
        "The {control} did not operate when it was called upon.",
        "The {control} had failed some time earlier and no replacement had been arranged.",
    ),
    "bypassed": (
        "The {control} had been bypassed to keep the job moving.",
        "Someone had jumpered out the {control}.",
        "The {control} was deliberately defeated by the crew.",
        "The {control} had been inhibited and never reset.",
        "The {control} was overridden without any approval.",
        "The {control} was tied back so it could not act.",
        "The crew had removed the {control} and worked around it.",
    ),
    "inadequate": (
        "The {control} was in place but was not adequate for the job.",
        "A {control} of the wrong rating had been used.",
        "The {control} was undersized for the duty.",
        "A makeshift {control} had been arranged instead of the specified one.",
        "The {control} covered only part of the exposure.",
        "The {control} was there but too degraded to be effective.",
    ),
    "not-verified": (
        "The {control} was assumed to be in place but was not verified.",
        "Nobody had confirmed the {control} before work began.",
        "The {control} was not cross-checked by a second person.",
        "The {control} was taken on trust; no try-out was carried out.",
        "There was no record that the {control} had been checked.",
        "The {control} had not been witnessed or signed for.",
    ),
}

#: Extra rules that the *wording* of a barrier clause implies, independent of the
#: frame. Labels must follow the text, so these are added at generation time.
_BARRIER_EXTRA_RULES: dict[str, tuple[str, ...]] = {
    "bypassed": ("bypassing-safety-controls",),
}

_MEASUREMENTS: dict[str, tuple[str, ...]] = {
    "pressure": (
        "The line was still live at {p} kg/cm2.",
        "Upstream pressure was recorded as {p} kg/cm2 at the time.",
        "The vessel was holding {p} bar when the work started.",
        "A gauge on the header read {p} kg/cm2.",
    ),
    "gravity": (
        "The working position was about {h} m above grade.",
        "The platform was at a height of roughly {h} metres.",
        "The fall exposure was measured at {h} m.",
        "He was working some {h} m up on the structure.",
    ),
    "chemical": (
        "A portable meter later read {g}% LEL at the work face.",
        "H2S was subsequently detected at {g} ppm inside the space.",
        "Oxygen in the space measured {g}% against a required 19.5%.",
        "The atmosphere tested {g}% LEL after the job was stopped.",
    ),
    "electrical": (
        "The feeder was an 11 kV supply and was still charged.",
        "The panel was fed from a 415 V bus that had not been made dead.",
        "The circuit was rated 6.6 kV and remained live.",
    ),
    "thermal": (
        "Surface temperature at the joint was about {t} deg C.",
        "The line was running at {t} deg C when work started.",
    ),
    "mechanical": (
        "The load being handled was around {kg} kg.",
        "The crane was working at about {pctload}% of its rated capacity.",
    ),
    "motion": (
        "The vehicle was doing an estimated {kmh} kmph on the approach road.",
        "Speed on the ROW stretch was around {kmh} kmph.",
    ),
}

_TEMPORAL = (
    "The job resumed after the meal break without a fresh check.",
    "This happened during the night shift.",
    "The work was going on across a crew change.",
    "It was the reliever's first day on that unit.",
    "The activity continued past the end of shift with a skeleton crew.",
    "The inspection was overdue by {d} days.",
    "The certificate had been overdue for {d} days.",
    "Calibration of the instrument was overdue by {d} days.",
    "Work was in progress on a Sunday with a single operator on the unit.",
    "The handover note did not mention the pending job.",
)

_OUTCOMES: dict[int, tuple[str, ...]] = {
    0: (
        "There was no injury and no damage.",
        "Nobody was hurt and the job was stopped immediately.",
        "The situation was corrected before anyone was exposed.",
        "No harm resulted; the observation was closed the same shift.",
    ),
    1: (
        "The worker sustained a minor graze and was given first aid.",
        "He suffered a small cut which needed first aid only.",
        "A bruise was reported and treated at the dispensary.",
    ),
    2: (
        "The worker sustained a laceration that needed stitches.",
        "He suffered a sprain and was put on light duty.",
        "An abrasion to the forearm was recorded as a first aid case.",
    ),
    3: (
        "The worker sustained a fracture and was hospitalised.",
        "He suffered a fractured wrist and lost time followed.",
        "A second degree burn was recorded and the man was hospitalised.",
    ),
    4: (
        "The worker suffered a crush injury to the hand.",
        "He was found unconscious and was shifted to hospital in critical condition.",
        "The injury resulted in a permanent disability.",
    ),
    5: (
        "The worker died at the scene.",
        "The incident resulted in a fatality.",
        "One person was killed and the site was cordoned off.",
    ),
}

#: Outcome sentences that sit genuinely on the line between two bands, listed
#: under BOTH of them.
#:
#: A cut that needed stitches is coded first-aid by one reviewer and recordable
#: by the next; "taken to hospital as a precaution" says nothing about whether an
#: injury was found. Without this overlap the recorded band can be read straight
#: off the vocabulary and the actual-severity head scores a meaningless 1.000 -
#: which measures the generator, not the model.
_OUTCOME_BOUNDARY: dict[tuple[int, int], tuple[str, ...]] = {
    (1, 2): (
        "The worker was cut on the hand and the wound was dressed on site.",
        "A soft tissue injury was reported and he returned to work after treatment.",
    ),
    (2, 3): (
        "He was taken to the dispensary with a suspected fracture.",
        "The worker was sent for an X-ray after complaining of pain in the arm.",
    ),
    (3, 4): (
        "The man was hospitalised and kept under observation overnight.",
        "He sustained burns to the arm and was referred to the base hospital.",
    ),
    (4, 5): (
        "He was shifted out unresponsive and the outcome was not known at the time of reporting.",
        "The worker suffered serious injuries and was airlifted from site.",
    ),
}

#: rank -> the boundary sentences that could also describe it
_BOUNDARY_BY_RANK: dict[int, tuple[str, ...]] = {}
for _pair, _lines in _OUTCOME_BOUNDARY.items():
    for _rank in _pair:
        _BOUNDARY_BY_RANK[_rank] = _BOUNDARY_BY_RANK.get(_rank, ()) + _lines

#: Chance an injury outcome is written with a boundary sentence instead of an
#: unambiguous one.
BOUNDARY_OUTCOME_RATE = 0.34


def _outcome_sentence(rng: random.Random, rank: int) -> str:
    """Pick the sentence that reports `rank`, sometimes an ambiguous one."""
    pool = _BOUNDARY_BY_RANK.get(rank)
    if pool and rng.random() < BOUNDARY_OUTCOME_RATE:
        return rng.choice(pool)
    return rng.choice(_OUTCOMES[rank])


#: Frames that genuinely carry a second hazard energy, so the recorded "primary"
#: energy is a coding decision rather than a fact. A suspended load is gravity to
#: one officer and mechanical to the next; a hot tap on a sour line is thermal or
#: chemical depending on which end of the job you were standing at. When the alt
#: is drawn the narrative is phrased from the alt energy's vocabulary too, so the
#: text leans the same way the label does - this is ambiguity, not label noise.
_ALT_ENERGY: dict[str, str] = {
    "crane-lift": "mechanical",       # suspended load, or the crane driving it
    "rig-floor-lift": "mechanical",   # tubulars on the drawworks
    "scaffold-height": "mechanical",  # dropped tools and swinging members
    "pump-seal": "pressure",          # a seal is a mechanical part on a pressured system
    "guard-removed": "motion",        # rotating equipment is also unguarded motion
    "hot-work-line": "chemical",      # welding on a hydrocarbon line
    "hot-work-height": "gravity",     # the same job, one floor up
    "vessel-entry": "chemical",       # atmosphere, or the agitator inside
    "tank-roof-entry": "gravity",     # roof edge as much as vapour
    "pressure-release": "chemical",   # what is in the line matters as much as the pressure
    "hose-whip": "motion",            # a failed hose is stored energy released as motion
    "well-intervention": "mechanical",
    "gas-leak-station": "pressure",
    "mcc-panel": "thermal",           # arc flash is heat as much as current
    "forklift-yard": "motion",
    "trip-inhibit": "chemical",
    "row-driving": "mechanical",
    "drain-cover": "motion",
}

#: Chance the alternate energy is the one recorded.
ALT_ENERGY_RATE = 0.24


def _draw_energy(rng: random.Random, frame: Frame) -> str:
    """The hazard energy as an officer would code it for this write-up."""
    alt = _ALT_ENERGY.get(frame.id)
    if alt and rng.random() < ALT_ENERGY_RATE:
        return alt
    return frame.energy

# ---------------------------------------------------------------------------
# scenario frames
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    """One kind of job, with the hazard it carries and the control that guards it."""

    id: str
    activity: str
    department: str
    rules: tuple[str, ...]
    energy: str
    potential: int  # 1..5 - what COULD happen if the barrier fails
    subjects: tuple[str, ...]
    gerunds: tuple[str, ...]
    pasts: tuple[str, ...]
    controls: tuple[str, ...]
    report_types: tuple[str, ...] = ("UA", "UC", "near-miss")
    #: extra rules only added when the barrier is not intact (a permit rule is
    #: not breached by a job that had a valid permit)
    rules_if_breached: tuple[str, ...] = field(default=())


FRAMES: tuple[Frame, ...] = (
    # ---------------- energy isolation (high) ----------------
    Frame(
        id="pump-seal",
        activity="Pump maintenance",
        department="Maintenance",
        rules=("energy-isolation",),
        energy="mechanical",
        potential=5,
        subjects=("a fitter", "a mechanical technician", "the maintenance crew", "a contract fitter"),
        gerunds=(
            "opening the mechanical seal housing of the transfer pump while the motor was still racked in",
            "dismantling the coupling guard of the running booster pump",
            "loosening the gland follower of a pump that was still on auto standby",
        ),
        pasts=(
            "had opened the seal housing of a pump whose motor was still racked in",
            "started dismantling a pump that could restart on auto",
            "began work on a pump breaker that had not been racked out",
        ),
        controls=("lock out tag out", "LOTO", "electrical isolation", "breaker racking out"),
        rules_if_breached=("work-authorisation",),
    ),
    Frame(
        id="mcc-panel",
        activity="Electrical maintenance",
        department="Electrical",
        rules=("energy-isolation",),
        energy="electrical",
        potential=5,
        subjects=("an electrician", "a junior electrician", "the electrical crew", "a contract electrician"),
        gerunds=(
            "working inside an MCC cubicle whose incomer was still charged",
            "replacing a contactor in a live 415 V panel",
            "opening a feeder compartment without proving it dead",
        ),
        pasts=(
            "opened a live MCC compartment",
            "worked on a feeder that had not been made dead",
            "replaced a relay in a panel that was still charged",
        ),
        controls=("LOTO", "electrical isolation", "earthing", "permit to work"),
        rules_if_breached=("work-authorisation",),
    ),
    # ---------------- energy isolation (low energy, similar words) ----------------
    Frame(
        id="loto-register",
        activity="Isolation audit",
        department="Maintenance",
        rules=("energy-isolation",),
        energy="mechanical",
        potential=2,
        subjects=("the shift in-charge", "the area operator", "an auditor"),
        gerunds=(
            "reviewing the LOTO register, where two locks were logged against a completed job",
            "checking the isolation register for the previous month",
        ),
        pasts=(
            "found two LOTO entries in the register without a closing signature",
            "noted that the isolation log had not been updated for a completed job",
        ),
        controls=("register entry", "closing signature", "LOTO record"),
        report_types=("UC",),
    ),
    # ---------------- hot work ----------------
    Frame(
        id="hot-work-line",
        activity="Welding / hot work",
        department="Maintenance",
        rules=("hot-work",),
        energy="thermal",
        potential=5,
        subjects=("a welder", "two contract welders", "the fabrication crew", "a contract welder"),
        gerunds=(
            "welding on the produced water line without a fresh gas test after the break",
            "cutting a bracket off a live process line with a grinder",
            "carrying out hot work next to a hydrocarbon drain that was still open",
        ),
        pasts=(
            "resumed welding on a process line after the meal break with no fresh gas test",
            "started grinding beside an open hydrocarbon drain",
            "carried out hot work within three metres of a live flange",
        ),
        controls=("hot work permit", "gas test", "fire watch", "PTW"),
        rules_if_breached=("work-authorisation",),
    ),
    Frame(
        id="hot-work-housekeeping",
        activity="Welding / hot work",
        department="Maintenance",
        rules=("hot-work",),
        energy="thermal",
        potential=2,
        subjects=("the welding crew", "a welder", "the fabricator"),
        gerunds=(
            "leaving welding cables strung across a walkway in the workshop bay",
            "storing gas cylinders without chains in the fabrication yard",
        ),
        pasts=(
            "left welding leads uncoiled across the workshop floor",
            "kept oxygen and acetylene cylinders together in the open yard",
        ),
        controls=("cylinder restraint", "housekeeping standard", "cable routing"),
        report_types=("UC",),
    ),
    # ---------------- confined space ----------------
    Frame(
        id="vessel-entry",
        activity="Vessel entry / cleaning",
        department="Production",
        rules=("confined-space",),
        energy="chemical",
        potential=5,
        subjects=("two contract workers", "a cleaning gang", "three contract hands", "a vessel cleaner"),
        gerunds=(
            "entering the test separator to remove sludge wearing only cloth masks",
            "climbing into a storage tank through the bottom manway to clean residue",
            "going inside the surge vessel while the inlet was still lined up",
        ),
        pasts=(
            "entered the test separator to remove sludge with only cloth masks on",
            "went inside a tank that had not been gas freed",
            "entered a vessel whose inlet valve was still open",
        ),
        controls=("entry permit", "continuous gas monitoring", "standby attendant", "CSE certificate", "blinding"),
        rules_if_breached=("work-authorisation",),
    ),
    Frame(
        id="cse-register",
        activity="Vessel entry / cleaning",
        department="Production",
        rules=("confined-space",),
        energy="chemical",
        potential=1,
        subjects=("the area operator", "the HSE officer", "an internal auditor"),
        gerunds=(
            "auditing the confined space register, which had two closed entries lacking a signature",
            "reviewing last quarter's confined space entry records",
        ),
        pasts=(
            "found two closed confined space entries without the closing signature",
            "noted that the CSE register was filled in a day late",
        ),
        controls=("closing signature", "register entry", "record keeping"),
        report_types=("UC",),
    ),
    # ---------------- working at height ----------------
    Frame(
        id="scaffold-height",
        activity="Scaffolding / work at height",
        department="Maintenance",
        rules=("working-at-height",),
        energy="gravity",
        potential=5,
        subjects=("a scaffolder", "a painter", "an insulation worker", "a contract rigger"),
        gerunds=(
            "working on the derrick monkey board with the harness lanyard unclipped",
            "standing on the top rail of a scaffold to reach a flange",
            "moving along an unboarded scaffold lift carrying a spanner set",
        ),
        pasts=(
            "worked at height with his lanyard hanging unclipped",
            "stepped onto the mid rail of a scaffold to reach a valve",
            "used an incomplete scaffold with no boards on the working lift",
        ),
        controls=("fall arrest anchor", "full body harness", "scaffold tag", "handrail", "lifeline"),
    ),
    Frame(
        id="ladder-low",
        activity="Scaffolding / work at height",
        department="Maintenance",
        rules=("working-at-height",),
        energy="gravity",
        potential=2,
        subjects=("a helper", "an electrician", "a painter"),
        gerunds=(
            "using a step ladder on level ground to change a light fitting at 1.5 m",
            "standing on the second rung of a short ladder to clean a panel",
        ),
        pasts=(
            "used a step ladder at about 1.2 m without anybody footing it",
            "stood on a low ladder to wipe down a junction box",
        ),
        controls=("ladder footing", "three point contact", "ladder inspection tag"),
        report_types=("UA", "UC"),
    ),
    # ---------------- line of fire ----------------
    Frame(
        id="pressure-release",
        activity="Line breaking",
        department="Production",
        rules=("line-of-fire", "energy-isolation"),
        energy="pressure",
        potential=5,
        subjects=("an operator", "a fitter", "the production crew", "a contract fitter"),
        gerunds=(
            "standing in line with a flange he was breaking on a line that was still packed",
            "opening a bleeder while facing the discharge path",
            "loosening flange bolts on a header without depressurising it first",
        ),
        pasts=(
            "broke a flange while standing directly in the discharge path",
            "opened a line that was still holding pressure",
            "cracked a bleeder with his face over the vent point",
        ),
        controls=("depressurisation", "double block and bleed", "pressure gauge check", "line breaking permit"),
        rules_if_breached=("work-authorisation",),
    ),
    Frame(
        id="hose-whip",
        activity="Pressure testing",
        department="Well Services",
        rules=("line-of-fire",),
        energy="pressure",
        potential=4,
        subjects=("the testing crew", "a pump operator", "two technicians"),
        gerunds=(
            "standing beside an unrestrained test hose during a pressure test",
            "walking through the test area while the line was being pressurised",
        ),
        pasts=(
            "stood next to a test hose with no whip check fitted",
            "crossed the barricade while the line was under test",
        ),
        controls=("whip check", "test barricade", "exclusion zone", "relief valve setting"),
    ),
    Frame(
        id="walkway-block",
        activity="Housekeeping",
        department="Production",
        rules=("line-of-fire",),
        energy="gravity",
        potential=2,
        subjects=("the area crew", "a helper", "the contractor's gang"),
        gerunds=(
            "stacking empty drums along the walkway near the manifold",
            "leaving scrap tubing across the escape route from the platform",
        ),
        pasts=(
            "stacked empty drums along a walkway, narrowing the escape route to about 600 mm",
            "left scrap material on the designated escape route",
        ),
        controls=("escape route width", "housekeeping standard", "material stacking plan"),
        report_types=("UC",),
    ),
    # ---------------- mechanical lifting ----------------
    Frame(
        id="crane-lift",
        activity="Crane / lifting operations",
        department="Logistics",
        rules=("safe-mechanical-lifting", "line-of-fire"),
        energy="gravity",
        potential=5,
        subjects=("the rigging crew", "a crane operator", "a rigger", "two slingers"),
        gerunds=(
            "walking under a suspended load while the crane held it on the hook",
            "guiding a load by hand with his body under the swing path",
            "lifting a bundle with a sling that had visible broken wires",
        ),
        pasts=(
            "passed under a suspended load during the lift",
            "used a frayed sling to lift a pipe bundle",
            "lifted a load without any tag line, guiding it by hand",
        ),
        controls=("lift plan", "certified sling", "tag line", "exclusion zone", "banksman"),
    ),
    Frame(
        id="forklift-yard",
        activity="Material handling",
        department="Logistics",
        rules=("safe-mechanical-lifting",),
        energy="mechanical",
        potential=3,
        subjects=("a forklift operator", "a store hand", "the yard crew"),
        gerunds=(
            "carrying a pallet with the forks raised too high across the yard",
            "reversing a forklift in the store without looking behind",
        ),
        pasts=(
            "drove a forklift with the load raised above axle height",
            "left a forklift with its forks up and the key in",
        ),
        controls=("travel height limit", "operator authorisation", "reverse alarm"),
    ),
    # ---------------- bypassing safety controls ----------------
    Frame(
        id="trip-inhibit",
        activity="Instrumentation work",
        department="Instrumentation",
        rules=("bypassing-safety-controls",),
        energy="pressure",
        potential=5,
        subjects=("an instrument technician", "the panel operator", "the DCS operator"),
        gerunds=(
            "keeping a high pressure trip inhibited so the compressor would not shut down",
            "running the unit with the ESD loop forced in the DCS",
            "silencing a repeating gas alarm instead of investigating it",
        ),
        pasts=(
            "left a high pressure trip inhibited overnight",
            "forced the ESD input in the DCS to keep the plant running",
            "muted a gas alarm that kept coming up",
        ),
        controls=("trip inhibit approval", "MOC", "override log", "alarm response procedure"),
        rules_if_breached=("work-authorisation",),
    ),
    Frame(
        id="guard-removed",
        activity="Rotating equipment work",
        department="Maintenance",
        rules=("bypassing-safety-controls", "energy-isolation"),
        energy="mechanical",
        potential=4,
        subjects=("a fitter", "the maintenance crew", "a helper"),
        gerunds=(
            "running a compressor with the coupling guard removed",
            "operating a belt drive whose guard had been taken off weeks earlier",
        ),
        pasts=(
            "ran the machine with the coupling guard removed",
            "operated a belt drive with the guard lying beside it",
        ),
        controls=("machine guard", "guard inspection", "pre-start check"),
    ),
    # ---------------- work authorisation ----------------
    Frame(
        id="permit-expired",
        activity="Contractor work",
        department="Projects",
        rules=("work-authorisation",),
        energy="mechanical",
        potential=4,
        subjects=("a contractor gang", "the piping crew", "four contract workers"),
        gerunds=(
            "continuing pipe fitting on a permit that had expired at noon",
            "working on a permit issued for a different location",
        ),
        pasts=(
            "carried on working after the permit had expired at 1200 hrs",
            "worked under a permit raised for another area",
            "started a job with no PTW at all",
        ),
        controls=("PTW", "permit extension", "area authority signature"),
    ),
    Frame(
        id="induction-gap",
        activity="Contractor work",
        department="Projects",
        rules=("work-authorisation",),
        energy="mechanical",
        potential=2,
        subjects=("two new contract hands", "a new helper", "a visiting technician"),
        gerunds=(
            "working in the yard without having attended the site induction",
            "entering the plant area on a gate pass issued the previous month",
        ),
        pasts=(
            "were working without having done the site induction",
            "entered the plant on an expired gate pass",
        ),
        controls=("site induction record", "gate pass validity", "competency check"),
        report_types=("UA", "UC"),
    ),
    # ---------------- driving ----------------
    Frame(
        id="row-driving",
        activity="Vehicle movement",
        department="Logistics",
        rules=("driving",),
        energy="motion",
        potential=5,
        subjects=("a light vehicle driver", "a tanker driver", "the crew bus driver", "a contract driver"),
        gerunds=(
            "overtaking on a blind bend of the ROW approach road",
            "driving a crew bus at speed on a wet unmetalled stretch",
            "using a mobile phone while driving a tanker on the field road",
        ),
        pasts=(
            "overtook another vehicle on a blind curve of the ROW road",
            "drove the crew bus well above the field speed limit",
            "was on a phone call while driving a loaded tanker",
        ),
        controls=("journey management plan", "seat belt", "IVMS", "speed limit", "defensive driving certificate"),
    ),
    Frame(
        id="parking-yard",
        activity="Vehicle movement",
        department="Logistics",
        rules=("driving",),
        energy="motion",
        potential=2,
        subjects=("a driver", "a visitor", "a contract driver"),
        gerunds=(
            "parking a pickup facing inwards in the yard instead of reverse parking",
            "leaving a vehicle idling unattended near the store",
        ),
        pasts=(
            "parked the vehicle nose-in against the site standard",
            "left the vehicle running with nobody in it",
        ),
        controls=("reverse parking standard", "vehicle checklist", "key control"),
        report_types=("UA", "UC"),
    ),
    # ---------------- multi-rule frames ----------------
    Frame(
        id="tank-roof-entry",
        activity="Tank inspection",
        department="Inspection",
        rules=("confined-space", "working-at-height"),
        energy="chemical",
        potential=5,
        subjects=("an inspector", "the inspection crew", "a contract inspector"),
        gerunds=(
            "climbing onto a tank roof and opening the gauge hatch without a harness or a gas test",
            "working on a floating roof tank with no lifeline and no atmosphere check",
        ),
        pasts=(
            "went onto a tank roof and opened the hatch with no harness and no gas test",
            "entered a floating roof rim space without any atmosphere check",
        ),
        controls=("gas test", "fall arrest anchor", "entry permit", "standby attendant"),
        rules_if_breached=("work-authorisation",),
    ),
    Frame(
        id="hot-work-height",
        activity="Welding / hot work",
        department="Projects",
        rules=("hot-work", "working-at-height"),
        energy="thermal",
        potential=5,
        subjects=("a welder", "the erection crew", "two contract welders"),
        gerunds=(
            "welding a pipe support from a scaffold with no fire watch below and no harness clipped",
            "cutting steel at height with sparks falling onto a hydrocarbon drain",
        ),
        pasts=(
            "welded at height with no fire watch stationed below",
            "cut structural steel above an open drain with no spark containment",
        ),
        controls=("hot work permit", "fire watch", "harness anchor", "spark containment"),
        rules_if_breached=("work-authorisation",),
    ),
    Frame(
        id="rig-floor-lift",
        activity="Drilling operations",
        department="Drilling",
        rules=("safe-mechanical-lifting", "line-of-fire", "working-at-height"),
        energy="gravity",
        potential=5,
        subjects=("the drill crew", "a derrickman", "two roustabouts"),
        gerunds=(
            "handling drill collars on the rig floor while standing inside the swing radius",
            "racking pipe with the elevators while the fingerboard latches were unproven",
        ),
        pasts=(
            "handled tubulars while standing inside the load swing radius",
            "racked pipe with a fingerboard latch that had not been proved",
        ),
        controls=("exclusion zone", "certified lifting gear", "latch verification", "lift plan"),
    ),
    Frame(
        id="well-intervention",
        activity="Well intervention",
        department="Well Services",
        rules=("line-of-fire", "energy-isolation", "bypassing-safety-controls"),
        energy="pressure",
        potential=5,
        subjects=("the wireline crew", "the well services team", "a wireline operator"),
        gerunds=(
            "rigging up over a live well with the BOP function test outstanding",
            "pulling tools with the pressure control equipment not function tested",
        ),
        pasts=(
            "rigged up on a live well before the BOP had been function tested",
            "continued the job with the pressure control equipment untested",
        ),
        controls=("BOP function test", "pressure control certificate", "well barrier verification"),
        rules_if_breached=("work-authorisation",),
    ),
    Frame(
        id="gas-leak-station",
        activity="Gas handling",
        department="Production",
        rules=("hot-work", "bypassing-safety-controls"),
        energy="chemical",
        potential=5,
        subjects=("the station crew", "the shift operator", "a gas plant operator"),
        gerunds=(
            "working at a station where a gas detector had been isolated from the logic",
            "attending a leaking flange while the detector head was under maintenance override",
        ),
        pasts=(
            "worked at a station with the gas detector isolated from the trip logic",
            "left a leaking flange in service with the detector overridden",
        ),
        controls=("gas detection", "detector override log", "leak repair plan", "MOC"),
    ),
    # ---------------- genuinely low-risk observations ----------------
    Frame(
        id="signage",
        activity="Housekeeping",
        department="HSE",
        rules=("work-authorisation",),
        energy="gravity",
        potential=1,
        subjects=("the HSE officer", "the area operator", "an auditor"),
        gerunds=(
            "noting that a mandatory PPE sign at the gate had faded",
            "finding an emergency assembly point board partly hidden by a shrub",
        ),
        pasts=(
            "found the PPE signage at the entrance faded and hard to read",
            "noted that an assembly point board was obscured by vegetation",
        ),
        controls=("signage standard", "signage inspection", "housekeeping round"),
        report_types=("UC",),
    ),
    Frame(
        id="eyewash",
        activity="Facility inspection",
        department="HSE",
        rules=("work-authorisation",),
        energy="chemical",
        potential=3,
        subjects=("the HSE officer", "a lab technician", "the area operator"),
        gerunds=(
            "finding the eyewash station in the lab with low flow",
            "checking a safety shower whose monthly test record was blank",
        ),
        pasts=(
            "found the eyewash station flowing weakly",
            "noted that the safety shower test record was not filled in",
        ),
        controls=("eyewash flow test", "monthly inspection record", "shower maintenance"),
        report_types=("UC",),
    ),
    Frame(
        id="ppe-minor",
        activity="Routine operations",
        department="Production",
        rules=("work-authorisation",),
        energy="mechanical",
        potential=2,
        subjects=("an operator", "a helper", "a contract hand"),
        gerunds=(
            "walking through the yard with his chin strap unfastened",
            "carrying out a routine round without safety glasses in a non-process area",
        ),
        pasts=(
            "was walking in the yard with an unfastened chin strap",
            "did a routine round without eye protection in the office block area",
        ),
        controls=("PPE standard", "PPE compliance round", "TBT briefing"),
        report_types=("UA",),
    ),
    Frame(
        id="drain-cover",
        activity="Housekeeping",
        department="Production",
        rules=("line-of-fire",),
        energy="gravity",
        potential=3,
        subjects=("the area crew", "an operator", "a helper"),
        gerunds=(
            "leaving a drain cover displaced next to the pump house walkway",
            "finding a floor grating lifted and not barricaded near the separator",
        ),
        pasts=(
            "left a drain cover off beside the walkway",
            "found a grating lifted with nothing around it",
        ),
        controls=("grating fixing", "barricading", "housekeeping round"),
        report_types=("UC",),
    ),
)

FRAME_BY_ID: dict[str, Frame] = {f.id: f for f in FRAMES}

# ---------------------------------------------------------------------------
# OIL geography, used for both training text and the served report stream
# ---------------------------------------------------------------------------

SITES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Duliajan", ("Central Workshop", "OCS Duliajan", "LPG Plant Duliajan", "Main Store")),
    ("Moran", ("OCS Moran", "GGS Moran", "Well MRN-118", "Workover Rig WR-7")),
    ("Naharkatiya", ("GCS Naharkatiya", "Well NHK-234", "Drilling Rig DR-42", "Pipeline ROW NHK")),
    ("Baghjan", ("Well BGJ-05", "GGS Baghjan", "Gas Compressor Station")),
    ("Kumchai", ("GGS Kumchai", "Well KMC-12", "Flow Station Kumchai")),
    ("Jorajan", ("OCS Jorajan", "Well JRJ-77", "Effluent Treatment Plant")),
    ("Hapjan", ("GGS Hapjan", "Well HPJ-31", "Pipeline ROW HPJ")),
    ("Shalmari", ("Water Injection Plant", "Well SLM-09", "OCS Shalmari")),
)

REPORTERS = (
    "N. Hazarika", "B. Gogoi", "P. Baruah", "R. Saikia", "A. Dutta", "S. Phukan",
    "M. Bora", "T. Chetia", "K. Rajkhowa", "J. Konwar", "D. Sonowal", "L. Moran",
)
REPORTER_ROLES = (
    "Area Operator", "HSE Officer", "Shift In-charge", "Maintenance Supervisor",
    "Safety Steward", "Contractor Supervisor", "Field Engineer", "Inspector",
)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def _measurement_clause(rng: random.Random, energy: str, potential: int) -> str:
    """A clause stating the energy magnitude, scaled to `potential`.

    This is the channel through which the drawn potential reaches the text. When
    the clause is omitted (see `_compose`) the potential is genuinely
    under-determined and the model has to fall back on the frame's prior - which
    is exactly the ambiguity a real report stream contains.
    """
    pool = _MEASUREMENTS.get(energy)
    if not pool:
        return ""
    template = rng.choice(pool)

    # index 0..4 for potential 1..5
    lvl = max(0, min(4, potential - 1))
    pick = lambda ladder: rng.choice(ladder[lvl])  # noqa: E731

    return template.format(
        p=pick(((1, 2), (3, 5), (8, 12), (28, 42), (56, 88))),
        h=pick(((0.8, 1.2), (1.5, 2), (3, 4), (7, 11), (14, 24))),
        g=pick(((1, 2), (3, 4), (6, 9), (14, 22), (35, 48))),
        t=pick(((40, 50), (60, 75), (95, 120), (180, 240), (280, 340))),
        kg=pick(((60, 90), (140, 220), (400, 700), (1800, 2600), (4200, 6500))),
        pctload=pick(((25, 35), (45, 55), (65, 75), (86, 92), (95, 99))),
        kmh=pick(((12, 18), (24, 30), (38, 46), (58, 68), (78, 92))),
    )


def _draw_potential(rng: random.Random, frame: Frame) -> int:
    """Jitter the frame's nominal potential.

    The same job is more or less dangerous depending on the pressure, the height
    or the gas concentration on the day. Without this jitter the potential band
    is a pure function of the frame and the head that predicts it just memorises
    which frame wrote the sentence.
    """
    jitter = rng.choices((-1, 0, 1), weights=(18, 64, 18))[0]
    return max(1, min(5, frame.potential + jitter))


def _shorthand(rng: random.Random, text: str) -> str:
    """Randomly write some controls the way a hurried observer would.

    This is what makes the normalisation step earn its place: about a third of
    the corpus uses shorthand, so a model trained without expansion would see
    two disjoint vocabularies for the same control.
    """
    if rng.random() > 0.34:
        return text
    swaps = (
        ("lock out tag out", "LOTO"),
        ("permit to work", "PTW"),
        ("hot work permit", "hot work PTW"),
        ("confined space entry", "CSE"),
        ("personal protective equipment", "PPE"),
        ("work at height", "WAH"),
        ("job safety analysis", "JSA"),
        ("toolbox talk", "TBT"),
        ("management of change", "MOC"),
        ("hydrogen sulphide", "H2S"),
    )
    out = text
    for long, short in swaps:
        if long in out.lower() and rng.random() < 0.7:
            idx = out.lower().index(long)
            out = out[:idx] + short + out[idx + len(long):]
    return out


def _compose(
    rng: random.Random,
    frame: Frame,
    energy: str,
    potential: int,
    barrier: str,
    outcome_rank: int,
) -> tuple[str, dict]:
    """Build one narrative and return it with the facts that generated it."""
    site, assets = rng.choice(SITES)
    asset = rng.choice(assets)
    place = f"{asset}, {site}" if rng.random() < 0.5 else asset

    # About a third of the time the hazard is described generically, so the text
    # cannot be traced back to a single frame. When the alternate energy was
    # drawn the generic phrasing is used more often, because that is what makes
    # the write-up lean toward the coding it was given.
    generic = _GENERIC_HAZARD.get(energy, ())
    generic_odds = 0.62 if energy != frame.energy else 0.32
    if generic and rng.random() < generic_odds:
        gerund = rng.choice(generic)
        past = "was " + gerund
    else:
        gerund = rng.choice(frame.gerunds)
        past = rng.choice(frame.pasts)

    opener = rng.choice(_OPENERS).format(
        place=place,
        subject=rng.choice(frame.subjects),
        gerund=gerund,
        past=past,
        activity_low=frame.activity.lower(),
    )

    control = rng.choice(frame.controls)
    barrier_clause = rng.choice(_BARRIER_CLAUSES[barrier]).format(control=control)

    parts = [opener, barrier_clause]

    # Around a third of reports quote no number at all, and then the severity
    # potential has to be read from the kind of job alone.
    #
    # But when this report's potential is NOT the one its job usually carries -
    # an unusually high pressure, an unusually short drop - the number is always
    # quoted. Observers do write down the figure that surprised them, and it also
    # keeps the model honest: an off-frame potential is something the text says,
    # not something the label knows and the narrative hides.
    off_frame = potential != frame.potential
    if off_frame or rng.random() < 0.62:
        clause = _measurement_clause(rng, energy, potential)
        if clause:
            parts.append(clause)

    if rng.random() < 0.40:
        parts.append(rng.choice(_TEMPORAL).format(d=rng.choice((7, 14, 21, 45, 60, 95))))

    parts.append(_outcome_sentence(rng, outcome_rank))

    text = _shorthand(rng, " ".join(p for p in parts if p))

    return text, {
        "frame": frame.id,
        "site": site,
        "asset": asset,
        "control": control,
        "barrier": barrier,
        "outcome_rank": outcome_rank,
    }


def _labels(
    frame: Frame,
    energy: str,
    potential: int,
    barrier: str,
    outcome_rank: int,
    rng: random.Random,
) -> dict:
    """Derive every target from the generative facts.

    The SIF rule is the DEKRA / EEI precursor definition: high energy AND a
    compromised barrier. Note that an intact barrier on a fatal-potential job is
    labelled negative - that is doctrinally correct, not a bug.
    """
    weight = BARRIER_WEIGHT[barrier]
    risk = potential + weight + rng.gauss(0.0, JUDGEMENT_SD)

    sif = risk >= SIF_THRESHOLD
    if rng.random() < LABEL_NOISE:
        sif = not sif

    primary = frame.rules[0]
    rules = set(frame.rules)
    if barrier != "intact":
        rules.update(frame.rules_if_breached)
        rules.update(_BARRIER_EXTRA_RULES.get(barrier, ()))

    # Secondary-tag disagreement. The primary rule is never dropped - an officer
    # reading a welding report always calls it hot work - but which additional
    # rules get ticked is a genuine judgement call.
    secondary = rules - {primary}
    for rule in list(secondary):
        if rng.random() < RULE_DROP_NOISE:
            rules.discard(rule)
    if rng.random() < RULE_ADD_NOISE:
        pool = _CONFUSABLE.get(primary, ())
        if pool:
            rules.add(rng.choice(pool))

    # Continuous severity, 0-10. Reads three things out of the text: how much
    # energy the job carries, how badly the barrier broke, and what actually
    # happened.
    severity = (
        1.05 * potential
        + 0.85 * weight
        + 1.15 * outcome_rank
        + rng.gauss(0.0, 0.45)
    )
    severity = max(0.0, min(10.0, severity))

    return {
        "sif": sif,
        "rules": sorted(rules),
        "severity": severity,
        "severity_actual": max(1, outcome_rank),
        "severity_potential": potential,
        "energy": energy,
        "barrier": barrier,
    }


def _draw_barrier(rng: random.Random) -> str:
    """Barrier-state prior.

    Weighted so the resulting SIF-positive share lands in the 20-25% band the
    problem statement cites for leading operators, rather than at whatever rate
    a uniform draw would produce.
    """
    return rng.choices(
        ("intact", "not-verified", "inadequate", "absent", "failed", "bypassed"),
        weights=(42, 20, 13, 11, 7, 7),
        k=1,
    )[0]


def _draw_outcome(rng: random.Random, potential: int, barrier: str) -> int:
    """Most reports are near-misses: nothing happened. That is the point.

    An outcome only becomes likely once the barrier is gone, and its ceiling is
    the drawn potential.
    """
    if barrier == "intact":
        return 0
    if rng.random() < 0.80:
        return 0
    return rng.randint(1, potential)


def synthetic_corpus(n: int = 9000, seed: int = 26165) -> tuple[list[str], list[dict]]:
    """Generate ``n`` labelled narratives.

    :returns: ``(texts, labels)`` where each label dict carries ``sif``,
        ``rules``, ``severity`` and the generative metadata.
    """
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[dict] = []

    for _ in range(n):
        frame = rng.choice(FRAMES)
        energy = _draw_energy(rng, frame)
        potential = _draw_potential(rng, frame)
        barrier = _draw_barrier(rng)
        outcome = _draw_outcome(rng, potential, barrier)

        text, facts = _compose(rng, frame, energy, potential, barrier, outcome)
        label = _labels(frame, energy, potential, barrier, outcome, rng)
        label.update(facts)

        texts.append(text)
        labels.append(label)

    return texts, labels
