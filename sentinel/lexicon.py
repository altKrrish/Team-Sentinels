"""
sentinel.lexicon
================
Canonical high-energy hazard lexicon for SIF (Serious Injury & Fatality) triage.

Design notes
------------
* The lexicon is *canonical-id centric*: every surface form (English, Hindi,
  Bengali, Assamese, Hinglish/romanized) maps to one canonical hazard id that
  carries an energy-source class and a tier.
* `tier` drives decision precedence, NOT the model probability:
    - INTERLOCK  : phrase is, on its own, sufficient to force SIF review.
    - CORROBORATE: phrase raises risk but must co-occur with an energy source,
                   an exposure verb, or a metadata energy breach.
    - CONTEXT    : useful as a model feature only; never fires a rule.
* Surface forms are stored *normalized* (see sentinel.text_norm.normalize) so
  matching is script- and spelling-robust.

Extending: add rows to _RAW. Do NOT add generic words (e.g. "unsafe", "danger")
at INTERLOCK tier -- that is how a rule-based interlock destroys precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# Energy source classes follow the E&P / HOP "energy wheel" taxonomy used by
# OSHA SIF precursor work and OISD/DGMS incident coding.
ENERGY_CLASSES = (
    "gravity",        # falls, dropped objects, suspended loads
    "motion",         # vehicles, moving equipment, struck-by
    "mechanical",     # rotating equipment, stored spring/tension
    "electrical",     # arc flash, contact with live conductors
    "pressure",       # blowout, pressurized release, hydro-jetting
    "temperature",    # fire, steam, hot work, molten
    "chemical",       # H2S, toxic release, corrosives
    "radiation",      # NORM, radiography sources
    "biological",     # confined-space atmosphere, pathogens
    "gravity_person", # working at height / person falling
)

TIER_INTERLOCK = "INTERLOCK"
TIER_CORROBORATE = "CORROBORATE"
TIER_CONTEXT = "CONTEXT"


@dataclass(frozen=True)
class LexEntry:
    canonical: str
    energy_class: str
    tier: str
    surfaces: Tuple[str, ...]
    # Minimum token length for fuzzy matching to be allowed on this entry.
    # Short forms are exact-match only (fuzzy on 3-letter tokens is noise).
    allow_fuzzy: bool = True
    note: str = ""


# ---------------------------------------------------------------------------
# The lexicon. Surfaces are raw; they are normalized at load time.
# HI = Hindi (Devanagari), BN = Bengali, AS = Assamese, RM = romanized/Hinglish
# ---------------------------------------------------------------------------
_RAW: List[LexEntry] = [
    # ---------------- PRESSURE ----------------
    LexEntry("blowout", "pressure", TIER_INTERLOCK, (
        "blowout", "blow out", "well blowout", "kick well control", "well kick",
        "loss of well control", "bop failure", "annulus pressure surge",
        "उड़ान विस्फोट", "कुआं फटना", "ব্লো আউট", "ব্ল' আউট",
        "blow-out", "bhalu blowout", "kuan phata",
    )),
    LexEntry("uncontrolled_release", "pressure", TIER_INTERLOCK, (
        "uncontrolled release", "high pressure release", "line rupture",
        "pipeline rupture", "flowline burst", "hose whip", "whipping hose",
        "pressure surge injury", "hydrostatic test failure",
        "पाइपलाइन फटना", "লাইন ফেটে", "লাইন ফাটি",
        "line fatna", "line phata", "pipeline phat gaya",
    )),
    LexEntry("gas_release_flammable", "pressure", TIER_CORROBORATE, (
        "gas leak", "gas release", "hydrocarbon release", "lel high",
        "gas detector alarm", "flammable vapour cloud",
        "गैस रिसाव", "গ্যাস লিক", "গেছ লিক", "gas leak ho gaya", "gas rasav",
    )),

    # ---------------- ELECTRICAL ----------------
    LexEntry("arc_flash", "electrical", TIER_INTERLOCK, (
        "arc flash", "arcing flash", "flashover", "arc blast",
        "electrical explosion", "switchgear blast", "breaker exploded",
        "आर्क फ्लैश", "आग की चिंगारी विस्फोट", "আর্ক ফ্ল্যাশ",
        "arc flash ho gaya", "spark blast",
    )),
    LexEntry("live_contact", "electrical", TIER_INTERLOCK, (
        "electrocution", "electrocuted", "electric shock", "live wire contact",
        "contact with live conductor", "hv contact", "11kv contact",
        "energised panel opened", "worked on live panel", "no loto",
        "lockout tagout not done", "loto violation", "without loto",
        "without lockout tagout", "without loto isolation", "no loto isolation",
        "करंट लगना", "बिजली का झटका", "কারেন্ট লাগা", "কাৰেণ্ট লগা",
        "current lag gaya", "current laga", "shock laga", "bijli ka jhatka",
    )),

    # ---------------- GRAVITY / HEIGHT ----------------
    LexEntry("fall_from_height", "gravity_person", TIER_INTERLOCK, (
        "fall from height", "fell from height", "fell from scaffold",
        "fell from derrick", "fell from monkey board", "fall from ladder",
        "fall through grating", "fall through opening", "no fall arrest",
        "unhooked lanyard", "working at height without harness",
        "ऊंचाई से गिरना", "उपर से गिर गया", "উচ্চতা থেকে পড়ে",
        "ওপৰৰ পৰা পৰি", "upar se gir gaya", "uchai se gira", "height se gira",
    )),
    LexEntry("suspended_load", "gravity", TIER_INTERLOCK, (
        "suspended load", "load suspended overhead", "under the load",
        "under suspended load", "under suspended", "under a suspended",
        "suspended drill pipe", "suspended casing", "suspended pipe",
        "crane load swung", "sling failure", "rigging failure", "dropped object",
        "dropped tongs", "falling object", "load fell", "boom collapse",
        "crane toppled", "crane overturned",
        "लटका हुआ भार", "क्रेन गिरना", "ঝুলন্ত লোড", "ক্ৰেন পৰি",
        "load gir gaya", "crane gir gaya", "sling toot gaya",
    )),

    # ---------------- MECHANICAL / MOTION ----------------
    LexEntry("caught_between", "mechanical", TIER_INTERLOCK, (
        "amputation", "amputated", "degloving", "crushed hand", "crush injury",
        "caught between", "caught in rotating", "entangled in rotating",
        "pinch point injury", "finger severed", "hand crushed in tongs",
        "drawworks entanglement", "top drive entanglement",
        "अंग कटना", "हाथ कट गया", "হাত কাটা", "হাত কটা",
        "haath kat gaya", "ungli kat gayi", "hand cut ho gaya",
    )),
    LexEntry("stored_mechanical", "mechanical", TIER_CORROBORATE, (
        "stored energy", "spring tension released", "tensioned cable snapped",
        "wire rope snapped", "guy wire failure", "recoil", "winch line fraying",
        "frayed winch line", "frayed wire rope", "frayed cable",
        "रस्सी टूटना", "তার ছিঁড়ে", "rope toot gaya", "tar tut gaya",
    )),
    LexEntry("vehicle_incident", "motion", TIER_CORROBORATE, (
        "vehicle rollover", "head on collision", "vehicle overturned",
        "run over", "struck by vehicle", "reversing accident", "loss of control",
        "गाड़ी पलटना", "গাড়ি উল্টে", "গাড়ী বাগৰি",
        "gaadi palat gayi", "gadi ulta gaya",
    )),

    # ---------------- TEMPERATURE / FIRE ----------------
    LexEntry("fire_explosion", "temperature", TIER_INTERLOCK, (
        "explosion", "detonation", "deflagration", "fireball", "flash fire",
        "jet fire", "pool fire", "vapour cloud explosion", "tank fire",
        "विस्फोट", "आग लग गई", "বিস্ফোরণ", "বিস্ফোৰণ",
        "blast ho gaya", "aag lag gayi", "visfot",
    )),
    LexEntry("hot_work_uncontrolled", "temperature", TIER_CORROBORATE, (
        "hot work without permit", "welding near hydrocarbon",
        "grinding near gas line", "steam burn", "scalded", "molten splash",
        "गर्म काम बिना अनुमति", "গরম কাজ অনুমতি ছাড়া",
        "permit ke bina welding", "bina permit hot work",
    )),

    # ---------------- CHEMICAL ----------------
    LexEntry("h2s_exposure", "chemical", TIER_INTERLOCK, (
        "h2s exposure", "hydrogen sulphide exposure", "hydrogen sulfide release",
        "sour gas exposure", "h2s alarm high", "collapsed near sour gas",
        "एच2एस रिसाव", "বিষাক্ত গ্যাস", "h2s leak", "sour gas leak",
    )),
    LexEntry("confined_space", "biological", TIER_INTERLOCK, (
        "confined space entry without permit", "entered vessel without gas test",
        "oxygen deficient atmosphere", "asphyxiation", "unconscious in tank",
        "rescue from tank", "nitrogen purge entry",
        "बंद स्थान में प्रवेश", "আবদ্ধ স্থানে প্রবেশ",
        "tank ke andar behosh", "bina test tank entry",
    )),
    LexEntry("toxic_corrosive", "chemical", TIER_CORROBORATE, (
        "acid splash", "caustic splash", "chemical burn", "mercaptan release",
        "तेजाब गिरना", "রাসায়নিক পোড়া", "acid gir gaya", "chemical jal gaya",
    )),

    # ---------------- RADIATION ----------------
    LexEntry("radiation_source", "radiation", TIER_INTERLOCK, (
        "radiography source stuck", "source not retracted", "norm exposure",
        "unshielded source", "gamma source exposed",
        "विकिरण स्रोत", "রেডিয়েশন উৎস", "radiation source khula",
    )),

    # ---------------- OUTCOME / SEVERITY MARKERS ----------------
    LexEntry("fatality", "gravity", TIER_INTERLOCK, (
        "fatality", "fatal injury", "died at site", "deceased", "death of worker",
        "declared dead", "brought dead",
        "मृत्यु", "मौत हो गई", "মৃত্যু", "মৃত্যু হৈছে",
        "maut ho gayi", "mrityu", "death ho gaya",
    )),
    LexEntry("hospitalisation_major", "gravity", TIER_CORROBORATE, (
        "admitted to icu", "life threatening", "critical condition",
        "multiple fractures", "internal bleeding", "airlifted",
        "आईसीयू में भर्ती", "গুরুতর আহত", "icu me bharti", "serious injury",
    )),

    # ---------------- CONTEXT-ONLY (features, never rules) ----------------
    LexEntry("near_miss_marker", "motion", TIER_CONTEXT, (
        "near miss", "near hit", "close call", "could have been serious",
        "बाल बाल बचा", "অল্পের জন্য রক্ষা", "baal baal bacha",
    ), allow_fuzzy=False),
    LexEntry("ppe_gap", "gravity_person", TIER_CONTEXT, (
        "without helmet", "no helmet", "without harness", "no gloves",
        "बिना हेलमेट", "হেলমেট ছাড়া", "bina helmet", "helmet nahi pehna",
    ), allow_fuzzy=False),
]


@dataclass
class Lexicon:
    entries: List[LexEntry] = field(default_factory=lambda: list(_RAW))

    def by_tier(self, tier: str) -> List[LexEntry]:
        return [e for e in self.entries if e.tier == tier]

    def canonical_index(self) -> Dict[str, LexEntry]:
        return {e.canonical: e for e in self.entries}

    def surface_pairs(self) -> List[Tuple[str, LexEntry]]:
        """(surface_form, entry) for every surface in the lexicon."""
        return [(s, e) for e in self.entries for s in e.surfaces]


LEXICON = Lexicon()

__all__ = [
    "LEXICON", "Lexicon", "LexEntry", "ENERGY_CLASSES",
    "TIER_INTERLOCK", "TIER_CORROBORATE", "TIER_CONTEXT",
]
