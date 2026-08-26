#!/usr/bin/env python3
"""
Real-World OSHA Severe Injury Data → SIF Precursor Training Pipeline
=====================================================================

This pipeline takes real OSHA Severe Injury Reports (106K+ records with
free-text narratives) and transforms them into a labeled dataset suitable
for training an AI/NLP model that:

  1. Classifies reports as SIF-Potential vs Non-SIF-Potential
  2. Tags to IOGP Life-Saving Rules
  3. Extracts precursor patterns (activity, location, barrier failure)

Data Source: OSHA Severe Injury Reports (Jan 2015 – Nov 2025)
  Downloaded from: https://www.osha.gov/severeinjury

Usage:
  python preprocess_realworld.py                           # all defaults
  python preprocess_realworld.py --industry oil_gas        # only oil & gas
  python preprocess_realworld.py --industry all            # all industries
  python preprocess_realworld.py --max-records 20000       # limit records

Author: SIH-2026 Team
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

csv.field_size_limit(10**7)

# ──────────────────────────────────────────────────────────────
# IOGP LIFE-SAVING RULES — keyword-based tagging
# ──────────────────────────────────────────────────────────────

LIFE_SAVING_RULES = {
    "Energy Isolation": {
        "keywords": [
            "loto", "lockout", "tagout", "lock out", "tag out",
            "isolation", "de-energize", "de-energise", "energized",
            "live equipment", "stored energy", "residual energy",
            "electrical isolation", "zero energy", "electrical panel",
            "switchgear", "energized circuit", "powered equipment",
            "electrical shock", "electrocuted", "electrocution",
            "arc flash", "high voltage", "live wire", "breaker",
            "electrical contact", "power supply",
        ],
        "description": "Verify isolation and zero energy before work begins",
    },
    "Confined Space": {
        "keywords": [
            "confined space", "vessel entry", "tank entry", "manhole",
            "enclosed space", "oxygen deficient", "toxic atmosphere",
            "gas test", "entry permit", "ventilation", "silo",
            "hopper", "pit", "vault", "tunnel", "trench",
            "asphyxiation", "suffocation", "engulfed", "engulfment",
            "atmospheric", "oxygen", "h2s", "methane buildup",
        ],
        "description": "Obtain authorization before entering a confined space",
    },
    "Driving": {
        "keywords": [
            "driving", "vehicle", "speeding", "seatbelt", "seat belt",
            "road", "collision", "rollover", "truck", "car accident",
            "forklift", "backing", "reversing", "pedestrian struck",
            "vehicle struck", "driver", "passenger", "motor vehicle",
            "traffic", "highway", "intersection", "rear-end",
            "side-impact", "head-on", "pickup truck", "van",
            "company vehicle", "fleet", "transport",
        ],
        "description": "Follow safe driving rules",
    },
    "Hot Work": {
        "keywords": [
            "welding", "cutting torch", "grinding", "brazing",
            "flame", "spark", "ignition", "flash fire",
            "fire watch", "hot work", "gas free", "lel",
            "flammable", "combustible", "oxy-acetylene", "arc welding",
            "fire", "explosion", "burn", "thermal", "torch",
            "ignited", "slag", "molten", "spatter", "weld",
        ],
        "description": "Control flammables and ignition sources",
    },
    "Line of Fire": {
        "keywords": [
            "struck by", "caught between", "pinch point",
            "crush", "falling object", "dropped object", "suspended load",
            "pressurized", "pressure release", "ejection",
            "projectile", "ricochet", "recoil",
            "caught in", "compressed", "pinned", "struck against",
            "flying debris", "kickback", "swinging", "rotating",
            "moving part", "machinery", "equipment",
            "impact", "crushed between",
        ],
        "description": "Keep yourself and others out of the line of fire",
    },
    "Safe Mechanical Lifting": {
        "keywords": [
            "crane", "lifting", "rigging", "sling", "shackle",
            "hoist", "overhead crane", "forklift",
            "chain block", "winch", "boom", "outrigger",
            "load chart", "overload", "tip over", "tipped over",
            "dropped load", "load fell", "swinging load",
            "lift plan", "lifting equipment", "material handling",
        ],
        "description": "Plan lifting operations and control the area",
    },
    "Work Authorization": {
        "keywords": [
            "permit to work", "work permit", "authorization",
            "risk assessment", "job safety analysis", "toolbox talk",
            "procedure", "unauthorized", "without authorization",
            "without permission", "no permit", "training",
            "untrained", "not trained", "certification expired",
            "not certified", "not qualified", "unqualified",
            "safety briefing", "method statement",
        ],
        "description": "Work with a valid permit when required",
    },
    "Working at Height": {
        "keywords": [
            "height", "scaffold", "scaffolding", "ladder",
            "harness", "fall protection", "fall arrest", "guardrail",
            "lanyard", "lifeline", "elevated", "roof",
            "platform", "cherry picker", "aerial lift",
            "mewp", "edge protection", "fall from", "fell from",
            "fell off", "fell down", "fell through",
            "step ladder", "extension ladder", "elevated work",
        ],
        "description": "Protect yourself against a fall when working at height",
    },
    "Bypassing Safety Controls": {
        "keywords": [
            "bypass", "override", "interlock", "safety device",
            "guard removed", "guard missing", "safety guard",
            "machine guard", "disabled", "defeated",
            "safety switch", "emergency stop", "e-stop",
            "alarm disabled", "safety valve", "relief valve",
            "inhibit", "tampered", "removed guard",
            "circumvent", "shield removed", "protective barrier",
        ],
        "description": "Do not bypass or disable safety controls",
    },
}

# ──────────────────────────────────────────────────────────────
# NAICS CODE MAPPING — Industry categories
# ──────────────────────────────────────────────────────────────

# Oil & Gas and related industries
OIL_GAS_NAICS = {
    "211": "Oil and Gas Extraction",
    "213": "Support Activities for Mining (incl. Oil & Gas)",
    "324": "Petroleum and Coal Products Manufacturing",
    "486": "Pipeline Transportation",
    "447": "Gasoline Stations",
}

# Heavy industry NAICS codes (relevant to SIF analysis)
HEAVY_INDUSTRY_NAICS = {
    "211": "Oil and Gas Extraction",
    "213": "Support Activities for Mining",
    "221": "Utilities",
    "236": "Construction of Buildings",
    "237": "Heavy and Civil Engineering Construction",
    "238": "Specialty Trade Contractors",
    "311": "Food Manufacturing",
    "321": "Wood Product Manufacturing",
    "324": "Petroleum and Coal Products",
    "325": "Chemical Manufacturing",
    "326": "Plastics and Rubber Products",
    "327": "Nonmetallic Mineral Products",
    "331": "Primary Metal Manufacturing",
    "332": "Fabricated Metal Products",
    "333": "Machinery Manufacturing",
    "336": "Transportation Equipment Manufacturing",
    "486": "Pipeline Transportation",
    "562": "Waste Management and Remediation",
}


# ──────────────────────────────────────────────────────────────
# SIF-POTENTIAL LABELING ENGINE (heuristic-based)
# ──────────────────────────────────────────────────────────────

# Keywords that strongly indicate SIF potential
SIF_HIGH_INDICATORS = [
    # Fatal/critical mechanism keywords
    r"\b(fatal|fatality|killed|death|died|deceased)\b",
    r"\b(explosion|exploded|detonation|blast)\b",
    r"\b(electrocuted|electrocution|arc flash)\b",
    r"\b(engulfed|engulfment|asphyxiat|suffocati)\b",
    r"\b(crushed|crushing|pinned between|caught between)\b",
    r"\b(amputat|sever|dismember)\b",
    r"\b(fell?\s+\d{2,}\s+feet)\b",  # fell 10+ feet
    r"\b(collapse[ds]?)\b",
    r"\b(trench collapse|wall collapse|structural collapse|roof collapse)\b",
    r"\b(loss of consciousness|unconscious|unresponsive)\b",
    r"\b(cardiac arrest|heart attack|stroke)\b",
    r"\b(high voltage|high pressure)\b",
    r"\b(caught in running|caught in machinery)\b",
    r"\b(struck by falling)\b",
    r"\b(rollover|roll-over|overturned)\b",
]

# Keywords that indicate moderate SIF potential
SIF_MEDIUM_INDICATORS = [
    r"\b(fractur|broken bone)\b",
    r"\b(hospitali[sz]|surgery|surgical|ICU)\b",
    r"\b(burn[s]?\b(?:.*?(?:degree|severe|chemical)))",
    r"\b(concussion|head injury|brain injury|traumatic brain)\b",
    r"\b(internal injur|internal bleed)\b",
    r"\b(spinal|vertebra|paralyz|paralys)\b",
    r"\b(lacerat|deep cut|severed tendon)\b",
    r"\b(chemical exposure|toxic|poisoning)\b",
    r"\b(confined space)\b",
    r"\b(excavation|trench)\b",
    r"\b(crane|scaffold|height)\b",
]

# Keywords that indicate LOW SIF potential
SIF_LOW_INDICATORS = [
    r"\b(minor|slight|small)\b",
    r"\b(first aid)\b",
    r"\b(bruise|abrasion|sprain|strain|twist)\b",
    r"\b(eye irritat|foreign body in eye)\b",
    r"\b(sting|bee|insect)\b",
    r"\b(headache|nausea|dizziness)\b",
    r"\b(ergonomic|repetitive|carpal tunnel)\b",
    r"\b(slip|trip)(?:.*?\b(same level)\b)",
]


def label_sif_potential(narrative: str, nature: str, event: str,
                        hospitalized: float, amputation: float,
                        loss_of_eye: float) -> Tuple[int, float, str]:
    """
    Heuristic SIF-potential labeling based on:
    1. Narrative text analysis (keyword matching)
    2. Injury severity (amputation, hospitalization)
    3. Event type classification
    4. Nature of injury

    Returns: (sif_binary, confidence_score, rationale)
    """
    text = f"{narrative} {nature} {event}".lower()
    score = 0.0
    reasons = []

    # --- Score based on injury outcome ---
    if amputation > 0:
        score += 0.35
        reasons.append("Amputation occurred")
    if loss_of_eye > 0:
        score += 0.30
        reasons.append("Loss of eye occurred")

    # --- Score based on HIGH severity keywords ---
    for pattern in SIF_HIGH_INDICATORS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            score += 0.20
            reasons.append(f"High-severity indicator: {matches[0] if isinstance(matches[0], str) else matches[0]}")
            break  # cap at one high-indicator boost

    # --- Score based on MEDIUM severity keywords ---
    medium_count = 0
    for pattern in SIF_MEDIUM_INDICATORS:
        if re.search(pattern, text, re.IGNORECASE):
            medium_count += 1
    if medium_count > 0:
        score += min(medium_count * 0.08, 0.20)
        reasons.append(f"{medium_count} medium-severity indicators")

    # --- Penalty for LOW severity keywords ---
    low_count = 0
    for pattern in SIF_LOW_INDICATORS:
        if re.search(pattern, text, re.IGNORECASE):
            low_count += 1
    if low_count > 0:
        score -= min(low_count * 0.10, 0.25)
        reasons.append(f"{low_count} low-severity indicators (penalty)")

    # --- Fall height analysis ---
    fall_match = re.search(r'fell?\s+(?:approximately\s+)?(\d+)\s+(?:feet|ft)', text)
    if fall_match:
        height = int(fall_match.group(1))
        if height >= 20:
            score += 0.25
            reasons.append(f"Fall from {height} feet (high)")
        elif height >= 10:
            score += 0.15
            reasons.append(f"Fall from {height} feet (moderate)")
        elif height >= 6:
            score += 0.08
            reasons.append(f"Fall from {height} feet")

    # --- Event type analysis ---
    high_sif_events = [
        "caught in running equipment", "caught in machinery",
        "struck by falling", "fall to lower level",
        "explosion", "fire", "electrical contact",
        "engulfment", "suffocation", "asphyxiation",
        "compressed or pinched", "vehicle collision",
    ]
    for evt in high_sif_events:
        if evt in event.lower():
            score += 0.10
            reasons.append(f"High-SIF event type: {evt}")
            break

    # --- Hospitalization without amputation (might still be SIF) ---
    if hospitalized > 0 and amputation == 0:
        score += 0.10
        reasons.append("Hospitalization (non-amputation)")

    # Normalize to [0, 1]
    score = max(0.0, min(1.0, score))

    # Threshold: >= 0.40 is SIF-Potential
    sif_binary = 1 if score >= 0.40 else 0
    rationale = "; ".join(reasons) if reasons else "No significant SIF indicators found"

    return sif_binary, round(score, 4), rationale


# ──────────────────────────────────────────────────────────────
# LIFE-SAVING RULE TAGGING ENGINE
# ──────────────────────────────────────────────────────────────

def tag_life_saving_rules(narrative: str, event: str, source: str) -> List[str]:
    """
    Auto-tag IOGP Life-Saving Rules based on narrative content.
    Returns list of matched rule names.
    """
    text = f"{narrative} {event} {source}".lower()
    matched_rules = []

    for rule_name, rule_data in LIFE_SAVING_RULES.items():
        for keyword in rule_data["keywords"]:
            if keyword in text:
                matched_rules.append(rule_name)
                break  # One keyword match per rule is enough

    return matched_rules


# ──────────────────────────────────────────────────────────────
# TEXT CLEANING
# ──────────────────────────────────────────────────────────────

def clean_narrative(text: str) -> str:
    """Clean OSHA narrative for NLP processing."""
    if not text or not text.strip():
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # Fix common OSHA data artifacts
    text = re.sub(r'\s+', ' ', text)  # collapse whitespace
    text = re.sub(r'[–—]', '-', text)  # normalize dashes
    text = re.sub(r'[""''‛‟]', '"', text)  # normalize quotes
    text = re.sub(r'\r\n|\r', '\n', text)  # normalize line endings
    text = re.sub(r'\n+', ' ', text)  # flatten to single line

    # Remove redundant ID numbers
    text = re.sub(r'\b\d{6,}\b', '', text)  # remove long ID numbers

    # Standardize measurements
    text = re.sub(r'(\d+)\s*feet', r'\1 feet', text)
    text = re.sub(r'(\d+)\s*ft\.?\b', r'\1 feet', text)
    text = re.sub(r'(\d+)\s*lbs?\.?\b', r'\1 pounds', text)

    return text.strip()


def tokenize(text: str, remove_stopwords: bool = False) -> str:
    """Tokenize cleaned text for NLP models."""
    text = text.lower()
    tokens = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text)

    if remove_stopwords:
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "shall", "should", "may", "might", "can", "could", "am",
            "its", "it", "i", "me", "my", "we", "our", "you", "your",
            "he", "him", "his", "she", "her", "they", "them", "their",
            "what", "which", "who", "whom", "this", "that", "these",
            "those", "here", "there", "when", "where", "why", "how",
            "all", "both", "each", "few", "more", "most", "other",
            "some", "any", "only", "own", "same", "so", "than", "too",
            "very", "just", "also", "now", "then", "once", "to", "of",
            "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "between",
            "about", "against", "and", "but", "or", "if", "while",
            "because", "until", "although", "though", "since",
        }
        # KEEP negations — critical for safety NLP
        keep = {"not", "no", "without", "never", "nor", "none"}
        stopwords -= keep
        tokens = [t for t in tokens if t not in stopwords]

    return " ".join(tokens)


# ──────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────

def extract_features(narrative: str, event: str, nature: str) -> Dict:
    """Extract NLP and safety-domain features from text."""
    text_lower = narrative.lower()

    # Sentence count
    sentences = re.split(r'[.!?]+', narrative)
    sentences = [s for s in sentences if s.strip()]
    sentence_count = len(sentences)

    # Word count
    word_count = len(narrative.split())

    # Negation count
    negation_words = ["not", "no", "without", "never", "nor", "none",
                       "didn't", "did not", "wasn't", "was not",
                       "weren't", "were not", "hadn't", "had not"]
    negation_count = sum(1 for w in negation_words if w in text_lower)

    # Severity word counts
    high_words = ["fatal", "death", "killed", "explosion", "electrocuted",
                   "crushed", "amputat", "engulfed", "collapsed", "unconscious"]
    medium_words = ["fracture", "burn", "hospital", "surgery", "lacerat",
                     "struck", "fell", "dropped", "trapped", "pinned",
                     "exposure", "inhalation", "fire", "leak"]
    low_words = ["minor", "first aid", "bruise", "sprain", "strain",
                  "observation", "housekeeping", "administrative"]

    high_count = sum(1 for w in high_words if w in text_lower)
    medium_count = sum(1 for w in medium_words if w in text_lower)
    low_count = sum(1 for w in low_words if w in text_lower)

    # Barrier failure indicators
    barrier_fail_words = ["failed", "failure", "absent", "missing", "bypassed",
                           "disabled", "not in place", "not installed",
                           "guard removed", "removed guard", "no guard",
                           "not wearing", "without", "expired"]
    barrier_fail_count = sum(1 for w in barrier_fail_words if w in text_lower)

    # Height extraction
    fall_height = 0
    height_match = re.search(r'(\d+)\s*(?:feet|ft)', text_lower)
    if height_match:
        fall_height = int(height_match.group(1))

    # Contains numbers (quantitative detail)
    number_count = len(re.findall(r'\b\d+(?:\.\d+)?\b', narrative))

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "negation_count": negation_count,
        "high_severity_word_count": high_count,
        "medium_severity_word_count": medium_count,
        "low_severity_word_count": low_count,
        "barrier_failure_count": barrier_fail_count,
        "fall_height_feet": fall_height,
        "number_count": number_count,
    }


# ──────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ──────────────────────────────────────────────────────────────

def run_pipeline(input_path: str, output_dir: str, industry: str = "all",
                 max_records: int = 0, seed: int = 42):
    """Run the complete real-world data preprocessing pipeline."""

    print("=" * 70)
    print("  REAL-WORLD OSHA DATA → SIF PRECURSOR TRAINING PIPELINE")
    print("=" * 70)
    print(f"  Input     : {input_path}")
    print(f"  Output    : {output_dir}")
    print(f"  Industry  : {industry}")
    print(f"  Max records: {max_records or 'all'}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    all_rules = sorted(LIFE_SAVING_RULES.keys())

    # ── Step 1: Load raw OSHA data ──
    print("\n📂 Step 1: Loading raw OSHA data...")
    raw_records = []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_records.append(row)
    print(f"   Loaded {len(raw_records):,} raw records")

    # ── Step 2: Filter by industry ──
    print(f"\n🏭 Step 2: Filtering by industry: {industry}...")

    if industry == "oil_gas":
        naics_filter = set(OIL_GAS_NAICS.keys())
        filtered = [r for r in raw_records if r.get("Primary NAICS", "")[:3] in naics_filter]
    elif industry == "heavy":
        naics_filter = set(HEAVY_INDUSTRY_NAICS.keys())
        filtered = [r for r in raw_records if r.get("Primary NAICS", "")[:3] in naics_filter]
    else:  # "all"
        filtered = raw_records

    print(f"   Filtered to {len(filtered):,} records")

    # ── Step 3: Apply max_records limit ──
    if max_records > 0 and len(filtered) > max_records:
        random.shuffle(filtered)
        filtered = filtered[:max_records]
        print(f"   Sampled {max_records:,} records")

    # ── Step 4: Clean & Transform ──
    print("\n🧹 Step 4: Cleaning narratives, labeling SIF, tagging rules...")

    processed = []
    sif_count = 0
    rule_counter = Counter()
    issue_count = 0

    for i, raw in enumerate(filtered):
        narrative = raw.get("Final Narrative", "").strip()

        # Skip records with empty or very short narratives
        if not narrative or len(narrative.split()) < 5:
            issue_count += 1
            continue

        # Parse numeric fields
        try:
            hospitalized = float(raw.get("Hospitalized", "0") or "0")
        except ValueError:
            hospitalized = 0.0
        try:
            amputation = float(raw.get("Amputation", "0") or "0")
        except ValueError:
            amputation = 0.0
        try:
            loss_of_eye = float(raw.get("Loss of Eye", "0") or "0")
        except ValueError:
            loss_of_eye = 0.0

        # Clean narrative
        cleaned = clean_narrative(narrative)
        tokenized = tokenize(cleaned)
        tokenized_no_sw = tokenize(cleaned, remove_stopwords=True)

        # Label SIF potential
        nature = raw.get("NatureTitle", "")
        event_title = raw.get("EventTitle", "")
        source_title = raw.get("SourceTitle", "")
        sif_binary, sif_score, sif_rationale = label_sif_potential(
            cleaned, nature, event_title, hospitalized, amputation, loss_of_eye
        )
        if sif_binary == 1:
            sif_count += 1

        # Tag Life-Saving Rules
        rules = tag_life_saving_rules(cleaned, event_title, source_title)
        for r in rules:
            rule_counter[r] += 1

        # Extract features
        features = extract_features(cleaned, event_title, nature)

        # Parse date
        event_date = raw.get("EventDate", "")
        try:
            dt = datetime.strptime(event_date, "%m/%d/%Y")
            year = dt.year
            month = dt.month
            quarter = f"Q{(month - 1) // 3 + 1}"
            day_of_week = dt.strftime("%A")
            hour_of_day = -1  # OSHA data doesn't include time
            is_monsoon = int(month in (6, 7, 8, 9))
            month_sin = round(math.sin(2 * math.pi * month / 12), 4)
            month_cos = round(math.cos(2 * math.pi * month / 12), 4)
        except ValueError:
            year = month = -1
            quarter = ""
            day_of_week = ""
            hour_of_day = -1
            is_monsoon = -1
            month_sin = month_cos = 0

        # Get NAICS sector
        naics = raw.get("Primary NAICS", "")
        naics_sector = naics[:3] if naics else ""
        industry_name = HEAVY_INDUSTRY_NAICS.get(naics_sector,
                         OIL_GAS_NAICS.get(naics_sector, "Other"))

        # Multi-label rule columns
        rule_set = set(rules)
        rule_cols = {
            f"rule_{r.lower().replace(' ', '_')}": int(r in rule_set)
            for r in all_rules
        }

        # Build record
        record = {
            "report_id": f"OSHA-SIR-{raw.get('ID', str(i+1))}",
            "event_date": event_date,
            "year": year,
            "month": month,
            "quarter": quarter,
            "day_of_week": day_of_week,
            # Location
            "city": raw.get("City", "").strip(),
            "state": raw.get("State", "").strip(),
            "zip_code": raw.get("Zip", "").strip(),
            "latitude": raw.get("Latitude", ""),
            "longitude": raw.get("Longitude", ""),
            # Industry
            "naics_code": naics,
            "naics_sector": naics_sector,
            "industry_name": industry_name,
            "employer": raw.get("Employer", "").strip(),
            # Text
            "original_narrative": narrative,
            "text_cleaned": cleaned,
            "text_tokenized": tokenized,
            "text_tokenized_no_stopwords": tokenized_no_sw,
            # Injury severity (raw OSHA)
            "hospitalized": int(hospitalized),
            "amputation": int(amputation),
            "loss_of_eye": int(loss_of_eye),
            "nature_of_injury": nature,
            "part_of_body": raw.get("Part of Body Title", ""),
            "event_type": event_title,
            "source": source_title,
            "secondary_source": raw.get("Secondary Source Title", ""),
            # ── TARGET VARIABLES ──
            "sif_potential": "SIF-Potential" if sif_binary else "Non-SIF-Potential",
            "sif_potential_binary": sif_binary,
            "sif_confidence_score": sif_score,
            "sif_rationale": sif_rationale,
            # Life-Saving Rules
            "life_saving_rules": "|".join(rules) if rules else "None",
            "life_saving_rules_count": len(rules),
            # Engineered features
            **features,
            # Rule multi-label columns
            **rule_cols,
            # Temporal features
            "month_sin": month_sin,
            "month_cos": month_cos,
            "is_monsoon": is_monsoon,
        }

        processed.append(record)

        if (i + 1) % 10000 == 0:
            print(f"   ... processed {i + 1:,} / {len(filtered):,}")

    print(f"\n   ✅ Processed {len(processed):,} records")
    print(f"   ⚠️  Skipped {issue_count} records (empty/short narratives)")
    print(f"   SIF-Potential: {sif_count:,} ({sif_count/len(processed)*100:.1f}%)")
    print(f"   Non-SIF: {len(processed) - sif_count:,} ({(len(processed)-sif_count)/len(processed)*100:.1f}%)")

    print("\n   Life-Saving Rule distribution:")
    for rule, count in rule_counter.most_common():
        print(f"     {rule}: {count:,} ({count/len(processed)*100:.1f}%)")

    # ── Step 5: Stratified Train/Val/Test Split ──
    print("\n✂️  Step 5: Splitting into train/val/test (70/15/15)...")

    sif_recs = [r for r in processed if r["sif_potential_binary"] == 1]
    nonsif_recs = [r for r in processed if r["sif_potential_binary"] == 0]
    random.shuffle(sif_recs)
    random.shuffle(nonsif_recs)

    def split_list(lst, train_r=0.70, val_r=0.15):
        n = len(lst)
        t_end = int(n * train_r)
        v_end = int(n * (train_r + val_r))
        return lst[:t_end], lst[t_end:v_end], lst[v_end:]

    s_train, s_val, s_test = split_list(sif_recs)
    n_train, n_val, n_test = split_list(nonsif_recs)

    train = s_train + n_train
    val = s_val + n_val
    test = s_test + n_test
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    for r in train: r["split"] = "train"
    for r in val: r["split"] = "val"
    for r in test: r["split"] = "test"

    all_records = train + val + test

    def _sif_pct(recs):
        s = sum(1 for r in recs if r["sif_potential_binary"] == 1)
        return f"{s}/{len(recs)} ({s/max(len(recs),1)*100:.1f}%)"

    print(f"   Train : {len(train):,}  — SIF: {_sif_pct(train)}")
    print(f"   Val   : {len(val):,}   — SIF: {_sif_pct(val)}")
    print(f"   Test  : {len(test):,}   — SIF: {_sif_pct(test)}")

    # ── Step 6: Write outputs ──
    print("\n💾 Step 6: Writing output files...")

    # Determine column order
    output_fields = [
        "report_id", "split", "event_date", "year", "month", "quarter", "day_of_week",
        "city", "state", "zip_code", "latitude", "longitude",
        "naics_code", "naics_sector", "industry_name", "employer",
        "original_narrative", "text_cleaned", "text_tokenized", "text_tokenized_no_stopwords",
        "hospitalized", "amputation", "loss_of_eye",
        "nature_of_injury", "part_of_body", "event_type", "source", "secondary_source",
        # Targets
        "sif_potential", "sif_potential_binary", "sif_confidence_score", "sif_rationale",
        "life_saving_rules", "life_saving_rules_count",
        # Multi-label rule columns
    ]
    output_fields += [f"rule_{r.lower().replace(' ', '_')}" for r in all_rules]
    output_fields += [
        "word_count", "sentence_count", "negation_count",
        "high_severity_word_count", "medium_severity_word_count", "low_severity_word_count",
        "barrier_failure_count", "fall_height_feet", "number_count",
        "month_sin", "month_cos", "is_monsoon",
    ]

    # 6a. Full preprocessed CSV
    csv_path = os.path.join(output_dir, "osha_sif_preprocessed.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    print(f"   ✅ Full CSV: {csv_path} ({len(all_records):,} records)")

    # 6b. Per-split CSVs
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(output_dir, f"osha_sif_{split_name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(split_data)
        print(f"   ✅ {split_name.upper()} CSV: {path} ({len(split_data):,})")

    # 6c. NLP-ready JSONL
    nlp_fields = [
        "report_id", "text_tokenized", "text_tokenized_no_stopwords",
        "sif_potential_binary", "sif_confidence_score",
        "life_saving_rules",
    ] + [f"rule_{r.lower().replace(' ', '_')}" for r in all_rules]

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(output_dir, f"osha_sif_{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in split_data:
                row = {}
                for field in nlp_fields:
                    field_val = rec.get(field, "")
                    if field.startswith("rule_") or field == "sif_potential_binary":
                        try:
                            field_val = int(field_val)
                        except (ValueError, TypeError):
                            field_val = 0
                    elif field == "sif_confidence_score":
                        field_val = float(field_val)
                    row[field] = field_val
                # Parse rules to list
                if row["life_saving_rules"] == "None":
                    row["life_saving_rules"] = []
                else:
                    row["life_saving_rules"] = row["life_saving_rules"].split("|")
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"   ✅ {split_name.upper()} JSONL: {path}")

    # 6d. Metadata & preprocessing report
    report = {
        "pipeline_name": "Real-World OSHA → SIF Precursor Dataset",
        "pipeline_version": "1.0.0",
        "run_at": datetime.now().isoformat(),
        "data_source": "OSHA Severe Injury Reports (Jan 2015 – Nov 2025)",
        "data_url": "https://www.osha.gov/severeinjury",
        "input_file": input_path,
        "industry_filter": industry,
        "total_raw_records": len(raw_records),
        "after_industry_filter": len(filtered),
        "after_cleaning": len(processed),
        "skipped_records": issue_count,
        "splits": {
            "train": {"total": len(train),
                      "sif": sum(1 for r in train if r["sif_potential_binary"] == 1),
                      "non_sif": sum(1 for r in train if r["sif_potential_binary"] == 0)},
            "val": {"total": len(val),
                    "sif": sum(1 for r in val if r["sif_potential_binary"] == 1),
                    "non_sif": sum(1 for r in val if r["sif_potential_binary"] == 0)},
            "test": {"total": len(test),
                     "sif": sum(1 for r in test if r["sif_potential_binary"] == 1),
                     "non_sif": sum(1 for r in test if r["sif_potential_binary"] == 0)},
        },
        "sif_distribution": {
            "sif_potential": sif_count,
            "non_sif_potential": len(processed) - sif_count,
            "sif_ratio": round(sif_count / len(processed), 4),
        },
        "life_saving_rules_distribution": {k: v for k, v in rule_counter.most_common()},
        "labeling_methodology": {
            "sif_potential": (
                "Heuristic labeling using: (1) Injury outcome severity (amputation, "
                "hospitalization, loss of eye), (2) High-severity narrative keywords "
                "(fatal, explosion, electrocuted, crushed, etc.), (3) Medium-severity "
                "indicators (fracture, burn, fall height), (4) Event type classification, "
                "(5) Fall height extraction. Threshold: score >= 0.40 → SIF-Potential."
            ),
            "life_saving_rules": (
                "Keyword-based matching against IOGP 9 Life-Saving Rules. Each rule "
                "has a curated keyword list covering domain terminology, equipment, "
                "activities, and hazard types."
            ),
        },
        "what_to_predict": {
            "task_1_sif_classification": {
                "column": "sif_potential_binary",
                "type": "Binary Classification",
                "description": "Predict whether a safety report has SIF (Serious Injury & Fatality) potential",
                "input": "text_tokenized or text_cleaned",
                "labels": {"0": "Non-SIF-Potential", "1": "SIF-Potential"},
            },
            "task_2_life_saving_rule_tagging": {
                "columns": [f"rule_{r.lower().replace(' ', '_')}" for r in all_rules],
                "type": "Multi-Label Classification",
                "description": "Tag each report to one or more IOGP Life-Saving Rules",
                "input": "text_tokenized or text_cleaned",
                "labels": all_rules,
            },
            "task_3_severity_scoring": {
                "column": "sif_confidence_score",
                "type": "Regression (0-1)",
                "description": "Predict the SIF severity/confidence score",
                "input": "text_tokenized + engineered features",
            },
        },
        "feature_columns": {
            "text_features": ["text_cleaned", "text_tokenized", "text_tokenized_no_stopwords"],
            "target_variables": ["sif_potential_binary", "sif_confidence_score"] +
                                [f"rule_{r.lower().replace(' ', '_')}" for r in all_rules],
            "engineered_numeric": [
                "word_count", "sentence_count", "negation_count",
                "high_severity_word_count", "medium_severity_word_count",
                "low_severity_word_count", "barrier_failure_count",
                "fall_height_feet", "number_count",
            ],
            "categorical": ["naics_sector", "industry_name", "state", "event_type",
                             "nature_of_injury", "part_of_body"],
            "temporal": ["year", "month", "quarter", "month_sin", "month_cos", "is_monsoon"],
            "geospatial": ["latitude", "longitude", "state", "city"],
        },
    }

    report_path = os.path.join(output_dir, "preprocessing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"   ✅ Report: {report_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  ✅ PIPELINE COMPLETE — REAL-WORLD DATA READY FOR TRAINING")
    print("=" * 70)
    print(f"  Total records    : {len(all_records):,}")
    print(f"  SIF-Potential    : {sif_count:,} ({sif_count/len(processed)*100:.1f}%)")
    print(f"  Non-SIF-Potential: {len(processed)-sif_count:,}")
    print(f"  Output directory : {output_dir}/")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-World OSHA Data → SIF Precursor Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str,
        default=str(Path(__file__).resolve().parent / "raw" / "osha_data" / "January2015toNovember2025.csv"),
        help="Path to raw OSHA CSV",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).resolve().parent / "processed"),
        help="Output directory",
    )
    parser.add_argument(
        "--industry", type=str, default="all",
        choices=["all", "oil_gas", "heavy"],
        help="Industry filter: all, oil_gas, or heavy",
    )
    parser.add_argument("--max-records", type=int, default=0, help="Max records (0=all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    run_pipeline(args.input, args.output_dir, args.industry, args.max_records, args.seed)


if __name__ == "__main__":
    main()
