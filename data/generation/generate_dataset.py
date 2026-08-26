#!/usr/bin/env python3
"""
OIL HSSE Synthetic Dataset Generator
======================================
Generates a large, realistic, analysis-ready dataset of Unsafe Act / Unsafe
Condition observations, near-miss reports, and incident records for Oil India
Limited's HSSE platform.

Output files:
  - oil_hsse_reports.csv         (main flat table)
  - oil_hsse_reports.json        (JSON-lines for NLP pipelines)
  - oil_hsse_reports_metadata.json  (dataset statistics & schema)
  - oil_hsse_life_saving_rules.csv  (normalized many-to-many mapping)
  - oil_hsse_evidence_snippets.csv  (explainability evidence per rule)

Usage:
  python generate_dataset.py                     # default 5000 records
  python generate_dataset.py --num-records 10000 # custom count
  python generate_dataset.py --seed 42           # reproducible
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import string
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# ── Import domain templates ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data.generation.templates import (
    ACTIVITY_TYPES,
    BARRIER_STATUSES,
    BARRIER_TYPES,
    CORRECTIVE_ACTIONS,
    EQUIPMENT_TYPES,
    IMMEDIATE_ACTIONS,
    LIFE_SAVING_RULES,
    NON_SIF_TEMPLATES,
    OBSERVATION_CATEGORIES,
    OIL_LOCATIONS,
    PERSONNEL_FIRST_NAMES,
    PERSONNEL_LAST_NAMES,
    REPORT_TYPES,
    REPORTER_DESIGNATIONS,
    SHIFTS,
    SIF_TEMPLATES,
    WEATHER_CONDITIONS,
)

# ──────────────────────────────────────────────────────────────
# NARRATIVE AUGMENTATION ENGINE
# ──────────────────────────────────────────────────────────────

# Sentence-level augmentations to add variety to reports
CONTEXTUAL_PREFIXES = [
    "During the morning safety round, ",
    "At approximately {time}, ",
    "While conducting routine inspection, ",
    "It was observed during the shift changeover that ",
    "A safety observation card was raised when ",
    "During a management safety walk, it was noted that ",
    "The area supervisor reported that ",
    "A contract worker reported that ",
    "Upon arriving at the worksite, the HSE officer observed that ",
    "Following a complaint from field personnel, an inspection revealed that ",
    "During pre-job preparation, it was discovered that ",
    "A CCTV review identified that ",
    "The control room alerted field personnel when ",
    "Post-incident investigation revealed that ",
    "A behavioral safety observation recorded that ",
]

CONTEXTUAL_SUFFIXES = [
    " The observation has been escalated to the area manager for review.",
    " A safety stand-down was conducted for the affected crew.",
    " This marks the {n}th similar observation at this location in the past quarter.",
    " The contractor's HSE performance is being tracked against KPIs.",
    " A safety alert has been issued to all sites regarding this observation.",
    " The incident has been logged in the HSSE database for trending.",
    " Follow-up inspection scheduled within 48 hours.",
    " The area safety committee will review this during the next monthly meeting.",
    " Photos and evidence have been attached to the observation card.",
    " The observation was discussed during the daily HSE meeting.",
    "",
    "",
    "",  # blank suffixes to add variety — not every report gets one
]

# Additional detail sentences to inject into narratives for variability
DETAIL_INJECTIONS = [
    "The ambient temperature at the time was approximately {temp}°C.",
    "Visibility conditions were {visibility}.",
    "Wind speed was measured at approximately {wind} km/h from the {direction} direction.",
    "The task had been ongoing for approximately {hours} hours at the time of observation.",
    "This is a repeat observation — a similar finding was reported {days_ago} days ago at the same location.",
    "The contractor involved has been working at this site for {months} months.",
    "The equipment involved was last inspected on {last_inspection} and is due for the next inspection on {next_inspection}.",
    "A total of {num_workers} workers were in the immediate area at the time.",
    "The nearest emergency assembly point is approximately {distance} meters from the observation location.",
    "The activity was part of a planned {scope} scope.",
]


def _random_time():
    """Generate a random time string HH:MM."""
    return f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}"


def _random_detail_injection():
    """Generate a random contextual detail sentence."""
    template = random.choice(DETAIL_INJECTIONS)
    return template.format(
        temp=random.randint(18, 45),
        visibility=random.choice(["good", "moderate", "poor due to fog", "poor due to dust"]),
        wind=random.randint(5, 50),
        direction=random.choice(["north", "south", "east", "west", "northeast", "southwest"]),
        hours=random.randint(1, 8),
        days_ago=random.randint(7, 120),
        months=random.randint(1, 24),
        last_inspection=datetime.now().strftime("%Y-%m-%d"),
        next_inspection=(datetime.now() + timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d"),
        num_workers=random.randint(2, 15),
        distance=random.randint(50, 500),
        scope=random.choice(["turnaround", "shutdown", "routine maintenance", "project construction", "commissioning"]),
    )


def augment_narrative(base_narrative: str) -> str:
    """Add contextual variety to a base narrative template."""
    result = base_narrative

    # 40% chance: add a contextual prefix
    if random.random() < 0.4:
        prefix = random.choice(CONTEXTUAL_PREFIXES).format(time=_random_time())
        # Lowercase the first char of the narrative if adding a prefix
        result = prefix + result[0].lower() + result[1:]

    # 50% chance: inject a contextual detail
    if random.random() < 0.5:
        sentences = result.split(". ")
        if len(sentences) > 2:
            insert_pos = random.randint(1, len(sentences) - 1)
            sentences.insert(insert_pos, _random_detail_injection())
            result = ". ".join(sentences)

    # 60% chance: add a contextual suffix
    if random.random() < 0.6:
        suffix = random.choice(CONTEXTUAL_SUFFIXES).format(n=random.randint(2, 8))
        result = result.rstrip() + suffix

    return result


# ──────────────────────────────────────────────────────────────
# EVIDENCE SNIPPET EXTRACTOR
# ──────────────────────────────────────────────────────────────

def extract_evidence_snippets(narrative: str, rules: list) -> dict:
    """
    Extract text snippets from the narrative that serve as evidence for
    each tagged Life-Saving Rule. Returns a dict: {rule_name: [snippets]}.
    """
    evidence = {}
    for rule in rules:
        if rule not in LIFE_SAVING_RULES:
            continue
        keywords = LIFE_SAVING_RULES[rule]["keywords"]
        snippets = []
        sentences = re.split(r'(?<=[.!?])\s+', narrative)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for kw in keywords:
                if kw in sentence_lower:
                    snippet = sentence.strip()
                    if snippet and snippet not in snippets:
                        snippets.append(snippet)
                    break  # one keyword match per sentence is enough
        evidence[rule] = snippets if snippets else [f"[Contextual inference based on overall scenario description]"]
    return evidence


# ──────────────────────────────────────────────────────────────
# CONFIDENCE SCORE MODEL
# ──────────────────────────────────────────────────────────────

def compute_confidence_score(
    is_sif: bool,
    num_rules: int,
    barrier_status: str,
    report_type: str,
) -> float:
    """
    Compute a simulated ML model confidence score for SIF classification.
    Higher scores for clear-cut SIF cases, lower for ambiguous ones.
    """
    base = 0.85 if is_sif else 0.80

    # More rules → higher confidence for SIF
    if is_sif:
        rule_boost = min(num_rules * 0.03, 0.10)
        base += rule_boost

    # Barrier failures increase SIF confidence
    if "Failed" in barrier_status or "Bypassed" in barrier_status:
        base += 0.04
    elif "Absent" in barrier_status:
        base += 0.03
    elif "Degraded" in barrier_status:
        base -= 0.02

    # Report type adjustments
    if report_type == "Incident":
        base += 0.05
    elif report_type == "Near Miss":
        base += 0.02
    elif report_type == "Positive Observation":
        base -= 0.10

    # Add noise
    noise = random.gauss(0, 0.05)
    return round(max(0.45, min(0.99, base + noise)), 4)


# ──────────────────────────────────────────────────────────────
# TEMPORAL PATTERN ENGINE
# ──────────────────────────────────────────────────────────────

def generate_date_with_patterns(start_date: datetime, end_date: datetime) -> datetime:
    """
    Generate dates with realistic temporal patterns:
    - Seasonal spikes (monsoon: Jun–Sep in Assam → more incidents)
    - Weekday bias (more reports Mon–Fri)
    - Time-of-day clustering
    """
    total_days = (end_date - start_date).days
    day_offset = random.randint(0, total_days)
    base_date = start_date + timedelta(days=day_offset)

    # Monsoon season bias: 30% more likely in Jun–Sep
    month = base_date.month
    if month in (6, 7, 8, 9) and random.random() < 0.3:
        # Re-roll to stay in monsoon
        monsoon_start = base_date.replace(month=6, day=1)
        monsoon_end = base_date.replace(month=9, day=30)
        monsoon_days = (monsoon_end - monsoon_start).days
        base_date = monsoon_start + timedelta(days=random.randint(0, monsoon_days))

    # Weekday bias: 80% weekday, 20% weekend
    while base_date.weekday() >= 5 and random.random() < 0.8:
        base_date += timedelta(days=random.choice([1, 2]))

    # Time of day — operational hours bias
    if random.random() < 0.7:
        # Day shift peak
        hour = random.choices(
            range(6, 18),
            weights=[3, 5, 8, 10, 10, 9, 8, 7, 8, 9, 5, 3],
        )[0]
    else:
        # Night shift
        hour = random.choices(
            range(18, 30),  # 18–05 (next day)
            weights=[6, 7, 5, 4, 3, 2, 2, 2, 3, 4, 5, 6],
        )[0]
        if hour >= 24:
            hour -= 24
            base_date += timedelta(days=1)

    minute = random.randint(0, 59)
    return base_date.replace(hour=hour, minute=minute, second=random.randint(0, 59))


# ──────────────────────────────────────────────────────────────
# RECURRING PATTERN INJECTION
# ──────────────────────────────────────────────────────────────

RECURRING_PATTERNS = [
    {
        "type": "location_cluster",
        "description": "Repeated scaffold deficiencies at Duliajan Central Tank Farm",
        "location": "Duliajan",
        "area": "Central Tank Farm",
        "rules": ["Working at Height"],
        "frequency": 0.03,  # 3% of records will have this pattern
    },
    {
        "type": "activity_cluster",
        "description": "Persistent LOTO violations during workover operations",
        "activity": "Workover Operations",
        "rules": ["Energy Isolation"],
        "frequency": 0.025,
    },
    {
        "type": "contractor_cluster",
        "description": "Contract welders repeatedly found without valid hot work permits",
        "activity": "Welding & Fabrication",
        "rules": ["Hot Work", "Work Authorization"],
        "frequency": 0.02,
    },
    {
        "type": "equipment_cluster",
        "description": "Recurring crane safety violations at multiple sites",
        "equipment_keyword": "Crane",
        "rules": ["Safe Mechanical Lifting"],
        "frequency": 0.02,
    },
    {
        "type": "time_cluster",
        "description": "Night shift confined space entry violations",
        "shift": "Night Shift (18:00–06:00)",
        "rules": ["Confined Space"],
        "frequency": 0.015,
    },
]


# ──────────────────────────────────────────────────────────────
# MAIN RECORD GENERATOR
# ──────────────────────────────────────────────────────────────

def generate_single_record(
    record_index: int,
    start_date: datetime,
    end_date: datetime,
    sif_ratio: float = 0.35,
) -> dict:
    """Generate a single HSSE report record."""

    # ── Determine SIF potential ──
    is_sif = random.random() < sif_ratio

    # ── Select template ──
    if is_sif:
        template_text, rules, severity_rationale = random.choice(SIF_TEMPLATES)
    else:
        template_text, rules, severity_rationale = random.choice(NON_SIF_TEMPLATES)

    # ── Select location ──
    location_name = random.choice(list(OIL_LOCATIONS.keys()))
    location_data = OIL_LOCATIONS[location_name]
    area = random.choice(location_data["areas"])
    state = location_data["state"]

    # ── Select other attributes ──
    activity = random.choice(ACTIVITY_TYPES)
    equipment = random.choice(EQUIPMENT_TYPES)
    personnel_name = f"{random.choice(PERSONNEL_FIRST_NAMES)} {random.choice(PERSONNEL_LAST_NAMES)}"

    # ── Generate report type with realistic distribution ──
    if is_sif:
        report_type = random.choices(
            REPORT_TYPES,
            weights=[15, 15, 40, 20, 8, 2],
        )[0]
    else:
        report_type = random.choices(
            REPORT_TYPES,
            weights=[25, 30, 10, 3, 20, 12],
        )[0]

    # ── Observation category ──
    if rules:
        # Bias category selection based on rules
        rule_category_map = {
            "Energy Isolation": ["Electrical Hazard", "Procedural Violation", "Process Safety Event"],
            "Confined Space": ["Confined Space Issue", "Chemical Exposure", "H2S Exposure Risk"],
            "Hot Work": ["Fire Hazard", "Procedural Violation", "Process Safety Event"],
            "Working at Height": ["Working at Height Issue", "Scaffolding Deficiency"],
            "Line of Fire": ["Dropped Object Potential", "Crane & Lifting", "Pressure System Issue"],
            "Safe Mechanical Lifting": ["Crane & Lifting", "Dropped Object Potential"],
            "Driving": ["Transportation Hazard"],
            "Work Authorization": ["Permit Violation", "Procedural Violation"],
            "Bypassing Safety Controls": ["Process Safety Event", "Equipment Defect"],
        }
        possible_cats = []
        for r in rules:
            possible_cats.extend(rule_category_map.get(r, []))
        obs_category = random.choice(possible_cats) if possible_cats else random.choice(OBSERVATION_CATEGORIES)
    else:
        obs_category = random.choice(OBSERVATION_CATEGORIES)

    # ── Fill template ──
    veh_num = f"{random.randint(100, 999)}"
    narrative = template_text.format(
        location=location_name,
        area=area,
        equipment=equipment,
        activity=activity,
        personnel=personnel_name,
        veh_num=veh_num,
    )

    # ── Augment narrative ──
    narrative = augment_narrative(narrative)

    # ── Generate date ──
    report_datetime = generate_date_with_patterns(start_date, end_date)

    # ── Barrier analysis ──
    if is_sif:
        barrier = random.choice(BARRIER_TYPES)
        barrier_status = random.choices(
            BARRIER_STATUSES,
            weights=[5, 15, 35, 25, 15, 5],
        )[0]
    else:
        barrier = random.choice(BARRIER_TYPES)
        barrier_status = random.choices(
            BARRIER_STATUSES,
            weights=[40, 25, 5, 5, 3, 22],
        )[0]

    # ── Risk level ──
    if is_sif:
        risk_level = random.choices(
            ["Critical", "High", "Medium", "Low"],
            weights=[30, 50, 18, 2],
        )[0]
    else:
        risk_level = random.choices(
            ["Critical", "High", "Medium", "Low"],
            weights=[1, 8, 40, 51],
        )[0]

    # ── Shift ──
    shift = random.choices(SHIFTS, weights=[65, 35])[0]

    # ── Weather ──
    weather = random.choice(WEATHER_CONDITIONS)

    # ── Confidence score ──
    confidence = compute_confidence_score(is_sif, len(rules), barrier_status, report_type)

    # ── Evidence snippets ──
    evidence = extract_evidence_snippets(narrative, rules)

    # ── Actions ──
    immediate_action = random.choice(IMMEDIATE_ACTIONS)
    corrective_action = random.choice(CORRECTIVE_ACTIONS)

    # ── Reporter ──
    reporter = random.choice(REPORTER_DESIGNATIONS)
    reporter_name = f"{random.choice(PERSONNEL_FIRST_NAMES)} {random.choice(PERSONNEL_LAST_NAMES)}"

    # ── Build record ──
    report_id = f"OIL-HSSE-{report_datetime.strftime('%Y%m%d')}-{record_index:05d}"

    record = {
        "report_id": report_id,
        "date_reported": report_datetime.strftime("%Y-%m-%d"),
        "time_reported": report_datetime.strftime("%H:%M:%S"),
        "datetime_reported": report_datetime.isoformat(),
        "year": report_datetime.year,
        "month": report_datetime.month,
        "quarter": f"Q{(report_datetime.month - 1) // 3 + 1}",
        "day_of_week": report_datetime.strftime("%A"),
        "report_type": report_type,
        "observation_category": obs_category,
        "location_site": location_name,
        "location_state": state,
        "location_area": area,
        "location_type": location_data["type"],
        "activity_type": activity,
        "equipment_involved": equipment,
        "free_text_description": narrative,
        "word_count": len(narrative.split()),
        "sif_potential": "SIF-Potential" if is_sif else "Non-SIF-Potential",
        "sif_potential_binary": 1 if is_sif else 0,
        "sif_confidence_score": confidence,
        "severity_rationale": severity_rationale,
        "risk_level": risk_level,
        "life_saving_rules": "|".join(rules) if rules else "None",
        "life_saving_rules_count": len(rules),
        "evidence_snippets_json": json.dumps(evidence),
        "barrier_type": barrier,
        "barrier_status": barrier_status,
        "immediate_action_taken": immediate_action,
        "corrective_action_recommended": corrective_action,
        "reporter_designation": reporter,
        "reporter_name": reporter_name,
        "shift": shift,
        "weather_conditions": weather,
    }

    return record


def inject_recurring_patterns(records: list) -> list:
    """
    Post-process records to inject recurring patterns — simulating
    real-world clustering that HSE teams need to detect.
    """
    for pattern in RECURRING_PATTERNS:
        count = max(1, int(len(records) * pattern["frequency"]))
        indices = random.sample(range(len(records)), min(count, len(records)))

        for idx in indices:
            rec = records[idx]

            if pattern["type"] == "location_cluster":
                rec["location_site"] = pattern["location"]
                loc_data = OIL_LOCATIONS[pattern["location"]]
                rec["location_state"] = loc_data["state"]
                rec["location_area"] = pattern["area"]
                rec["location_type"] = loc_data["type"]

            elif pattern["type"] == "activity_cluster":
                rec["activity_type"] = pattern["activity"]

            elif pattern["type"] == "contractor_cluster":
                rec["activity_type"] = pattern["activity"]
                rec["reporter_designation"] = "Contractor Foreman"

            elif pattern["type"] == "equipment_cluster":
                equip_candidates = [e for e in EQUIPMENT_TYPES if pattern["equipment_keyword"] in e]
                if equip_candidates:
                    rec["equipment_involved"] = random.choice(equip_candidates)

            elif pattern["type"] == "time_cluster":
                rec["shift"] = pattern["shift"]

            # Tag with pattern's rules if the record doesn't already have them
            existing_rules = set(rec["life_saving_rules"].split("|")) if rec["life_saving_rules"] != "None" else set()
            new_rules = existing_rules | set(pattern["rules"])
            rec["life_saving_rules"] = "|".join(sorted(new_rules))
            rec["life_saving_rules_count"] = len(new_rules)

            # Add a pattern marker for analysis
            rec["_recurring_pattern"] = pattern["description"]

    return records


# ──────────────────────────────────────────────────────────────
# DATASET VALIDATION
# ──────────────────────────────────────────────────────────────

def validate_dataset(records: list) -> dict:
    """Run quality checks on the generated dataset."""
    issues = []

    # Check for empty narratives
    empty_narratives = sum(1 for r in records if not r["free_text_description"].strip())
    if empty_narratives > 0:
        issues.append(f"{empty_narratives} records have empty narratives")

    # Check for duplicate report IDs
    ids = [r["report_id"] for r in records]
    if len(ids) != len(set(ids)):
        issues.append("Duplicate report IDs found")

    # Check SIF ratio
    sif_count = sum(1 for r in records if r["sif_potential_binary"] == 1)
    sif_ratio = sif_count / len(records)
    if sif_ratio < 0.25 or sif_ratio > 0.45:
        issues.append(f"SIF ratio {sif_ratio:.2%} is outside expected range (25-45%)")

    # Check narrative length distribution
    word_counts = [r["word_count"] for r in records]
    avg_words = sum(word_counts) / len(word_counts)

    # Check for valid dates
    date_issues = 0
    for r in records:
        try:
            datetime.fromisoformat(r["datetime_reported"])
        except ValueError:
            date_issues += 1
    if date_issues > 0:
        issues.append(f"{date_issues} records have invalid dates")

    validation = {
        "total_records": len(records),
        "sif_count": sif_count,
        "non_sif_count": len(records) - sif_count,
        "sif_ratio": round(sif_ratio, 4),
        "avg_word_count": round(avg_words, 1),
        "min_word_count": min(word_counts),
        "max_word_count": max(word_counts),
        "unique_locations": len(set(r["location_site"] for r in records)),
        "unique_activities": len(set(r["activity_type"] for r in records)),
        "unique_equipment": len(set(r["equipment_involved"] for r in records)),
        "issues": issues if issues else ["None — dataset passed all checks"],
    }

    return validation


# ──────────────────────────────────────────────────────────────
# OUTPUT WRITERS
# ──────────────────────────────────────────────────────────────

def write_csv(records: list, filepath: str):
    """Write main dataset CSV."""
    fieldnames = [
        "report_id", "date_reported", "time_reported", "datetime_reported",
        "year", "month", "quarter", "day_of_week",
        "report_type", "observation_category",
        "location_site", "location_state", "location_area", "location_type",
        "activity_type", "equipment_involved",
        "free_text_description", "word_count",
        "sif_potential", "sif_potential_binary", "sif_confidence_score",
        "severity_rationale", "risk_level",
        "life_saving_rules", "life_saving_rules_count",
        "evidence_snippets_json",
        "barrier_type", "barrier_status",
        "immediate_action_taken", "corrective_action_recommended",
        "reporter_designation", "reporter_name",
        "shift", "weather_conditions",
    ]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"  ✅ CSV written: {filepath} ({len(records):,} records)")


def write_jsonl(records: list, filepath: str):
    """Write JSON-lines for NLP pipeline ingestion."""
    fieldnames_to_include = [
        "report_id", "datetime_reported", "report_type",
        "observation_category", "location_site", "location_area",
        "activity_type", "equipment_involved",
        "free_text_description",
        "sif_potential", "sif_potential_binary", "sif_confidence_score",
        "life_saving_rules", "life_saving_rules_count",
        "evidence_snippets_json", "risk_level",
        "barrier_type", "barrier_status",
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        for rec in records:
            filtered = {k: rec[k] for k in fieldnames_to_include if k in rec}
            # Parse evidence JSON back to dict for cleaner JSONL
            if "evidence_snippets_json" in filtered:
                filtered["evidence_snippets"] = json.loads(filtered.pop("evidence_snippets_json"))
            # Parse rules to list
            if filtered["life_saving_rules"] == "None":
                filtered["life_saving_rules"] = []
            else:
                filtered["life_saving_rules"] = filtered["life_saving_rules"].split("|")
            f.write(json.dumps(filtered, ensure_ascii=False) + "\n")
    print(f"  ✅ JSONL written: {filepath} ({len(records):,} records)")


def write_life_saving_rules_table(records: list, filepath: str):
    """Write normalized many-to-many table: report_id ↔ life_saving_rule."""
    rows = []
    for rec in records:
        rules_str = rec["life_saving_rules"]
        if rules_str == "None":
            continue
        for rule in rules_str.split("|"):
            evidence_dict = json.loads(rec["evidence_snippets_json"])
            snippets = evidence_dict.get(rule, [])
            rows.append({
                "report_id": rec["report_id"],
                "life_saving_rule": rule.strip(),
                "evidence_snippet": " | ".join(snippets) if snippets else "",
                "sif_potential": rec["sif_potential"],
            })

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["report_id", "life_saving_rule", "evidence_snippet", "sif_potential"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  ✅ Life-Saving Rules table written: {filepath} ({len(rows):,} rows)")


def write_metadata(records: list, validation: dict, filepath: str, generation_time: float):
    """Write dataset metadata and statistics."""
    # Compute detailed statistics
    from collections import Counter

    location_dist = Counter(r["location_site"] for r in records)
    rule_counter = Counter()
    for r in records:
        if r["life_saving_rules"] != "None":
            for rule in r["life_saving_rules"].split("|"):
                rule_counter[rule.strip()] += 1

    report_type_dist = Counter(r["report_type"] for r in records)
    risk_dist = Counter(r["risk_level"] for r in records)
    barrier_status_dist = Counter(r["barrier_status"] for r in records)
    monthly_dist = Counter(r["datetime_reported"][:7] for r in records)

    metadata = {
        "dataset_name": "OIL India HSSE Safety Reports — Synthetic Dataset",
        "version": "1.0.0",
        "generated_at": datetime.now().isoformat(),
        "generation_time_seconds": round(generation_time, 2),
        "description": (
            "Synthetic dataset of Unsafe Act/Unsafe Condition observations, "
            "near-miss reports, and incident records for Oil India Limited's "
            "HSSE platform. Designed for training and evaluating AI/NLP models "
            "for SIF precursor detection."
        ),
        "schema": {
            "report_id": "Unique report identifier (format: OIL-HSSE-YYYYMMDD-NNNNN)",
            "date_reported": "Date of report (YYYY-MM-DD)",
            "time_reported": "Time of report (HH:MM:SS)",
            "datetime_reported": "ISO 8601 datetime",
            "year": "Year extracted from date",
            "month": "Month extracted from date (1-12)",
            "quarter": "Fiscal quarter (Q1-Q4)",
            "day_of_week": "Day name (Monday-Sunday)",
            "report_type": "Type: Unsafe Act, Unsafe Condition, Near Miss, Incident, Hazard Observation, Positive Observation",
            "observation_category": "HSE observation category",
            "location_site": "OIL India site name",
            "location_state": "Indian state",
            "location_area": "Specific area within site",
            "location_type": "Site type: Oilfield, Refinery, Pipeline, etc.",
            "activity_type": "Oil & gas activity being performed",
            "equipment_involved": "Equipment type involved",
            "free_text_description": "Full narrative text of the observation/report",
            "word_count": "Number of words in the narrative",
            "sif_potential": "Classification: SIF-Potential or Non-SIF-Potential",
            "sif_potential_binary": "Binary label: 1=SIF, 0=Non-SIF",
            "sif_confidence_score": "Simulated ML confidence score (0-1)",
            "severity_rationale": "Explanation of severity assessment",
            "risk_level": "Risk level: Critical, High, Medium, Low",
            "life_saving_rules": "Pipe-delimited IOGP Life-Saving Rules",
            "life_saving_rules_count": "Number of applicable rules",
            "evidence_snippets_json": "JSON dict mapping each rule to supporting text excerpts",
            "barrier_type": "Safety barrier type",
            "barrier_status": "Barrier status: Intact, Degraded, Failed, Absent, Bypassed",
            "immediate_action_taken": "Immediate corrective action",
            "corrective_action_recommended": "Recommended follow-up action",
            "reporter_designation": "Job title of reporter",
            "reporter_name": "Name of reporter (synthetic)",
            "shift": "Shift: Day or Night",
            "weather_conditions": "Weather at time of observation",
        },
        "statistics": {
            "total_records": len(records),
            "validation": validation,
            "distributions": {
                "sif_potential": {"SIF-Potential": validation["sif_count"], "Non-SIF-Potential": validation["non_sif_count"]},
                "report_type": dict(report_type_dist.most_common()),
                "risk_level": dict(risk_dist.most_common()),
                "location_site": dict(location_dist.most_common()),
                "life_saving_rules": dict(rule_counter.most_common()),
                "barrier_status": dict(barrier_status_dist.most_common()),
                "monthly_volume": dict(sorted(monthly_dist.items())),
            },
        },
        "intended_use": {
            "primary": "Training & evaluating NLP models for SIF precursor classification",
            "secondary": [
                "Life-Saving Rule auto-tagging with explainable evidence",
                "Recurring precursor pattern detection and clustering",
                "Time-series trend analysis of safety observations",
                "Dashboard prototyping for HSE decision support",
            ],
        },
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Metadata written: {filepath}")


# ──────────────────────────────────────────────────────────────
# CLI ENTRYPOINT
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic OIL India HSSE safety report dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_dataset.py                      # 5000 records, default settings
  python generate_dataset.py --num-records 10000  # 10k records
  python generate_dataset.py --seed 42            # reproducible generation
  python generate_dataset.py --sif-ratio 0.30     # 30% SIF-potential
  python generate_dataset.py --output-dir ./data/output
        """,
    )
    parser.add_argument("--num-records", type=int, default=5000, help="Number of records to generate (default: 5000)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--sif-ratio", type=float, default=0.35, help="Fraction of SIF-potential records (default: 0.35)")
    parser.add_argument("--start-date", type=str, default="2022-01-01", help="Start date for report range (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2025-12-31", help="End date for report range (YYYY-MM-DD)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: ./data/output)")

    args = parser.parse_args()

    # ── Setup ──
    if args.seed is not None:
        random.seed(args.seed)

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    output_dir = args.output_dir or str(Path(__file__).resolve().parent.parent / "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  OIL INDIA HSSE — SYNTHETIC DATASET GENERATOR")
    print("=" * 70)
    print(f"  Records to generate : {args.num_records:,}")
    print(f"  SIF ratio           : {args.sif_ratio:.0%}")
    print(f"  Date range          : {args.start_date} → {args.end_date}")
    print(f"  Random seed         : {args.seed or 'None (random)'}")
    print(f"  Output directory    : {output_dir}")
    print("=" * 70)

    # ── Generate ──
    print("\n🔄 Generating records...")
    gen_start = datetime.now()

    records = []
    for i in range(args.num_records):
        record = generate_single_record(
            record_index=i + 1,
            start_date=start_date,
            end_date=end_date,
            sif_ratio=args.sif_ratio,
        )
        records.append(record)

        if (i + 1) % 1000 == 0:
            print(f"  ... generated {i + 1:,} / {args.num_records:,} records")

    # ── Inject recurring patterns ──
    print("🔄 Injecting recurring precursor patterns...")
    records = inject_recurring_patterns(records)

    # ── Sort by date ──
    records.sort(key=lambda r: r["datetime_reported"])

    # ── Re-assign sequential IDs after sorting ──
    for i, rec in enumerate(records):
        rec["report_id"] = f"OIL-HSSE-{rec['date_reported'].replace('-', '')}-{i + 1:05d}"

    gen_time = (datetime.now() - gen_start).total_seconds()

    # ── Validate ──
    print("🔄 Validating dataset...")
    validation = validate_dataset(records)

    # ── Write outputs ──
    print("\n📁 Writing output files...")

    csv_path = os.path.join(output_dir, "oil_hsse_reports.csv")
    jsonl_path = os.path.join(output_dir, "oil_hsse_reports.jsonl")
    rules_path = os.path.join(output_dir, "oil_hsse_life_saving_rules.csv")
    meta_path = os.path.join(output_dir, "oil_hsse_reports_metadata.json")

    write_csv(records, csv_path)
    write_jsonl(records, jsonl_path)
    write_life_saving_rules_table(records, rules_path)
    write_metadata(records, validation, meta_path, gen_time)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  ✅ DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Total records        : {validation['total_records']:,}")
    print(f"  SIF-Potential        : {validation['sif_count']:,} ({validation['sif_ratio']:.1%})")
    print(f"  Non-SIF-Potential    : {validation['non_sif_count']:,} ({1 - validation['sif_ratio']:.1%})")
    print(f"  Avg. word count      : {validation['avg_word_count']}")
    print(f"  Word count range     : {validation['min_word_count']} – {validation['max_word_count']}")
    print(f"  Unique locations     : {validation['unique_locations']}")
    print(f"  Unique activities    : {validation['unique_activities']}")
    print(f"  Unique equipment     : {validation['unique_equipment']}")
    print(f"  Generation time      : {gen_time:.2f}s")
    print(f"  Quality issues       : {', '.join(validation['issues'])}")
    print("=" * 70)


if __name__ == "__main__":
    main()
