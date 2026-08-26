#!/usr/bin/env python3
"""
OIL HSSE Dataset — Cleaning, Preprocessing & Feature Engineering Pipeline
==========================================================================
Takes the raw generated CSV and produces:
  1. A cleaned & preprocessed master CSV
  2. Train / Validation / Test splits (70/15/15) — stratified on SIF label
  3. A multi-label Life-Saving-Rule classification table
  4. Tokenized-ready JSONL files for NLP model ingestion
  5. Feature-engineered columns for dashboard analytics

Usage:
  python preprocess_pipeline.py                            # defaults
  python preprocess_pipeline.py --input ./data/output/oil_hsse_reports.csv
  python preprocess_pipeline.py --output-dir ./data/processed
  python preprocess_pipeline.py --augment                  # enable text augmentation

Author: SIH-2026 Team
"""

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────

ALL_LIFE_SAVING_RULES = [
    "Bypassing Safety Controls",
    "Confined Space",
    "Driving",
    "Energy Isolation",
    "Hot Work",
    "Line of Fire",
    "Safe Mechanical Lifting",
    "Work Authorization",
    "Working at Height",
]

RISK_LEVEL_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}

BARRIER_STATUS_MAP = {
    "Intact – Functioning as designed": 0,
    "Degraded – Partially effective": 1,
    "Not Applicable": 2,
    "Failed – Not functioning": 3,
    "Absent – Not installed / Not in place": 4,
    "Bypassed – Intentionally defeated": 5,
}

# Oil & Gas domain stop-words to KEEP (important for SIF detection)
DOMAIN_KEEP_WORDS = {
    "not", "no", "without", "never", "nor", "none",  # negations are critical
    "failed", "absent", "bypassed", "missing", "expired",
    "unauthorized", "invalid", "defective", "damaged",
    "under", "over", "above", "below", "near", "within",
}

# Common English stop-words to REMOVE (minus domain-critical ones)
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "am", "its", "it", "i",
    "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "this", "that", "these",
    "those", "such", "here", "there", "when", "where", "why", "how",
    "all", "both", "each", "few", "more", "most", "other", "some",
    "any", "only", "own", "same", "so", "than", "too", "very",
    "just", "also", "now", "then", "once", "again", "further",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "between",
    "about", "against", "and", "but", "or", "if", "while", "because",
    "until", "although", "though", "since", "unless",
}

# Remove domain-critical words from the stop-word set
STOP_WORDS = STOP_WORDS - DOMAIN_KEEP_WORDS


# ──────────────────────────────────────────────────────────────
# TEXT CLEANING
# ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean free-text narrative for NLP processing.
    Preserves domain-critical terms, negations, measurements.
    """
    if not text or not text.strip():
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # 2. Standardize common abbreviations in oil & gas domain
    abbreviation_map = {
        r'\bLOTO\b': 'lockout tagout',
        r'\bPTW\b': 'permit to work',
        r'\bJSA\b': 'job safety analysis',
        r'\bSCBA\b': 'self contained breathing apparatus',
        r'\bLEL\b': 'lower explosive limit',
        r'\bH2S\b': 'hydrogen sulfide',
        r'\bBOP\b': 'blowout preventer',
        r'\bESD\b': 'emergency shutdown',
        r'\bMEWP\b': 'mobile elevating work platform',
        r'\bSWL\b': 'safe working load',
        r'\bPSV\b': 'pressure safety valve',
        r'\bSIS\b': 'safety instrumented system',
        r'\bSCE\b': 'safety critical element',
        r'\bMOC\b': 'management of change',
        r'\bMCC\b': 'motor control center',
        r'\bVFD\b': 'variable frequency drive',
        r'\bRCA\b': 'root cause analysis',
        r'\bMAWP\b': 'maximum allowable working pressure',
        r'\bPPE\b': 'personal protective equipment',
        r'\bHPSD\b': 'high pressure shutdown',
        r'\bLMI\b': 'load moment indicator',
        r'\bSIMOPS\b': 'simultaneous operations',
        r'\bCDU\b': 'crude distillation unit',
        r'\bEPS\b': 'early production system',
        r'\bGGS\b': 'group gathering station',
        r'\bGCS\b': 'gas collecting station',
        r'\bO2\b': 'oxygen',
        r'\bppm\b': 'parts per million',
        r'\bpsi\b': 'pounds per square inch',
    }
    for pattern, replacement in abbreviation_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # 3. Normalize measurements (keep numbers + units together)
    text = re.sub(r'(\d+)\s*(meters?|m)\b', r'\1_meters', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*(km/?h)\b', r'\1_kmh', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*(%)\b', r'\1_percent', text)
    text = re.sub(r'(\d+)\s*(°C)\b', r'\1_celsius', text)
    text = re.sub(r'(\d+)\s*(kg)\b', r'\1_kg', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*(ton[s]?)\b', r'\1_tons', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*(V)\b', r'\1_volts', text)
    text = re.sub(r'(\d+)\s*(dBA)\b', r'\1_dba', text, flags=re.IGNORECASE)

    # 4. Normalize special characters
    text = re.sub(r'[–—]', '-', text)  # em/en dashes to hyphens
    text = re.sub(r'[""''‛‟]', '"', text)  # smart quotes to regular
    text = re.sub(r'[′′]', "'", text)

    # 5. Remove report template artifacts
    text = re.sub(r'\bOIL-HSSE-\d{8}-\d{5}\b', '', text)  # report IDs
    text = re.sub(r'\bFE-\d{3}\b', 'fire_extinguisher', text)  # equipment codes
    text = re.sub(r'\bOIL-\d{3}\b', '', text)  # vehicle numbers

    # 6. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_for_nlp(text: str, remove_stopwords: bool = False) -> str:
    """
    Tokenize cleaned text for NLP model input.
    Lowercases, optionally removes stop-words.
    """
    text = text.lower()

    # Keep alphanumeric, underscores (for compound terms), hyphens in compounds
    tokens = re.findall(r'[a-z0-9_]+(?:-[a-z0-9_]+)*', text)

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS]

    return " ".join(tokens)


# ──────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────

def extract_text_features(narrative: str) -> Dict:
    """Extract NLP-relevant features from free-text narrative."""
    text_lower = narrative.lower()

    # Sentence count
    sentences = re.split(r'[.!?]+', narrative)
    sentence_count = len([s for s in sentences if s.strip()])

    # Average sentence length
    word_counts = [len(s.split()) for s in sentences if s.strip()]
    avg_sentence_len = sum(word_counts) / max(len(word_counts), 1)

    # Negation count (critical for SIF detection)
    negation_words = ["not", "no", "without", "never", "nor", "none",
                       "didn't", "did not", "wasn't", "was not",
                       "weren't", "were not", "hadn't", "had not",
                       "hasn't", "has not", "haven't", "have not",
                       "couldn't", "could not", "wouldn't", "would not"]
    negation_count = sum(1 for w in negation_words if w in text_lower)

    # Severity indicator words
    severity_high_words = [
        "fatal", "fatality", "death", "killed", "deceased", "died",
        "explosion", "blast", "catastrophic", "collapse",
        "amputation", "severed", "crushed", "electrocuted",
        "unconscious", "unresponsive", "critical condition",
        "multiple casualties", "mass casualty",
    ]
    severity_medium_words = [
        "fracture", "burn", "injury", "hospitalized", "medical treatment",
        "struck", "fell", "dropped", "trapped", "pinned",
        "exposure", "inhalation", "contamination",
        "fire", "ignition", "smoke", "leak", "spill",
    ]
    severity_low_words = [
        "minor", "first aid", "bruise", "scratch", "discomfort",
        "observation", "noted", "reminder", "awareness",
        "housekeeping", "hygiene", "welfare", "administrative",
    ]

    high_severity_count = sum(1 for w in severity_high_words if w in text_lower)
    medium_severity_count = sum(1 for w in severity_medium_words if w in text_lower)
    low_severity_count = sum(1 for w in severity_low_words if w in text_lower)

    # Barrier failure indicators
    barrier_failure_words = [
        "failed", "failure", "absent", "missing", "bypassed",
        "disabled", "defeated", "overridden", "not in place",
        "not installed", "expired", "non-functional",
        "not functioning", "out of service", "damaged",
        "not applied", "not verified", "not conducted",
        "not performed", "not maintained", "not present",
    ]
    barrier_failure_count = sum(1 for w in barrier_failure_words if w in text_lower)

    # Compliance violation indicators
    violation_words = [
        "without", "unauthorized", "invalid", "expired permit",
        "not wearing", "removed", "did not", "was not",
        "had not", "no valid", "not secured", "not locked",
        "not signed", "blank", "incomplete", "overdue",
    ]
    violation_count = sum(1 for w in violation_words if w in text_lower)

    # Presence of measurements (quantitative reporting = higher quality report)
    has_measurements = bool(re.search(r'\d+\s*(?:meters?|m|kg|tons?|psi|ppm|%|°C|V|dBA|km/?h)', text_lower))

    # Presence of numbers (quantitative detail)
    number_count = len(re.findall(r'\b\d+(?:\.\d+)?\b', narrative))

    # Contains positive observation keywords
    is_positive = any(w in text_lower for w in [
        "positive observation", "exemplary", "commended", "safe work practice",
        "proactive", "good practice", "best practice",
    ])

    return {
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_len, 1),
        "negation_count": negation_count,
        "high_severity_word_count": high_severity_count,
        "medium_severity_word_count": medium_severity_count,
        "low_severity_word_count": low_severity_count,
        "severity_word_ratio": round(
            (high_severity_count * 3 + medium_severity_count * 2 + low_severity_count) /
            max(len(narrative.split()), 1), 4
        ),
        "barrier_failure_count": barrier_failure_count,
        "violation_count": violation_count,
        "has_measurements": int(has_measurements),
        "number_count": number_count,
        "is_positive_observation": int(is_positive),
    }


def create_multi_label_columns(rules_str: str) -> Dict[str, int]:
    """Convert pipe-delimited rule string to binary multi-label columns."""
    active_rules = set()
    if rules_str and rules_str != "None":
        active_rules = {r.strip() for r in rules_str.split("|")}

    return {
        f"rule_{rule.lower().replace(' ', '_')}": int(rule in active_rules)
        for rule in ALL_LIFE_SAVING_RULES
    }


def encode_categorical(record: Dict) -> Dict:
    """Encode categorical features as integers."""
    encodings = {}

    # Risk level ordinal encoding
    encodings["risk_level_encoded"] = RISK_LEVEL_MAP.get(record.get("risk_level", ""), -1)

    # Barrier status ordinal encoding
    encodings["barrier_status_encoded"] = BARRIER_STATUS_MAP.get(record.get("barrier_status", ""), -1)

    # Time-based features
    try:
        dt = datetime.fromisoformat(record["datetime_reported"])
        encodings["hour_of_day"] = dt.hour
        encodings["is_night_shift"] = int(dt.hour < 6 or dt.hour >= 18)
        encodings["is_weekend"] = int(dt.weekday() >= 5)
        encodings["month_sin"] = round(__import__("math").sin(2 * __import__("math").pi * dt.month / 12), 4)
        encodings["month_cos"] = round(__import__("math").cos(2 * __import__("math").pi * dt.month / 12), 4)
        encodings["is_monsoon"] = int(dt.month in (6, 7, 8, 9))
    except (ValueError, KeyError):
        encodings.update({
            "hour_of_day": -1, "is_night_shift": -1, "is_weekend": -1,
            "month_sin": 0, "month_cos": 0, "is_monsoon": -1,
        })

    return encodings


# ──────────────────────────────────────────────────────────────
# DATA QUALITY CHECKS & CLEANING
# ──────────────────────────────────────────────────────────────

def quality_check_record(record: Dict) -> Tuple[Dict, List[str]]:
    """
    Apply data quality checks and fixes. Returns (cleaned_record, issues_list).
    """
    issues = []

    # 1. Check for empty or too-short narratives
    narrative = record.get("free_text_description", "")
    if not narrative.strip():
        issues.append("EMPTY_NARRATIVE")
    elif len(narrative.split()) < 10:
        issues.append("SHORT_NARRATIVE")

    # 2. Validate date
    try:
        dt = datetime.fromisoformat(record["datetime_reported"])
        if dt.year < 2020 or dt.year > 2027:
            issues.append("DATE_OUT_OF_RANGE")
    except (ValueError, KeyError):
        issues.append("INVALID_DATE")

    # 3. Check SIF label consistency
    sif_label = record.get("sif_potential", "")
    sif_binary = record.get("sif_potential_binary", "")
    try:
        sif_binary = int(sif_binary)
    except (ValueError, TypeError):
        sif_binary = -1

    if sif_label == "SIF-Potential" and sif_binary != 1:
        record["sif_potential_binary"] = 1
        issues.append("SIF_LABEL_MISMATCH_FIXED")
    elif sif_label == "Non-SIF-Potential" and sif_binary != 0:
        record["sif_potential_binary"] = 0
        issues.append("SIF_LABEL_MISMATCH_FIXED")

    # 4. Validate life-saving rules
    rules_str = record.get("life_saving_rules", "None")
    if rules_str != "None":
        rules = [r.strip() for r in rules_str.split("|")]
        valid_rules = [r for r in rules if r in ALL_LIFE_SAVING_RULES]
        invalid_rules = [r for r in rules if r not in ALL_LIFE_SAVING_RULES]
        if invalid_rules:
            issues.append(f"INVALID_RULES: {invalid_rules}")
            record["life_saving_rules"] = "|".join(valid_rules) if valid_rules else "None"
            record["life_saving_rules_count"] = len(valid_rules)

    # 5. Validate word count matches actual
    actual_word_count = len(narrative.split())
    recorded_word_count = int(record.get("word_count", 0))
    if abs(actual_word_count - recorded_word_count) > 5:
        record["word_count"] = actual_word_count
        issues.append("WORD_COUNT_CORRECTED")

    # 6. Ensure confidence score is in valid range
    try:
        conf = float(record.get("sif_confidence_score", 0))
        if conf < 0 or conf > 1:
            record["sif_confidence_score"] = max(0, min(1, conf))
            issues.append("CONFIDENCE_CLAMPED")
    except (ValueError, TypeError):
        record["sif_confidence_score"] = 0.5
        issues.append("CONFIDENCE_DEFAULTED")

    return record, issues


# ──────────────────────────────────────────────────────────────
# TEXT AUGMENTATION (optional, for training data expansion)
# ──────────────────────────────────────────────────────────────

def synonym_augment(text: str) -> str:
    """
    Simple synonym replacement for data augmentation.
    Replaces common safety terms with synonyms to increase text diversity.
    """
    synonym_map = {
        "worker": ["personnel", "employee", "technician", "crew member"],
        "observed": ["noted", "witnessed", "identified", "found"],
        "immediately": ["promptly", "at once", "right away", "without delay"],
        "hazard": ["danger", "risk", "threat", "peril"],
        "defective": ["faulty", "damaged", "malfunctioning", "impaired"],
        "conducted": ["performed", "carried out", "executed", "undertaken"],
        "area": ["zone", "location", "site", "vicinity"],
        "inspection": ["examination", "check", "audit", "review"],
        "violation": ["breach", "infringement", "non-compliance", "transgression"],
        "equipment": ["machinery", "apparatus", "device", "tool"],
    }

    words = text.split()
    augmented = []
    for word in words:
        word_lower = word.lower().strip(".,;:!?")
        if word_lower in synonym_map and __import__("random").random() < 0.3:
            replacement = __import__("random").choice(synonym_map[word_lower])
            # Preserve original casing
            if word[0].isupper():
                replacement = replacement.capitalize()
            augmented.append(replacement)
        else:
            augmented.append(word)
    return " ".join(augmented)


# ──────────────────────────────────────────────────────────────
# STRATIFIED TRAIN/VAL/TEST SPLIT
# ──────────────────────────────────────────────────────────────

def stratified_split(
    records: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Stratified split on sif_potential_binary to maintain label distribution.
    """
    import random as rng
    rng.seed(seed)

    # Separate by class
    sif_records = [r for r in records if int(r["sif_potential_binary"]) == 1]
    non_sif_records = [r for r in records if int(r["sif_potential_binary"]) == 0]

    rng.shuffle(sif_records)
    rng.shuffle(non_sif_records)

    def split_list(lst):
        n = len(lst)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        return lst[:train_end], lst[train_end:val_end], lst[val_end:]

    sif_train, sif_val, sif_test = split_list(sif_records)
    nonsif_train, nonsif_val, nonsif_test = split_list(non_sif_records)

    train = sif_train + nonsif_train
    val = sif_val + nonsif_val
    test = sif_test + nonsif_test

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ──────────────────────────────────────────────────────────────
# PIPELINE ORCHESTRATOR
# ──────────────────────────────────────────────────────────────

def run_pipeline(input_path: str, output_dir: str, augment: bool = False, seed: int = 42):
    """Run the full preprocessing pipeline."""

    print("=" * 70)
    print("  OIL HSSE — DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"  Input  : {input_path}")
    print(f"  Output : {output_dir}")
    print(f"  Augment: {augment}")
    print(f"  Seed   : {seed}")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # ── Step 1: Load raw data ──
    print("\n📂 Step 1: Loading raw data...")
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_records = list(reader)
    print(f"   Loaded {len(raw_records):,} records")

    # ── Step 2: Data quality checks ──
    print("\n🔍 Step 2: Running data quality checks...")
    all_issues = Counter()
    cleaned_records = []
    dropped_count = 0

    for rec in raw_records:
        rec, issues = quality_check_record(rec)
        for issue in issues:
            all_issues[issue] += 1

        # Drop records with critical issues
        if "EMPTY_NARRATIVE" in issues or "INVALID_DATE" in issues:
            dropped_count += 1
            continue

        cleaned_records.append(rec)

    print(f"   Quality issues found:")
    for issue, count in all_issues.most_common():
        print(f"     {issue}: {count}")
    print(f"   Dropped {dropped_count} records with critical issues")
    print(f"   Remaining: {len(cleaned_records):,} records")

    # ── Step 3: Text cleaning ──
    print("\n🧹 Step 3: Cleaning text narratives...")
    for rec in cleaned_records:
        raw_text = rec["free_text_description"]
        rec["text_cleaned"] = clean_text(raw_text)
        rec["text_tokenized"] = tokenize_for_nlp(rec["text_cleaned"])
        rec["text_tokenized_no_stopwords"] = tokenize_for_nlp(rec["text_cleaned"], remove_stopwords=True)
        rec["word_count_cleaned"] = len(rec["text_tokenized"].split())

    avg_words = sum(r["word_count_cleaned"] for r in cleaned_records) / len(cleaned_records)
    print(f"   Average word count (cleaned): {avg_words:.1f}")

    # ── Step 4: Feature engineering ──
    print("\n⚙️  Step 4: Engineering features...")
    for rec in cleaned_records:
        # Text-derived features
        text_features = extract_text_features(rec["free_text_description"])
        rec.update(text_features)

        # Multi-label rule columns
        rule_columns = create_multi_label_columns(rec.get("life_saving_rules", "None"))
        rec.update(rule_columns)

        # Categorical encodings
        cat_features = encode_categorical(rec)
        rec.update(cat_features)

    feature_cols = list(text_features.keys()) + list(rule_columns.keys()) + list(cat_features.keys())
    print(f"   Added {len(feature_cols)} engineered features")

    # ── Step 5: Text augmentation (optional) ──
    if augment:
        print("\n📝 Step 5: Augmenting SIF-potential texts...")
        sif_records = [r for r in cleaned_records if int(r["sif_potential_binary"]) == 1]
        augmented_records = []

        for rec in sif_records:
            aug_rec = rec.copy()
            aug_rec["text_cleaned"] = synonym_augment(rec["text_cleaned"])
            aug_rec["text_tokenized"] = tokenize_for_nlp(aug_rec["text_cleaned"])
            aug_rec["text_tokenized_no_stopwords"] = tokenize_for_nlp(aug_rec["text_cleaned"], remove_stopwords=True)
            aug_rec["report_id"] = rec["report_id"] + "-AUG"
            aug_rec["is_augmented"] = 1
            augmented_records.append(aug_rec)

        for rec in cleaned_records:
            rec["is_augmented"] = 0

        cleaned_records.extend(augmented_records)
        print(f"   Added {len(augmented_records)} augmented records")
        print(f"   Total records: {len(cleaned_records):,}")
    else:
        for rec in cleaned_records:
            rec["is_augmented"] = 0
        print("\n⏩ Step 5: Skipping augmentation (use --augment to enable)")

    # ── Step 6: Stratified train/val/test split ──
    print("\n✂️  Step 6: Splitting into train/val/test...")
    # Only use non-augmented records for split assignment
    non_aug = [r for r in cleaned_records if r["is_augmented"] == 0]
    aug_only = [r for r in cleaned_records if r["is_augmented"] == 1]

    train, val, test = stratified_split(non_aug, seed=seed)

    # Add augmented records to training set only
    train_ids = {r["report_id"] for r in train}
    for aug_rec in aug_only:
        orig_id = aug_rec["report_id"].replace("-AUG", "")
        if orig_id in train_ids:
            train.append(aug_rec)

    # Assign split labels
    for r in train:
        r["split"] = "train"
    for r in val:
        r["split"] = "val"
    for r in test:
        r["split"] = "test"

    all_records = train + val + test

    def _sif_pct(recs):
        sif = sum(1 for r in recs if int(r["sif_potential_binary"]) == 1)
        return f"{sif}/{len(recs)} ({sif/len(recs)*100:.1f}%)"

    print(f"   Train : {len(train):,} records — SIF: {_sif_pct(train)}")
    print(f"   Val   : {len(val):,}  records — SIF: {_sif_pct(val)}")
    print(f"   Test  : {len(test):,}  records — SIF: {_sif_pct(test)}")

    # ── Step 7: Write outputs ──
    print("\n💾 Step 7: Writing output files...")

    # 7a. Full preprocessed CSV
    output_fields = [
        "report_id", "split", "datetime_reported", "year", "month", "quarter", "day_of_week",
        "report_type", "observation_category",
        "location_site", "location_state", "location_area", "location_type",
        "activity_type", "equipment_involved",
        "free_text_description", "text_cleaned", "text_tokenized", "text_tokenized_no_stopwords",
        "word_count", "word_count_cleaned",
        # ── Target variables ──
        "sif_potential", "sif_potential_binary", "sif_confidence_score",
        "life_saving_rules", "life_saving_rules_count",
        # ── Multi-label rule columns ──
        "rule_bypassing_safety_controls", "rule_confined_space", "rule_driving",
        "rule_energy_isolation", "rule_hot_work", "rule_line_of_fire",
        "rule_safe_mechanical_lifting", "rule_work_authorization", "rule_working_at_height",
        # ── Engineered features ──
        "sentence_count", "avg_sentence_length", "negation_count",
        "high_severity_word_count", "medium_severity_word_count", "low_severity_word_count",
        "severity_word_ratio", "barrier_failure_count", "violation_count",
        "has_measurements", "number_count", "is_positive_observation",
        # ── Categorical encodings ──
        "risk_level", "risk_level_encoded",
        "barrier_type", "barrier_status", "barrier_status_encoded",
        "hour_of_day", "is_night_shift", "is_weekend", "month_sin", "month_cos", "is_monsoon",
        # ── Metadata ──
        "severity_rationale", "reporter_designation", "shift", "weather_conditions",
        "is_augmented",
    ]

    csv_path = os.path.join(output_dir, "oil_hsse_preprocessed.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_records)
    print(f"   ✅ Full preprocessed CSV: {csv_path}")

    # 7b. Split-specific CSVs
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(output_dir, f"oil_hsse_{split_name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(split_data)
        print(f"   ✅ {split_name.upper()} CSV: {path} ({len(split_data):,} records)")

    # 7c. NLP-ready JSONL files
    nlp_fields = [
        "report_id", "text_tokenized", "text_tokenized_no_stopwords",
        "sif_potential_binary", "life_saving_rules",
        "rule_bypassing_safety_controls", "rule_confined_space", "rule_driving",
        "rule_energy_isolation", "rule_hot_work", "rule_line_of_fire",
        "rule_safe_mechanical_lifting", "rule_work_authorization", "rule_working_at_height",
    ]

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(output_dir, f"oil_hsse_{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in split_data:
                row = {}
                for field in nlp_fields:
                    val = rec.get(field, "")
                    # Convert binary fields to int
                    if field.startswith("rule_") or field == "sif_potential_binary":
                        try:
                            val = int(val)
                        except (ValueError, TypeError):
                            val = 0
                    row[field] = val
                # Convert life_saving_rules to list
                if row["life_saving_rules"] == "None":
                    row["life_saving_rules"] = []
                else:
                    row["life_saving_rules"] = row["life_saving_rules"].split("|")
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"   ✅ {split_name.upper()} JSONL: {path}")

    # 7d. Preprocessing report
    report = {
        "pipeline_version": "2.0.0",
        "run_at": datetime.now().isoformat(),
        "input_file": input_path,
        "total_raw_records": len(raw_records),
        "total_after_cleaning": len(non_aug),
        "dropped_records": dropped_count,
        "augmented_records": len(aug_only),
        "quality_issues": dict(all_issues.most_common()),
        "splits": {
            "train": {
                "count": len(train),
                "sif_count": sum(1 for r in train if int(r["sif_potential_binary"]) == 1),
                "non_sif_count": sum(1 for r in train if int(r["sif_potential_binary"]) == 0),
            },
            "val": {
                "count": len(val),
                "sif_count": sum(1 for r in val if int(r["sif_potential_binary"]) == 1),
                "non_sif_count": sum(1 for r in val if int(r["sif_potential_binary"]) == 0),
            },
            "test": {
                "count": len(test),
                "sif_count": sum(1 for r in test if int(r["sif_potential_binary"]) == 1),
                "non_sif_count": sum(1 for r in test if int(r["sif_potential_binary"]) == 0),
            },
        },
        "feature_columns": feature_cols,
        "text_stats": {
            "avg_word_count_raw": round(sum(int(r["word_count"]) for r in non_aug) / len(non_aug), 1),
            "avg_word_count_cleaned": round(sum(r["word_count_cleaned"] for r in non_aug) / len(non_aug), 1),
        },
    }

    report_path = os.path.join(output_dir, "preprocessing_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"   ✅ Preprocessing report: {report_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  ✅ PREPROCESSING PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total output records   : {len(all_records):,}")
    print(f"  Quality issues fixed   : {sum(all_issues.values())}")
    print(f"  Features engineered    : {len(feature_cols)}")
    print(f"  Output files           : {output_dir}/")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OIL HSSE Data Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str,
        default=str(Path(__file__).resolve().parent / "output" / "oil_hsse_reports.csv"),
        help="Path to raw CSV (default: data/output/oil_hsse_reports.csv)",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(Path(__file__).resolve().parent / "processed"),
        help="Output directory (default: data/processed/)",
    )
    parser.add_argument("--augment", action="store_true", help="Enable synonym augmentation for SIF class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    run_pipeline(args.input, args.output_dir, augment=args.augment, seed=args.seed)


if __name__ == "__main__":
    main()
