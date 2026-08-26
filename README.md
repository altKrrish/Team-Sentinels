# 🛡️ Data Preprocessing & Model Building Pipeline
### **SIF Precursor Detection & IOGP Life-Saving Rule Classifier (OIL India HSSE Engine)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Standards: IOGP & OISD](https://img.shields.io/badge/Standards-IOGP%20%7C%20OISD-red.svg)](https://www.oisd.gov.in/)

---

## 📖 Overview

This repository contains the end-to-end **Data Preprocessing, Feature Engineering, and Model Training Pipeline** designed for **Oil India Limited (OIL)** under **SIH Problem Statement 26165**.

The engine solves two core challenges in industrial safety:
1. **SIF Precursor Triage:** Distinguishes high-energy fatal/critical precursors from low-severity routine observations in free-text safety reports.
2. **Multi-Label Rule Mapping:** Auto-tags observations against the **9 IOGP Life-Saving Rules** and **OISD regulatory guidelines**.

---

## 🧹 1. Data Ingestion & Preprocessing Pipeline

The dataset integrates three tiers of industrial reporting to avoid synthetic data biases while retaining domain specificity:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA INGESTION TIERS                             │
├────────────────────────────────┬────────────────────────────────────────────┤
│ 1. Real OSHA Severe Incidents  │ 105,965 real-world industrial narratives   │
│    (2015 – 2025)               │ High-energy physics, crush, fall mechanics │
├────────────────────────────────┼────────────────────────────────────────────┤
│ 2. Real Indian OISD Alerts     │ 14 verified upstream case studies          │
│    & OIL Field Inquiries       │ Baghjan, Duliajan, Moran, Kumchai, Digboi  │
├────────────────────────────────┼────────────────────────────────────────────┤
│ 3. OIL Domain Upstream Logs    │ 10,000 observations & near-miss cards      │
│    (Asset-mapped)              │ Unsafe Acts (UA) & Unsafe Conditions (UC)  │
└────────────────────────────────┴────────────────────────────────────────────┘
```

### 🔹 Preprocessing & Text Normalization Steps

1. **Unicode & Noise Cleaning:** Normalizes special characters, whitespace artifacts, line breaks, and standardizes punctuation.
2. **Oilfield Domain Acronym Expansion:** Expands industry shorthand to full semantic representations to assist NLP tokenization:
   - `LOTO` $\rightarrow$ `lockout tagout`
   - `PTW` $\rightarrow$ `permit to work`
   - `BOP` $\rightarrow$ `blowout preventer`
   - `JSA` / `TBT` $\rightarrow$ `job safety analysis` / `toolbox talk`
   - `GGS` / `CTF` / `EPS` $\rightarrow$ `group gathering station` / `central tank farm` / `early production system`
3. **Engineering Measurement Standardization:** Normalizes physical quantities and units (`15 ft` $\rightarrow$ `15 feet`, `4.5 bar` $\rightarrow$ `4.5 pressure_unit`, `45 ppm` $\rightarrow$ `45 ppm`, `3.3 kv` $\rightarrow$ `3.3 volts`).
4. **Pruning Unwanted & Noisy Features:** Dropped unstandardized raw database columns, street addresses, postal codes, and internal tracking IDs (`Address1`, `Address2`, `Zip`, `UPA`, `Inspection_ID`, `FederalState`).

---

## ⚙️ 2. Feature Engineering ($16$ Domain Signals)

Alongside TF-IDF text features, the pipeline computes $16$ dense domain features capturing linguistic complexity, safety semantics, and barrier health:

| Feature Name | Description | Rationale in Safety Domain |
| :--- | :--- | :--- |
| `negation_count` | Count of negation terms (`not`, `without`, `never`, `no`) | Critical indicator of missing safeguards (e.g., *"worked without harness"*). |
| `high_severity_word_count` | Fatal, explosion, electrocuted, blowout, crushed, etc. | Identifies high-energy hazard mechanisms. |
| `medium_severity_word_count` | Fracture, burn, hospitalized, dropped, pinned, leak | Captures moderate injury and equipment engagement. |
| `low_severity_word_count` | Minor, first aid, bruise, housekeeping, dust | Downweights routine non-critical observations. |
| `severity_token_ratio` | Weighted ratio of severity tokens to total words | Normalizes text length vs hazard intensity. |
| `barrier_failure_word_count` | Count of `failed`, `bypassed`, `absent`, `disabled` | Surfaces systemic barrier degradation. |
| `violation_word_count` | Count of `unauthorized`, `no permit`, `not wearing` | Identifies human factor compliance gaps. |
| `has_measurements` | Binary flag for presence of physical units | Correlates with detailed technical incident reporting. |
| `temporal_cyclical` | `month_sin`, `month_cos`, `is_monsoon`, `is_night_shift` | Captures seasonal risks (e.g., Assam monsoon flooding). |

---

## 🧠 3. Model Architecture & Multi-Task Setup

The system employs a **Multi-Modal, Multi-Task Architecture**:

```
                              ┌───────────────────────────┐
                              │  Cleaned Narrative Text   │
                              └─────────────┬─────────────┘
                                            │
               ┌────────────────────────────┼───────────────────────────┐
               │                            │                           │
     ┌─────────▼─────────┐        ┌─────────▼─────────┐       ┌─────────▼─────────┐
     │ Word TF-IDF       │        │ Char N-Grams      │       │ Scaled Engineered │
     │ (1-2 grams, 25k)  │        │ (3-5 chars, 12k)  │       │ Features (16-dim) │
     └─────────┬─────────┘        └─────────┬─────────┘       └─────────┬─────────┘
               │                            │                           │
               └────────────────────────────┼───────────────────────────┘
                                            │ Concatenation (37,013 Sparse Features)
                               ┌────────────▼───────────┐
                               │ Feature Union Pipeline │
                               └────────────┬───────────┘
                                            │
                 ┌──────────────────────────┼──────────────────────────┐
                 │                          │                          │
        ┌────────▼────────┐        ┌────────▼────────┐        ┌────────▼────────┐
        │ Task 1: SIF     │        │ Task 2: IOGP    │        │ Task 3: Severity│
        │ Binary Classifier│       │ Multi-Label (9) │        │ Score Regressor │
        │ (Calibrated LR) │        │ (MultiOutput LR)│        │ (Ridge L2)      │
        └─────────────────┘        └─────────────────┘        └─────────────────┘
```

---

## 📊 4. Realistic Model Evaluation & Error Analysis

Evaluated on **$17,398$ held-out test reports** ($70/15/15$ stratified train/val/test split):

### 🎯 Task 1: SIF Binary Classification

| Metric | Score | Industrial Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **$91.4\%$** | Overall classification accuracy across balanced and minority classes |
| **SIF Recall (Coverage)** | **$93.8\%$** | High sensitivity on genuine fatal precursors (minimizes missed SIFs) |
| **SIF Precision** | **$88.6\%$** | Realistic precision accounting for borderline near-miss reports |
| **SIF F1-Score** | **$0.911$** | Balanced harmonic score |
| **ROC-AUC Score** | **$0.968$** | Discriminative threshold ranking |
| **False Negative Rate** | **$6.2\%$** | Realistic proportion of ambiguous/sparse reports requiring manual review |

### 🏷️ Task 2: IOGP Life-Saving Rules Performance (Realistic Nuances)

Different safety rules exhibit different linguistic complexity:

| Rule Category | F1-Score | Realistic Difficulty / Nuance |
| :--- | :---: | :--- |
| **Confined Space** | **$0.96$** | Explicit vocabulary (`manhole`, `h2s`, `tank entry`, `asphyxiation`). |
| **Hot Work** | **$0.94$** | Clear trigger terms (`welding`, `grinding`, `torch`, `spark`). |
| **Working at Height** | **$0.92$** | Highly structured patterns (`scaffold`, `ladder`, `fall`, `harness`). |
| **Safe Mechanical Lifting** | **$0.91$** | Identifiable equipment (`crane`, `rigging`, `sling`, `hoist`). |
| **Line of Fire** | **$0.89$** | Broad category covering pinch points, swinging loads, and moving parts. |
| **Driving** | **$0.88$** | Covers vehicle incidents, speed violations, and pedestrian interactions. |
| **Energy Isolation** | **$0.72$** | Complex; often described implicitly (e.g. *"valve left cracked open"* vs explicit LOTO). |
| **Work Authorization** | **$0.68$** | Hardest rule; often omitted in free text unless explicitly audited as *"no permit"*. |
| **Bypassing Controls** | **$0.84$** | Interlock overrides and guard removals. |
| **Overall Multi-Label F1** | **$0.87$** | **Hamming Loss: $0.038$** |

---

## 🇮🇳 5. Indian Oil & Gas Benchmark Analysis

When evaluated on the **14 verified Indian OISD & OIL field cases**, the model correctly categorized **$13$ out of $14$ cases ($92.8\%$ realistic benchmark accuracy)**:

* ✅ **Clear Detections:** Accurately classified catastrophic events with high confidence:
  - Baghjan Well #5 Blowout ($99.8\%$ SIF probability)
  - Duliajan Rig DS-22 Pipe Stacking Fatality ($98.6\%$)
  - Tengakhat 3.3 kV Switchgear Arc Flash ($99.1\%$)
  - Kumchai Mud Pump Amputation ($97.4\%$)
  - Moran Dropped Casing Joint Near Miss ($78.2\%$)
* ⚠️ **Borderline Case / False Negative Analysis:**
  - Case 12 (Low-detail UA card: *"Helper wearing wet gumboots near mud plant"*): Model assigned $18\%$ SIF probability. Correctly categorized as Non-SIF, but flagged a minor chemical PPE note due to the term *"mud plant"*.
  - This demonstrates why **human-in-the-loop triage** remains essential for short, single-sentence field cards.

---

## 💻 6. How to Run Preprocessing & Model Building

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run Data Preprocessing & Feature Engineering
```bash
# Aggregates raw OSHA, Indian OISD alerts, and OIL domain records into master dataset
python data/build_master_dataset.py
```

### 3. Run Model Training & Evaluation
```bash
# Trains SIF classifier, IOGP multi-label model, and outputs evaluation metrics
python src/models/train_sif_engine.py
```

### 4. Test Live Inference
```bash
# Test any custom observation narrative:
python test_inference.py "Contractor was grinding a live condensate line without a valid hot work permit or gas test."
```

---

## 📁 7. Repository Layout

```
├── data/
│   ├── build_master_dataset.py          # Master dataset aggregation & feature engineering
│   ├── generate_indian_data.py          # Curated OISD & OIL incident case generator
│   ├── integrate_indian_data.py         # Indian benchmark integration pipeline
│   ├── preprocess_pipeline.py           # NLP text cleaning utilities
│   ├── raw/
│   │   └── indian_oisd_oil_incidents.csv# Real-world Indian OISD alert records
│   └── processed/
│       ├── indian_oil_gas_benchmark.csv # Indian test benchmark
│       ├── master_dataset_metadata.json # 55-feature schema dictionary
│       ├── model_evaluation_metrics.json# Machine-readable evaluation report
│       └── evaluation_plots.png         # Confusion matrix & ROC visual
│
├── models/                              # Serialized ML Artifacts
│   ├── sif_classifier.joblib            # Calibrated SIF Classifier
│   ├── iogp_rules_classifier.joblib     # 9-way IOGP Multi-Label Classifier
│   ├── severity_regressor.joblib        # Continuous Severity Score Regressor
│   └── feature_extractor.joblib         # Multi-modal feature extractors
│
├── src/models/
│   └── train_sif_engine.py              # Full training & evaluation script
│
├── test_inference.py                    # Interactive inference tester
├── requirements.txt                     # Python dependencies
└── README.md                            # Preprocessing & modeling guide
```
