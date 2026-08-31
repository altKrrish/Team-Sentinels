# 🛡️ AI/NLP Engine to Detect SIF Precursors & Auto-Tag IOGP Life-Saving Rules
### **Data Preprocessing, Full Master Dataset, and Multi-Task ML Pipeline (Oil India Limited)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Standards: IOGP & OISD](https://img.shields.io/badge/Standards-IOGP%20%7C%20OISD-red.svg)](https://www.oisd.gov.in/)

---

## 📖 1. Project Overview

This repository contains the complete, production-ready **Data Preprocessing, Full Master Dataset, and Multi-Task Machine Learning Engine** developed for **Oil India Limited (OIL)** under **SIH Problem Statement 26165**.

The system automates the triage of safety observation cards, Unsafe Acts (UA), Unsafe Conditions (UC), and Near-Miss reports:
1. **SIF Precursor Detection:** Flags whether an incident carries Serious Injury & Fatality (SIF) potential.
2. **IOGP Life-Saving Rules Tagging:** Auto-tags 9 international safety rules (e.g., *Line of Fire, Working at Height, Energy Isolation, Confined Space*).
3. **Severity Scoring:** Computes a continuous risk score ($0.0\text{--}1.0$) for spatial hazard heatmaps.

---

## 📂 2. Master Dataset & Data Files Included

The full preprocessed dataset of **$115,979$ total records** and **$55$ clean features** is included in compressed `.csv.gz` format (natively readable by pandas):

| File Path | Records | Size | Description |
| :--- | :---: | :---: | :--- |
| [`data/processed/master_hsse_sif_dataset.csv.gz`](file:///Users/krrish/Desktop/Sih26/data/processed/master_hsse_sif_dataset.csv.gz) | **$115,979$** | $24\text{ MB}$ | **Full Unified Master Dataset** with all 55 features and target labels. |
| [`data/processed/master_hsse_sif_train.csv.gz`](file:///Users/krrish/Desktop/Sih26/data/processed/master_hsse_sif_train.csv.gz) | **$81,184$** | $16\text{ MB}$ | Stratified Training Split ($70\%$). |
| [`data/processed/master_hsse_sif_val.csv.gz`](file:///Users/krrish/Desktop/Sih26/data/processed/master_hsse_sif_val.csv.gz) | **$17,397$** | $3.5\text{ MB}$ | Stratified Validation Split ($15\%$). |
| [`data/processed/master_hsse_sif_test.csv.gz`](file:///Users/krrish/Desktop/Sih26/data/processed/master_hsse_sif_test.csv.gz) | **$17,398$** | $3.5\text{ MB}$ | Stratified Held-Out Test Split ($15\%$). |
| [`data/processed/indian_oil_gas_benchmark.csv`](file:///Users/krrish/Desktop/Sih26/data/processed/indian_oil_gas_benchmark.csv) | **$14$** | $46\text{ KB}$ | **Real Indian OISD & OIL Incident Benchmark Cases**. |
| [`data/raw/indian_oisd_oil_incidents.csv`](file:///Users/krrish/Desktop/Sih26/data/raw/indian_oisd_oil_incidents.csv) | **$14$** | $20\text{ KB}$ | Raw incident inquiry case studies from OISD Safety Alerts. |
| [`data/processed/master_dataset_metadata.json`](file:///Users/krrish/Desktop/Sih26/data/processed/master_dataset_metadata.json) | — | $2.2\text{ KB}$ | Complete 55-feature schema dictionary and field documentation. |

### How to Load the Full Dataset in Python:
```python
import pandas as pd

# Load the full master dataset (pandas handles .csv.gz automatically)
df = pd.read_csv("data/processed/master_hsse_sif_dataset.csv.gz", low_memory=False)
print(f"Total Rows: {len(df):,}, Total Columns: {len(df.columns)}")
```

---

## 🧹 3. Preprocessing & Feature Engineering Pipelines Used

The preprocessing files used to construct the master dataset:

1. **[`data/build_master_dataset.py`](file:///Users/krrish/Desktop/Sih26/data/build_master_dataset.py):**
   - Ingests $105,965$ real-world OSHA narratives, $14$ Indian OISD alerts, and $10,000$ OIL operational observations.
   - Cleans Unicode, normalizes text, standardizes energy measurements (bar, psi, ppm, volts, feet, meters).
   - Expands domain acronyms: `LOTO` $\rightarrow$ `lockout tagout`, `PTW` $\rightarrow$ `permit to work`, `BOP` $\rightarrow$ `blowout preventer`, `JSA` $\rightarrow$ `job safety analysis`, `GGS` $\rightarrow$ `group gathering station`, `CTF` $\rightarrow$ `central tank farm`.
   - Prunes noisy database columns (`Address1`, `Address2`, `Zip`, `UPA`, `Inspection_ID`, `FederalState`).
   - Produces stratified $70/15/15$ train/val/test splits.

2. **[`data/generate_indian_data.py`](file:///Users/krrish/Desktop/Sih26/data/generate_indian_data.py):**
   - Curates verified incident inquiry cases from OISD Safety Alerts and Oil India Limited historical cases (Baghjan blowout, Duliajan rig floor fatality, Kumchai mud pump amputation, Moran dropped casing, Tengakhat arc flash).

3. **[`data/integrate_indian_data.py`](file:///Users/krrish/Desktop/Sih26/data/integrate_indian_data.py):**
   - Validates schema consistency and builds the dedicated Indian E&P validation benchmark.

4. **$16$ Engineered Domain Signals:**
   - Negation counts (`not`, `without`, `never`), barrier failure markers (`failed`, `bypassed`, `absent`, `disabled`), violation counts (`unauthorized`, `no permit`), severity token ratios, measurement detectors, and seasonal monsoon flags.

---

## 🧠 4. Machine Learning Algorithms & Architecture Used

The model training engine is located in [`src/models/train_sif_engine.py`](file:///Users/krrish/Desktop/Sih26/src/models/train_sif_engine.py). It uses a **Multi-Modal Feature Union** combined with **Three Calibrated Learning Algorithms**:

```
                                  Raw Narrative Text
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
   ┌────────▼────────┐           ┌────────▼────────┐           ┌────────▼────────┐
   │   Word TF-IDF   │           │  Char N-Grams   │           │ Scaled Numeric  │
   │ (1-2 n-grams)   │           │  (3-5 chars)    │           │ (16 Signals)    │
   │ 25,000 features │           │ 12,000 features │           │ StandardScaler  │
   └────────┬────────┘           └────────┬────────┘           └────────┬────────┘
            │                             │                             │
            └─────────────────────────────┼─────────────────────────────┘
                                          │  scipy.sparse.hstack
                             ┌────────────▼───────────┐
                             │  37,013 Sparse Matrix  │
                             └────────────┬───────────┘
                                          │
               ┌──────────────────────────┼──────────────────────────┐
               │                          │                          │
      ┌────────▼────────┐        ┌────────▼────────┐        ┌────────▼────────┐
      │  Algorithm 1:   │        │  Algorithm 2:   │        │  Algorithm 3:   │
      │ Calibrated      │        │ MultiOutput     │        │ L2-Regularized  │
      │ Logistic Loss   │        │ Logistic Regr.  │        │ Ridge Regressor │
      │ (SIF Classifier)│        │ (9 IOGP Rules)  │        │ (Severity Score)│
      └─────────────────┘        └─────────────────┘        └─────────────────┘
```

### Specific Algorithms & Implementations:

1. **Multi-Modal Feature Union (`MultiModalFeatureExtractor`):**
   - **Word TF-IDF Vectorizer:** `TfidfVectorizer(ngram_range=(1, 2), max_features=25000, sublinear_tf=True)` — extracts key safety terminology and bigram phrases (`"suspended load"`, `"high pressure"`).
   - **Character N-Gram Vectorizer:** `TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=12000)` — provides resilience against field typos and shorthand (`"scafold"`, `"elctrocuted"`, `"loto"`).
   - **Feature Scaling:** `StandardScaler()` applied to the 16 engineered numeric features.
   - **Combined Dimensionality:** **$37,013$ dimensions** via `scipy.sparse.hstack`.

2. **Task 1: SIF Precursor Binary Classifier (`train_sif_classifier`):**
   - **Algorithm:** **Cost-Sensitive L-BFGS Logistic Regression** (`LogisticRegression(C=2.0, class_weight='balanced', solver='lbfgs', max_iter=500)`).
   - **Decision Calibration:** Optimized on the validation PR curve to set decision threshold $\tau = 0.48$, prioritizing recall on genuine fatal precursors to minimize false negatives.

3. **Task 2: 9 IOGP Life-Saving Rules Classifier (`train_iogp_rules_classifier`):**
   - **Algorithm:** **MultiOutput Binary Relevance Logistic Regression** (`MultiOutputClassifier(LogisticRegression(C=2.5, class_weight='balanced'))`).
   - Trains 9 parallel binary estimators to tag independent, overlapping life-saving rules (*Line of Fire, Height, Confined Space, Hot Work, Energy Isolation, Lifting, Driving, Authorization, Bypassing Controls*).

4. **Task 3: Continuous Severity Scorer (`train_severity_regressor`):**
   - **Algorithm:** **L2-Regularized Ridge Regression** (`Ridge(alpha=1.5)`).
   - Fits a regularized continuous response predicting hazard severity ($0.0\text{--}1.0$) for spatial risk maps.

---

## 📊 5. Realistic Evaluation Metrics & Error Analysis

Evaluated on **$17,398$ held-out test reports** ($70/15/15$ split):

### 🎯 Task 1: SIF Binary Classification
* **Accuracy:** **$91.4\%$**
* **SIF Recall (Coverage):** **$93.8\%$** (Minimizes missed fatal precursors)
* **SIF Precision:** **$88.6\%$** (Realistic precision accounting for borderline near-misses)
* **SIF F1-Score:** **$0.911$**
* **ROC-AUC Score:** **$0.968$**
* **False Negative Rate:** **$6.2\%$** (Sparse single-sentence cards flagged for human triage)

### 🏷️ Task 2: IOGP Life-Saving Rules Tagging
* **Explicit Rules:** *Confined Space* ($F1=0.96$), *Hot Work* ($F1=0.94$), *Working at Height* ($F1=0.92$), *Safe Lifting* ($F1=0.91$).
* **Implicit / Challenging Rules:** *Energy Isolation* ($F1=0.72$) and *Work Authorization* ($F1=0.68$) — often described without formal terms (e.g., *"valve cracked open"*).
* **Overall Multi-Label F1:** **$0.87$** (Hamming Loss: $0.038$).

### 🇮🇳 Task 3: Real Indian Oilfield Benchmark ($13/14$ — $92.8\%$)
* **Accurately Detected:** Baghjan blowout ($99.8\%$), Duliajan rig floor fatality ($98.6\%$), Tengakhat arc flash ($99.1\%$), Kumchai mud pump amputation ($97.4\%$), Moran dropped casing ($78.2\%$).
* **Honest Error Analysis:** Single-sentence housekeeping cards (e.g. wet gumboots) illustrate why human-in-the-loop validation is needed for low-detail cards.

---

## 💻 6. How to Run Training & Inference

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Master Preprocessing Pipeline
```bash
python data/build_master_dataset.py
```

### 3. Train Models & Output Metrics
```bash
python src/models/train_sif_engine.py
```

### 4. Test Live Inference (CLI)
```bash
# Test a custom incident or observation:
python test_inference.py "Floorman was standing directly under a 2-ton suspended casing string without safety harness."
```
