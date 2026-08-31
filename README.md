# 🛡️ AI/NLP Engine to Detect SIF Precursors & Auto-Tag IOGP Life-Saving Rules
### **Data Preprocessing, Full Master Dataset, and Improved Multi-Task ML Pipeline (Oil India Limited)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Standards: IOGP & OISD](https://img.shields.io/badge/Standards-IOGP%20%7C%20OISD-red.svg)](https://www.oisd.gov.in/)

---

## 📖 1. Project Overview

This repository contains the complete, production-ready **Data Preprocessing, Full Master Dataset, and Improved Multi-Task Machine Learning Engine (v2)** developed for **Oil India Limited (OIL)** under **SIH Problem Statement 26165**.

The system automates the triage of safety observation cards, Unsafe Acts (UA), Unsafe Conditions (UC), and Near-Miss reports:
1. **SIF Precursor Detection:** Flags whether an incident carries Serious Injury & Fatality (SIF) potential with a high sensitivity of $98.55\%$ and only $1.45\%$ false negative rate.
2. **IOGP Life-Saving Rules Tagging:** Auto-tags 9 international safety rules (e.g., *Line of Fire, Working at Height, Energy Isolation, Confined Space*) with optimized per-rule probability thresholds.
3. **Continuous Severity Scoring:** Predicts a continuous risk score ($0.0\text{--}1.0$) for spatial hazard heatmaps.

---

## 📦 2. Final Model Files & Artifacts

The final, trained production model files live in the `models/` and `src/models/` directories:

| Model File | Location | Purpose & Description |
| :--- | :--- | :--- |
| **SIF Binary Classifier** | [`models/sif_classifier.joblib`](file:///Users/krrish/Desktop/Sih26/models/sif_classifier.joblib) | **Soft-Voting Ensemble** combining L2-Logistic Regression, SGD (Modified Huber), and L1-Logistic Regression for high-precision SIF triage. |
| **IOGP Multi-Label Classifier** | [`models/iogp_rules_classifier.joblib`](file:///Users/krrish/Desktop/Sih26/models/iogp_rules_classifier.joblib) | **9-way MultiOutput Classifier** mapping narratives to IOGP Life-Saving Rules with calibrated per-rule thresholds. |
| **Severity Regressor** | [`models/severity_regressor.joblib`](file:///Users/krrish/Desktop/Sih26/models/severity_regressor.joblib) | **L2-Ridge Regressor** predicting continuous hazard severity indices ($0.0\text{--}1.0$). |
| **Feature Extractor Pipeline** | [`models/feature_extractor.joblib`](file:///Users/krrish/Desktop/Sih26/models/feature_extractor.joblib) | Serialized multi-modal pipeline ($45,013$ dimensions: Word TF-IDF + Char N-Grams + StandardScaler for 13 domain features). |
| **Calibrated Thresholds** | [`models/optimal_threshold.json`](file:///Users/krrish/Desktop/Sih26/models/optimal_threshold.json) | Stores optimal SIF decision threshold ($0.47$) and per-rule thresholds for all 9 IOGP rules. |
| **Training Engine Code** | [`src/models/train_sif_engine.py`](file:///Users/krrish/Desktop/Sih26/src/models/train_sif_engine.py) | Full training, threshold calibration, and evaluation pipeline script. |
| **Live Inference Script** | [`test_inference.py`](file:///Users/krrish/Desktop/Sih26/test_inference.py) | Interactive CLI tool for real-time inference on arbitrary safety observation text. |

---

## 📂 3. Master Dataset & Data Files Included

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

## 🧹 4. Preprocessing & Feature Engineering Pipelines

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

## 🧠 5. Machine Learning Algorithms & Architecture (v2 Improved)

The improved model training engine ([`src/models/train_sif_engine.py`](file:///Users/krrish/Desktop/Sih26/src/models/train_sif_engine.py)) employs a **Multi-Modal Feature Union** combined with **Three Calibrated Learning Algorithms**:

```
                                  Raw Narrative Text
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
   ┌────────▼────────┐           ┌────────▼────────┐           ┌────────▼────────┐
   │   Word TF-IDF   │           │  Char N-Grams   │           │ Scaled Numeric  │
   │ (1-3 n-grams)   │           │  (3-6 chars)    │           │ (13 Signals)    │
   │ 30,000 features │           │ 15,000 features │           │ StandardScaler  │
   └────────┬────────┘           └────────┬────────┘           └────────┬────────┘
            │                             │                             │
            └─────────────────────────────┼─────────────────────────────┘
                                          │  scipy.sparse.hstack
                             ┌────────────▼───────────┐
                             │  45,013 Sparse Matrix  │
                             └────────────┬───────────┘
                                          │
               ┌──────────────────────────┼──────────────────────────┐
               │                          │                          │
      ┌────────▼────────┐        ┌────────▼────────┐        ┌────────▼────────┐
      │  Algorithm 1:   │        │  Algorithm 2:   │        │  Algorithm 3:   │
      │ Soft-Voting     │        │ MultiOutput     │        │ L2-Regularized  │
      │ Ensemble        │        │ Logistic Regr.  │        │ Ridge Regressor │
      │ (LR+SGD+L1)     │        │ (9 IOGP Rules)  │        │ (Severity Score)│
      └─────────────────┘        └─────────────────┘        └─────────────────┘
```

### Specific Algorithms & Implementations:

1. **Multi-Modal Feature Union (`MultiModalFeatureExtractor`):**
   - **Word TF-IDF Vectorizer:** `TfidfVectorizer(ngram_range=(1, 3), max_features=30000, sublinear_tf=True, min_df=2)` — captures phrases up to trigrams (`"suspended drill pipe"`, `"high pressure gas"`).
   - **Character N-Gram Vectorizer:** `TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), max_features=15000, min_df=3)` — handles misspellings and oilfield shorthand (`"scafold"`, `"elctrocuted"`, `"loto"`).
   - **Feature Scaling:** `StandardScaler()` applied to the 13 engineered numeric features.
   - **Combined Dimensionality:** **$45,013$ dimensions** via `scipy.sparse.hstack`.

2. **Task 1: SIF Precursor Binary Classifier (`VotingClassifier` Ensemble):**
   - **Ensemble Members:**
     1. `LogisticRegression(C=2.0, penalty='l2', solver='lbfgs', class_weight='balanced')` — smooth probabilistic boundary.
     2. `SGDClassifier(loss='modified_huber', alpha=5e-5, class_weight='balanced')` — fast, outlier-resilient margin.
     3. `LogisticRegression(C=1.5, penalty='l1', solver='liblinear', class_weight='balanced')` — sparse feature selector.
   - **Soft Voting:** Averages predicted class probabilities across all 3 models.
   - **Composite Threshold Calibration:** Optimized using Youden's J statistic ($J = \text{Sensitivity} + \text{Specificity} - 1$) + F1 composite at $\tau = 0.47$.

3. **Task 2: 9 IOGP Life-Saving Rules Classifier (`train_iogp_rules_classifier`):**
   - **Algorithm:** `MultiOutputClassifier(LogisticRegression(C=3.0, class_weight='balanced'))`.
   - **Per-Rule Threshold Calibration:** Optimizes individual decision thresholds per rule (e.g., Confined Space $\tau=0.34$, Height $\tau=0.64$, Energy Isolation $\tau=0.68$, Safe Lifting $\tau=0.68$).

4. **Task 3: Continuous Severity Scorer (`train_severity_regressor`):**
   - **Algorithm:** **L2-Regularized Ridge Regression** (`Ridge(alpha=1.0)`).
   - Fits a regularized continuous response predicting hazard severity ($0.0\text{--}1.0$) for spatial risk maps.

---

## 📊 6. Updated Model Performance Metrics (Held-Out Test Set: 17,398 Reports)

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          IMPROVED AI MODEL EVALUATION DASHBOARD                        │
├────────────────────────────────┬───────────────────────────────┬───────────────────────┤
│ Task 1: SIF Classification     │ Task 2: IOGP Life-Saving Rules│ Task 3: Severity Regr.│
│  • Overall Accuracy : 98.34%   │  • Micro F1-Score : 0.9328    │  • R² Score    : 0.9348│
│  • ROC-AUC Score    : 0.9951   │  • Macro F1-Score : 0.9213    │  • MAE         : 0.0434│
│  • PR-AUC (Avg Prec): 0.9864   │  • Hamming Loss   : 0.0215    │  • RMSE        : 0.0624│
│  • SIF Recall       : 98.55%   │  • Exact Match Acc: 82.73%    │  • Spearman rs : 0.9448│
│  • SIF Precision    : 96.17%   │                               │                       │
│  • False Negative   : 1.45%    │                               │                       │
└────────────────────────────────┴───────────────────────────────┴───────────────────────┘
```

### 🎯 Confusion Matrix (17,398 Test Reports):
* **True SIF Caught ($TP$):** **$5,291$** ($98.55\%$ Recall)
* **Missed SIF ($FN$):** **$78$** ($1.45\%$ False Negative Rate)
* **Correct Non-SIF ($TN$):** **$11,818$**
* **False Alarms ($FP$):** **$211$**

### 🏷️ Per-Rule Performance Breakdown:

| Rule Display Name | F1-Score | Precision | Recall | ROC-AUC | Tuned Threshold | Support |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Confined Space** | **$0.9932$** | $99.77\%$ | $98.87\%$ | $0.9992$ | $0.34$ | $4,425$ |
| **Hot Work** | **$0.9638$** | $98.02\%$ | $94.80\%$ | $0.9971$ | $0.66$ | $1,251$ |
| **Safe Mechanical Lifting** | **$0.9556$** | $96.99\%$ | $94.16\%$ | $0.9966$ | $0.68$ | $1,028$ |
| **Bypassing Safety Controls**| **$0.9433$** | $95.20\%$ | $93.48\%$ | $0.9987$ | $0.58$ | $276$ |
| **Working at Height** | **$0.9370$** | $96.46\%$ | $91.09\%$ | $0.9932$ | $0.64$ | $2,481$ |
| **Line of Fire** | **$0.9283$** | $89.53\%$ | $96.37\%$ | $0.9786$ | $0.40$ | $8,022$ |
| **Driving** | **$0.8980$** | $90.11\%$ | $89.49\%$ | $0.9761$ | $0.48$ | $6,711$ |
| **Energy Isolation** | **$0.8488$** | $80.80\%$ | $89.38\%$ | $0.9916$ | $0.68$ | $405$ |
| **Work Authorization** | **$0.8237$** | $75.37\%$ | $90.81\%$ | $0.9962$ | $0.68$ | $283$ |

### 🇮🇳 Real-World Indian Oilfield Benchmark:
* **Accuracy:** **$100.0\%$ ($14/14$ Cases Correctly Classified)** on verified OISD & Oil India Limited incident inquiries (including Baghjan blowout, Duliajan fatality, Tengakhat arc flash, Kumchai amputation, Moran dropped casing, Jaisalmer crane tilt).

---

## 💻 7. How to Run Training & Inference

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Master Preprocessing Pipeline
```bash
python data/build_master_dataset.py
```

### 3. Train Improved v2 Models & Output Metrics
```bash
python src/models/train_sif_engine.py
```

### 4. Test Live Inference (CLI)
```bash
# Test a custom incident or observation:
python test_inference.py "Roughneck working under suspended casing joint without safety harness on rig floor near Duliajan."
```
