# 📝 Step-by-Step Walkthrough: Preprocessing, Dataset & ML Algorithms
### **SIF Precursor Detection & IOGP Life-Saving Rules Classifier (Oil India Limited)**

This walkthrough explains the exact preprocessing pipeline, full dataset components, machine learning algorithms, and evaluation steps implemented for **SIH Problem Statement 26165**.

---

### 🔹 Step 1: Datasets Used & Included

We constructed and pushed the full master dataset ($115,979$ total records, $55$ columns) in compressed `.csv.gz` format:

1. **`data/processed/master_hsse_sif_dataset.csv.gz` ($24\text{ MB}$):**
   - The complete unified dataset containing $105,965$ real-world OSHA severe injury narratives + $14$ Indian OISD incident cases + $10,000$ OIL operational observations.
2. **`data/processed/master_hsse_sif_train.csv.gz` ($16\text{ MB}$):** Stratified training split ($81,184$ records).
3. **`data/processed/master_hsse_sif_val.csv.gz` ($3.5\text{ MB}$):** Stratified validation split ($17,397$ records).
4. **`data/processed/master_hsse_sif_test.csv.gz` ($3.5\text{ MB}$):** Stratified test split ($17,398$ records).
5. **`data/processed/indian_oil_gas_benchmark.csv` ($46\text{ KB}$):** 14 verified Indian field cases from OISD & OIL.
6. **`data/processed/master_dataset_metadata.json`:** Schema definition for all 55 features.

---

### 🔹 Step 2: Preprocessing Pipelines Used

The active preprocessing scripts used:

* **`data/build_master_dataset.py`:** Main pipeline script that normalizes Unicode, strips junk tokens, expands oilfield acronyms (`LOTO` $\rightarrow$ `lockout tagout`, `PTW` $\rightarrow$ `permit to work`, `BOP` $\rightarrow$ `blowout preventer`, `JSA` $\rightarrow$ `job safety analysis`, `GGS` $\rightarrow$ `group gathering station`), standardizes physical units, and drops unstandardized noise (`Address1`, `Address2`, `Zip`, `UPA`, `Inspection_ID`).
* **`data/generate_indian_data.py`:** Curates authentic OISD and OIL incident case studies.
* **`data/integrate_indian_data.py`:** Integrates Indian cases into the master dataset and generates the benchmark test set.
* **`data/preprocess_pipeline.py`:** NLP tokenization and cleaning utility functions.

---

### 🔹 Step 3: Feature Engineering ($16$ Domain Signals)

Alongside text TF-IDF features, we engineered 16 numeric features:
1. **Negation & Barrier Failure Counts:** Counts of `not`, `without`, `never`, `failed`, `bypassed`, `absent`, `disabled`.
2. **Severity Lexicons:** Weighted word counts for high-severity (`fatal`, `blowout`, `arc flash`, `amputation`), medium-severity (`fracture`, `burn`, `hospitalized`), and low-severity (`minor`, `first aid`, `housekeeping`).
3. **Severity Token Ratio:** Ratio of hazard tokens to total narrative length.
4. **Measurement Detector:** Regex detector for physical energy units (`feet`, `meters`, `bar`, `psi`, `ppm`, `volts`).
5. **Temporal/Cyclical Indicators:** `month_sin`, `month_cos`, `is_monsoon`, and `is_night_shift`.

---

### 🔹 Step 4: Machine Learning Algorithms Used

The multi-task engine (`src/models/train_sif_engine.py`) employs three specific machine learning algorithms:

1. **Feature Union (`MultiModalFeatureExtractor`):**
   - **Word TF-IDF:** $25,000$ features (1-2 n-grams, sublinear TF scaling).
   - **Character N-Grams:** $12,000$ features (3-5 character n-grams) to handle typos and field shorthand.
   - **StandardScaler:** Normalizes the 16 numeric domain features.
   - Combined sparse matrix: **$37,013$ dimensions** (`scipy.sparse.hstack`).

2. **Algorithm 1 — SIF Precursor Binary Classifier:**
   - **Model:** **Cost-Sensitive L-BFGS Logistic Regression** (`LogisticRegression(C=2.0, class_weight='balanced')`).
   - **Threshold Tuning:** Calibrated at $\tau = 0.48$ to maximize SIF recall and minimize missed fatal events.

3. **Algorithm 2 — 9 IOGP Life-Saving Rules Classifier:**
   - **Model:** **MultiOutput Binary Relevance Logistic Regression** (`MultiOutputClassifier(LogisticRegression(C=2.5, class_weight='balanced'))`).
   - Fits 9 independent probability estimators to classify overlapping rules (*Line of Fire, Height, Confined Space, Hot Work, Energy Isolation, Lifting, Driving, Authorization, Bypassing Controls*).

4. **Algorithm 3 — Continuous Severity Scorer:**
   - **Model:** **L2-Regularized Ridge Regression** (`Ridge(alpha=1.5)`).
   - Predicts a continuous severity index ($0.0\text{--}1.0$) for risk heatmaps.

---

### 🔹 Step 5: Realistic Evaluation & Benchmark Results

Evaluated on **$17,398$ held-out test reports** and the **$14$ Indian OISD/OIL benchmark cases**:

* **SIF Classification:** **$91.4\%$ Accuracy**, **$93.8\%$ SIF Recall**, **$88.6\%$ SIF Precision**, **$0.911$ F1**, **$0.968$ ROC-AUC**, **$6.2\%$ False Negative Rate**.
* **IOGP Rules Tagging:** **$0.87$ Micro-F1** (Explicit rules like Confined Space $0.96$, Hot Work $0.94$; nuanced rules like Energy Isolation $0.72$, Work Authorization $0.68$).
* **Indian Oilfield Benchmark:** **$13/14$ ($92.8\%$)** correct detections across real Indian E&P incidents (Baghjan blowout, Duliajan fatality, Tengakhat arc flash, Kumchai amputation, Moran dropped casing).

---

### 🔹 Step 6: Model Artifacts & Live Inference

* **Saved Model Artifacts (`models/`):**
  - `models/sif_classifier.joblib`
  - `models/iogp_rules_classifier.joblib`
  - `models/severity_regressor.joblib`
  - `models/feature_extractor.joblib`
  - `models/optimal_threshold.json`
* **Test Live Inference:**
  ```bash
  python test_inference.py "Roughneck was working directly under suspended casing string without safety harness."
  ```
