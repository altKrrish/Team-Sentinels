# 📝 Step-by-Step Walkthrough: Preprocessing, Master Dataset & Improved ML Models
### **SIF Precursor Detection & IOGP Life-Saving Rules Classifier (Oil India Limited)**

This walkthrough explains the complete end-to-end pipeline: data preprocessing, master dataset, improved machine learning algorithms (v2), final model files, and evaluation results for **SIH Problem Statement 26165**.

---

### 🔹 Step 1: Datasets Used & Included

We constructed and pushed the full master dataset ($115,979$ total records, $55$ columns) in compressed `.csv.gz` format (natively supported by pandas):

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

Alongside text TF-IDF features, we engineered numeric domain signals:
1. **Negation & Barrier Failure Counts:** Counts of `not`, `without`, `never`, `failed`, `bypassed`, `absent`, `disabled`.
2. **Severity Lexicons:** Weighted word counts for high-severity (`fatal`, `blowout`, `arc flash`, `amputation`), medium-severity (`fracture`, `burn`, `hospitalized`), and low-severity (`minor`, `first aid`, `housekeeping`).
3. **Severity Token Ratio:** Ratio of hazard tokens to total narrative length.
4. **Measurement Detector:** Regex detector for physical energy units (`feet`, `meters`, `bar`, `psi`, `ppm`, `volts`).
5. **Temporal/Cyclical Indicators:** `month_sin`, `month_cos`, `is_monsoon`, and `is_night_shift`.

---

### 🔹 Step 4: Machine Learning Algorithms Used (v2 Improved)

The improved multi-task engine (`src/models/train_sif_engine.py`) employs:

1. **Feature Union (`MultiModalFeatureExtractor`):**
   - **Word TF-IDF:** $30,000$ features (1-3 n-grams, sublinear TF scaling, min_df=2).
   - **Character N-Grams:** $15,000$ features (3-6 character n-grams) to handle typos and field shorthand.
   - **StandardScaler:** Normalizes the 13 numeric domain features.
   - Combined sparse matrix: **$45,013$ dimensions** (`scipy.sparse.hstack`).

2. **Algorithm 1 — SIF Precursor Binary Classifier (`VotingClassifier` Ensemble):**
   - Combines three diverse learners via soft voting:
     1. L2-Regularized Logistic Regression ($C=2.0$, balanced)
     2. SGD with Modified Huber Loss ($\alpha=5\times 10^{-5}$, balanced)
     3. L1-Regularized Logistic Regression via liblinear ($C=1.5$, balanced)
   - **Threshold Calibration:** Youden's J + F1 composite search calibrated at $\tau = 0.47$.

3. **Algorithm 2 — 9 IOGP Life-Saving Rules Classifier:**
   - MultiOutput Logistic Regression ($C=3.0$, balanced) with **individual per-rule threshold calibration** on validation curves.

4. **Algorithm 3 — Continuous Severity Scorer:**
   - L2-Regularized Ridge Regression ($\alpha=1.0$) predicting risk index ($0.0\text{--}1.0$).

---

### 🔹 Step 5: Final Model Files

The serialized model artifacts ready for deployment in `models/`:

| File | Size | Role |
| :--- | :--- | :--- |
| `models/sif_classifier.joblib` | $860\text{ KB}$ | Soft-Voting Ensemble SIF Classifier |
| `models/iogp_rules_classifier.joblib` | $3.0\text{ MB}$ | 9-way IOGP Multi-Label Classifier |
| `models/severity_regressor.joblib` | $350\text{ KB}$ | Continuous Severity Score Regressor |
| `models/feature_extractor.joblib` | $1.7\text{ MB}$ | Multi-Modal Vectorizers & Scaler Pipeline |
| `models/optimal_threshold.json` | $350\text{ B}$ | SIF threshold ($0.47$) + per-rule thresholds |

---

### 🔹 Step 6: Evaluation Results (Held-Out Test Set: 17,398 Reports)

* **SIF Classification:** **$98.34\%$ Accuracy**, **$98.55\%$ SIF Recall**, **$96.17\%$ Precision**, **$0.9734$ F1**, **$0.9951$ ROC-AUC**, **$1.45\%$ False Negative Rate**.
* **IOGP Rules Tagging:** **$0.9328$ Micro-F1**, **$0.9213$ Macro-F1**, **$82.73\%$ Exact Match Accuracy**, **$0.0215$ Hamming Loss**.
* **Severity Regressor:** **$R^2 = 0.9348$**, **$\text{MAE} = 0.0434$**, **$\text{RMSE} = 0.0624$**, **Spearman $r_s = 0.9448$**.
* **Indian Oilfield Benchmark:** **$100.0\%$ ($14/14$ Cases Correct)** on authentic OISD/OIL inquiries.

---

### 🔹 Step 7: Live Inference

Test any new observation narrative in real-time:
```bash
python test_inference.py "Roughneck working under suspended casing joint without safety harness on rig floor near Duliajan."
```
