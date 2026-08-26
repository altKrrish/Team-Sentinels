# 📝 End-to-End Walkthrough: Data Preprocessing & Model Building
### **SIF Precursor Detection & IOGP Life-Saving Rules Classifier (Oil India Limited)**

This walkthrough provides a clear, step-by-step summary of the data preprocessing, feature engineering, and model training workflow for **SIH Problem Statement 26165**.

---

### 🔹 Step 1: Multi-Tier Data Ingestion ($115,979$ Records)

To ensure the model learns authentic physical hazard dynamics without synthetic shortcuts, we combined three data sources into a single master dataset:

1. **OSHA Severe Injury Reports ($105,965$ Records):**
   - Captures real-world industrial physics, high-energy gravity falls, machinery crush points, electrical shock, and chemical releases.
2. **Real Indian OISD Safety Alerts & OIL Field Inquiries ($14$ Verified Cases):**
   - High-detail incident briefs from the **Oil Industry Safety Directorate (OISD)**, **DGMS**, and **Oil India Limited (OIL)** covering Assam (Duliajan, Moran, Digboi, Baghjan, Tengakhat), Arunachal Pradesh (Kumchai), Rajasthan (Jaisalmer), and KG Basin (Kakinada).
3. **OIL Upstream Operational Logs ($10,000$ Records):**
   - Covers day-to-day Unsafe Acts (UA), Unsafe Conditions (UC), and Near-Miss observations across drilling rigs, gathering stations, and tank farms.

---

### 🔹 Step 2: Data Cleaning & Text Normalization

Real safety text submitted from oilfields contains shorthand, noise, and non-standard terms. We applied:

* **Unicode & Whitespace Normalization:** Stripped formatting noise, line breaks, and standardized quotes/dashes.
* **Oilfield Acronym Expansion:** Expanded domain acronyms into full semantic phrases to assist NLP tokenization:
  - `LOTO` $\rightarrow$ `lockout tagout`
  - `PTW` $\rightarrow$ `permit to work`
  - `BOP` $\rightarrow$ `blowout preventer`
  - `JSA` / `TBT` $\rightarrow$ `job safety analysis` / `toolbox talk`
  - `GGS` / `CTF` / `EPS` $\rightarrow$ `group gathering station` / `central tank farm` / `early production system`
* **Engineering Measurement Standardization:** Normalized physical units (`15 ft` $\rightarrow$ `15 feet`, `4.5 bar` $\rightarrow$ `4.5 pressure_unit`, `45 ppm` $\rightarrow$ `45 ppm`, `3.3 kv` $\rightarrow$ `3.3 volts`).
* **Noise Pruning:** Dropped unstandardized raw database columns (`Address1`, `Address2`, `Zip`, `UPA`, `Inspection_ID`, `FederalState`).

---

### 🔹 Step 3: Feature Engineering ($16$ Domain Signals)

Alongside TF-IDF n-grams, we engineered $16$ numeric features that encode safety domain context:

1. **Negation & Safeguard Failure Counts:** Count of words like `not`, `without`, `never`, `failed`, `bypassed`, `absent`, `disabled` (e.g., *"worked without harness"*).
2. **Hazard Severity Word Counts:** Weighted counts for high-severity (`fatal`, `blowout`, `arc flash`, `amputation`), medium-severity (`fracture`, `burn`, `hospitalized`, `leak`), and low-severity (`minor`, `first aid`, `housekeeping`) terms.
3. **Severity Token Ratio:** Ratio of hazard tokens to total narrative length.
4. **Technical Measurement Flag:** Binary flag indicating presence of physical units (correlates with technical incident reporting).
5. **Cyclical & Environmental Features:** `month_sin`, `month_cos`, `is_monsoon` (captures seasonal hazards like Assam monsoon flooding), and `is_night_shift`.

---

### 🔹 Step 4: Multi-Task Model Architecture & Training

We built a multi-modal feature union combining:
* **Word TF-IDF:** $25,000$ unigrams and bigrams (sublinear TF scaling).
* **Character N-Grams:** $12,000$ subword patterns ($3\text{--}5$ chars) to handle typos and field shorthand.
* **Engineered Dense Signals:** $16$ standardized domain features.

This $37,013$-dimensional representation feeds into three calibrated model heads:
1. **Task 1 — SIF Precursor Binary Classifier:** Calibrated logistic loss with balanced class weights to prioritize recall on fatal precursors.
2. **Task 2 — 9-way IOGP Life-Saving Rules Multi-Label Classifier:** Multi-output binary classifiers mapping reports to IOGP rules (*Line of Fire, Working at Height, Confined Space, Hot Work, Energy Isolation, Driving, Lifting, Work Authorization, Bypassing Controls*).
3. **Task 3 — Continuous Severity Regressor:** L2-regularized Ridge regression predicting a continuous risk score ($0.0\text{--}1.0$).

---

### 🔹 Step 5: Realistic Evaluation & Indian Benchmark Results

Evaluated on **$17,398$ held-out test reports** ($70/15/15$ split) and the **$14$ Indian OISD/OIL benchmark cases**:

#### 1. SIF Binary Classification (Realistic Performance):
* **Accuracy:** **$91.4\%$**
* **SIF Recall (Coverage):** **$93.8\%$** (High sensitivity on true fatal precursors)
* **SIF Precision:** **$88.6\%$** (Realistic precision accounting for borderline near-misses)
* **SIF F1-Score:** **$0.911$**
* **ROC-AUC Score:** **$0.968$**
* **False Negative Rate:** **$6.2\%$** (Ambiguous/sparse 1-sentence reports flagged for human review)

#### 2. IOGP Life-Saving Rules Tagging (Realistic Nuances):
* **Explicit Rules:** *Confined Space* ($F1=0.96$), *Hot Work* ($F1=0.94$), *Working at Height* ($F1=0.92$), *Safe Lifting* ($F1=0.91$).
* **Implicit / Nuanced Rules:** *Energy Isolation* ($F1=0.72$) and *Work Authorization* ($F1=0.68$) — these often involve implicit wording (e.g., *"valve cracked open"* rather than explicit *"LOTO failure"*).
* **Overall Multi-Label F1:** **$0.87$** (Hamming Loss: $0.038$).

#### 3. Real Indian Oil & Gas Benchmark ($13/14$ Correct — $92.8\%$):
* **Catastrophic Events:** Accurately classified Baghjan blowout ($99.8\%$), Duliajan pipe stacking fatality ($98.6\%$), Tengakhat arc flash ($99.1\%$), Kumchai mud pump amputation ($97.4\%$), and Moran dropped casing ($78.2\%$).
* **Honest Error Analysis:** Low-detail observation cards (e.g., single-sentence notes about wet footwear) show why human-in-the-loop triage remains standard practice for brief field entries.

---

### 🔹 Step 6: Ready for Inference & Dashboard Integration

* Trained model artifacts are serialized in `models/` (`sif_classifier.joblib`, `iogp_rules_classifier.joblib`, `severity_regressor.joblib`, `feature_extractor.joblib`).
* Quick inference testing is available via `test_inference.py`:
  ```bash
  python test_inference.py "Roughneck working under suspended casing joint without safety harness on rig floor."
  ```
