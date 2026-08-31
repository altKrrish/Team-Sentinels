# Model Card & Limitations
### **AI/NLP SIF Precursor Detection & IOGP Life-Saving Rules Engine**
*Target Organization: Oil India Limited (OIL) | SIH Problem Statement ID: 26165*

---

## 📌 Executive Summary & Purpose

This Model Card documents the architecture, data provenance, operating constraints, explainability mechanisms, and known boundaries of the **SIF Precursor & IOGP Rule Classification Engine**. It is designed to be displayed alongside the HSSE dashboard so safety officers, operations managers, and hackathon reviewers can evaluate predictions in proper context.

---

## 🧠 Module 2 — SIF Precursor Classifier

### APPROACH
* **Architecture:** Multi-Modal Feature Union ($45,013$ dimensions) feeding a **Soft-Voting Ensemble (`VotingClassifier`)** combining three diverse estimators:
  1. **L2-Regularized Logistic Regression ($C=2.0$, L-BFGS solver):** Provides smooth, well-calibrated class probability estimates.
  2. **SGD Classifier with Modified Huber Loss ($\alpha=5\times 10^{-5}$):** Fast, outlier-resilient linear margin estimator.
  3. **L1-Regularized Logistic Regression ($C=1.5$, `liblinear` solver):** Imposes strict sparsity for domain-specific feature selection.
* **Feature Extraction:**
  - $30,000$ Word TF-IDF n-grams ($1\text{--}3$ grams, sublinear TF scaling).
  - $15,000$ Subword Character N-Grams ($3\text{--}6$ characters) for typo and field shorthand resilience.
  - $13$ Dense engineered domain features (negation counts, barrier failure terms, severity ratios, energy measurement indicators).

### OPERATING POINT
* **Optimal Decision Threshold:** $\tau = 0.47$, calibrated via **Youden’s J Statistic** ($J = \text{Sensitivity} + \text{Specificity} - 1$) and F1 optimization on validation PR curves.
* **Triage Philosophy (Recall-First):** In industrial process safety, a missed fatal precursor (**False Negative**) carries catastrophic risk, whereas investigating a false alarm (**False Positive**) costs ~2 minutes of an HSE officer's review.

### TARGET RECALL & PERFORMANCE (Held-Out Test Set: 17,398 Reports)
* **SIF Recall (Sensitivity):** **$98.55\%$** (Catches $5,291$ of $5,369$ true fatal precursors).
* **False Negative Rate:** **$1.45\%$** (Only 78 edge-case reports missed).
* **SIF Precision:** **$96.17\%$** | **Overall Accuracy:** **$98.34\%$** | **ROC-AUC:** **$0.9951$** | **PR-AUC:** **$0.9864$**.

### EXPLAINABILITY & AUDITABILITY
* Every prediction decomposes into **token-level feature attributions** showing the positive SIF trigger phrases (e.g., `+12.02` for `amputated`, `+9.02` for `amputating`, `+5.73` for `severing`, `+4.50` for `collapse`) and negative routine terms (e.g., `-7.11` for `housekeeping`, `-2.19` for `minor`).

---

## 🏷️ Module 3 — IOGP Life-Saving Rules Tagger

### APPROACH
* **Architecture:** Shared $45,013$-dimensional sparse embeddings feeding a **9-way MultiOutput Classifier** (`MultiOutputClassifier(LogisticRegression(C=3.0, class_weight='balanced'))`).
* **Per-Rule Threshold Tuning:** Rather than applying an arbitrary $0.50$ cutoff, each rule utilizes an independently calibrated decision threshold tuned on validation curves:
  - *Confined Space:* $\tau = 0.34$ ($F1 = 0.9932$, $\text{Recall} = 98.87\%$)
  - *Line of Fire:* $\tau = 0.40$ ($F1 = 0.9283$, $\text{Recall} = 96.37\%$)
  - *Driving:* $\tau = 0.48$ ($F1 = 0.8980$, $\text{Recall} = 89.49\%$)
  - *Bypassing Safety Controls:* $\tau = 0.58$ ($F1 = 0.9433$, $\text{Recall} = 93.48\%$)
  - *Working at Height:* $\tau = 0.64$ ($F1 = 0.9370$, $\text{Recall} = 91.09\%$)
  - *Hot Work:* $\tau = 0.66$ ($F1 = 0.9638$, $\text{Recall} = 94.80\%$)
  - *Safe Mechanical Lifting:* $\tau = 0.68$ ($F1 = 0.9556$, $\text{Recall} = 94.16\%$)
  - *Energy Isolation (LOTO):* $\tau = 0.68$ ($F1 = 0.8488$, $\text{Recall} = 89.38\%$)
  - *Work Authorization (PTW):* $\tau = 0.68$ ($F1 = 0.8237$, $\text{Recall} = 90.81\%$)

### GROUND TRUTH & TAXONOMY
* Aligned with **IOGP Report 459** (9 Life-Saving Rules) and **OISD standards** (`OISD-STD-105`, `OISD-STD-115`, `OISD-STD-189`).
* Ground truth is constructed via multi-tiered domain rules, safety barrier mapping, and hand-audited inquiry cases.

### MULTI-LABEL NATURE
* Observations frequently trigger multiple overlapping rules simultaneously (e.g., hot work inside an unblinded crude tank excavation without a valid gas permit triggers *Hot Work*, *Confined Space*, and *Work Authorization*).
* **Overall Metrics:** **$0.9328$ Micro-F1**, **$0.9213$ Macro-F1**, **$82.73\%$ Exact Match Accuracy**, **$0.0215$ Hamming Loss**.

---

## 📈 Module 4 — Continuous Severity & Precursor Pattern Metrics

### CONTINUOUS SEVERITY REGRESSOR
* **Algorithm:** L2-regularized Ridge Regression ($\alpha=1.0$) mapped across $[0.0000, 1.0000]$.
* **Performance:** **$R^2 = 0.9348$**, **$\text{MAE} = 0.0434$**, **$\text{RMSE} = 0.0624$**, **Spearman Rank Correlation $r_s = 0.9448$**.

### SIF PRECURSOR DENSITY INDEX (SPDI)
To prevent high-activity drilling sites from dominating hazard rankings solely due to higher reporting volume:
$$\text{SPDI}_{\text{site}} = \frac{\sum \text{Flagged SIF Reports}}{\text{Total Reports Submitted from Site}} \times 100$$
* Normalizes reporting volume across field assets (e.g., comparing Duliajan Central Tank Farm vs Remote Exploration Well in Rajasthan).

---

## 🌐 Data Provenance

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA PROVENANCE PIPELINE                           │
├────────────────────────────────┬────────────────────────────────────────────┤
│ 1. Real OSHA Severe Incidents  │ 105,965 real-world industrial narratives   │
│    (2015 – 2025)               │ Physical crush, fall, pressure mechanics   │
├────────────────────────────────┼────────────────────────────────────────────┤
│ 2. Real Indian OISD Alerts     │ 14 verified upstream case studies          │
│    & OIL Field Inquiries       │ Baghjan, Duliajan, Moran, Kumchai, Digboi  │
├────────────────────────────────┼────────────────────────────────────────────┤
│ 3. OIL Domain Upstream Logs    │ 10,000 operational observations & cards    │
│    (Asset-mapped)              │ Unsafe Acts (UA) & Unsafe Conditions (UC)  │
└────────────────────────────────┴────────────────────────────────────────────┘
```

* **Training Set ($81,184$ rows):** Stratified $70\%$ partition of the master unified dataset.
* **Validation Set ($17,397$ rows):** Used exclusively for hyperparameter tuning and threshold calibration.
* **Held-Out Test Set ($17,398$ rows):** Stratified $15\%$ unseen test split for final metric reporting.
* **Out-of-Domain Indian Benchmark ($14$ Verified Cases):** Real-world inquiry briefs from OISD Safety Alerts & Oil India Limited historical inquiries (**$100.0\%$ Classification Accuracy** across all 14 cases).

---

## ⚠️ Known Limitations & Operational Boundaries

1. **Production HSSE Platform Data Access:**
   - In accordance with hackathon constraints, no internal proprietary Oil India Limited database credentials were provided. Models were trained on authentic OSHA industrial incident narratives + curated OISD inquiry reports + standardized OIL asset profiles. Performance on live, raw SAP/HSSE internal tables should be recalibrated during enterprise onboarding.
2. **Short / Sparse Observation Cards:**
   - Single-phrase entries (e.g., *"wet footwear near pump"*) provide insufficient semantic context to infer stored energy levels. For such edge cases, the system defaults to flagging a low-confidence recommendation requiring safety officer verification.
3. **Language Scope:**
   - The current NLP pipeline is English-first (including oilfield abbreviations like `LOTO`, `PTW`, `BOP`, `JSA`, `CTF`, `GGS`). Field observations submitted in regional Indian scripts (Assamese, Hindi, Bengali) or romanized colloquial slang are not natively tokenized without translation.
4. **Decision Support System (Not an Automated Adjudicator):**
   - This AI engine acts as a **Level-1 triage assistant** to prioritize high-risk cards in the queue. It does **not** replace mandatory statutory incident investigations, root cause analyses (RCA), or DGMS/OISD regulatory reporting.

---

## 🔮 Deferred Scope (Enterprise Roadmap)

* **Direct Live API Integration:** Two-way connector with Oil India's enterprise HSSE software (e.g., SAP EHS / custom internal portals).
* **Automated Corrective & Preventive Actions (CAPA):** Automatic generation of work-permit freeze suggestions and equipment isolation audit triggers based on recurring barrier failures.
* **Multilingual / Indic NLP:** Fine-tuning multilingual IndicBERT models to support mixed English-Assamese-Hindi field reports and voice-to-text safety cards.
* **Native Offline-First Mobile App:** Edge-deployed observation app for roughnecks on remote derrick floors with asynchronous synchronization upon network reconnect.
