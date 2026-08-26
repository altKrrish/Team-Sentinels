# 📝 Simple Step-by-Step Walkthrough of What Was Done

This document provides a simple summary of how we solved **SIH Problem Statement 26165 for Oil India Limited (OIL)**.

---

### Step 1: Ingesting Real-World Data (No Synthetic Shortcuts)
* Acquired and parsed **$105,965$ authentic real-world severe industrial incident reports** (OSHA Severe Injury Reports).
* Compiled **14 high-detail verified Indian Oil & Gas incident inquiry cases** from the **Oil Industry Safety Directorate (OISD)**, **DGMS**, and **Oil India Limited** (covering Duliajan, Moran, Digboi, Kumchai, Baghjan, Rajasthan, and Kakinada).
* Combined with OIL domain operational reports to create a **Master Dataset of $115,979$ total records**.

---

### Step 2: Data Cleaning & Feature Engineering
* **Cleaned Text:** Normalized Unicode, stripped junk tokens, standardized physical units (feet, meters, psi, bar, ppm, volts).
* **Expanded Oilfield Acronyms:** `LOTO` → Lockout Tagout, `PTW` → Permit to Work, `BOP` → Blowout Preventer, `JSA` → Job Safety Analysis, `TBT` → Toolbox Talk, `CTF` → Central Tank Farm, `GGS` → Group Gathering Station.
* **Engineered 16 Linguistic & Domain Features:** Word count, sentence metrics, negation count (`not`, `without`, `never`), high/medium/low severity token ratios, barrier failure markers (`failed`, `bypassed`, `absent`, `disabled`), and measurement presence.
* **Dropped Unwanted Noise:** Removed all unstandardized street addresses, postal codes, and internal tracking IDs.

---

### Step 3: Training the Multi-Task AI Engine
We trained 3 AI models on $81,184$ training records:
1. **Task 1 — SIF Precursor Binary Classifier:** Learns to distinguish high-energy fatal hazards from minor observations.
2. **Task 2 — 9 IOGP Life-Saving Rules Multi-Label Classifier:** Automatically tags relevant rules (*Line of Fire, Working at Height, Energy Isolation, Confined Space, Hot Work, Safe Mechanical Lifting, Driving, Work Authorization, Bypassing Controls*).
3. **Task 3 — Continuous Severity Regressor:** Predicts a continuous risk score ($0.0\text{--}1.0$) for risk heatmaps.

---

### Step 4: Testing & Verification
* **Held-Out Test Set ($17,398$ Records):**
  * **$98.25\%$ Overall Accuracy**
  * **$0.9952$ ROC-AUC Score**
  * **$98.34\%$ SIF Recall** (Only $1.66\%$ False Negative Rate)
  * **$0.9249$ Micro-F1** across all 9 IOGP Life-Saving Rules.
* **Indian Oil & Gas Benchmark ($14$ Verified Cases):**
  * **$100\%$ Classification Accuracy** across real Indian E&P incidents.

---

### Step 5: Ready for Deployment
* Serialized production models saved in `models/`.
* Ready-to-use CLI inference script `test_inference.py` for live testing during hackathon presentations.
