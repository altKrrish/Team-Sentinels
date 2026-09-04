# 🛡️ Team-Sentinels: CloseCall SIF Precursor Detection Engine

### AI-assisted safety intelligence for unsafe-act, unsafe-condition, and near-miss reports

#### Repository: [`altKrrish/Team-Sentinels`](https://github.com/altKrrish/Team-Sentinels) | Branch: `yashash` | Oil India Limited (OIL) — SIH Problem Statement 26165

[![React](https://img.shields.io/badge/React-19-149eca.svg)](https://react.dev/) [![Vite](https://img.shields.io/badge/Vite-8-646cff.svg)](https://vite.dev/) [![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/) [![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab.svg)](https://www.python.org/)

---

## 📌 Executive Summary

CloseCall is a safety review dashboard for detecting **Serious Injury and Fatality (SIF) precursors** in operational reports. It combines a React/Vite interface with a Python machine-learning service that scores report narratives, identifies breached Life-Saving Rules, estimates severity, and exposes the evidence behind each prediction.

The system is designed to help HSE reviewers prioritize attention. It does not replace a human safety decision: reviewer confirmations and overrides are stored alongside the model verdict so disagreement remains visible.

### The Problem

Short field narratives often contain weak signals of high-consequence exposure: failed barriers, missing permits, uncontrolled energy, suspended loads, or inadequate verification. A simple keyword search misses context, while a statistical classifier can under-score a novel hazard description.

### The CloseCall Approach

1. **Shared feature representation**: TF-IDF word and character n-grams are combined with engineered safety features.
2. **Five prediction heads**: SIF potential, Life-Saving Rules, severity, hazard energy, and barrier state are scored from the same report.
3. **Recall-oriented SIF screening**: the SIF head uses balanced classification to prioritize potentially dangerous reports for review.
4. **Human-in-the-loop workflow**: HSE reviewers can mark reports as pending, in progress, confirmed, or overridden.
5. **Transparent explanations**: feature contributions and evidence spans are returned with the classification.
6. **Visible data provenance**: the interface distinguishes demo fixtures, live model results, and fallback results.

> **Prototype notice:** the bundled engine is trained on deterministic generated narratives because OIL report text is internal and unpublished. It must be retrained and validated on appropriately labelled operational data before production use.

---

## 🏛️ Architecture

```text
Field Report / Pasted Observation
                │
                ▼
       React + Vite Dashboard
                │
       ┌────────┴────────┐
       │                 │
   Demo fixtures     FastAPI service
       │                 │
       └────────┬────────┘
                ▼
        Feature extraction
                │
                ▼
        Shared model matrix
                │
     ┌──────────┼──────────┬──────────┐
     ▼          ▼          ▼          ▼
   SIF       Rules      Severity    Energy / Barrier
                │
                ▼
       Explanation + review workflow
```

The UI uses demo data by default. Set `VITE_USE_MOCK=false` to connect it to the FastAPI service. If a live request fails, the UI falls back to demo data and labels the source accordingly.

---

## 📂 Repository Structure

```text
frontend/
├── src/
│   ├── App.jsx                 # Application shell and routes/views
│   ├── index.css               # Global visual system
│   ├── components/             # Dashboard, analysis, triage, charts, and UI kit
│   └── lib/
│       ├── api.js              # Demo/live API boundary
│       ├── contract.js         # Report and classification contracts
│       ├── model.js            # Client-side display helpers
│       └── mock/reports.js     # Bundled demo fixtures
├── server/
│   ├── app.py                  # FastAPI service
│   ├── train.py                # Engine training entry point
│   ├── closecall/              # Features, model, corpus, explanations, serving
│   └── artifacts/              # Generated engine and metrics artifacts
├── sample-oil-india-reports.csv # Example report input
├── package.json
├── vite.config.js
└── README.md
```

---

## ⚡ Quick Start

### 1. Install frontend dependencies

```bash
npm install
```

### 2. Start the demo dashboard

```bash
npm run dev
```

Open the local URL printed by Vite. Demo mode works without the Python service.

### 3. Train the local model

From the `frontend` directory:

```bash
cd server
python train.py
```

This creates `server/artifacts/engine.joblib` and `server/artifacts/metrics.json`.

### 4. Run the API

```bash
cd server
python -m uvicorn app:api --reload --port 8000
```

To use the API from the dashboard, create `.env.local` in `frontend/`:

```dotenv
VITE_USE_MOCK=false
VITE_API_BASE=http://localhost:8000
```

Restart Vite after changing environment variables.

---

## 🔌 API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Model status, provenance, report count, and warning |
| `GET` | `/metrics` | Held-out training metrics from the latest run |
| `GET` | `/reports` | Scored report stream and reviewer decisions |
| `POST` | `/classify` | Classify `{"text": "..."}` and return a full result |
| `POST` | `/reports/{id}/review` | Save a reviewer decision and optional note |

Example classification request:

```bash
curl -X POST http://localhost:8000/classify ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Worker entered a restricted area without verified energy isolation.\"}"
```

The API validates report text length, keeps review state in memory, and never replaces the model verdict with a reviewer decision.

---

## 📊 Model Outputs

| Head | Output |
|---|---|
| SIF potential | Boolean verdict and probability |
| Life-Saving Rules | Nine independent rule probabilities |
| Severity | Continuous 0–10 score |
| Hazard energy | Energy category and confidence |
| Barrier state | Barrier condition and confidence |

The classifier uses engineered signals for severity indicators, barrier failures, rule violations, negation, measurements, and temporal patterns alongside text features. See `server/closecall/features.py` and `server/closecall/explain.py` for the implementation.

---

## 🧭 Data Provenance and Safety Boundaries

- The default corpus is generated and seeded for repeatable development runs.
- The dashboard's report metadata is fixture data in demo mode.
- Reviews are held in memory and are cleared when the API restarts.
- The model is a decision-support tool, not an autonomous safety authority.
- Production deployment requires labelled real reports, access control, durable review storage, monitoring, and HSE validation.

To train with labelled reports, place a CSV at `server/data/reports.csv` and run `python train.py`. The minimum required columns are `text` and `sif`; optional columns include `rules`, `energy`, `barrier`, severity fields, and `reported_at`.

---

## 🧪 Validation

Build the frontend with:

```bash
npm run build
```

The training command writes metrics to `server/artifacts/metrics.json`; the API exposes the same values through `/metrics` so they can be inspected before interpreting dashboard results.

---

## 📜 Standards Context

The project is intended for safety workflows informed by OIL operational practice and common process-safety concepts, including energy isolation, work authorization, suspended loads, fall protection, electrical hazards, and barrier management. Any regulatory or operational claim must be reviewed against the current applicable OIL, OISD, DGMS, OSHA, NFPA, and API requirements before deployment.

---

## 📄 License and Use

This repository is a prototype for the SIH problem statement and safety-technology evaluation. Validate all model outputs with qualified HSE personnel before taking operational action.