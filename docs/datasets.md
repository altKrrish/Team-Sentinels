# Dataset Evaluation

No single public dataset matches OIL's exact need (Indian oil & gas, free-text UA/UC + near-miss
reports mapped to SIF potential and IOGP Life-Saving Rules). Plan is to combine several datasets,
each doing a different job. Everything below was checked against the live source as of Aug 2026;
notes flag what's confirmed vs. what still needs manual verification.

## Recommended combination

| Dataset | Role | Status |
|---|---|---|
| [OISD Safety Alerts](https://www.oisd.gov.in/en-in/SafetyAlerts) | Indian oil & gas domain vocabulary + small hand-labeled validation set | Confirmed real — ~45 incident PDFs (2021–2026), pipeline fires, crane toppling, falls, equipment failure |
| [OSHA Severe Injury Reports](https://www.osha.gov/severeinjury) | Volume of labeled severe-injury narratives to pretrain the SIF/non-SIF classifier | Confirmed real; mirrors on [Figshare](https://figshare.com/articles/dataset/OSHA_Severe_Injury_Reports_Jan_2015_-_Sep_2023/28669604) and [Kaggle](https://www.kaggle.com/datasets/krist0phersmith/osha-severe-incident-reports) |
| [PHMSA Pipeline Incident data](https://www.phmsa.dot.gov/data-and-statistics/pipeline/pipeline-incident-flagged-files) | Oil/gas-sector-specific incident causes (pipeline ops, matches OIL's business) | Confirmed real; more tabular than narrative-heavy — verify text-field richness before relying on it |
| [Industrial Safety and Health Analytics Database (Kaggle)](https://www.kaggle.com/datasets/ihmstefanini/industrial-safety-and-health-analytics-database) | General industrial accident narratives + a "Potential Accident Level" field that maps closely to SIF-potential | Well-known dataset (~425 Brazil manufacturing-plant records) — **verify exact schema yourself**, Kaggle's page is JS-rendered and could not be scraped for confirmation here |
| IOGP Report 459 (Life-Saving Rules) | Taxonomy for the multi-label tagger — no labeled dataset exists, build weak-labeling rules from the 9 rules' "I Will" statements | Confirmed real, official IOGP document |

## Explicitly ruled out / needs caution

- **`energysafetystat/oil-facility-accidents-2010-present` (Kaggle)** — description only says "accidents and their associated costs," suggesting tabular/financial fields rather than free-text narratives. Uploader has no other footprint online. Do not use until manually verified to contain real narrative text.
- **"SmartQHSE IOGP Life-Saving Rules Machine-Readable Reference" (HuggingFace/Zenodo)** — does not appear to exist. SmartQHSE is a real org with 34 HF datasets on generic HSE topics, but none specifically for IOGP Life-Saving Rules. Do not cite this as a source.

## Fact-checked academic reference (architecture blueprint, not a dataset to reuse directly)

[VelocityEHS PSIF paper (Nature Scientific Reports, 2024)](https://www.nature.com/articles/s41598-024-58824-y) — trained on ~2.2M internal VelocityEHS incident records (not public). Useful for the *method* (weak supervision + DistilBERT/BERT embeddings + XGBoost, F2-optimized), not as a data source. Actual reported results (Table 4), corrected from an earlier inflated draft:

| Model | Precision | Recall | F1 | F2 |
|---|---|---|---|---|
| TFIDF + Logistic Regression | 0.41 | 0.72 | 0.52 | 0.63 |
| TFIDF + XGBoost | 0.46 | 0.77 | 0.58 | 0.69 |
| DistilBERT + Logistic Regression | 0.48 | 0.75 | 0.59 | 0.69 |
| GloVe + XGBoost | 0.50 | 0.75 | 0.60 | 0.69 |
| DistilBERT + XGBoost (primary) | 0.52 | 0.77 | 0.62 | 0.71 |
| BERT + XGBoost | 0.52 | 0.77 | 0.62 | 0.71 |

## Next steps

1. Manually open the Kaggle "Industrial Safety and Health Analytics Database" and confirm schema/row count.
2. Scrape and hand-label the OISD PDFs as a validation set.
3. Decide whether PHMSA data has enough narrative text to be worth the ingestion effort.
