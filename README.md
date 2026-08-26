# SIH 2026 — PS 26165: AI/NLP Engine for SIF Precursor Detection

**Problem Statement ID:** 26165
**Title:** AI/NLP Engine to Detect Serious Injury & Fatality (SIF) Precursors in OIL's Unsafe-Act/Unsafe-Condition and Near-Miss Reports
**Organization:** Oil India Limited (OIL)

## Goal

Build a prototype that ingests OIL's free-text UA/UC observation, near-miss, and incident reports and:

1. Classifies each report as **SIF-potential vs. non-SIF-potential**.
2. Tags it to the relevant **IOGP Life-Saving Rule** (Energy Isolation, Hot Work, Confined Space, Line of Fire, etc.).
3. Surfaces **recurring precursor patterns** (activity, location, barrier failure) via an interactive dashboard.

See `docs/` for research notes, dataset evaluation, architecture plan, and the IOGP taxonomy reference used for labeling.

## Repo layout

```
docs/                  research notes, dataset evaluation, architecture, roadmap
data/                  raw / processed datasets (not committed — see data/README)
src/preprocessing/     text normalization, jargon expansion, tokenization
src/models/            SIF/non-SIF classifier, IOGP multi-label tagger, clustering
src/dashboard/         interactive dashboard (site/activity risk ranking, precursor density)
notebooks/             exploratory analysis
```

## Status

Early-stage scaffold — research and dataset selection in progress. No trained model yet.
