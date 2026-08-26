# Problem Statement ID

26165

# Problem Statement Title

AI / NLP Engine to Detect Serious Injury & Fatality (SIF) Precursors in OIL's Unsafe-Act / Unsafe-Condition and Near‑Miss Reports

## Background

OIL’s HSSE platform collects large volumes of free-text Unsafe Act / Unsafe Condition (UA/UC) observations, near-miss reports, and incident records. These reports are typically triaged manually at scheduled intervals (for example, monthly). Manual review can delay identification of recurring high-risk patterns and slows the response to emerging SIF (Serious Injury & Fatality) precursors, making it harder for HSE teams to prioritize interventions effectively.

## Problem Description

The goal is to build a prototype AI/NLP system that automatically processes OIL’s free-text safety reports to achieve three primary capabilities:

1. Classify each report as either "SIF‑potential" or "Non‑SIF‑potential".
2. Tag each report with one or more applicable IOGP Life‑Saving Rules (for example: Energy Isolation, Hot Work, Confined Space, Line of Fire), providing traceable textual evidence for each tag.
3. Identify and surface recurring precursor patterns—such as repeated unsafe activities, frequent problem locations, or barrier failures—and present those findings via an interactive dashboard so HSE teams can spot trends and clusters quickly.

## Expected Outcome / Solution

Deliverables should include a working AI/NLP prototype and an interactive dashboard that together provide the following:

- Automated ingestion and processing of free-text reports (supporting near real-time or batch workflows).
- Accurate classification of reports for SIF potential, with evaluation metrics (precision, recall, F1) and confidence scores.
- Automatic mapping of reports to one or more Life‑Saving Rules, along with highlighted text snippets that justify each mapping to aid explainability.
- Discovery and visualization of recurring precursor patterns (by activity, location, asset, barrier failure), with time-series trends, clustering, and the ability to filter and drill down.
- Prioritization and ranking of sites, teams, or activities by SIF‑precursor density to guide where interventions will have the greatest impact.
- Exportable summaries and filters for HSE workflows, and basic model monitoring indicators to detect concept drift or falling performance.

## Relevant Data Availability

Available inputs include OIL’s UA/UC observations, near-miss reports, and incident records. Typical fields include free-text descriptions plus any structured metadata such as date/time, location, site/asset, category, reporter, and severity. If historical labels or expert annotations for SIF relevance are available, they will significantly accelerate model training and validation.


