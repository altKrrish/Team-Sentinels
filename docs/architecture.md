# Prototype Architecture

Five-module pipeline, adapted from the VelocityEHS PSIF paper's method and scaled down for a
hackathon prototype timeline.

## 1. Preprocessing & domain normalization
Jargon/acronym expansion (e.g. "PTW" -> "Permit to Work"), boilerplate removal from mobile-submitted
reports, domain-augmented tokenizer.

## 2. Binary SIF-potential classifier
Fine-tuned transformer (DistilBERT to start — cheaper to train/serve than full BERT, per the
paper's own finding that DistilBERT+XGBoost matched BERT+XGBoost performance) + XGBoost head on
concatenated text embeddings + metadata. Loss/threshold tuned for high recall (F2-weighted) since
missing a real SIF precursor is far costlier than a false positive.

## 3. Multi-label IOGP Life-Saving Rule tagger
Shared transformer embeddings -> 9 independent sigmoid outputs (one per rule), trained initially
on weak labels from `docs/iogp_life_saving_rules.md`, refined with any hand-labeled data collected.

## 4. Precursor pattern clustering
NER (spaCy or BERT-NER) to extract activity / location / barrier-failure entities from flagged
reports. Cluster with BERTopic or HDBSCAN to surface recurring precursor patterns without hardcoding
categories upfront.

## 5. Dashboard
Site/activity risk ranking by SIF-precursor density (not raw incident count), Life-Saving Rule
breakdown, barrier-failure correlation view. See `src/dashboard/`.

## Known constraints to design around
- Extreme class imbalance (SIF-potential is a minority class) — plan for weak supervision to
  expand a small labeled seed set, same as the reference paper did.
- No labeled data for IOGP rule mapping exists anywhere — the weak-label rules ARE the initial
  ground truth; treat early multi-label accuracy skeptically until validated against real OIL text.
- Confidence-threshold routing (e.g. 0.4–0.6 predictions) to human review, if time allows — improves
  trust and gives a feedback loop for retraining.
