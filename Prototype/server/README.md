# CloseCall — the engine

SIF-precursor screening for free-text unsafe-act, unsafe-condition and near-miss
reports. SIH problem statement **26165** (Oil India Limited).

> **This is a prototype.** The engine is trained on **generated narratives**, not
> on OIL's reports — see [Provenance](#provenance). Every safety field the UI
> shows is real model output; the corpus behind it is not real data. `GET /health`
> says so out loud so nothing downstream can quietly imply otherwise.

## Run it

Two commands. Train once, then serve.

```bash
cd server && python train.py
```

```bash
cd server && python -m uvicorn app:api --port 8000
```

`train.py` writes `artifacts/engine.joblib` and `artifacts/metrics.json`. It takes
roughly a minute. The API refuses to start without the engine, rather than serving
an empty dashboard.

Then start the UI from the project root — it reads `.env.local`, which points it at
`http://localhost:8000`:

```bash
npm run dev
```

The header badge reads **Live model** when it is talking to this API and **Demo
data** when it has fallen back to fixtures, so a dead backend is visible rather
than silent.

## What it predicts

Five heads over one shared feature matrix:

| Head | Output | Held-out |
|---|---|---|
| SIF potential | boolean + probability | recall 0.786 · precision 0.590 · F1 0.674 · ROC-AUC 0.870 |
| Life-Saving Rules | 9 independent probabilities | micro F1 0.857 · subset acc 0.580 |
| Severity (continuous) | 0–10 regression | R² 0.892 · MAE 0.529 |
| Hazard energy | 7 classes | 0.980 |
| Barrier state | 5 classes | 0.996 |

Numbers come from `artifacts/metrics.json` and are also served at `GET /metrics`.

The SIF head is deliberately **recall-oriented** (`class_weight="balanced"`). A
precursor screen that misses a fatality precursor has failed at its only job;
one that over-flags has merely cost a reviewer a minute. The consequence is
visible on the dashboard: the model flags ~35% of the stream where the ground
truth rate is ~24%, and the meter labels that as above the industry band rather
than hiding it.

The rule head is one-vs-rest, so its probabilities do **not** sum to 1 — a report
that breaches energy isolation *and* work authorisation should score high on both.

## Why it is not just TF-IDF

TF-IDF reads "no" and "gas test" as two independent features. A safety officer
reads "no gas test" as *a barrier that was never applied* — one fact, and the most
important one in the sentence. So 16 hand-built features sit beside the n-grams,
grouped into 6 families the UI can name: severity indicators, barrier failures,
rule violations, negation handling, measurements, temporal patterns.
See `closecall/features.py`.

Every head is **linear**, which is the point: a feature's push on the logit is
exactly `coefficient × value`. The explanation the reviewer sees is not a
surrogate, an approximation, or a sample — it is the arithmetic the model did.
See `closecall/explain.py`.

## Layout

```
server/
  train.py              build the engine, write metrics
  app.py                the FastAPI service
  artifacts/            engine.joblib, metrics.json   (generated)
  data/reports.csv      real reports, if you have them  (optional)
  closecall/
    corpus.py           the generated narratives + their labels
    normalize.py        OIL shorthand -> canonical terms (GGS, LOTO, PTW, ...)
    features.py         the 16 engineered features, in 6 families
    model.py            the shared matrix + the five heads
    explain.py          feature pushes and verbatim evidence spans
    serve.py            the scored report stream the dashboard reads
    dataset.py          load real labelled reports from CSV
```

## Endpoints

| | |
|---|---|
| `GET /health` | is the model loaded, and **what was it trained on** |
| `GET /metrics` | the held-out numbers, verbatim from the last run |
| `GET /reports` | the scored stream, with reviewer decisions merged over it |
| `POST /classify` | `{"text": "..."}` → one scored report |
| `POST /reports/{id}/review` | record a confirm or override |

A review never alters the model's verdict — the reviewer's decision sits alongside
it, so disagreement stays visible instead of being overwritten. Reviews live in
memory; a restart clears them. That is the one thing to replace with a table when
this goes anywhere real.

## Provenance

OIL's UA/UC and near-miss narratives are internal and unpublished, so there is no
corpus to train on. The public safety datasets that do exist — OSHA Severe Injury
Reports, MSHA, PHMSA incident reports, BSEE, NASA ASRS — carry none of the four
labels this model predicts (SIF potential, Life-Saving Rule, hazard energy,
barrier state). They would have to be hand-labelled first, which is the same
problem one step removed.

So `corpus.py` generates narratives from 29 hazard frames in OIL's own idiom, and
labels them by the rule the SIF literature actually states: **fatal potential =
high hazard energy AND a barrier that was absent, failed, bypassed, inadequate or
never verified.** High energy behind an *intact* barrier is labelled negative on
purpose. That is the doctrine (DEKRA; EEI's SIF-precursor model), not a
convenience.

The generator is seeded (`26165`) and its window ends on a fixed date, so every
run produces the same corpus and the same numbers.

### Swapping in real reports

It is a file drop and one command — no code change.

Put a CSV at `data/reports.csv`, then:

```bash
cd server && python train.py
```

Real data wins automatically whenever that file is present. `--data other.csv`
points elsewhere; `--synthetic` forces the generated corpus even if real data
exists.

Required columns are `text` and `sif`. Everything else is optional and degrades
gracefully. Headers are matched loosely — `narrative`, `description`,
`observation` all resolve to `text`; `sif_potential` and `is_sif` both resolve to
`sif`. See `_ALIASES` in `closecall/dataset.py` for the full list.

| column | values |
|---|---|
| `text` | the narrative, verbatim as the observer wrote it |
| `sif` | `1/0`, `true/false`, `yes/no` |
| `rules` | `;`-separated IOGP ids, e.g. `energy-isolation;hot-work` |
| `energy` | `gravity` `pressure` `electrical` `thermal` `mechanical` `chemical` `motion` |
| `barrier` | `intact` `absent` `failed` `bypassed` `inadequate` `not-verified` |
| `severity_actual`, `severity_potential` | 1–5 |
| `reported_at` | ISO date |

If most rows carry `reported_at`, the split switches from random to
**chronological** 70/15/15 — train on the past, test on the future. That is the
only honest split for real narratives, because reporting language drifts and a
random split lets the model peek at it.

Unknown Life-Saving Rule ids raise rather than being silently dropped, so a typo
in a column of 4,000 rows surfaces immediately.
