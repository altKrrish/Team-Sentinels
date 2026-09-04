"""
Train the engine and measure it on a held-out split.

    python train.py                     real data if data/reports.csv exists,
                                        otherwise the synthetic corpus
    python train.py --data path.csv     train on a specific export
    python train.py --synthetic         force the generator, ignore any CSV

Writes two files into `artifacts/`:

    engine.joblib   the fitted pipeline - vectorisers, heads, column geometry
    metrics.json    everything measured on the held-out test set

Nothing here is interactive and nothing depends on the wall clock, so two runs on
the same machine produce byte-identical metrics.

On the split. Synthetic rows are independent, so a stratified 70/15/15 is fine.
REAL reports are not: the same job gets written up several times in near-identical
language, so a random split leaks and inflates every number. When the CSV carries
a `reported_at` column this script switches to a DATE split - test set strictly
later than train, which is also the question you actually care about ("would it
have caught next quarter's precursors?"). `metrics.json` records which split ran.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    hamming_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from closecall import dataset
from closecall.corpus import LABEL_NOISE, SIF_THRESHOLD as CORPUS_SIF_THRESHOLD, synthetic_corpus
from closecall.features import FEATURE_COUNT, FEATURE_FAMILIES
from closecall.model import LSR_THRESHOLD, SIF_THRESHOLD, SifEngine

ARTIFACTS = Path(__file__).parent / "artifacts"
SEED = 26165
N_SAMPLES = 9000


def _split(texts, labels):
    """Stratified 70/15/15 on the SIF label."""
    y = [bool(l["sif"]) for l in labels]
    idx = np.arange(len(texts))

    train_idx, rest_idx = train_test_split(
        idx, test_size=0.30, random_state=SEED, stratify=y
    )
    rest_y = [y[i] for i in rest_idx]
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.50, random_state=SEED, stratify=rest_y
    )

    def take(ii):
        return [texts[i] for i in ii], [labels[i] for i in ii]

    return take(train_idx), take(val_idx), take(test_idx)


def _date_split(texts, labels):
    """Chronological 70/15/15 - the honest split for real narratives.

    Undated rows sort to the front, so they land in train rather than
    contaminating the test set.
    """
    order = sorted(range(len(texts)), key=lambda i: labels[i].get("reported_at") or "")
    n = len(order)
    a, b = int(n * 0.70), int(n * 0.85)

    def take(ii):
        return [texts[i] for i in ii], [labels[i] for i in ii]

    return take(order[:a]), take(order[a:b]), take(order[b:])


def _load(args) -> tuple[list[str], list[dict], dict]:
    """Return `(texts, labels, provenance)`.

    Real data wins whenever it is present. That is the whole point of
    `closecall/dataset.py`: swapping the prototype for the real thing is dropping
    a CSV in, not editing code.
    """
    if not args.synthetic:
        loaded = dataset.load(args.data)
        if loaded is not None:
            texts, labels = loaded
            path = Path(args.data) if args.data else dataset.DEFAULT_PATH
            print(f"data           REAL - {path}")
            print(f"               {dataset.describe(labels)}")
            return texts, labels, {"status": "REAL-DATA", "source": str(path)}
        if args.data:
            raise SystemExit(f"no such file: {args.data}")

    texts, labels = synthetic_corpus(N_SAMPLES, seed=SEED)
    print("data           SYNTHETIC - closecall/corpus.py")
    print(f"               drop a labelled CSV at {dataset.DEFAULT_PATH} to train on real reports")
    return texts, labels, {"status": "PROTOTYPE", "source": "closecall/corpus.py"}


def _binary_metrics(y_true, prob) -> dict:
    pred = prob >= SIF_THRESHOLD
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[False, True]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return {
        "confusion": {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)},
        "n": int(tp + fn + fp + tn),
        "positives": int(tp + fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / (tp + fn + fp + tn)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "rocAuc": float(roc_auc_score(y_true, prob)),
        "threshold": SIF_THRESHOLD,
    }


def _multilabel_metrics(engine, y_true_sets, prob) -> dict:
    classes = list(engine.mlb.classes_)
    y_true = engine.mlb.transform(y_true_sets)
    y_pred = (prob >= LSR_THRESHOLD).astype(int)

    # Never predict the empty set - mirrors what `SifEngine.predict` serves.
    empty = y_pred.sum(axis=1) == 0
    if empty.any():
        y_pred[empty, prob[empty].argmax(axis=1)] = 1

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    per_rule = [
        {
            "id": classes[i],
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(classes))
    ]
    micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    return {
        "perRule": sorted(per_rule, key=lambda r: -r["f1"]),
        "microF1": float(micro[2]),
        "macroF1": float(macro[2]),
        "hammingLoss": float(hamming_loss(y_true, y_pred)),
        "subsetAccuracy": float(accuracy_score(y_true, y_pred)),
        "labels": len(classes),
        "tagThreshold": LSR_THRESHOLD,
        "tagsPerReport": float(y_pred.sum(axis=1).mean()),
        "multiRuleShare": float((y_pred.sum(axis=1) > 1).mean()),
    }


def _severity_metrics(y_true, pred, bands) -> dict:
    band_rows = []
    for band in sorted(set(bands)):
        mask = np.asarray(bands) == band
        if not mask.any():
            continue
        band_rows.append(
            {
                "band": int(band),
                "mae": float(mean_absolute_error(y_true[mask], pred[mask])),
                "support": int(mask.sum()),
            }
        )
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "r2": float(r2_score(y_true, pred)),
        "scale": 10,
        "calibration": band_rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the CloseCall engine.")
    parser.add_argument("--data", help="CSV of labelled reports (see closecall/dataset.py)")
    parser.add_argument(
        "--synthetic", action="store_true", help="force the generator, ignore any CSV"
    )
    args = parser.parse_args(argv)

    print("CloseCall - training the engine")
    print("-" * 62)

    texts, labels, provenance = _load(args)

    dated = sum(1 for l in labels if l.get("reported_at"))
    by_date = dated > len(labels) * 0.8
    if by_date:
        (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = _date_split(texts, labels)
        split_name = "chronological 70/15/15 by reportedAt"
    else:
        (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = _split(texts, labels)
        split_name = "stratified 70/15/15 on the SIF label"

    positives = sum(1 for l in labels if l["sif"])
    print(f"corpus         {len(texts):>6} narratives  ({positives / len(texts):.1%} SIF-positive)")
    print(f"split          {len(tr_x):>6} train / {len(va_x)} val / {len(te_x)} test  ({split_name})")

    engine = SifEngine(seed=SEED)
    print("fitting        vectorisers + 7 linear heads ...")
    engine.fit(tr_x, tr_y)

    word_dim = engine.block_offsets["word"][1]
    char_dim = engine.block_offsets["char"][1] - engine.block_offsets["char"][0]
    print(f"features       {word_dim} word + {char_dim} char + {FEATURE_COUNT} engineered")

    # ---- held-out evaluation -------------------------------------------
    X_test = engine.transform(te_x)
    y_sif = np.asarray([bool(l["sif"]) for l in te_y])
    prob_sif = engine.sif_head.predict_proba(X_test)[:, 1]
    prob_rules = np.asarray(engine.rule_head.predict_proba(X_test))
    pred_sev = np.clip(engine.severity_head.predict(X_test), 0.0, 10.0)
    true_sev = np.asarray([float(l["severity"]) for l in te_y])

    binary = _binary_metrics(y_sif, prob_sif)
    multilabel = _multilabel_metrics(engine, [l["rules"] for l in te_y], prob_rules)
    severity = _severity_metrics(
        true_sev, pred_sev, [int(l["severity_potential"]) for l in te_y]
    )

    auxiliary = {
        "hazardEnergy": float(
            accuracy_score([l["energy"] for l in te_y], engine.energy_head.predict(X_test))
        ),
        "severityActualBand": float(
            accuracy_score(
                [int(l["severity_actual"]) for l in te_y], engine.actual_head.predict(X_test)
            )
        ),
        "severityPotentialBand": float(
            accuracy_score(
                [int(l["severity_potential"]) for l in te_y],
                engine.potential_head.predict(X_test),
            )
        ),
    }
    breached = np.asarray([l["barrier"] != "intact" for l in te_y])
    if breached.any():
        auxiliary["barrierState"] = float(
            accuracy_score(
                [l["barrier"] for l, m in zip(te_y, breached) if m],
                engine.barrier_head.predict(X_test[breached]),
            )
        )

    # ---- report ---------------------------------------------------------
    print("-" * 62)
    c = binary["confusion"]
    print("SIF head (binary, logistic regression)")
    print(f"  recall       {binary['recall']:.3f}   <- the one that matters; a miss is a missed precursor")
    print(f"  precision    {binary['precision']:.3f}")
    print(f"  F1           {binary['f1']:.3f}")
    print(f"  ROC-AUC      {binary['rocAuc']:.3f}")
    print(f"  confusion    tp={c['tp']} fn={c['fn']} fp={c['fp']} tn={c['tn']}")
    print("Life-Saving Rules (multi-label, one-vs-rest)")
    print(f"  micro F1     {multilabel['microF1']:.3f}")
    print(f"  macro F1     {multilabel['macroF1']:.3f}")
    print(f"  hamming      {multilabel['hammingLoss']:.3f}")
    print(f"  subset acc   {multilabel['subsetAccuracy']:.3f}")
    print(f"  tags/report  {multilabel['tagsPerReport']:.2f}  ({multilabel['multiRuleShare']:.1%} multi-rule)")
    print("  weakest rules")
    for row in multilabel["perRule"][-3:]:
        print(f"    {row['id']:<28} F1 {row['f1']:.3f}  (n={row['support']})")
    print("Severity (ridge regression, 0-10 continuous)")
    print(f"  MAE          {severity['mae']:.3f}")
    print(f"  RMSE         {severity['rmse']:.3f}")
    print(f"  R2           {severity['r2']:.3f}")
    print("Auxiliary heads (accuracy)")
    for key, value in auxiliary.items():
        print(f"  {key:<22} {value:.3f}")

    # ---- persist --------------------------------------------------------
    ARTIFACTS.mkdir(exist_ok=True)
    joblib.dump(engine, ARTIFACTS / "engine.joblib", compress=3)

    real = provenance["status"] == "REAL-DATA"
    metrics = {
        "provenance": {
            "status": provenance["status"],
            "trainedOn": provenance["source"],
            "warning": (
                "Measured on a held-out slice of real labelled reports."
                if real
                else "These figures measure whether the pipeline learns the pattern it was "
                "shown on generated narratives. They are NOT evidence about performance "
                "on OIL's live reports. Drop a labelled CSV at data/reports.csv and "
                "re-run train.py; no other file changes."
            ),
            "labelRule": (
                "Supplied by the dataset."
                if real
                else "SIF potential = high hazard energy AND a barrier that was absent, "
                "failed, bypassed, inadequate or unverified (DEKRA / EEI precursor "
                "model). Latent risk threshold "
                f"{CORPUS_SIF_THRESHOLD}, with {LABEL_NOISE:.0%} label noise standing in "
                "for reviewer disagreement."
            ),
            "split": split_name,
            "splitCaveat": (
                None
                if by_date
                else "Random split. Real narratives need a DATE split - the same job is "
                "written up repeatedly in near-identical words and a random split leaks. "
                "Include a reportedAt column and train.py switches automatically."
            ),
            "seed": SEED,
        },
        "corpus": {
            "total": len(texts),
            "train": len(tr_x),
            "validation": len(va_x),
            "test": len(te_x),
            "sifPositiveRate": positives / len(texts),
        },
        "pipeline": {
            "wordTfidfDims": int(word_dim),
            "charNgramDims": int(char_dim),
            "engineeredDims": FEATURE_COUNT,
            "engineeredFamilies": {k: v for k, v in FEATURE_FAMILIES.items()},
            "heads": {
                "sif": "LogisticRegression (class_weight=balanced)",
                "rules": "OneVsRestClassifier(LogisticRegression)",
                "severity": "Ridge",
                "auxiliary": "LogisticRegression x4 (energy, barrier, actual band, potential band)",
            },
        },
        "binary": binary,
        "multilabel": multilabel,
        "severity": severity,
        "auxiliary": auxiliary,
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("-" * 62)
    print(f"wrote {ARTIFACTS / 'engine.joblib'}")
    print(f"wrote {ARTIFACTS / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
