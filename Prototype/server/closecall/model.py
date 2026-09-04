"""
The engine: one shared feature matrix, several linear heads on top of it.

    text
      |
      +-- normalize()                      standardise industry shorthand
      |
      +-- FeatureUnion
      |     word TF-IDF   (1-2 grams)      phrases: "no gas test", "permit expired"
      |     char TF-IDF   (3-5 grams)      survives 11kV, H2S, kg/cm2, typos
      |     16 engineered features         what a safety officer reads for
      |
      +-- heads (all linear, all on the same matrix)
            sif           LogisticRegression            -> sifPotential, sifConfidence
            rules         OneVsRest LogisticRegression  -> lsrTags[] (multi-label)
            severity      Ridge                         -> severityScore, continuous 0-10
            + auxiliary   LogisticRegression x4         -> hazard energy, barrier state,
                                                            actual band, potential band

The three named heads are the ones in the pitch. The four auxiliary heads exist
because the dashboard's precursor and severity-band fields should come from the
model rather than be invented in the UI; they are trained on the same matrix and
cost almost nothing.

Every head being linear is a deliberate choice, not a limitation. It means an
explanation is exact: the push a feature applies to a prediction is
``coefficient x value``, not an approximation from a surrogate model. That is
what `explain.py` reads, and it is why an HSE reviewer can be shown *why* a
report was flagged.

PROTOTYPE. Trained on the synthetic corpus in `corpus.py` - see the provenance
note there. The contract this serves is stable; the estimator behind it is meant
to be replaced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

from .features import EngineeredFeatures, FEATURE_COUNT
from .normalize import normalize

#: Rules below this probability are suppressed rather than shown. Mirrors
#: LSR_TAG_THRESHOLD in src/lib/contract.js - keep the two in step.
LSR_THRESHOLD = 0.35

#: Decision threshold for the binary head. Left at 0.50; recall is bought with
#: `class_weight="balanced"` instead, which is the more honest lever.
SIF_THRESHOLD = 0.50

HAZARD_ENERGIES = ("gravity", "pressure", "electrical", "thermal", "mechanical", "chemical", "motion")
BARRIER_STATES = ("absent", "failed", "bypassed", "inadequate", "not-verified")


def build_vectoriser() -> FeatureUnion:
    """The shared text -> matrix step.

    ``preprocessor=normalize`` puts the standardisation step inside the fitted
    artifact, so training and serving cannot drift apart: there is no way to
    score a report without normalising it first.
    """
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    preprocessor=normalize,
                    ngram_range=(1, 2),
                    min_df=3,
                    sublinear_tf=True,
                    dtype=np.float64,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    preprocessor=normalize,
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    sublinear_tf=True,
                    dtype=np.float64,
                ),
            ),
            (
                "eng",
                Pipeline([("extract", EngineeredFeatures()), ("scale", MaxAbsScaler())]),
            ),
        ]
    )


@dataclass
class Prediction:
    """One report's model output, before it is dressed in contract field names."""

    sif: bool
    sif_confidence: float
    rules: list[tuple[str, float]]
    severity_score: float
    severity_actual: int
    severity_potential: int
    hazard_energy: str
    barrier_failure: str


class SifEngine:
    """Fit-once, score-many. Pickled whole by `train.py`."""

    def __init__(self, *, seed: int = 26165) -> None:
        self.seed = seed
        self.vectoriser = build_vectoriser()
        self.mlb = MultiLabelBinarizer()

        common = dict(max_iter=4000, solver="liblinear")
        self.sif_head = LogisticRegression(C=4.0, class_weight="balanced", **common)
        self.rule_head = OneVsRestClassifier(
            LogisticRegression(C=6.0, class_weight="balanced", **common)
        )
        self.severity_head = Ridge(alpha=1.0, random_state=seed)

        aux = dict(max_iter=3000, C=3.0)
        self.energy_head = LogisticRegression(**aux)
        self.barrier_head = LogisticRegression(**aux)
        self.actual_head = LogisticRegression(**aux)
        self.potential_head = LogisticRegression(**aux)

        # filled in by fit()
        self.block_offsets: dict[str, tuple[int, int]] = {}
        self.word_vocab_inv: dict[int, str] = {}
        self.contribution_scale: float = 1.0

    # -- fitting ----------------------------------------------------------

    def _record_geometry(self, matrix_width: int) -> None:
        """Where each vectoriser's columns live in the concatenated matrix.

        `explain.py` needs this to read the engineered block's coefficients and
        to map word-TF-IDF columns back to terms.
        """
        word = self.vectoriser.transformer_list[0][1]
        char = self.vectoriser.transformer_list[1][1]
        word_dim = len(word.vocabulary_)
        char_dim = len(char.vocabulary_)
        eng_dim = FEATURE_COUNT

        assert word_dim + char_dim + eng_dim == matrix_width, (
            f"column geometry mismatch: {word_dim}+{char_dim}+{eng_dim} != {matrix_width}"
        )

        self.block_offsets = {
            "word": (0, word_dim),
            "char": (word_dim, word_dim + char_dim),
            "eng": (word_dim + char_dim, matrix_width),
        }
        self.word_vocab_inv = {idx: term for term, idx in word.vocabulary_.items()}

    def fit(self, texts: list[str], labels: list[dict]) -> "SifEngine":
        X = self.vectoriser.fit_transform(texts)
        self._record_geometry(X.shape[1])

        y_sif = np.asarray([bool(l["sif"]) for l in labels])
        y_rules = self.mlb.fit_transform([l["rules"] for l in labels])
        y_sev = np.asarray([float(l["severity"]) for l in labels])

        self.sif_head.fit(X, y_sif)
        self.rule_head.fit(X, y_rules)
        self.severity_head.fit(X, y_sev)

        self.energy_head.fit(X, [l["energy"] for l in labels])
        # The barrier head is only meaningful where a barrier actually broke; an
        # intact barrier has no failure mode to name.
        mask = np.asarray([l["barrier"] != "intact" for l in labels])
        self.barrier_head.fit(X[mask], [l["barrier"] for l, m in zip(labels, mask) if m])

        self.actual_head.fit(X, [int(l["severity_actual"]) for l in labels])
        self.potential_head.fit(X, [int(l["severity_potential"]) for l in labels])

        self._calibrate_contributions(X)
        return self

    def _calibrate_contributions(self, X) -> None:
        """Fix the scale that turns a raw logit push into the contract's -1..1.

        Uses the 95th percentile of per-family absolute contribution across the
        training set, so a typical strong feature reads near +/-1 and only genuine
        outliers clip. Without this the UI's bars would be scaled per report and
        two reports' bars would not be comparable.
        """
        from .features import FAMILY_SLICES

        start, stop = self.block_offsets["eng"]
        coef = self.sif_head.coef_.ravel()[start:stop]
        eng = X[:, start:stop].toarray()

        # contribution of a family = sum over its columns of value * coefficient
        pushes = [
            np.abs(eng[:, np.asarray(cols)] @ coef[np.asarray(cols)])
            for cols in FAMILY_SLICES.values()
        ]

        allpush = np.concatenate(pushes) if pushes else np.asarray([1.0])
        scale = float(np.percentile(allpush, 95))
        self.contribution_scale = scale if scale > 1e-6 else 1.0

    # -- scoring ----------------------------------------------------------

    def transform(self, texts: list[str]):
        return self.vectoriser.transform(texts)

    def predict(self, X) -> list[Prediction]:
        """Score a pre-transformed matrix. Batched, so `GET /reports` is one pass."""
        sif_prob = self.sif_head.predict_proba(X)[:, 1]
        rule_prob = self._rule_proba(X)
        severity = np.clip(self.severity_head.predict(X), 0.0, 10.0)

        energy = self.energy_head.predict(X)
        barrier = self.barrier_head.predict(X)
        actual = self.actual_head.predict(X)
        potential = self.potential_head.predict(X)

        classes = list(self.mlb.classes_)
        out: list[Prediction] = []
        for i in range(X.shape[0]):
            tags = [
                (classes[j], float(rule_prob[i, j]))
                for j in range(len(classes))
                if rule_prob[i, j] >= LSR_THRESHOLD
            ]
            tags.sort(key=lambda t: -t[1])
            if not tags:
                # Never emit an untagged report: show the single best guess so the
                # UI always has a rule to group by, and let the low confidence
                # speak for itself.
                j = int(np.argmax(rule_prob[i]))
                tags = [(classes[j], float(rule_prob[i, j]))]

            act = int(actual[i])
            pot = int(potential[i])
            out.append(
                Prediction(
                    sif=bool(sif_prob[i] >= SIF_THRESHOLD),
                    sif_confidence=float(sif_prob[i]),
                    rules=tags,
                    severity_score=float(severity[i]),
                    severity_actual=act,
                    # A precursor's potential cannot be below what already
                    # happened; the ordering is what the dashboard is built on.
                    severity_potential=max(pot, act),
                    hazard_energy=str(energy[i]),
                    barrier_failure=str(barrier[i]),
                )
            )
        return out

    def _rule_proba(self, X) -> np.ndarray:
        """Per-rule probabilities from the one-vs-rest head.

        Independent per rule and deliberately not normalised - a report that
        breaches three rules should read high on all three.
        """
        probs = self.rule_head.predict_proba(X)
        return np.asarray(probs)

    def predict_one(self, text: str) -> Prediction:
        return self.predict(self.transform([text]))[0]
