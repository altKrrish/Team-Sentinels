"""
sentinel.features
==================
Replacement/extension for `MultiModalFeatureExtractor`. Combines:

  1. Word-level TF-IDF (existing 30k-feature vectorizer slot) over the
     NORMALIZED text (transliterated + abbreviation-expanded), so Hindi/
     Bengali/Assamese and Hinglish reports land in the same vocabulary space
     as English OSHA narratives instead of being unseen tokens.
  2. Character n-gram TF-IDF, n in [3, 6] (widened from a narrower default)
     with `analyzer="char_wb"` so it captures subword hazard morphemes
     ("electr-", "-cution", "hydraul-") across spelling variants and romanized
     shorthand without needing every surface form enumerated in the lexicon.
  3. A phonetic-key channel: the same char n-gram vectorizer refit over the
     phonetic-key stream, so "current lag gaya" and "karrent laga gya" hash
     close together even when the raw character n-grams diverge.
  4. Dense interlock/metadata features appended as extra columns: energy
     classes hit, interlock fired flag, corroborate count, word count (to
     let the model itself learn to distrust very short reports), metadata
     energy-triggered flag.

This is deliberately a superset, not a replacement of the existing
pipeline's word TF-IDF -- retraining only the word vectorizer (as originally
proposed) leaves the sparse/short/romanized failure modes untouched, because
the failure is in *what text reaches the vectorizer*, not only in its
vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from . import interlock as interlock_mod
from .energy_metadata import assess as assess_metadata
from .lexicon import ENERGY_CLASSES
from .text_norm import normalize, phonetic_phrase_key

CHAR_NGRAM_RANGE = (3, 6)   # widened per hardening spec
WORD_MAX_FEATURES = 30_000
CHAR_MAX_FEATURES = 20_000
PHONETIC_MAX_FEATURES = 8_000


@dataclass
class ExtractedRow:
    normalized_text: str
    phonetic_text: str
    interlock: interlock_mod.InterlockResult
    metadata: Optional[Dict]
    word_count: int


class MultiModalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Drop-in replacement with the same fit/transform contract as the
    original class, so it can be swapped into an existing sklearn Pipeline
    without touching downstream code.
    """

    def __init__(
        self,
        word_max_features: int = WORD_MAX_FEATURES,
        char_max_features: int = CHAR_MAX_FEATURES,
        phonetic_max_features: int = PHONETIC_MAX_FEATURES,
        char_ngram_range: tuple = CHAR_NGRAM_RANGE,
        word_ngram_range: tuple = (1, 2),
        min_df: int = 2,
    ):
        self.word_max_features = word_max_features
        self.char_max_features = char_max_features
        self.phonetic_max_features = phonetic_max_features
        self.char_ngram_range = char_ngram_range
        self.word_ngram_range = word_ngram_range
        self.min_df = min_df

    # -- sklearn plumbing -----------------------------------------------
    def _rows(self, texts: Sequence[str], metadata_list: Optional[Sequence[Dict]]) -> List[ExtractedRow]:
        rows = []
        meta_iter = metadata_list if metadata_list is not None else [None] * len(texts)
        for text, meta in zip(texts, meta_iter):
            norm = normalize(text)
            phon = phonetic_phrase_key(norm)
            il = interlock_mod.scan(text)
            rows.append(ExtractedRow(
                normalized_text=norm,
                phonetic_text=phon,
                interlock=il,
                metadata=meta,
                word_count=len(norm.split()),
            ))
        return rows

    def fit(self, X: Sequence[str], y=None, metadata: Optional[Sequence[Dict]] = None):
        rows = self._rows(X, metadata)
        norm_texts = [r.normalized_text for r in rows]
        phon_texts = [r.phonetic_text for r in rows]

        self.word_vectorizer_ = TfidfVectorizer(
            max_features=self.word_max_features,
            ngram_range=self.word_ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
        ).fit(norm_texts)

        self.char_vectorizer_ = TfidfVectorizer(
            max_features=self.char_max_features,
            analyzer="char_wb",
            ngram_range=self.char_ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
        ).fit(norm_texts)

        self.phonetic_vectorizer_ = TfidfVectorizer(
            max_features=self.phonetic_max_features,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        ).fit([t for t in phon_texts if t] or ["_empty_"])

        self.energy_class_list_ = list(ENERGY_CLASSES)
        return self

    def transform(self, X: Sequence[str], metadata: Optional[Sequence[Dict]] = None):
        rows = self._rows(X, metadata)
        norm_texts = [r.normalized_text for r in rows]
        phon_texts = [r.phonetic_text or "_empty_" for r in rows]

        word_mat = self.word_vectorizer_.transform(norm_texts)
        char_mat = self.char_vectorizer_.transform(norm_texts)
        phon_mat = self.phonetic_vectorizer_.transform(phon_texts)

        dense_cols = []
        for r in rows:
            energy_flags = [1.0 if ec in r.interlock.energy_classes_hit else 0.0
                             for ec in self.energy_class_list_]
            meta_assessed = assess_metadata(r.metadata) if r.metadata else None
            row_feats = energy_flags + [
                1.0 if r.interlock.fired else 0.0,
                float(len(r.interlock.matches)),
                float(len(r.interlock.corroborate_only)),
                float(r.word_count),
                1.0 if r.word_count < 8 else 0.0,     # sparse-report flag
                1.0 if (meta_assessed and meta_assessed.any_triggered) else 0.0,
            ]
            dense_cols.append(row_feats)

        dense_mat = sparse.csr_matrix(np.array(dense_cols, dtype=np.float64))
        return sparse.hstack([word_mat, char_mat, phon_mat, dense_mat], format="csr")

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = list(self.word_vectorizer_.get_feature_names_out())
        names += [f"char__{f}" for f in self.char_vectorizer_.get_feature_names_out()]
        names += [f"phon__{f}" for f in self.phonetic_vectorizer_.get_feature_names_out()]
        names += [f"energy__{ec}" for ec in self.energy_class_list_]
        names += ["interlock_fired", "interlock_match_count", "corroborate_count",
                  "word_count", "is_sparse_report", "metadata_energy_flag"]
        return np.array(names)


__all__ = ["MultiModalFeatureExtractor", "CHAR_NGRAM_RANGE"]
