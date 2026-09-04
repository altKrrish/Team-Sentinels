import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinel.explainability import explain, ExplanationResult
import numpy as np
from scipy import sparse


class _MockLinearModel:
    """Minimal mock that mimics a linear model with .coef_ attribute."""
    def __init__(self, n_features=10):
        self.coef_ = np.array([0.5, -0.3, 0.8, 0.0, -0.1, 0.9, 0.2, -0.4, 0.1, 0.0])


def test_explain_returns_result_for_linear_model():
    model = _MockLinearModel()
    # Feature matrix with some non-zero values
    x = sparse.csr_matrix(np.array([[0.5, 0.3, 0.0, 0.7, 0.0, 0.4, 0.0, 0.2, 0.0, 0.0]]))
    result = explain(model, x, top_k_positive=3, top_k_negative=2)
    assert isinstance(result, ExplanationResult)
    assert result.available is True
    assert len(result.top_positive) > 0
    # Top positive contributor should have positive contribution
    assert result.top_positive[0].contribution > 0


def test_explain_handles_stub_model():
    """Stub models without .coef_ should return available=False gracefully."""
    class StubModel:
        pass
    model = StubModel()
    x = sparse.csr_matrix(np.array([[1.0, 0.0, 0.5]]))
    result = explain(model, x)
    assert result.available is False


def test_explain_to_dict():
    model = _MockLinearModel()
    x = sparse.csr_matrix(np.array([[0.5, 0.3, 0.0, 0.7, 0.0, 0.4, 0.0, 0.2, 0.0, 0.0]]))
    result = explain(model, x)
    d = result.to_dict()
    assert "method" in d
    assert "available" in d
    assert "top_positive_contributors" in d
    assert "top_negative_contributors" in d
