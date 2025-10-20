# Copyright (c) ContextualFairness contributors.
# Licensed under the MIT License.

import pytest

import pandas as pd

from contextualfairness.norms import RegressionEqualityNorm


def test_regression_normalizer_before_calling_once():
    norm = RegressionEqualityNorm()

    with pytest.raises(RuntimeError) as e1:
        norm._normalizer(100)
    assert (
        str(e1.value)
        == "Regression equality norm must have been called at least once before being able to compute normalizer."
    )


def test_regression_normalizer_after_calling_once():
    X = pd.DataFrame({"attr": [1, 2, 3, 4, 5, 6]})
    y_pred = [50, 100, 30, 100, 250, 175]
    norm = RegressionEqualityNorm()

    norm(X, y_pred, None)

    assert norm._normalizer(10) == 2_200


def test_regression_call_no_normalize():
    X = pd.DataFrame({"attr": [1, 2, 3, 4, 5, 6]}, index=["A", "B", "C", "D", "E", "F"])
    y_pred = [50, 100, 30, 100, 250, 175]
    norm = RegressionEqualityNorm()

    result = norm(X, y_pred, None, normalize=False)

    assert sum(result) == 795
    assert result.equals(
        pd.Series([200, 150, 220, 150, 0, 75], index=["A", "B", "C", "D", "E", "F"])
    )


def test_regression_call_with_normalize():
    X = pd.DataFrame({"attr": [1, 2, 3, 4, 5, 6]})
    y_pred = [50, 100, 30, 100, 250, 175]
    norm = RegressionEqualityNorm()

    result = norm(X, y_pred, None, normalize=True)

    assert sum(result) == pytest.approx(795 / 1320)
    assert result.equals(
        pd.Series([200 / 1320, 150 / 1320, 220 / 1320, 150 / 1320, 0 / 1320, 75 / 1320])
    )
