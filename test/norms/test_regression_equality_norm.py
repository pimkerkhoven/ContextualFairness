# Copyright (c) ContextualFairness contributors.
# Licensed under the MIT License.

import numpy as np

import polars as pl
from polars.testing import assert_frame_equal

from contextualfairness.norms import RegressionEqualityNorm


def test_regression_call_no_normalize():
    data = {"attr": [1, 2, 3, 4, 5, 6], "predictions": [50, 100, 30, 100, 250, 175]}
    X = pl.LazyFrame(data)
    norm = RegressionEqualityNorm()

    result = norm(X, normalize=False)

    data["equality"] = np.array([200, 150, 220, 150, 0, 75], dtype=np.float64)
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())


def test_regression_call_with_normalize():
    data = {"attr": [1, 2, 3, 4, 5, 6], "predictions": [50, 100, 30, 100, 250, 175]}
    X = pl.LazyFrame(data)
    norm = RegressionEqualityNorm()

    result = norm(X, normalize=True)

    data["equality"] = [
        200 / 1320,
        150 / 1320,
        220 / 1320,
        150 / 1320,
        0 / 1320,
        75 / 1320,
    ]
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())

    # assert sum(result) == pytest.approx(795 / 1320)
    # assert result.equals(
    #     pl.Series([200 / 1320, 150 / 1320, 220 / 1320, 150 / 1320, 0 / 1320, 75 / 1320])
    # )
