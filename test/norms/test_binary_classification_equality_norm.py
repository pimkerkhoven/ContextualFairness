# Copyright (c) ContextualFairness contributors.
# Licensed under the MIT License.

import pytest
import numpy as np

import polars as pl
from polars.testing import assert_frame_equal

from contextualfairness.norms import BinaryClassificationEqualityNorm


def test_binary_classification_call_no_positive_class():
    data = {"attr": [1, 2, 3, 4, 5, 6], "predictions": ["A", "A", "A", "B", "B", "A"]}
    X = pl.LazyFrame(data)

    norm = BinaryClassificationEqualityNorm()

    result = norm(X)

    data["equality"] = np.array([0, 0, 0, 1 / 3, 1 / 3, 0], dtype=np.float64)
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())


def test_binary_classification_call_no_positive_class_no_normalize():
    data = {"attr": [1, 2, 3, 4, 5, 6], "predictions": ["A", "A", "A", "B", "B", "A"]}
    X = pl.LazyFrame(data)

    norm = BinaryClassificationEqualityNorm()

    result = norm(X, normalize=False)

    data["equality"] = np.array([0, 0, 0, 1, 1, 0], dtype=np.float64)
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())


def test_binary_classification_call_with_positive_class():
    data = {"attr": [1, 2, 3, 4, 5, 6], "predictions": ["A", "A", "A", "B", "B", "A"]}
    X = pl.LazyFrame(data)

    norm = BinaryClassificationEqualityNorm(positive_class_value="B")

    result = norm(X)

    data["equality"] = np.array([1 / 6, 1 / 6, 1 / 6, 0, 0, 1 / 6], dtype=np.float64)
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())



def test_binary_classification_call_only_one_class_present_no_positive_class():
    data = {"attr": [1, 2, 3, 4, 5, 6], "predictions": ["A", "A", "A", "A", "A", "A"]}
    X = pl.LazyFrame(data)

    norm = BinaryClassificationEqualityNorm()

    result = norm(X)

    data["equality"] = np.array([0,0,0,0,0,0], dtype=np.float64)
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())


def test_binary_classification_call_only_one_class_present_with_positive_class():
    data = {"attr": [1, 2, 3, 4, 5, 6], "predictions": ["A", "A", "A", "A", "A", "A"]}
    X = pl.LazyFrame(data)

    norm = BinaryClassificationEqualityNorm(positive_class_value="B")

    result = norm(X)

    data["equality"] = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6], dtype=np.float64)
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())


def test_binary_classification_call_three_classes_present_no_positive_class():
    data = {"attr": [1, 2, 3, 4, 5, 6], "predictions": ["A", "A", "A", "B", "B", "C"]}
    X = pl.LazyFrame(data)

    norm = BinaryClassificationEqualityNorm()

    with pytest.raises(ValueError) as e1:
        norm(X)

    assert (
        str(e1.value)
        == "y_pred must not contain more than two classes for binary classification."
    )


def test_binary_classification_call_only_three_classes_present_with_positive_class():
    data = {"attr": [1, 2, 3, 4, 5, 6], "predictions": ["A", "A", "A", "B", "B", "C"]}
    X = pl.LazyFrame(data)

    norm = BinaryClassificationEqualityNorm(positive_class_value="B")

    with pytest.raises(ValueError) as e1:
        norm(X)

    assert (
        str(e1.value)
        == "y_pred must not contain more than two classes for binary classification."
    )
