import pytest

import pandas as pd

from contextualfairness.norms import BinaryClassificationEqualityNorm


def test_binary_classification_normalizer():
    norm = BinaryClassificationEqualityNorm(1)

    assert norm._normalizer(101) == 50
    assert norm._normalizer(100) == 50

    norm.positive_class_value = "P"

    assert norm._normalizer(101) == 101
    assert norm._normalizer(100) == 100


def test_binary_classification_call_no_positive_class():
    X = pd.DataFrame(
        {"attr": [1, 2, 3, 4, 5, 6]},
        index=["A", "B", "C", "D", "E", "F"],
    )
    y_pred = ["A", "A", "A", "B", "B", "A"]
    norm = BinaryClassificationEqualityNorm(1)

    result = norm(X, y_pred, None)

    assert sum(result) == pytest.approx(2 / 3)
    assert result.equals(
        pd.Series([0, 0, 0, 1 / 3, 1 / 3, 0], index=["A", "B", "C", "D", "E", "F"])
    )


def test_binary_classification_call_no_positive_class_no_normalize():
    X = pd.DataFrame({"attr": [1, 2, 3, 4, 5, 6]})
    y_pred = ["A", "A", "A", "B", "B", "A"]
    norm = BinaryClassificationEqualityNorm(1)

    result = norm(X, y_pred, None, normalize=False)

    assert sum(result) == 2
    assert result.equals(pd.Series([0, 0, 0, 1, 1, 0]))


def test_binary_classification_call_with_positive_class():
    X = pd.DataFrame({"attr": [1, 2, 3, 4, 5, 6]})
    y_pred = ["A", "A", "A", "B", "B", "A"]

    norm = BinaryClassificationEqualityNorm(1, positive_class_value="B")

    result = norm(X, y_pred, None)

    assert sum(result) == 4 / 6
    assert result.equals(pd.Series([1 / 6, 1 / 6, 1 / 6, 0, 0, 1 / 6]))


def test_binary_classification_call_only_one_class_present_no_positive_class():
    X = pd.DataFrame({"attr": [1, 2, 3, 4, 5, 6]})
    y_pred = ["A", "A", "A", "A", "A", "A"]

    norm = BinaryClassificationEqualityNorm(1)

    result = norm(X, y_pred, None)

    assert sum(result) == 0
    assert result.equals(pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))


def test_binary_classification_call_only_one_class_present_with_positive_class():
    X = pd.DataFrame({"attr": [1, 2, 3, 4, 5, 6]})
    y_pred = ["A", "A", "A", "A", "A", "A"]

    norm = BinaryClassificationEqualityNorm(1, positive_class_value="B")

    result = norm(X, y_pred, None)

    assert sum(result) == pytest.approx(6 / 6)
    assert result.equals(pd.Series([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6]))


def test_binary_classification_call_three_classes_present_no_positive_class():
    X = pd.DataFrame({"attr": [1, 2, 3, 4, 5, 6]})
    y_pred = ["A", "A", "A", "B", "B", "C"]

    norm = BinaryClassificationEqualityNorm(1)

    with pytest.raises(ValueError) as e1:
        norm(X, y_pred, None)

    assert (
        str(e1.value)
        == "y_pred must not contain more than two classes for binary classification."
    )


def test_binary_classification_call_only_three_classes_present_with_positive_class():
    X = pd.DataFrame({"attr": [1, 2, 3, 4, 5, 6]})
    y_pred = ["A", "A", "A", "B", "B", "C"]

    norm = BinaryClassificationEqualityNorm(1, positive_class_value="B")

    with pytest.raises(ValueError) as e1:
        norm(X, y_pred, None)

    assert (
        str(e1.value)
        == "y_pred must not contain more than two classes for binary classification."
    )
