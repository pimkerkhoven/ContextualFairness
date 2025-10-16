import pytest

from contextualfairness.norms import BinaryClassificationEqualityNorm


def test_binary_classification_normalizer():
    norm = BinaryClassificationEqualityNorm(1)

    assert norm.normalizer(101) == 50
    assert norm.normalizer(100) == 50

    norm.positive_class_value = "P"

    assert norm.normalizer(101) == 101
    assert norm.normalizer(100) == 100


def test_binary_classification_call_no_positive_class():
    y_pred = ["A", "A", "A", "B", "B", "A"]
    norm = BinaryClassificationEqualityNorm(1)

    result = norm(None, y_pred, None)

    assert sum(result) == 2
    assert result == [0, 0, 0, 1, 1, 0]


def test_binary_classification_call_with_positive_class():
    y_pred = ["A", "A", "A", "B", "B", "A"]

    norm = BinaryClassificationEqualityNorm(1, positive_class_value="B")

    result = norm(None, y_pred, None)

    assert sum(result) == 4
    assert result == [1, 1, 1, 0, 0, 1]


def test_binary_classification_call_only_one_class_present_no_positive_class():
    y_pred = ["A", "A", "A", "A", "A", "A"]

    norm = BinaryClassificationEqualityNorm(1)

    result = norm(None, y_pred, None)

    assert sum(result) == 0
    assert result == [0, 0, 0, 0, 0, 0]


def test_binary_classification_call_only_one_class_present_with_positive_class():
    y_pred = ["A", "A", "A", "A", "A", "A"]

    norm = BinaryClassificationEqualityNorm(1, positive_class_value="B")

    result = norm(None, y_pred, None)

    assert sum(result) == 6
    assert result == [1, 1, 1, 1, 1, 1]


def test_binary_classification_call_three_classes_present_no_positive_class():
    y_pred = ["A", "A", "A", "B", "B", "C"]

    norm = BinaryClassificationEqualityNorm(1)

    with pytest.raises(ValueError) as e1:
        norm(None, y_pred, None)

    assert (
        str(e1.value)
        == "y_pred must not contain more than two classes for binary classification."
    )


def test_binary_classification_call_only_three_classes_present_with_positive_class():
    y_pred = ["A", "A", "A", "B", "B", "C"]

    norm = BinaryClassificationEqualityNorm(1, positive_class_value="B")

    with pytest.raises(ValueError) as e1:
        norm(None, y_pred, None)

    assert (
        str(e1.value)
        == "y_pred must not contain more than two classes for binary classification."
    )
