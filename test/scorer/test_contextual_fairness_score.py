import pytest

import pandas as pd

from collections import namedtuple

from contextualfairness.scorer import contexual_fairness_score
from contextualfairness.norms import BinaryClassificationEqualityNorm, RankNorm


Norm = namedtuple("Norm", ["weight", "name"])


def test_empty_norms():
    with pytest.raises(ValueError) as e1:
        contexual_fairness_score([], None, None)
    assert str(e1.value) == "Must specify at least one norm."


def test_norm_with_invalid_weight():
    with pytest.raises(ValueError) as e1:
        contexual_fairness_score([Norm(weight=-1, name="dummy_norm")], None, None)
    assert (
        str(e1.value) == "Weight for norm `dummy_norm` is -1. Must be between 0 and 1."
    )


def test_sum_norm_weights_smaller_than_one():
    with pytest.raises(ValueError) as e1:
        contexual_fairness_score(
            [Norm(weight=0.5, name="dummy_norm"), Norm(weight=0.4, name="dummy_norm")],
            None,
            None,
        )
    assert str(e1.value) == "Norm weights must sum to 1."


def test_sum_norm_weights_larger_than_one():
    with pytest.raises(ValueError) as e1:
        contexual_fairness_score(
            [Norm(weight=0.5, name="dummy_norm"), Norm(weight=0.6, name="dummy_norm")],
            None,
            None,
        )
    assert str(e1.value) == "Norm weights must sum to 1."


def test_X_and_y_pred_not_same_length():
    with pytest.raises(ValueError) as e1:
        contexual_fairness_score(
            [Norm(weight=0.5, name="dummy_norm"), Norm(weight=0.5, name="dummy_norm")],
            [1, 5, 8],
            [1, 2],
        )
    assert str(e1.value) == "X and y_pred must have the same length."


def test_X_and_y_pred_probas_not_same_length():
    with pytest.raises(ValueError) as e1:
        contexual_fairness_score(
            [Norm(weight=0.5, name="dummy_norm"), Norm(weight=0.5, name="dummy_norm")],
            [1, 5, 8],
            [1, 2, 5],
            [
                1,
                2,
            ],
        )
    assert str(e1.value) == "X and y_pred_probas must have the same length."


def test_result_is_correct():
    X = pd.DataFrame(
        data={"attr": [10, 5, 2, 1], "sex": ["M", "F", "F", "M"]},
        index=["A", "B", "C", "D"],
    )
    y_pred = [1, 1, 0, 0]
    y_pred_probas = [3, 1, 2, 4]

    def attr_norm_function(x):
        return x["attr"]

    norms = [
        BinaryClassificationEqualityNorm(weight=0.5, positive_class_value=1),
        RankNorm(weight=0.5, norm_function=attr_norm_function),
    ]

    result = contexual_fairness_score(norms, X, y_pred, y_pred_probas).df

    assert list(result.columns) == [
        "attr",
        "sex",
        "Equality",
        "attr_norm_function",
        "total",
    ]

    assert result["Equality"]["A"] == 0
    assert result["Equality"]["B"] == 0
    assert result["Equality"]["C"] == 0.125
    assert result["Equality"]["D"] == 0.125

    assert result["attr_norm_function"]["A"] == 0.5 / 6
    assert result["attr_norm_function"]["B"] == 1 / 6
    assert result["attr_norm_function"]["C"] == 0.5 / 6
    assert result["attr_norm_function"]["D"] == 0

    assert result["total"]["A"] == 0 + 0.5 / 6
    assert result["total"]["B"] == 0 + 1 / 6
    assert result["total"]["C"] == 0.125 + 0.5 / 6
    assert result["total"]["D"] == 0.125 + 0
