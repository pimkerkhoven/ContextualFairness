# Copyright (c) ContextualFairness contributors.
# Licensed under the MIT License.

import pytest

import pandas as pd

from collections import namedtuple

from contextualfairness.scorer import contextual_fairness_score
from contextualfairness.norms import BinaryClassificationEqualityNorm, RankNorm


Norm = namedtuple("Norm", ["name"])


def test_empty_norms():
    with pytest.raises(ValueError) as e1:
        contextual_fairness_score([], None, None)
    assert str(e1.value) == "Must specify at least one norm."


def test_norm_with_invalid_weight():
    with pytest.raises(ValueError) as e1:
        contextual_fairness_score([Norm(name="dummy_norm")], None, None, weights=[-1])
    assert str(e1.value) == "All weights must be between 0 and 1."


def test_not_enough_weights():
    with pytest.raises(ValueError) as e1:
        contextual_fairness_score([Norm(name="dummy_norm")], None, None, weights=[])
    assert str(e1.value) == "Weights must have same length as norms."


def test_too_many_weights():
    with pytest.raises(ValueError) as e1:
        contextual_fairness_score(
            [Norm(name="dummy_norm")], None, None, weights=[0.5, 0.5]
        )
    assert str(e1.value) == "Weights must have same length as norms."


def test_sum_norm_weights_smaller_than_one():
    with pytest.raises(ValueError) as e1:
        contextual_fairness_score(
            [Norm(name="dummy_norm"), Norm(name="dummy_norm_2")],
            None,
            None,
            weights=[0.5, 0.4],
        )
    assert str(e1.value) == "Norm weights must sum to 1."


def test_sum_norm_weights_larger_than_one():
    with pytest.raises(ValueError) as e1:
        contextual_fairness_score(
            [Norm(name="dummy_norm"), Norm(name="dummy_norm_2")],
            None,
            None,
            weights=[0.5, 0.6],
        )
    assert str(e1.value) == "Norm weights must sum to 1."


def test_sum_norm_names_cannot_be_the_same():
    with pytest.raises(ValueError) as e1:
        contextual_fairness_score(
            [Norm(name="dummy_norm"), Norm(name="dummy_norm")],
            None,
            None,
            weights=[0.5, 0.4],
        )
    assert str(e1.value) == "Norm names must be unique."


def test_X_and_y_pred_not_same_length():
    with pytest.raises(ValueError) as e1:
        contextual_fairness_score(
            [Norm(name="dummy_norm"), Norm(name="dummy_norm_2")],
            [1, 5, 8],
            [1, 2],
            weights=[0.5, 0.5],
        )
    assert str(e1.value) == "X and y_pred must have the same length."


def test_X_and_y_pred_probas_not_same_length():
    with pytest.raises(ValueError) as e1:
        contextual_fairness_score(
            [Norm(name="dummy_norm"), Norm(name="dummy_norm_2")],
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
        BinaryClassificationEqualityNorm(positive_class_value=1),
        RankNorm(norm_function=attr_norm_function),
    ]

    result = contextual_fairness_score(
        norms, X, y_pred, y_pred_probas, weights=[0.5, 0.5]
    ).df

    assert list(result.columns) == [
        "attr",
        "sex",
        "Equality",
        "attr_norm_function",
        "total",
    ]

    print(result)

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


def test_result_passed_in_data_remains_the_same():
    X = pd.DataFrame(
        data={"attr": [10, 5, 2, 1], "sex": ["M", "F", "F", "M"]},
        index=["A", "B", "C", "D"],
    )
    y_pred = [1, 1, 0, 0]
    y_pred_probas = [3, 1, 2, 4]

    def attr_norm_function(x):
        return x["attr"]

    norms = [
        BinaryClassificationEqualityNorm(positive_class_value=1),
        RankNorm(norm_function=attr_norm_function),
    ]

    result = contextual_fairness_score(
        norms, X, y_pred, y_pred_probas, weights=[0.5, 0.5]
    ).df

    assert list(result.columns) == [
        "attr",
        "sex",
        "Equality",
        "attr_norm_function",
        "total",
    ]

    assert list(X.columns) == [
        "attr",
        "sex",
    ]


def test_result_uniform_norms():
    X = pd.DataFrame(
        data={"attr": [10, 5, 2, 1], "sex": ["M", "F", "F", "M"]},
        index=["A", "B", "C", "D"],
    )
    y_pred = [1, 1, 0, 0]
    y_pred_probas = [3, 1, 2, 4]

    def attr_norm_function(x):
        return x["attr"]

    norms = [
        BinaryClassificationEqualityNorm(positive_class_value=1),
        RankNorm(norm_function=attr_norm_function),
        RankNorm(norm_function=attr_norm_function, name="Second rank"),
    ]

    result = contextual_fairness_score(norms, X, y_pred, y_pred_probas).df

    assert list(result.columns) == [
        "attr",
        "sex",
        "Equality",
        "attr_norm_function",
        "Second rank",
        "total",
    ]

    assert list(X.columns) == [
        "attr",
        "sex",
    ]


def test_uniform_norms_should_not_sum_to_one_if_no_weights():
    X = pd.DataFrame(
        data={"attr": [10, 5, 2, 1], "sex": ["M", "F", "F", "M"]},
        index=["A", "B", "C", "D"],
    )
    y_pred = [1, 1, 0, 0]
    y_pred_probas = [3, 1, 2, 4]

    def attr_norm_function(x):
        return x["attr"]

    norms = [
        RankNorm(norm_function=attr_norm_function, name="1"),
        RankNorm(norm_function=attr_norm_function, name="2"),
        RankNorm(norm_function=attr_norm_function, name="3"),
        RankNorm(norm_function=attr_norm_function, name="4"),
        RankNorm(norm_function=attr_norm_function, name="5"),
        RankNorm(norm_function=attr_norm_function, name="6"),
    ]

    result = contextual_fairness_score(norms, X, y_pred, y_pred_probas).df

    assert list(result.columns) == [
        "attr",
        "sex",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "total",
    ]

    assert list(X.columns) == [
        "attr",
        "sex",
    ]


def test_uniform_norms_should_sum_to_one_if_weights():
    with pytest.raises(ValueError) as e1:
        contextual_fairness_score(
            [
                Norm(name="dummy_norm_1"),
                Norm(name="dummy_norm_2"),
                Norm(name="dummy_norm_3"),
                Norm(name="dummy_norm_4"),
                Norm(name="dummy_norm_5"),
                Norm(name="dummy_norm_6"),
            ],
            None,
            None,
            weights=6 * [1 / 6],
        )
    assert str(e1.value) == "Norm weights must sum to 1."
