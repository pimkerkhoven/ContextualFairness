# Copyright (c) ContextualFairness contributors.
# Licensed under the MIT License.

import pytest
import numpy as np

import polars as pl
from polars.testing import assert_frame_equal

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
    data = {"attr": [1, 2, 3]}
    with pytest.raises(ValueError) as e1:
        contextual_fairness_score(
            [Norm(name="dummy_norm"), Norm(name="dummy_norm_2")],
            data,
            [1, 2],
            weights=[0.5, 0.5],
        )
    assert str(e1.value) == "X and y_pred must have the same length."


def test_X_and_y_pred_probas_not_same_length():
    data = {"attr": [1, 2, 3]}
    with pytest.raises(ValueError) as e1:
        contextual_fairness_score(
            [Norm(name="dummy_norm"), Norm(name="dummy_norm_2")],
            data,
            [1, 2, 5],
            [1, 2],
        )
    assert str(e1.value) == "X and outcome_scores must have the same length."


def test_result_is_correct():
    data = {"attr": [10, 5, 2, 1], "sex": ["M", "F", "F", "M"]}
    y_pred = [1, 1, 0, 0]
    y_pred_probas = [3, 1, 2, 4]

    attr_norm_stmt = pl.col("attr")

    norms = [
        BinaryClassificationEqualityNorm(positive_class_value=1),
        RankNorm(norm_statement=attr_norm_stmt, name="attr_norm"),
    ]

    result = contextual_fairness_score(
        norms, data, y_pred, y_pred_probas, weights=[0.5, 0.5]
    ).df

    data["predictions"] = y_pred
    data["outcomes"] = y_pred_probas
    data["equality"] = np.array([0, 0, 0.125, 0.125], dtype=np.float64)
    data["attr_norm"] = np.array([0.5 / 6, 1 / 6, 0.5 / 6, 0], dtype=np.float64)
    data["total"] = np.array(
        [
            0 + 0.5 / 6,
            0 + 1 / 6,
            0.125 + 0.5 / 6,
            0.125 + 0,
        ],
        dtype=np.float64,
    )
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())


def test_result_uniform_norms():
    data = {"attr": [10, 5, 2, 1], "sex": ["M", "F", "F", "M"]}
    y_pred = [1, 1, 0, 0]
    y_pred_probas = [3, 1, 2, 4]

    attr_norm_stmt = pl.col("attr")

    norms = [
        BinaryClassificationEqualityNorm(positive_class_value=1),
        RankNorm(norm_statement=attr_norm_stmt, name="attr_norm"),
        RankNorm(norm_statement=attr_norm_stmt, name="attr_norm_2"),
    ]

    result = contextual_fairness_score(norms, data, y_pred, y_pred_probas).df.collect()

    assert list(result.columns) == [
        "attr",
        "sex",
        "predictions",
        "outcomes",
        "equality",
        "attr_norm",
        "attr_norm_2",
        "total",
    ]


def test_uniform_norms_should_not_sum_to_one_if_no_weights():
    data = {"attr": [10, 5, 2, 1], "sex": ["M", "F", "F", "M"]}
    y_pred = [1, 1, 0, 0]
    y_pred_probas = [3, 1, 2, 4]

    attr_norm_stmt = pl.col("attr")

    norms = [
        RankNorm(norm_statement=attr_norm_stmt, name="1"),
        RankNorm(norm_statement=attr_norm_stmt, name="2"),
        RankNorm(norm_statement=attr_norm_stmt, name="3"),
        RankNorm(norm_statement=attr_norm_stmt, name="4"),
        RankNorm(norm_statement=attr_norm_stmt, name="5"),
        RankNorm(norm_statement=attr_norm_stmt, name="6"),
    ]

    result = contextual_fairness_score(norms, data, y_pred, y_pred_probas).df.collect()

    assert list(result.columns) == [
        "attr",
        "sex",
        "predictions",
        "outcomes",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "total",
    ]


def test_uniform_norms_should_sum_to_one_if_weights():
    with pytest.raises(ValueError) as e1:
        contextual_fairness_score(
            [Norm(name=f"dumm_norm_{i}") for i in range(49)],
            None,
            None,
            weights=49 * [1 / 49],
        )
    assert str(e1.value) == "Norm weights must sum to 1."
