# Copyright (c) ContextualFairness contributors.
# Licensed under the MIT License.

import pytest
import numpy as np

import polars as pl
from polars.testing import assert_frame_equal

from contextualfairness.scorer import ContextualFairnessResult


@pytest.fixture
def result_obj():
    data = {
        "sex": ["M", "M", "F", "M", "F", "M", "M", "F"],
        "age": ["O", "O", "Y", "Y", "Y", "O", "Y", "O"],
        "total": [0.1, 0.05, 0.2, 0.3, 0.1, 0.025, 0.06, 0.02],
    }

    df = pl.LazyFrame(data)

    return ContextualFairnessResult(df)


def test_most_total_score(result_obj):
    assert result_obj.total_score() == 0.855


def test_group_scores_no_attrs(result_obj):
    with pytest.raises(ValueError) as e1:
        result_obj.group_scores([])
    assert str(e1.value) == "Must specify at least one attribute."


def test_group_scores_non_existent_attr(result_obj):
    with pytest.raises(ValueError) as e1:
        result_obj.group_scores(["attr"])
    assert (
        str(e1.value)
        == "Column with name `attr` does not exist in ContextualFairnessResult."
    )


def test_group_score_single_attribute(result_obj):
    result = result_obj.group_scores(["sex"])

    data = {
        "sex": ["M", "F"],
        "total": [0.1 + 0.05 + 0.3 + 0.025 + 0.06, 0.2 + 0.1 + 0.02],
        "indices": [
            np.array([0, 1, 3, 5, 6], dtype=np.uint32),
            np.array([2, 4, 7], dtype=np.uint32),
        ],
    }
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect(), check_row_order=False)


def test_group_score_two_attributes(result_obj):
    result = result_obj.group_scores(["sex", "age"])

    data = {
        "sex": ["F", "F", "M", "M"],
        "age": ["O", "Y", "O", "Y"],
        "total": [0.02, 0.2 + 0.1, 0.1 + 0.05 + 0.025, 0.3 + 0.06],
        "indices": [
            np.array([7], dtype=np.uint32),
            np.array([2, 4], dtype=np.uint32),
            np.array([0, 1, 5], dtype=np.uint32),
            np.array([3, 6], dtype=np.uint32),
        ],
    }
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect(), check_row_order=False)


def test_group_score_scaled_single_attribute(result_obj):
    result = result_obj.group_scores(["sex"], scaled=True)

    denominator = (0.2 + 0.1 + 0.02) / 3 + (0.1 + 0.05 + 0.3 + 0.025 + 0.06) / 5
    data = {
        "sex": ["M", "F"],
        "total": [
            ((0.1 + 0.05 + 0.3 + 0.025 + 0.06) / 5 / denominator) * 0.855,
            ((0.2 + 0.1 + 0.02) / 3 / denominator) * 0.855,
        ],
        "indices": [
            np.array([0, 1, 3, 5, 6], dtype=np.uint32),
            np.array([2, 4, 7], dtype=np.uint32),
        ],
    }
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect(), check_row_order=False)


def test_group_score_scaled_with_empty_group():
    df = pl.LazyFrame(
        {
            "sex": ["M", "M", "F", "M", "F", "M", "M", "F"],
            "age": ["O", "O", "Y", "Y", "Y", "O", "Y", "Y"],
            "total": [0.1, 0.05, 0.2, 0.3, 0.1, 0.025, 0.06, 0.02],
        }
    )

    result_obj = ContextualFairnessResult(df)
    result = result_obj.group_scores(["sex", "age"], scaled=True)

    denominator = (0.2 + 0.1 + 0.02) / 3 + (0.1 + 0.05 + 0.025) / 3 + (0.3 + 0.06) / 2

    data = {
        "sex": ["F", "F", "M", "M"],
        "age": ["O", "Y", "O", "Y"],
        "total": [
            0.0,
            ((0.2 + 0.1 + 0.02) / 3 / denominator) * 0.855,
            ((0.1 + 0.05 + 0.025) / 3 / denominator) * 0.855,
            ((0.3 + 0.06) / 2 / denominator) * 0.855,
        ],
        "indices": [
            np.array([], dtype=np.uint32),
            np.array([2, 4, 7], dtype=np.uint32),
            np.array([0, 1, 5], dtype=np.uint32),
            np.array([3, 6], dtype=np.uint32),
        ],
    }
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect(), check_row_order=False)


def test_group_score_scaled_with_group_with_score_of_0():
    df = pl.LazyFrame(
        {
            "income": [50, 80, 30, 100, 30],
            "age": ["young", "young", "old", "young", "old"],
            "sex": ["male", "female", "male", "female", "male"],
            "y true": [1, 0, 0, 1, 0],
            "Equality": [0.0, 0.25, 0.0, 0.25, 0.0],
            "richer_is_better": [0.05, 0.15, 0.0, 0.15, 0.0],
            "total": [0.05, 0.4, 0.0, 0.4, 0.0],
        }
    )

    result_obj = ContextualFairnessResult(df)
    result = result_obj.group_scores(["sex", "age"], scaled=True)

    print(result.collect())

    denominator = (0.05) / 1 + (0.4 + 0.4) / 2

    data = {
        "sex": ["male", "male", "female", "female"],
        "age": ["old", "young", "old", "young"],
        "total": [
            0.0,
            ((0.05) / 1 / denominator) * 0.85,
            0.0,
            ((0.4 + 0.4) / 2 / denominator) * 0.85,
        ],
        "indices": [
            np.array([2, 4], dtype=np.uint32),
            np.array([0], dtype=np.uint32),
            np.array([], dtype=np.uint32),
            np.array([1, 3], dtype=np.uint32),
        ],
    }
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect(), check_row_order=False)
