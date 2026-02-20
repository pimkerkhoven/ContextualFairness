# Copyright (c) ContextualFairness contributors.
# Licensed under the MIT License.

import pytest

import numpy as np

import polars as pl
from polars.testing import assert_frame_equal

from contextualfairness.norms import RankNorm


def test_rank_call_norm_function_call_non_existent_attribute():
    dummy_norm_stmt = pl.col("dummy")

    df = pl.LazyFrame({"attr": [10, 5, 2], "outcomes": [1, 2, 3]})

    norm = RankNorm(dummy_norm_stmt, "dummy")

    with pytest.raises(Exception):
        norm(df).collect()


def test_rank_call_three_items():
    attr_norm_stmt = pl.col("attr")

    data = {"attr": [10, 5, 2], "outcomes": [1, 2, 3]}

    df = pl.LazyFrame(data)

    norm = RankNorm(attr_norm_stmt, "attr_norm")
    result = norm(df)

    data["attr_norm"] = np.array([2 / 3, 1 / 3, 0 / 3])

    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())


def test_rank_call_three_items_no_normalize():
    attr_norm_stmt = pl.col("attr")

    data = {"attr": [10, 5, 2], "outcomes": [1, 2, 3]}
    df = pl.LazyFrame(data)

    norm = RankNorm(attr_norm_stmt, "attr_norm")
    result = norm(df, normalize=False)

    data["attr_norm"] = np.array([2, 1, 0], dtype=np.float64)
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())


def test_rank_call_four_items():
    attr_norm_stmt = pl.col("attr")

    data = {"attr": [10, 5, 2, 1], "outcomes": [3, 1, 2, 4]}
    df = pl.LazyFrame(data)

    norm = RankNorm(attr_norm_stmt, "attr_norm")
    result = norm(df)

    data["attr_norm"] = np.array([1 / 6, 2 / 6, 1 / 6, 0 / 6])
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())


def test_rank_call_equal_attrs():
    attr_norm_stmt = pl.col("attr")

    data = {"attr": [10, 5, 5, 1], "outcomes": [3, 1, 2, 4]}
    df = pl.LazyFrame(data)

    norm = RankNorm(attr_norm_stmt, "attr_norm")
    result = norm(df)

    data["attr_norm"] = np.array([1 / 6, 1 / 6, 1 / 6, 0 / 6])
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())


def test_rank_call_equal_outcome():
    attr_norm_stmt = pl.col("attr")

    data = {"attr": [10, 5, 2, 1], "outcomes": [3, 2, 2, 4]}
    df = pl.LazyFrame(data)

    norm = RankNorm(attr_norm_stmt, "attr_norm")
    result = norm(df)

    data["attr_norm"] = np.array([1 / 6, 1 / 6, 1 / 6, 0 / 6])
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())


def test_rank_call_eight_items_trivial():
    attr_norm_stmt = pl.col("attr")

    data = {"attr": [8, 7, 6, 5, 4, 3, 2, 1], "outcomes": [8, 7, 6, 5, 4, 3, 2, 1]}
    df = pl.LazyFrame(data)

    norm = RankNorm(attr_norm_stmt, "attr_norm")
    result = norm(df)

    data["attr_norm"] = [0 / 28, 0 / 28, 0 / 28, 0 / 28, 0 / 28, 0 / 28, 0 / 28, 0 / 28]
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())


def test_rank_call_eight_items_complex():
    attr_norm_stmt = pl.col("attr")

    data = {"attr": [8, 5, 3, 7, 5, 4, 2, 4], "outcomes": [6, 8, 3, 4, 3, 6, 5, 2]}
    df = pl.LazyFrame(data)

    norm = RankNorm(attr_norm_stmt, "attr_norm")
    result = norm(df)

    data["attr_norm"] = [1 / 28, 0 / 28, 1 / 28, 3 / 28, 2 / 28, 0 / 28, 0 / 28, 2 / 28]
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())


def test_rank_call_non_numerical_norm_function():
    attr_norm_stmt = pl.col("attr")

    data = {
        "attr": ["H", "E", "C", "G", "E", "D", "B", "D"],
        "outcomes": [6, 8, 3, 4, 3, 6, 5, 2],
    }
    df = pl.LazyFrame(data)

    norm = RankNorm(attr_norm_stmt, "attr_norm")
    result = norm(df)

    data["attr_norm"] = [1 / 28, 0 / 28, 1 / 28, 3 / 28, 2 / 28, 0 / 28, 0 / 28, 2 / 28]
    base = pl.LazyFrame(data)

    assert_frame_equal(result.collect(), base.collect())
