import pytest

import pandas as pd

from contextualfairness.norms import RankNorm


def test_rank_normalizer():
    def dummy_norm_function(x):
        x["attr"]

    norm = RankNorm(dummy_norm_function)

    assert norm._normalizer(300) == 44_850


# test not functioning norm function


def test_rank_call_norm_function_call_non_existent_attribute():
    def dummy_norm_function(x):
        return x["dummy"]

    data = pd.DataFrame(data={"attr": [10, 5, 2]})
    outcome_scores = [1, 2, 3]

    norm = RankNorm(dummy_norm_function)

    with pytest.raises(Exception) as e1:
        norm(data, None, outcome_scores)

    assert (
        str(e1.value)
        == "Error occured when applying norm_function for `dummy_norm_function`."
    )


# #TODO: rename norm function to value function? -> Check paper?


def test_rank_call_three_items():
    def attr_norm_function(x):
        return x["attr"]

    data = pd.DataFrame(data={"attr": [10, 5, 2]}, index=["A", "B", "C"])
    outcome_scores = [1, 2, 3]

    norm = RankNorm(attr_norm_function)
    result = norm(data, None, outcome_scores)

    assert result["A"] == 2 / 3
    assert result["B"] == 1 / 3
    assert result["C"] == 0 / 3


def test_rank_call_three_items_no_normalize():
    def attr_norm_function(x):
        return x["attr"]

    data = pd.DataFrame(data={"attr": [10, 5, 2]}, index=["A", "B", "C"])
    outcome_scores = [1, 2, 3]

    norm = RankNorm(attr_norm_function)
    result = norm(data, None, outcome_scores, normalize=False)

    assert result["A"] == 2
    assert result["B"] == 1
    assert result["C"] == 0


def test_rank_call_four_items():
    def attr_norm_function(x):
        return x["attr"]

    data = pd.DataFrame(data={"attr": [10, 5, 2, 1]}, index=["A", "B", "C", "D"])
    outcome_scores = [3, 1, 2, 4]

    norm = RankNorm(attr_norm_function)
    result = norm(data, None, outcome_scores)

    assert result.equals(
        pd.Series([2 / 6, 1 / 6, 1 / 6, 0 / 6], index=["B", "C", "A", "D"])
    )


def test_rank_call_equal_attrs():
    def attr_norm_function(x):
        return x["attr"]

    data = pd.DataFrame(data={"attr": [10, 5, 5, 1]}, index=["A", "B", "C", "D"])
    outcome_scores = [3, 1, 2, 4]

    norm = RankNorm(attr_norm_function)
    result = norm(data, None, outcome_scores)

    assert result.equals(
        pd.Series([1 / 6, 1 / 6, 1 / 6, 0 / 6], index=["B", "C", "A", "D"])
    )


def test_rank_call_equal_outcome():
    def attr_norm_function(x):
        return x["attr"]

    data = pd.DataFrame(data={"attr": [10, 5, 2, 1]}, index=["A", "B", "C", "D"])
    outcome_scores = [3, 2, 2, 4]

    norm = RankNorm(attr_norm_function)
    result = norm(data, None, outcome_scores)

    assert result.equals(
        pd.Series([1 / 6, 1 / 6, 1 / 6, 0 / 6], index=["B", "C", "A", "D"])
    )


def test_rank_call_eight_items_trivial():
    def attr_norm_function(x):
        return x["attr"]

    data = pd.DataFrame(
        data={"attr": [8, 7, 6, 5, 4, 3, 2, 1]},
        index=["A", "B", "C", "D", "E", "F", "G", "H"],
    )
    outcome_scores = [8, 7, 6, 5, 4, 3, 2, 1]

    norm = RankNorm(attr_norm_function)
    result = norm(data, None, outcome_scores)

    assert result.equals(
        pd.Series(
            [0 / 28, 0 / 28, 0 / 28, 0 / 28, 0 / 28, 0 / 28, 0 / 28, 0 / 28],
            index=["H", "G", "F", "E", "D", "C", "B", "A"],
        )
    )


def test_rank_call_eight_items_complex():
    def attr_norm_function(x):
        return x["attr"]

    data = pd.DataFrame(
        data={"attr": [8, 5, 3, 7, 5, 4, 2, 4]},
        index=["A", "B", "C", "D", "E", "F", "G", "H"],
    )
    outcome_scores = [6, 8, 3, 4, 3, 6, 5, 2]

    norm = RankNorm(attr_norm_function)
    result = norm(data, None, outcome_scores)

    assert result.equals(
        pd.Series(
            [2 / 28, 1 / 28, 2 / 28, 3 / 28, 0 / 28, 1 / 28, 0 / 28, 0 / 28],
            index=["H", "C", "E", "D", "G", "A", "F", "B"],
        )
    )


def test_rank_call_non_numerical_norm_function():
    def attr_norm_function(x):
        return x["attr"]

    data = pd.DataFrame(
        data={"attr": ["H", "E", "C", "G", "E", "D", "B", "D"]},
        index=["A", "B", "C", "D", "E", "F", "G", "H"],
    )
    outcome_scores = [6, 8, 3, 4, 3, 6, 5, 2]

    norm = RankNorm(attr_norm_function)
    result = norm(data, None, outcome_scores)

    assert result.equals(
        pd.Series(
            [2 / 28, 1 / 28, 2 / 28, 3 / 28, 0 / 28, 1 / 28, 0 / 28, 0 / 28],
            index=["H", "C", "E", "D", "G", "A", "F", "B"],
        )
    )
