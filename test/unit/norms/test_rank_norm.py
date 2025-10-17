import pytest

import pandas as pd

from contextualfairness.norms import RankNorm


def test_rank_normalizer():
    def dummy_norm_function(x):
        x["attr"]

    norm = RankNorm(1, dummy_norm_function)

    assert norm.normalizer(300) == 44_850


# test not functioning norm function


def test_rank_call_norm_function_call_non_existent_attribute():
    def dummy_norm_function(x):
        return x["dummy"]

    data = pd.DataFrame(data={"attr": [10, 5, 2]})
    outcome_scores = [1, 2, 3]

    norm = RankNorm(1, dummy_norm_function)

    with pytest.raises(Exception) as e1:
        norm(data, None, outcome_scores)

    assert (
        str(e1.value)
        == "Error occured when applying norm_function for `dummy_norm_function`."
    )


# test non-numerical output for norm function attribute
#
# #TODO: rename norm function to value function? -> Check paper?


def test_rank_call_three_items():
    def attr_norm_function(x):
        return x["attr"]

    data = pd.DataFrame(data={"attr": [10, 5, 2]}, index=["A", "B", "C"])
    outcome_scores = [1, 2, 3]

    norm = RankNorm(1, attr_norm_function)
    result = norm(data, None, outcome_scores)

    assert result["A"] == 2
    assert result["B"] == 1
    assert result["C"] == 0


def test_rank_call_four_items():
    def attr_norm_function(x):
        return x["attr"]

    data = pd.DataFrame(data={"attr": [10, 5, 2, 1]}, index=["A", "B", "C", "D"])
    outcome_scores = [3, 1, 2, 4]

    norm = RankNorm(1, attr_norm_function)
    result = norm(data, None, outcome_scores)

    assert result["A"] == 1
    assert result["B"] == 2
    assert result["C"] == 1
    assert result["D"] == 0


def test_rank_call_equal_attrs():
    def attr_norm_function(x):
        return x["attr"]

    data = pd.DataFrame(data={"attr": [10, 5, 5, 1]}, index=["A", "B", "C", "D"])
    outcome_scores = [3, 1, 2, 4]

    norm = RankNorm(1, attr_norm_function)
    result = norm(data, None, outcome_scores)

    assert result["A"] == 1
    assert result["B"] == 1
    assert result["C"] == 1
    assert result["D"] == 0


def test_rank_call_equal_outcome():
    def attr_norm_function(x):
        return x["attr"]

    data = pd.DataFrame(data={"attr": [10, 5, 2, 1]}, index=["A", "B", "C", "D"])
    outcome_scores = [3, 2, 2, 4]

    norm = RankNorm(1, attr_norm_function)
    result = norm(data, None, outcome_scores)

    assert result["A"] == 1
    assert result["B"] == 1
    assert result["C"] == 1
    assert result["D"] == 0


def test_rank_call_eight_items_trivial():
    def attr_norm_function(x):
        return x["attr"]

    data = pd.DataFrame(
        data={"attr": [8, 7, 6, 5, 4, 3, 2, 1]},
        index=["A", "B", "C", "D", "E", "F", "G", "H"],
    )
    outcome_scores = [8, 7, 6, 5, 4, 3, 2, 1]

    norm = RankNorm(1, attr_norm_function)
    result = norm(data, None, outcome_scores)

    assert result["A"] == 0
    assert result["B"] == 0
    assert result["C"] == 0
    assert result["D"] == 0
    assert result["E"] == 0
    assert result["F"] == 0
    assert result["G"] == 0
    assert result["H"] == 0


def test_rank_call_eight_items_complex():
    def attr_norm_function(x):
        return x["attr"]

    data = pd.DataFrame(
        data={"attr": [8, 5, 3, 7, 5, 4, 2, 4]},
        index=["A", "B", "C", "D", "E", "F", "G", "H"],
    )
    outcome_scores = [6, 8, 3, 4, 3, 6, 5, 2]

    norm = RankNorm(1, attr_norm_function)
    result = norm(data, None, outcome_scores)

    assert result["A"] == 1
    assert result["B"] == 0
    assert result["C"] == 1
    assert result["D"] == 3
    assert result["E"] == 2
    assert result["F"] == 0
    assert result["G"] == 0
    assert result["H"] == 2
