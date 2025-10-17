import pytest

import pandas as pd

from contextualfairness.scorer import Result


@pytest.fixture
def result_obj():
    df = pd.DataFrame(
        {
            "sex": ["M", "M", "F", "M", "F", "M", "M", "F"],
            "age": ["O", "O", "Y", "Y", "Y", "O", "Y", "O"],
            "total": [0.1, 0.05, 0.2, 0.3, 0.1, 0.025, 0.06, 0.02],
        }
    )

    return Result(df)


def test_most_unfairly_treated(result_obj):
    result = result_obj.most_unfairly_treated()

    assert list(result.index.values) == [3, 2, 0, 4, 6, 1, 5, 7]
    assert list(result.columns) == ["sex", "age", "total"]


def test_most_total_score(result_obj):
    assert result_obj.total_score() == 0.855


def test_group_scores_no_attrs(result_obj):
    with pytest.raises(ValueError) as e1:
        result_obj.group_scores([])
    assert str(e1.value) == "Must specify at least one attribute."


def test_group_scores_non_existent_attr(result_obj):
    with pytest.raises(ValueError) as e1:
        result_obj.group_scores(["attr"])
    assert str(e1.value) == "Column with name `attr` does not exist in Result."


def test_group_score_single_attribute(result_obj):
    result = result_obj.group_scores(["sex"])

    assert "sex=F" in result
    assert "sex=M" in result

    assert result["sex=F"]["score"] == 0.2 + 0.1 + 0.02
    assert result["sex=M"]["score"] == 0.1 + 0.05 + 0.3 + 0.025 + 0.06

    assert list(result["sex=F"]["data"].index.values) == [2, 4, 7]
    assert list(result["sex=M"]["data"].index.values) == [0, 1, 3, 5, 6]

    assert sum(result["sex=F"]["data"]) == result["sex=F"]["score"]
    assert sum(result["sex=M"]["data"]) == result["sex=M"]["score"]


def test_group_score_two_attributes(result_obj):
    result = result_obj.group_scores(["sex", "age"])

    assert "sex=F;age=O" in result
    assert "sex=F;age=Y" in result
    assert "sex=M;age=O" in result
    assert "sex=M;age=Y" in result

    assert result["sex=F;age=O"]["score"] == 0.02
    assert result["sex=F;age=Y"]["score"] == 0.2 + 0.1
    assert result["sex=M;age=O"]["score"] == 0.1 + 0.05 + 0.025
    assert result["sex=M;age=Y"]["score"] == 0.3 + 0.06

    assert list(result["sex=F;age=O"]["data"].index.values) == [7]
    assert list(result["sex=F;age=Y"]["data"].index.values) == [2, 4]
    assert list(result["sex=M;age=O"]["data"].index.values) == [0, 1, 5]
    assert list(result["sex=M;age=Y"]["data"].index.values) == [3, 6]

    assert sum(result["sex=F;age=O"]["data"]) == result["sex=F;age=O"]["score"]
    assert sum(result["sex=F;age=Y"]["data"]) == result["sex=F;age=Y"]["score"]
    assert sum(result["sex=M;age=O"]["data"]) == result["sex=M;age=O"]["score"]
    assert sum(result["sex=M;age=Y"]["data"]) == result["sex=M;age=Y"]["score"]


def test_group_score_scaled_single_attribute(result_obj):
    result = result_obj.group_scores(["sex"], scaled=True)

    denominator = (0.2 + 0.1 + 0.02) / 3 + (0.1 + 0.05 + 0.3 + 0.025 + 0.06) / 5

    assert "sex=F" in result
    assert "sex=M" in result

    assert result["sex=F"]["score"] == ((0.2 + 0.1 + 0.02) / 3 / denominator) * 0.855
    assert (
        result["sex=M"]["score"]
        == ((0.1 + 0.05 + 0.3 + 0.025 + 0.06) / 5 / denominator) * 0.855
    )

    assert list(result["sex=F"]["data"].index.values) == [2, 4, 7]
    assert list(result["sex=M"]["data"].index.values) == [0, 1, 3, 5, 6]

    assert sum(result["sex=F"]["data"]) == pytest.approx(result["sex=F"]["score"])
    assert sum(result["sex=M"]["data"]) == pytest.approx(result["sex=M"]["score"])
