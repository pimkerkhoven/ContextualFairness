import pytest

from contextualfairness.norms import RegressionEqualityNorm


def test_regression_normalizer_before_calling_once():
    norm = RegressionEqualityNorm(1)

    with pytest.raises(RuntimeError) as e1:
        norm.normalizer(100)
    assert (
        str(e1.value)
        == "Regression equality norm must have been called at least once before being able to compute normalizer."
    )


def test_regression_normalizer_after_calling_once():
    y_pred = [50, 100, 30, 100, 250, 175]
    norm = RegressionEqualityNorm(1)

    norm(None, y_pred, None)

    assert norm.normalizer(10) == 2_200


def test_regression_call():
    y_pred = [50, 100, 30, 100, 250, 175]
    norm = RegressionEqualityNorm(1)

    result = norm(None, y_pred, None)

    assert sum(result) == 795
    assert result == [200, 150, 220, 150, 0, 75]
