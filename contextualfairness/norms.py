# TODO: copyright information

import math

import numpy as np
import pandas as pd


class BinaryClassificationEqualityNorm:
    """Equality norm for binary classification tasks
    This class is used for calculating the equality score for each sample in
    a dataset given the binary classification prediction for each sample.

    The score for a sample is either 0 or 1, based on whether the prediction
    for the sample is respectively equal or not equal to the majority class or
    a user-defined positive class.

    Parameters
    ----------
    positive_class_value : obj, default=None
        The value of the class that is considered to be the postive class,
        i.e., the class that people want to be predicted. For example, in a
        loan approval setting with outcomes True (get the loan) and False
        (not getting the loan), True would (usually) be considerd the postive
        outcome.

    Attributes
    ----------
    name : str
        The (human-readable) name of the norm.
    """

    def __init__(
        self,
        positive_class_value=None,
    ):
        self.name = "Equality"
        self.positive_class_value = positive_class_value

    def __call__(
        self,
        X,
        y_pred,
        _,
        normalize=True,
    ):
        """Calculate the equality score for each sample in X given binary
        classification predictions y_pred.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, _)
            The samples for which the equality score is calculated.

        y_pred : array-like of shape (n_samples,)
            The binary classification predictions for the samples in X.

        _ : obj
            Not applicable for equality norm

        normalize : bool, default=True
            Flag that states whether or not the score is normalized based on
            n_samples.

        Returns
        -------
        pd.Series of shape (n_samples,)
            The equality score (0 or 1) for each sample in X.
        """
        values, counts = np.unique(
            y_pred,
            return_counts=True,
        )

        if len(values) > 2:
            raise ValueError(
                "y_pred must not contain more than two classes for binary classification."
            )

        if self.positive_class_value is None:
            ind = np.argmax(counts)
            reference_class = values[ind]
        else:
            reference_class = self.positive_class_value

        result = pd.Series(
            [0 if y == reference_class else 1 for y in y_pred],
            index=X.index,
        )

        if normalize:
            return result / self._normalizer(len(X))

        return result

    def _normalizer(self, n):
        if self.positive_class_value is None:
            return math.floor(n / 2)

        return n


class RegressionEqualityNorm:
    """Equality norm for regression tasks
    This class is used for calculating the equality score for each sample in
    a dataset given the regression prediction for each sample.

    Equality is defined as all samples having the maximum prediction in
    y_pred. Therefore, the equality score for a sample is the (absolute)
    difference between the prediction for a sample and max(y_pred).

    Parameters
    ----------
    positive_class_value : obj, default=None
        The value of the class that is considered to be the postive class,
        i.e., the class that people want to be predicted. For example, in a
        loan approval setting with outcomes True (get the loan) and False
        (not getting the loan), True would (usually) be considerd the postive
        outcome.

    Attributes
    ----------
    name : str
        The (human-readable) name of the norm.
    """

    def __init__(self):
        self.name = "Equality"

        self._normalizer_val = None

    def __call__(
        self,
        X,
        y_pred,
        _,
        normalize=True,
    ):
        """Calculate the equality score for each sample in X given regression
        predictions y_pred.


        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, _)
            The samples for which the equality score is calculated.

        y_pred : array-like of shape (n_samples,)
            The regression predictions for the samples in X.

        _ : obj
            Not applicable for equality norm.

        normalize : bool, default=True
            Flag that states whether or not the score is normalized based on
            n_samples.

        Returns
        -------
        pd.Series of shape (n_samples,)
            The equality score for each sample in X.
        """
        y_max = np.max(y_pred)
        self._normalizer_val = abs(y_max - np.min(y_pred))

        result = pd.Series(
            [abs(v - y_max) for v in y_pred],
            index=X.index,
        )
        if normalize:
            return result / self._normalizer(len(X))

        return result

    def _normalizer(self, n):
        if self._normalizer_val is None:
            raise RuntimeError(
                "Regression equality norm must have been called at least once before being able to compute normalizer."
            )

        return n * self._normalizer_val


class RankNorm:
    """Rank norm
    This class is used for calculating the rank norm score for each sample in
    a dataset given a ranking based on predictions (outcome ranking) of a model
    and a function for ranking each sample with respect to a certain norm (norm
    ranking). An example of such a norm is equity, which, in a specific context,
    could mean ranking individuals on income.

    The outcome ranking ranks all samples based on the predicitions of a model,
    for example, the probability of being predicted the positive class in a
    binary classification setting or the predictions of a regression model.

    The norm ranking ranks all samples based on a specific function for a norm.
    This function calculates a norm score for each sample, e.g., income.

    To calculate a score for a sample, for each sample we find the number of
    samples that are ranked lower on the norm ranking but higher on the outcome
    ranking. When summing the results for all samples, this is equivalent to
    calculating the kendall-tau distance between the two rankings.

    Parameters
    ----------
    norm_function : Callable
        Function to calculate the norm score for a sample. Takes a sample as
        input and returns a value that can be sorted in order to create the
        norm ranking

    name : str, default=None
        The name of the norm, if None the name of the norm function will be used.
    """

    def __init__(
        self,
        norm_function,
        name=None,
    ):
        self.name = name if name is not None else norm_function.__name__
        self.norm_function = norm_function

    def __call__(
        self,
        X,
        _,
        outcome_scores,
        normalize=True,
    ):
        """Calculate the rank score for each sample in X given the
        outcome_scores and the norm_function.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_samples, _)
            The samples for which the rank score is calculated.

        _ : obj
            Not applicable for rank norm.

        outcome_scores : array-like of shape (n_samples,)
            The outcome scores for the samples in X.

        normalize : bool, default=True
            Flag that states whether or not the score is normalized based on
            n_samples.

        Returns
        -------
        pd.Series of shape (n_samples,)
            The rank score for each sample in X.
        """
        scores = []
        X = X.copy()

        try:
            X["norm_score"] = X.apply(
                self.norm_function,
                axis=1,
            )
        except Exception as e:
            raise RuntimeError(
                f"Error occured when applying norm_function for `{self.name}`."
            ) from e

        X["outcome_scores"] = outcome_scores
        X.sort_values(
            by=["outcome_scores"],
            inplace=True,
        )

        X_norm_sorted = X["norm_score"].copy()
        X_norm_sorted.sort_values(
            inplace=True,
            ascending=False,
        )

        for i in range(len(X) - 1):
            outcome_value_i = X.iloc[i]["outcome_scores"]
            outcome_ranking_offset = 1

            while (
                i + outcome_ranking_offset < len(X)
                and outcome_value_i
                == X.iloc[i + outcome_ranking_offset]["outcome_scores"]
            ):
                outcome_ranking_offset += 1

            higher_outcome_individuals = X.iloc[i + outcome_ranking_offset :].index

            # Map individual from outcome to norm value
            outcome_rank_i = X.iloc[i : i + 1].index[0]
            norm_value_rank_i = X_norm_sorted.index.get_loc(outcome_rank_i)

            norm_value_i = X_norm_sorted.iloc[norm_value_rank_i]
            norm_value_offset = 1

            while (
                norm_value_rank_i + norm_value_offset < len(X_norm_sorted)
                and norm_value_i
                == X_norm_sorted.iloc[norm_value_rank_i + norm_value_offset]
            ):
                norm_value_offset += 1

            lower_norm_value_individuals = X_norm_sorted.iloc[
                norm_value_rank_i + norm_value_offset :
            ].index

            individual_score = len(
                lower_norm_value_individuals.intersection(higher_outcome_individuals)
            )

            scores.append(individual_score)

        # Lowest outcome score always has score 0 (TODO: check claim)
        scores.append(0)

        result = pd.Series(
            scores,
            index=X.index,
        )

        if normalize:
            return result / self._normalizer(len(X))

        return result

    def _normalizer(self, n):
        return n * (n - 1) / 2
