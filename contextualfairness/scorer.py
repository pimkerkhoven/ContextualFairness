# Copyright (c) ContextualFairness contributors.
# Licensed under the MIT License.

import itertools

import polars as pl


class ContextualFairnessResult:
    """Results for contextual fairness
    This class stores the results for the calculations of contextual fairness
    for a specific instances. It provides methods for calculating common
    metrics. As well as access to the dataframe containing the results.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the results of contextual fairness.
    """

    def __init__(self, df):
        self.df = df

    def total_score(self):
        """Retrieve the total contextual fairness score

        Returns
        -------
        np.float64
            The sum of the contextual fairness score for each sample.

        """
        return self.df.select(pl.col("total").sum()).collect().item()

    def group_scores(self, attributes, scaled=False):
        """Calculate the contextual fairness score for each group based on a
        list of attributes defining the groups.

        The considered groups are based on the product of the values for each
        attribute in the result. For example, suppose the attributes are 'sex'
        and 'age', and the values in the result are 'male' and 'female' for
        'sex' and 'young' and 'old' for 'age', the considered groups are:
            - 'male',   'old'
            - 'male',   'young'
            - 'female', 'old'
            - 'female', 'young'

        The score for each group can either be unscaled or scaled. When
        unscaled the score for each group is simply the sum of the scores for
        each sample belonging to the group. When scaled the score for each
        group is scaled relative to the number of samples belonging to the
        group. This scaling is done such that the sum of the scores for each
        group still sums to the total score.

        Parameters
        ----------
        attributes : list[str]
            Attributes used for defining the groups.

        scaled : boolean, default=False
            Flag stating whether the score is scaled or not.

        Returns
        -------
        dict
            Contains a key for each group that is generated based on the specified
            attributes (e.g. sex=male;age=young). For each group, a dict is
            defined containing the contextual fairness "score" and the "data"
            containing the scores for each sample in the group.
        """
        if len(attributes) == 0:
            raise ValueError("Must specify at least one attribute.")

        for attr in attributes:
            if attr not in self.df.collect_schema().names():
                raise ValueError(
                    f"Column with name `{attr}` does not exist in ContextualFairnessResult."
                )

        result = self.df.select(pl.col(attributes[0]).unique())

        for attr in attributes[1:]:
            result = result.join(self.df.select(pl.col(attr).unique()), how="cross")

        result = result.join(
            self.df.with_row_index()
            .group_by(attributes)
            .agg(pl.col("total").sum(), pl.col("index").alias("indices")),
            how="left",
            on=attributes,
        ).with_columns(pl.col("total").fill_null(0), pl.col("indices").fill_null([]))

        if scaled:
            num_in_group = pl.col("indices").list.len()
            denominator = (pl.col("total") / num_in_group).fill_nan(0).sum()
            sum_total = pl.col("total").sum()

            result = result.with_columns(
                total=(pl.col("total") / num_in_group / denominator).fill_nan(0) * sum_total
            )

        return result


def contextual_fairness_score(norms, data, y_pred, outcome_scores=None, weights=None):
    """Calculate contexual fairness scores for each sample.
    This function calculates the contextual fairness for each sample in X by
    first calculating score for each norm. Then, the total contextual fairness
    score for a sample is calculated by taking the weighted sum of the scores
    for each norm.

    Parameters
    ----------
    norms : list[Norm]
        Norms used in calculating the contextual fairness score.

    X : pandas.Dataframe of shape (n_samples, _)
        The samples for which contextual fairness is calculated.

    y_pred : array-like of shape (n_samples,)
        The predictions for the samples.

    outcome_scores : array-like of shape (n_samples,), default=None
        The outcome score for each sample being predicted a specific class,
        when specified this usually is the probabilities for the positive
        class. In case of regression, not specifying outcome_scores will result
        in setting outcome_scores equal to y_pred.

    weights : list[float]
        The weight for each norm.


    Returns
    -------
    ContextualFairnessResult
    """
    if len(norms) < 1:
        raise ValueError("Must specify at least one norm.")

    norm_names = [norm.name for norm in norms]
    if len(norm_names) != len(set(norm_names)):
        raise ValueError("Norm names must be unique.")

    uniform_weights = False
    if weights is None:
        weights = len(norms) * [1 / len(norms)]
        uniform_weights = True

    if len(weights) != len(norms):
        raise ValueError("Weights must have same length as norms.")

    if max(weights) > 1 or min(weights) < 0:
        raise ValueError("All weights must be between 0 and 1.")

    if not uniform_weights and not sum(weights) == 1:
        raise ValueError("Norm weights must sum to 1.")

    if not len(data[list(data.keys())[0]]) == len(y_pred):
        raise ValueError("X and y_pred must have the same length.")

    if outcome_scores is not None and not len(data[list(data.keys())[0]]) == len(
        outcome_scores
    ):
        raise ValueError("X and outcome_scores must have the same length.")

    outcome_scores = y_pred if outcome_scores is None else outcome_scores

    data["predictions"] = y_pred
    data["outcomes"] = outcome_scores

    df = pl.LazyFrame(data)

    for i, norm in enumerate(norms):
        df = norm(df, normalize=True).with_columns(pl.col(norm.name) * weights[i])

    df = df.with_columns(total=pl.sum_horizontal((norm.name for norm in norms)))

    return ContextualFairnessResult(df.collect().lazy())
