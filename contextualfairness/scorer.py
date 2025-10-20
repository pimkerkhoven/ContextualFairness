# TODO: copyright information

import itertools


# TODO: add documentation
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

    def most_unfairly_treated(self, n_samples=10):
        """Retrieve the n_samples with the highest total score.

        Parameters
        ----------
        n_samples : int
            Number of samples to include in result.

        Returns
        -------
        pandas.DataFrame
            The n_samples with the highest total score.
        """
        return self.df.sort_values(by=["total"], ascending=False).head(n_samples)

    def total_score(self):
        """Retrieve the total contextual fairness score

        Returns
        -------
        np.float64
            The sum of the contextual fairness score for each sample.

        """
        return self.df["total"].sum()

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
            containg the samples that belong to the group.
        """
        if len(attributes) == 0:
            raise ValueError("Must specify at least one attribute.")

        for attr in attributes:
            if attr not in self.df.columns:
                raise ValueError(
                    f"Column with name `{attr}` does not exist in ContextualFairnessResult."
                )

        values = [sorted(self.df[attr].unique()) for attr in attributes]
        groups = itertools.product(*values)

        result = dict()
        for group in groups:
            indexer = True
            group_name = ""
            for attr, val in zip(attributes, group):
                indexer = indexer & (self.df[attr] == val)
                group_name += f"{attr}={val};"
            group_name = group_name[:-1]

            result[group_name] = dict()
            result[group_name]["data"] = self.df[indexer]["total"].copy()
            result[group_name]["score"] = result[group_name]["data"].sum()

        if scaled:
            denominator = 0
            for group_name in result.keys():
                if len(result[group_name]["data"]) > 0:
                    denominator += result[group_name]["score"] / len(
                        result[group_name]["data"]
                    )

            for group_name in result.keys():
                if len(result[group_name]["data"]) > 0:
                    scaled_score = (
                        (result[group_name]["score"] / len(result[group_name]["data"]))
                        / denominator
                    ) * self.total_score()

                    if result[group_name]["score"] == 0:
                        ratio = 0
                    else:
                        ratio = scaled_score / result[group_name]["score"]

                    result[group_name]["data"] *= ratio
                    result[group_name]["score"] = scaled_score

        return result


def contextual_fairness_score(norms, X, y_pred, y_pred_probas=None, weights=None):
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

    y_pred_probas : array-like of shape (n_samples,), default=None
        The probabilities for each sample being predicted a specific class,
        usually this is the positive class. In case of regression, not
        specifying y_pred_probas will result in setting y_pred_probas equal to
        y_pred.

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

    if not len(X) == len(y_pred):
        raise ValueError("X and y_pred must have the same length.")

    if y_pred_probas is not None and not len(X) == len(y_pred_probas):
        raise ValueError("X and y_pred_probas must have the same length.")

    outcome_scores = y_pred.copy() if y_pred_probas is None else y_pred_probas

    result_df = X.copy()

    for i, norm in enumerate(norms):
        result_df[norm.name] = norm(X, y_pred, outcome_scores, normalize=True)
        result_df[norm.name] = result_df[norm.name].astype("float64")

        result_df[norm.name] = result_df.loc[:, norm.name] * weights[i]

    result_df["total"] = result_df[(norm.name for norm in norms)].sum(axis=1)

    return ContextualFairnessResult(result_df)
