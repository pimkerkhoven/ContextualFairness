# TODO: copyright information

import functools
import itertools


# TODO: add documentation
class Result:
    def __init__(self, df):
        self.df = df

    def most_unfairly_treated(self, n_individuals=10):
        return self.df.sort_values(by=["total"], ascending=False).head(n_individuals)

    def total_score(self):
        return self.df["total"].sum()

    def group_scores(self, attributes, scaled=False):
        if len(attributes) == 0:
            raise ValueError("Must specify at least one attribute.")

        for attr in attributes:
            if attr not in self.df.columns:
                raise ValueError(f"Column with name `{attr}` does not exist in Result.")

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
                    ratio = scaled_score / result[group_name]["score"]

                    result[group_name]["data"] *= ratio
                    result[group_name]["score"] = scaled_score

        return result


def contextual_fairness_score(norms, X, y_pred, y_pred_probas=None):
    """Calculate the contexual fairness scores.
    TODO: description

    Parameters
    ----------

    Returns
    -------

    Examples?
    --------

    """
    if len(norms) < 1:
        raise ValueError("Must specify at least one norm.")

    total_norm_weight = 0
    for norm in norms:
        if norm.weight < 0 or norm.weight > 1:
            raise ValueError(
                f"Weight for norm `{norm.name}` is {norm.weight}. Must be between 0 and 1."
            )

        total_norm_weight += norm.weight

    if not total_norm_weight == 1:
        raise ValueError("Norm weights must sum to 1.")

    if not len(X) == len(y_pred):
        raise ValueError("X and y_pred must have the same length.")

    if y_pred_probas is not None and not len(X) == len(y_pred_probas):
        raise ValueError("X and y_pred_probas must have the same length.")

    outcome_scores = y_pred.copy() if y_pred_probas is None else y_pred_probas

    result = X.copy()

    for norm in norms:
        result.loc[:, norm.name] = norm(X, y_pred, outcome_scores)
        result.loc[:, norm.name] = (
            result.loc[:, norm.name] * norm.weight / norm.normalizer(len(X))
        )

    result.loc[:, "total"] = functools.reduce(
        lambda a, b: a + b, [result[norm.name] for norm in norms]
    )

    return Result(result)
