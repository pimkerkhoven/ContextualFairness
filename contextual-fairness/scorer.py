# TODO: copyright information

import functools


# TODO: add documentation
class Result:
    def __init__(self, df):
        self.df = df

    def most_unfairly_treated(self, n_individuals=10):
        return self.df.sort_values(by=['total'], ascending=False).head(n_individuals)
        
    def total_score(self):
        return self.df['total'].sum()

    def scaled_group_scores(self, attributes=None):
        return self.group_scores(attributes, scaled=True)

    def group_scores(self, attributes, scaled=False):
        if len(attributes) == 0:
            raise ValueError("Must specify at least one attribute")

        for attr in attributes:
            if not attr in X.columns:
                raise ValueError(f"Colum with name `{attr}` does not exist in ....")

        values = [sorted(self.df[attr].unique()) for attr in attributes]
        groups = itertools.product(*values)
        
        result = dict()
        for group in groups:
            indexer = True
            group_name = ""
            for attr, val in zip(attributes, group):
                indexer = indexer & (self.df[attr] == val)
                group_name += f"{attr}={val}"

            result[group_name] = dict()
            result[group_name]['data'] = self.df[indexer]['total'].copy()
            result[group_name]['score'] = result[group_name]['data'].sum()

        if scaled:
            denominator = 0
            for group_name in result.keys():
                denominator += result[group_name]['score'] / len(result[group_name]['data'])

            for group_name in result.keys():
                scaled_score = ((result[group_name]['score'] / len(result[group_name]['data'])) / denominator) * self.total_score()
                ratio = scaled_score / result[group_name]['score']

                result[group_name]['data'] *= ratio
                result[group_name]['score'] = scaled_score

        return result

    

def contexual_fairness_score(norms: ..., X: DataFrame,y_pred:..., y_pred_probas:...=None):
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

    if not sum((norm.weight for norm in norms)) == 1:
        raise UserWarning("Norm weights must sum to one.")

    if not len(X) == len(y_pred):
        raise ValueError("X and y_pred must have the same length.")

    if y_pred_probas is not None and not len(X) == len(y_pred_probas):
        raise ValueError("X and y_pred_probas must have the same length.")

    if y_pred_probas is not None and not len(y_pred) == len(y_pred_probas):
        raise ValueError("y_pred and y_pred_probas must have the same length.")

    outcome_scores = y_pred.copy() if y_pred_probas is None else y_pred_probas

    result = X.copy()

    for norm in norms:
        result.loc[:, norm.name] = norm(X, y_pred, outcome_scores) * norm.weight / norm.normalizer(len(X))

    result.loc[:, 'total'] = functools.reduce(lambda a,b: a + b, [result[norm.name] for norm in norms])

    return Result(result)


    
    
    