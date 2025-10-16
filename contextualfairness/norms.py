# TODO: copyright information

import math

import numpy as np
import pandas as pd


class BinaryClassificationEqualityNorm:
    def __init__(self, weight, positive_class_value=None):
        self.weight = weight
        self.name = "Equality"
        self.positive_class_value = positive_class_value

    def __call__(self, X, y_pred, _):
        values, counts = np.unique(y_pred, return_counts=True)

        if len(values) > 2:
            raise ValueError(
                "y_pred must not contain more than two classes for binary classification."
            )

        if self.positive_class_value is None:
            ind = np.argmax(counts)
            reference_class = values[ind]
        else:
            reference_class = self.positive_class_value

        return [0 if y == reference_class else 1 for y in y_pred]

    def normalizer(self, n):
        if self.positive_class_value is None:
            return math.floor(n / 2)

        return n


class RegressionEqualityNorm:
    def __init__(self, weight):
        self.weight = weight
        self.name = "Equality"

        self._normalizer_val = None

    def __call__(self, X, y_pred, _):
        y_max = np.max(y_pred)
        self._normalizer_val = abs(y_max - np.min(y_pred))

        return [abs(v - y_max) for v in y_pred]

    def normalizer(self, n):
        if self._normalizer_val is None:
            raise RuntimeError(
                "Regression equality norm must have been called at least once before being able to compute normalizer."
            )

        # print(n, self._normalizer_val)
        return n * self._normalizer_val


class RankNorm:
    def __init__(self, weight, norm_function, name=None):
        self.weight = weight
        self.name = name if name is not None else norm_function.__name__
        self.norm_function = norm_function

    def __call__(self, X, _, outcome_score):
        scores = []

        X = X.copy()
        X["norm_score"] = X.apply(self.norm_function, axis=1)
        X["outcome_score"] = outcome_score
        X.sort_values(by=["outcome_score"], inplace=True)

        X_norm_sorted = X["norm_score"].copy()
        X_norm_sorted.sort_values(inplace=True, ascending=False)

        for i in range(len(X) - 1):
            indices_j_to_n = X.iloc[i + 1 :].index
            index_i = X.iloc[i : i + 1].index[0]
            index_i_row_in_norms = X_norm_sorted.index.get_loc(index_i)

            norm_indices = X_norm_sorted.iloc[index_i_row_in_norms + 1 :].index

            # print(norm_indices.intersection(indices_j_to_n))

            individual_score = len(norm_indices.intersection(indices_j_to_n))

            scores.append(individual_score)

        # Lowest outcome score always has score 0 (TODO: check claim)
        scores.append(0)
        # print(scores)
        return pd.Series(scores, index=X.index)

    def normalizer(self, n):
        return n * (n - 1) / 2


class AttributeRankNorm(RankNorm):
    def __init__(self, weight, attribute, name=None, lower_is_better=False):
        if lower_is_better:

            def norm_function(x):
                return -x[attribute]
        else:

            def norm_function(x):
                return x[attribute]

        if name is None:
            if lower_is_better:
                name = f"Lower {attribute} is better"
            else:
                name = f"Higher {attribute} is better"

        super().__init__(weight, norm_function, name)
        self.attribute = attribute

    def __call__(self, X, _, outcome_score):
        # check attribute exists and is numeric
        if self.attribute not in X:
            raise ValueError(f"Colum with name `{self.attribute}` does not exist in X.")

        # TODO: check is numeric

        super().__call__(X, _, outcome_score)
