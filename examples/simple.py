import pandas as pd

from contextualfairness.scorer import contextual_fairness_score
from contextualfairness.norms import BinaryClassificationEqualityNorm, RankNorm

if __name__ == "__main__":
    ages = ["young", "young", "old", "young", "old"]
    y_true = [1, 0, 0, 1, 0]

    d = {
        "income": [50, 80, 30, 100, 30],
        "age": ages,
        "sex": ["male", "female", "male", "female", "male"],
        "y true": y_true,
    }
    X = pd.DataFrame(data=d)
    y_pred = [1, 0, 1, 0, 1]

    y_pred_probas = [0.5, 0.2, 0.6, 0.4, 0.5]

    def richer_is_better(x):
        return x["income"]

    norms = [BinaryClassificationEqualityNorm(0.5), RankNorm(0.5, richer_is_better)]

    scorer = contextual_fairness_score(
        norms=norms,
        X=X,
        y_pred=y_pred,
        y_pred_probas=y_pred_probas,
    )

    # print(scorer.df.head())
    # print(scorer.total_score())

    print(scorer.group_scores(["sex", "age"]))

    scorer.group_scores(["sex", "age"], scaled=True)
