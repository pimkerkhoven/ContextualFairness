
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?color=g&style=plastic)](https://opensource.org/licenses/MIT)
[![pypy: v](https://img.shields.io/pypi/v/contextualfairness.svg)](https://pypi.python.org/pypi/contextualfairness)


**ContextualFairness** is a Python package for assessing machine learning fairness with multiple contextual norms. The packages provides functions and classes for defining contextual norms and calculating a fairness score for a model based on these norms for binary classification and regression tasks using tabular data.

The contextual norms allow for not only considering the equality norms, but also other norms such as equity or need. This is important because depening on the context equality is not the only fairness norm that is suitable or even suitable at all.

ContextualFairness allows for a fairness analysis on three levels, the global level, the between-group level (e.g., old vs young people), and the in-group level. As shown in this [paper](), this three level analysis allows for a more detailed fairness analysis and combined with the contextual norms, ContextualFairness allows for more nuanced evalutions of fairness with respect to the societal context an ML system operates in.


## Contents
1. [Installation with pip](#installation-with-pip)
2. [Usage](#usage)
3. [Example](#examples)
4. [Limitations](#limitations)
5. [Citing ContextualFairness](#citing-contextualfairness)


## Installation with pip
1. (Optionally) create a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```
2. Install via pip
```
pip install contextualfairness
```

## Usage


### Formulating contextual norms
The first step in using ContextualFairness is to elicit and define the relevant norms for a specific ML model in a specific context. This is not a technical step, but rather a societal step. For this elicitation, all relevant stakeholders in a specific context should be considered. By using stakeholder elicitation techniques, fairness norms can formulated. Note that this is not a straightforward step and requires careful consideration of the societal context and stakeholders.

An example of formulated norms for an income prediction scenario could be:

- Everybody should get the same prediction.
- People who work more hours should earn more.
- People with a lower education level should earn more.

### Operationalizing norms

To use ContextualFairness, we must first operationalize the norms, to this end ContextualFairness provides three classes: `BinaryClassificationEqualityNorm`, `RegressionEqualityNorm`, and `RankNorm`. The first two are specific for the ML task at hand and the last one can be used for both binary classification and regression.

The `BinaryClassificationEqualityNorm` is operationalized as follows:

```python
from contextualfairness import BinaryClassificationEqualityNorm

binary_classification_equality_norm = BinaryClassificationEqualityNorm()
```

in this case equality means being equal to the majority class in the predictions and we calculate a score for this norm by counting the number of samples that are not predicted the majority class.
Alternatively, we can also specify a positive class. In this case, equality means being predicted the positive class.

```python
binary_classification_equality_norm = BinaryClassificationEqualityNorm(positive_class_value=1)
```

The `RegressionEqualityNorm` is operationalized as follows:
```python
from contextualfairness import RegressionEqualityNorm

regression_equality_norm = RegressionEqualityNorm()
```

Here equality, means having the maximum predicted value. Therefore, we calculate a score for this norm by taking the (absolute) difference between the prediction for each sample and the maximum prediction.

To operationalize a `RankNorm`, we must first specify a function for ranking all samples in the dataset with respect to the norm. For example, for the norms defined above, rank by hours worked if *people who work more hours should earn more*. This gives te following operationalization:

```python
from contextualfairness import RankNorm

def more_hours_worked_is_preferred(x):
    x.hours_worked # assuming x has the attribute `hours_worked`

rank_norm = RegressionEqualityNorm(norm_function=more_hours_worked_is_preferred, name="Work more hours")
```

Alternative, we can also not specify a name, in this case the function name is used to name the norm:
```python
rank_norm = RegressionEqualityNorm(norm_function=more_hours_worked_is_preferred)
```

For rank norms we calculate a score, by for each sample counting the number of samples that are ranked lower with respect to the `norm_function` but higher with respect to an `output_score`. This `output_score` is the probability of being predicted a (positive) class for binary classification or the predicted value for regression.


### Calculating contextual fairness
After operationalizing the norms, we provide these norms to `contextual_fairness_score` to calculate the contextual fairness score for a specific model. We also specifiy a list of  `weights` that will weigh the results for each norm for the total score.

For binary classification this looks as follows:
```python
from contextualfairness import contextual_fairness_score

norms = [binary_classification_equality_norm, rank_norm]

result = contextual_fairness_score(
    norms=norms,
    X=X, # Assume the existence of some pandas.DataFrame dataset
    y_pred=y_pred, # Assume the existence of some array-like of predictions
    outcome_scores=y_pred_probas, # Assume the existence of some array-like of outcome_scores
    weights=[0.5, 0.4]
)
```

Alternatively, we can also not specify the weights to get uniform weights:
```python
result = contextual_fairness_score(norms=norms, X=X, y_pred=y_pred, outcome_scores=y_pred_probas)
```

For regression, we do it as follows:
```python
norms = [regression_equality_norm, rank_norm]

result = contextual_fairness_score(
    norms=norms,
    X=X, # Assume the existence of some pandas.DataFrame dataset
    y_pred=y_pred, # Assume the existence of some array-like of predictions
)
```
Note, that for regression not specifying the `outcome_scores` results in setting `outcome_scores=y_pred`.


### Analyze the results
After calculating the score, we can analyze the results on three levels:

The total score:
```python
result.total_score()
```

The between-group and in-group level:
```python
group_scores = result.group_scores(attributes=["sex", "age"]) # assuming existence of `sex` and `age` attribute in X

print(group_scores["sex=male;age=young"]["score"])
print(group_scores["sex=male;age=young"]["data"])
```
This gives the score for all groups in the dataset (where a group is combination of values for the specified attributes, e.g., sex=male;age=young). These scores an be compared between the different groups. Additionally, the data used for calculating the group scores is also provided to analyze the scores with-in a group.

The group scores can also be scaled relative to their group sizes, as follows:
```python
group_scores = result.group_scores(attributes=["sex", "age"], scaled=True)
```

Finally, for additional analyses the `pandas.DataFrame` containing the results can be accessed as follows:
```python
result.df
```


## Example
We show a short example on the [ACSIncome data](https://github.com/socialfoundations/folktables) using a [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), with the three following norms:

- Everybody should get the same prediction.
- People who work more hours should earn more.
- People with a lower education level should earn more.




```python
from folktables import ACSDataSource, ACSIncome
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from contextualfairness import contextual_fairness_score BinaryClassificationEqualityNorm, RankNorm,


# load and prepare data
data_source = ACSDataSource(survey_year="2016", horizon="1-Year", survey="person", root_dir="data/raw")
acs_data_ca = data_source.get_data(states=["CA"], download="True")
X, y, _ = ACSIncome.df_to_pandas(acs_data_ca)
y = y["PINCP"]
sensitive_attribute = X["SEX"].copy()

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(X, y, sensitive_attribute, test_size=0.2, random_state=0)


# Train model
clf = LogisticRegression(max_iter=10_000, penalty="l2", random_state=42)
clf.fit(X_train, y_train)

# Predict for test data
y_pred = clf.predict(X_test)
y_pred_probas = clf.predict_proba(X_test)[:, 1]


# Define norms
def work_more_hours(x):
    return x["WKHP"]


def lower_education(x):
    return -x["SCHL"]


norms = [
    BinaryClassificationEqualityNorm(positive_class_value=True),
    RankNorm(norm_function=work_more_hours),
    RankNorm(norm_function=lower_education),
]

# Calculate contextual fairness
result = contextual_fairness_score(
    norms=norms, X=X_test, y_pred=y_pred, outcome_scores=y_pred_probas
)

# Analysis
print(result.total_score())
print(result.group_scores(attributes=["SEX"], scaled=True))
```

Additional examples can be found in the `examples` folder. This folder also contains the experiments used for evaluation in the paper [*Assessing machine learning fairness with multiple contextual norms*]().

## Limitations

The most important limitations of the current implementation are:

- Linear norms
- Computational limitation for rank norms on large datasets
- Only tabular data

Further limitations of ContextualFairness can be found in the [paper]().

## Citing ContextualFairness
ContextualFairness is proposed in this [paper](), wich you can cite as follows:

```
@article{kerkhoven2025assessing,
  title={Assessing machine learning fairness with multiple contextual norms},
  author={Kerkhoven, Pim and Dignum, Virginia and Bhuyan, Monowar},
  journal={},
  volume={},
  year={2025}
}
```
