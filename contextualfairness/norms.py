# Copyright (c) ContextualFairness contributors.
# Licensed under the MIT License.

import polars as pl


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
        self.name = "equality"
        self.positive_class_value = positive_class_value

    def __call__(
        self,
        df,
        normalize=True,
    ):
        """Calculate the equality score for each sample in X given binary
        classification predictions y_pred.

        Parameters
        ----------
        df : polars.lazyFrame of shape (n_cols, n_samples,)
            The dataset for which the equality score is calculated.
            Must contain a `predictions` column containing binary predictions.

        normalize : bool, default=True
            Flag that states whether or not the score is normalized based on
            n_samples.

        Returns
        -------
        df.LazyFrme of shape (n_cols + 1, n_samples)
            The input data extended with the equality score (0 or 1) for each
            sample in df.
        """
        pred_column_name = "predictions"

        if df.unique(pred_column_name).select(pl.len()).collect().item() > 2:
            raise ValueError(
                "y_pred must not contain more than two classes for binary classification."
            )

        if self.positive_class_value is None:
            reference_class = pl.col(pred_column_name).mode().first()
        else:
            reference_class = self.positive_class_value

        normalizer = 1
        if normalize and self.positive_class_value is None:
            normalizer = (pl.len() / 2).floor()
        elif normalize:
            normalizer = pl.len()

        return df.with_columns(
            equality=(
                pl.when(pl.col(pred_column_name) == reference_class)
                .then(0)
                .otherwise(1)
            )
            / normalizer
        )


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

    def __init__(self, lower_is_better=False):
        self.name = "equality"
        self.lower_is_better = lower_is_better

    def __call__(
        self,
        df,
        normalize=True,
    ):
        """Calculate the equality score for each sample in X given regression
        predictions y_pred.


        Parameters
        ----------
        df : polars.lazyFrame of shape (n_cols, n_samples,)
            The dataset for which the equality score is calculated.
            Must contain a `predictions` column containing regression
            predictions.

        normalize : bool, default=True
            Flag that states whether or not the score is normalized based on
            n_samples.

        Returns
        -------
        df.LazyFrme of shape (n_cols + 1, n_samples)
            The input data extended with the equality score for each
            sample in df.
        """
        pred_column_name = "predictions"

        normalizer = pl.lit(1)
        if normalize:
            normalizer = (
                pl.col(pred_column_name).max() - pl.col(pred_column_name).min()
            ) * pl.len()

        if self.lower_is_better:
            return df.with_columns(
                equality=(
                    pl.col(pred_column_name).min() - pl.col(pred_column_name)
                ).abs()
                / normalizer
            )

        return df.with_columns(
            equality=(pl.col(pred_column_name).max() - pl.col(pred_column_name))
            / normalizer
        )


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
        norm_statement,
        name,
    ):
        self.name = name
        self.norm_statement = norm_statement

    def __call__(
        self,
        df,
        normalize=True,
    ):
        """Calculate the rank score for each sample in X given the
        outcome_scores and the norm_function.

        Parameters
        ----------
        df : polars.lazyFrame of shape (n_cols, n_samples,)
            The dataset for which the equality score is calculated.
            Must contain an `outcomes` column.

        normalize : bool, default=True
            Flag that states whether or not the score is normalized based on
            n_samples.

        Returns
        -------
        df.LazyFrme of shape (n_cols + 1, n_samples)
            The input data extended with the rank norm score for each
            sample in df.
        """
        outcome_column_name = "outcomes"
        out_columns = df.collect_schema().names() + [self.name]

        normalizer = 1
        if normalize:
            normalizer = pl.len() * (pl.len() - 1) / 2

        df = df.with_columns(norm=self.norm_statement)

        df_score = (
            df.with_row_index()
            .join_where(
                df,
                (pl.col("norm") > pl.col("norm_right"))
                & (
                    pl.col(outcome_column_name) < pl.col(f"{outcome_column_name}_right")
                ),
            )
            .group_by("index")
            .len()
        )

        return (
            df.with_row_index()
            .join(df_score, on="index", how="left")
            .fill_null(strategy="zero")
            .drop("index")
            .with_columns((pl.col("len") / normalizer).alias(self.name))
            .select(out_columns)
        )
