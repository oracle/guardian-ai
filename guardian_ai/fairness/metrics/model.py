#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Fairness metrics for evaluating a model"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

from guardian_ai.fairness.utils.lazy_loader import LazyLoader
from guardian_ai.fairness.metrics.utils import (
    DEFAULT_DISTANCE,
    DEFAULT_REDUCTION,
    _DistanceMetric,
    _FairnessScorer,
    _get_check_arrays,
    _get_check_distance,
    _get_check_inputs,
    _get_check_reduction,
    _get_score_group_from_metrics,
    _place_space_before_capital_letters,
    _y_to_aifm_ds,
)
from guardian_ai.utils.exception import GuardianAIValueError

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from aif360.metrics import ClassificationMetric
else:
    np = LazyLoader("numpy")
    pd = LazyLoader("pandas")
    ClassificationMetric = LazyLoader(
        "aif360.metrics", "ClassificationMetric", suppress_import_warnings=True
    )


def _model_metric(
    y_true: Optional[Union[pd.Series, np.ndarray, List]],
    y_pred: Union[pd.Series, np.ndarray, List],
    subgroups: pd.DataFrame,
    metric: str,
    distance_measure: Optional[str],
    reduction: Optional[str],
    allow_y_true_none: bool,
    allow_distance_measure_none: bool,
):
    """
    Compute engine for all model metrics.

    This computes a given metric on all subgroup pairs for a specified ``subgroups`` input.

    Parameters
    ----------
    y_true : pandas.Series, numpy.ndarray, list or None
        Array of groundtruth labels.
    y_pred : pandas.Series, numpy.ndarray, list
        Array of model predictions.
    subgroups : pandas.DataFrame
        Dataframe containing protected attributes for each instance.
    metric : str
        Name of the base metric to be called.
    distance_measure : str or None
        Determines the distance used to compare a subgroup's metric
        against the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.
            Only allowed if `allow_distance_measure_none` is set to True
    reduction : str or None
        Determines how to reduce scores on all subgroups to
        a single output. Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.
    allow_y_true_none : bool
        Whether or not to allow `y_true` to be set to ``None``.
    allow_distance_measure_none : bool
        Whether or not to allow ``distance_measure`` to be set
        to ``None``.


    Returns
    -------
    float, dict
        The computed metric value, with format according to `reduction`.

    """
    y_true, y_pred = _get_check_arrays(y_true, y_pred, allow_y_true_none)
    (
        reduction,
        distance,
        attr_vals_to_idx,
        attr_idx_to_vals,
        subgroup_divisions,
    ) = _get_check_inputs(
        reduction, distance_measure, subgroups, allow_distance_measure_none
    )

    ds_pred = _y_to_aifm_ds(y_pred, subgroups, attr_vals_to_idx)

    # Certain metrics like statistical disparity don't use ground truth labels.
    # AIF360 still needs a labels dataset, so we copy the predicted dataset.
    if y_true is None:
        ds_true = ds_pred.copy()
    else:
        ds_true = _y_to_aifm_ds(y_true, subgroups, attr_vals_to_idx)

    groups = []
    scores = []
    # subgroup_divisions is a list of all subgroup pairs,
    # e.g. [([{'sex': 0, 'race': 0}], [{'sex': 0, 'race': 1}]), ...]
    for unpriv_group, priv_group in subgroup_divisions:
        subgroup_metrics = ClassificationMetric(
            ds_true, ds_pred, unpriv_group, priv_group
        )

        score, unpriv_group_repr = _get_score_group_from_metrics(
            subgroup_metrics, distance, metric, unpriv_group, attr_idx_to_vals
        )

        scores.append(score)
        groups.append(unpriv_group_repr)

    return reduction(groups, scores)


class _ModelFairnessScorer(_FairnessScorer):
    """
    Common base object for all model metrics.

    This stores settings to pass on to the ``_model_metric`` compute
    engine and does subgroups generation from a `protected_attributes` array on
    an input array of instances ``X``.

    Parameters
    ----------
    protected_attributes: pandas.Series, numpy.ndarray, list, str
        Array of attributes or single attribute that should be treated as
        protected. If an attribute is protected, then all of its unique
        values are considered as subgroups.
    metric : str or Callable
        Name of the base metric to be called.
    distance_measure : str or None, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroup pairs to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    allow_distance_measure_none : bool, default=True
        Whether or not to allow ``distance_measure`` to be set to ``None``.
    """

    def __init__(
        self,
        protected_attributes: Union[pd.Series, np.ndarray, List, str],
        metric: Union[str, Callable],
        distance_measure: Optional[str] = DEFAULT_DISTANCE,
        reduction: Optional[str] = DEFAULT_REDUCTION,
        allow_distance_measure_none: bool = True,
    ):
        super().__init__(protected_attributes, metric)

        self.distance_measure = _get_check_distance(
            distance_measure, allow_distance_measure_none
        )
        self.reduction = _get_check_reduction(reduction)

    def __call__(  # type: ignore[override]
        self,
        model: Any,
        X: pd.DataFrame,
        y_true: Union[pd.Series, np.ndarray, List],
        supplementary_features: Optional[pd.DataFrame] = None,
    ):
        """
        Compute the metric using a model's predictions on a given array
        of instances ``X``.

        Parameters
        ----------
        model: Any
            Object that implements a `predict(X)` function to collect
            categorical predictions.
        X : pandas.DataFrame
            Array of instances to compute the metric on.
        y_true : pandas.Series, numpy.ndarray, list
            Array of groundtruth labels.
        supplementary_features : pandas.DataFrame or None, default=None
            Array of supplementary features for each instance. Used in case
            one attribute in ``self.protected_attributes`` is not contained by
            ``X`` (e.g. if the protected attribute is not used by the model).

        Returns
        -------
        float, dict
            The computed metric value, with format according to ``self.reduction``.


        Raises
        ------
        GuardianAIValueError
            - if a feature is present in both ``X``
              and ``supplementary_features``.

        """
        y_pred = model.predict(X)

        subgroups = self._get_check_subgroups(X, supplementary_features)

        return self.metric(
            y_true, y_pred, subgroups, self.distance_measure, self.reduction
        )

    @property
    def display_name(self):
        base_display_name = super().display_name

        fullname = " ".join(
            [
                self.reduction.display_name,
                base_display_name,
                self.distance_measure.display_name,
                self._display_name_protected_attributes,
            ]
        )

        fullname = " ".join(fullname.split())

        return _place_space_before_capital_letters(fullname)


class ModelStatisticalParityScorer(_ModelFairnessScorer):  # noqa: D412
    """
    Measure the statistical parity [1] of a model's output between all subgroup pairs.

    Statistical parity (also known as Base Rate or Disparate Impact) states that
    a predictor is unbiased if the prediction is independent of the protected
    attribute.

    Statistical Parity is calculated as PP / N, where PP and N are the number of
    Positive Predictions and total Number of predictions made, respectively.

    Perfect score
        A perfect score for this metric means that the model does not predict
        positively any of the subgroups at a different rate than it does for the
        rest of the subgroups. For example, if the protected attributes are race
        and sex, then a perfect statistical parity would mean that all combinations
        of values for race and sex have identical ratios of positive predictions.
        Perfect values are:

        - 1 if using ``'ratio'`` as ``distance_measure``.
        - 0 if using ``'diff'`` as ``distance_measure``.

    Parameters
    ----------
    protected_attributes: pandas.Series, numpy.ndarray, list, str
        Array of attributes or single attribute that should be treated as
        protected. If an attribute is protected, then all of its unique
        values are considered as subgroups.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.


    References
    ----------
    [1] `Cynthia Dwork et al. "Fairness Through Awareness". Innovations in
    Theoretical Computer Science. 2012. <https://arxiv.org/abs/1104.3913>`_

    Examples
    --------

    .. code-block:: python

        from guardian_ai.fairness.metrics import ModelStatisticalParityScorer

        scorer = ModelStatisticalParityScorer(['race', 'sex'])
        scorer(model, X, y_true)

    This metric does not require `y_true`. It can also be called using

    .. code-block:: python

        scorer(model, X)
    """  # noqa: D412

    def __init__(
        self,
        protected_attributes: Union[pd.Series, np.ndarray, List, str],
        distance_measure: str = DEFAULT_DISTANCE,
        reduction: Optional[str] = DEFAULT_REDUCTION,
    ):
        super().__init__(
            protected_attributes=protected_attributes,
            metric=model_statistical_parity,
            distance_measure=distance_measure,
            reduction=reduction,
            allow_distance_measure_none=False,
        )

    def __call__(
        self,
        model: Any,
        X: pd.DataFrame,
        y_true: Optional[Union[pd.Series, np.ndarray, List]] = None,
        supplementary_features: Optional[pd.DataFrame] = None,
    ):
        """
        Compute the metric using a model's predictions on a given array
        of instances ``X``.

        Parameters
        ----------
        model: Any
            Object that implements a `predict(X)` function to collect
            categorical predictions.
        X : pandas.DataFrame
            Array of instances to compute the metric on.
        y_true : pandas.Series, numpy.ndarray, list, or None, default=None
            Array of groundtruth labels.
        supplementary_features : pandas.DataFrame, or None, default=None
            Array of supplementary features for each instance. Used in case
            one attribute in ``self.protected_attributes`` is not contained by
            ``X`` (e.g. if the protected attribute is not used by the model).

        Returns
        -------
        float, dict
            The computed metric value, with format according to ``self.reduction``.


        Raises
        ------
        GuardianAIValueError
            - if a feature is present in both ``X``
              and ``supplementary_features``.

        """
        y_pred = model.predict(X)

        subgroups = self._get_check_subgroups(X, supplementary_features)

        return self.metric(
            y_true, y_pred, subgroups, self.distance_measure, self.reduction
        )


# This function has the same signature as other model metrics even though it
# does not need nor use y_true.
# We use default values of None for the unused `y_true` and required `y_pred`
# and `subgroups` arguments. This way this function can be called using
# `model_statistical_parity(y_pred=y_pred, subgroups=subgroups)`.


def model_statistical_parity(
    y_true: Optional[Union[pd.Series, np.ndarray, List]] = None,
    y_pred: Optional[Union[pd.Series, np.ndarray, List]] = None,
    subgroups: Optional[pd.DataFrame] = None,
    distance_measure: str = DEFAULT_DISTANCE,
    reduction: Optional[str] = DEFAULT_REDUCTION,
):
    """
    Measure the statistical parity of a model's output between all subgroup pairs.

    For more details, refer to :class:`.ModelStatisticalParityScorer`.

    Parameters
    ----------
    y_true : pandas.Series, numpy.ndarray, list or None, default=None
        Array of groundtruth labels.
    y_pred : pandas.Series, numpy.ndarray, list or None, default=None
        Array of model predictions.
    subgroups : pandas.DataFrame or None, default=None
        Dataframe containing protected attributes for each instance.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    Returns
    -------
    float, dict
        The computed metric value, with format according to `reduction`.

    Raises
    ------
    GuardianAIValueError
        If Value of None is received for either `y_pred` or `subgroups`.

    Examples
    --------

    .. code-block:: python

        from guardian_ai.fairness.metrics import model_statistical_parity
        subgroups = X[['race', 'sex']]
        model_statistical_parity(y_true, y_pred, subgroups)

    This metric does not require `y_true`. It can also be called using

    .. code-block:: python

        model_statistical_parity(None, y_pred, subgroups)
        model_statistical_parity(y_pred=y_pred, subgroups=subgroups)
    """  # noqa: D412

    if y_pred is None or subgroups is None:
        raise GuardianAIValueError(
            "Value of None was received for either `y_pred` or `subgroups`. "
            "This may be due to calling the metric using only 2 positional "
            "arguments. If this is the case, either call the function by "
            "passing ``None`` as the first argument or use named arguments for "
            "`y_pred` and `subgroups`."
        )

    return _model_metric(
        None,
        y_pred,
        subgroups,
        metric="selection_rate",
        distance_measure=distance_measure,
        reduction=reduction,
        allow_y_true_none=True,
        allow_distance_measure_none=False,
    )


class TruePositiveRateScorer(_ModelFairnessScorer):
    """
    Measures the disparity of a model's true positive rate between
    all subgroup pairs (also known as equal opportunity).

    For each subgroup, the disparity is measured by comparing the true positive
    rate on instances of a subgroup against the rest of the subgroups.

    True Positive Rate [1] (also known as TPR, recall, or sensitivity) is
    calculated as TP / (TP + FN), where TP and FN are the number of true
    positives and false negatives, respectively.


    Perfect score
        A perfect score for this metric means that the model does not correctly
        predict the positive class for any of the subgroups more often than it
        does for the rest of the subgroups. For example, if the protected
        attributes are race and sex, then a perfect true positive rate disparity
        would mean that all combinations of values for race and sex have
        identical true positive rates. Perfect values are:

        - 1 if using ``'ratio'`` as ``distance_measure``.
        - 0 if using ``'diff'`` as ``distance_measure``.

    Parameters
    ----------
    protected_attributes: pandas.Series, numpy.ndarray, list, str
        Array of attributes or single attribute that should be treated as
        protected. If an attribute is protected, then all of its unique
        values are considered as subgroups.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    References
    ----------
    [1] `Moritz Hardt et al. "Equality of Opportunity in Supervised Learning".
    Advances in Neural Information Processing Systems. 2016.
    <https://arxiv.org/pdf/1610.02413.pdf>`_

    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import TruePositiveRateScorer
        scorer = TruePositiveRateScorer(['race', 'sex'])
        scorer(model, X, y_true)
    """

    def __init__(
        self,
        protected_attributes: Union[pd.Series, np.ndarray, List, str],
        distance_measure: str = DEFAULT_DISTANCE,
        reduction: Optional[str] = DEFAULT_REDUCTION,
    ):
        super().__init__(
            protected_attributes=protected_attributes,
            metric=true_positive_rate,
            distance_measure=distance_measure,
            reduction=reduction,
            allow_distance_measure_none=False,
        )


def true_positive_rate(
    y_true: Union[pd.Series, np.ndarray, List],
    y_pred: Union[pd.Series, np.ndarray, List],
    subgroups: pd.DataFrame,
    distance_measure: str = DEFAULT_DISTANCE,
    reduction: Optional[str] = DEFAULT_REDUCTION,
):
    """
    Measures the disparity of a model's true positive rate between all subgroup pairs.

    For more details, refer to :class:`.TruePositiveRateScorer`.

    Parameters
    ----------
    y_true : pandas.Series, numpy.ndarray, list
        Array of groundtruth labels.
    y_pred : pandas.Series, numpy.ndarray, list
        Array of model predictions.
    subgroups : pandas.DataFrame
        Dataframe containing protected attributes for each instance.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.
    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    Returns
    -------
    float, dict
        The computed metric value, with format according to `reduction`.


    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import true_positive_rate
        subgroups = X[['race', 'sex']]
        true_positive_rate(y_true, y_pred, subgroups)
    """
    return _model_metric(
        y_true,
        y_pred,
        subgroups,
        metric="true_positive_rate",
        distance_measure=distance_measure,
        reduction=reduction,
        allow_y_true_none=False,
        allow_distance_measure_none=False,
    )


class FalsePositiveRateScorer(_ModelFairnessScorer):
    """
    Measures the disparity of a model's false positive rate between all subgroup pairs.

    For each subgroup, the disparity is measured by comparing the false
    positive rate on instances of a subgroup against the rest of the subgroups.

    False Positive Rate [1] (also known as FPR or fall-out) is calculated as
    FP / (FP + TN), where FP and TN are the number of false positives and
    true negatives, respectively.

    Perfect score
        A perfect score for this metric means that the model does not incorrectly
        predict the positive class for any of the subgroups more often than it
        does for the rest of the subgroups. For example, if the protected
        attributes are race and sex, then a perfect false positive rate disparity
        would mean that all combinations of values for race and sex have identical
        false positive rates. Perfect values are:

        - 1 if using ``'ratio'`` as ``distance_measure``.
        - 0 if using ``'diff'`` as ``distance_measure``.

    Parameters
    ----------
    protected_attributes: pandas.Series, numpy.ndarray, list, str
        Array of attributes or single attribute that should be treated as
        protected. If an attribute is protected, then all of its unique
        values are considered as subgroups.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    References
    ----------
    [1] `Alexandra Chouldechova. "Fair Prediction with Disparate Impact: A Study
    of Bias in Recidivism Prediction Instruments". Big Data (2016).
    <https://www.liebertpub.com/doi/10.1089/big.2016.0047>`_

    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import FalsePositiveRateScorer
        scorer = FalsePositiveRateScorer(['race', 'sex'])
        scorer(model, X, y_true)
    """

    def __init__(
        self,
        protected_attributes: Union[pd.Series, np.ndarray, List, str],
        distance_measure: str = DEFAULT_DISTANCE,
        reduction: Optional[str] = DEFAULT_REDUCTION,
    ):
        super().__init__(
            protected_attributes=protected_attributes,
            metric=false_positive_rate,
            distance_measure=distance_measure,
            reduction=reduction,
            allow_distance_measure_none=False,
        )


def false_positive_rate(
    y_true: Union[pd.Series, np.ndarray, List],
    y_pred: Union[pd.Series, np.ndarray, List],
    subgroups: pd.DataFrame,
    distance_measure: str = DEFAULT_DISTANCE,
    reduction: Optional[str] = DEFAULT_REDUCTION,
):
    """
    Measures the disparity of a model's false positive rate between all subgroup pairs.

    For more details, refer to :class:`.FalsePositiveRateScorer`.

    Parameters
    ----------
    y_true : pandas.Series, numpy.ndarray, list
        Array of groundtruth labels.
    y_pred : pandas.Series, numpy.ndarray, list
        Array of model predictions.
    subgroups : pandas.DataFrame
        Dataframe containing protected attributes for each instance.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    Returns
    -------
    float, dict
        The computed metric value, with format according to `reduction`.


    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import false_positive_rate
        subgroups = X[['race', 'sex']]
        false_positive_rate(y_true, y_pred, subgroups)
    """
    return _model_metric(
        y_true,
        y_pred,
        subgroups,
        metric="false_positive_rate",
        distance_measure=distance_measure,
        reduction=reduction,
        allow_y_true_none=False,
        allow_distance_measure_none=False,
    )


class FalseNegativeRateScorer(_ModelFairnessScorer):
    """
    Measures the disparity of a model's false negative rate between all subgroup pairs.

    For each subgroup, the disparity is measured by comparing the false
    negative rate on instances of a subgroup against the rest of the subgroups.

    False Negative Rate [1] (also known as FNR or miss rate) is calculated as
    FN / (FN + TP), where FN and TP are the number of false negatives and
    true positives, respectively.

    Perfect score
        A perfect score for this metric means that the model does not incorrectly
        predict the negative class for any of the subgroups more often than it
        does for the rest of the subgroups. For example, if the protected
        attributes are race and sex, then a perfect false negative rate disparity
        would mean that all combinations of values for race and sex have identical
        false negative rates. Perfect values are:

        - 1 if using ``'ratio'`` as ``distance_measure``.
        - 0 if using ``'diff'`` as ``distance_measure``.

    Parameters
    ----------
    protected_attributes: pandas.Series, numpy.ndarray, list, str
        Array of attributes or single attribute that should be treated as
        protected. If an attribute is protected, then all of its unique
        values are considered as subgroups.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    References
    ----------
    [1] `Alexandra Chouldechova. "Fair Prediction with Disparate Impact: A Study
    of Bias in Recidivism Prediction Instruments". Big Data (2016).
    <https://www.liebertpub.com/doi/10.1089/big.2016.0047>`_

    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import FalseNegativeRateScorer
        scorer = FalseNegativeRateScorer(['race', 'sex'])
        scorer(model, X, y_true)
    """

    def __init__(
        self,
        protected_attributes: Union[pd.Series, np.ndarray, List, str],
        distance_measure: str = DEFAULT_DISTANCE,
        reduction: Optional[str] = DEFAULT_REDUCTION,
    ):
        super().__init__(
            protected_attributes=protected_attributes,
            metric=false_negative_rate,
            distance_measure=distance_measure,
            reduction=reduction,
            allow_distance_measure_none=False,
        )


def false_negative_rate(
    y_true: Union[pd.Series, np.ndarray, List],
    y_pred: Union[pd.Series, np.ndarray, List],
    subgroups: pd.DataFrame,
    distance_measure: str = DEFAULT_DISTANCE,
    reduction: Optional[str] = DEFAULT_REDUCTION,
):
    """
    Measures the disparity of a model's false negative rate between all subgroup pairs.

    For more details, refer to :class:`.FalseNegativeRateScorer`.

    Parameters
    ----------
    y_true : pandas.Series, numpy.ndarray, list
        Array of groundtruth labels.
    y_pred : pandas.Series, numpy.ndarray, list
        Array of model predictions.
    subgroups : pandas.DataFrame
        Dataframe containing protected attributes for each instance.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    Returns
    -------
    float, dict
        The computed metric value, with format according to `reduction`.


    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import false_negative_rate
        subgroups = X[['race', 'sex']]
        false_negative_rate(y_true, y_pred, subgroups)
    """
    return _model_metric(
        y_true,
        y_pred,
        subgroups,
        metric="false_negative_rate",
        distance_measure=distance_measure,
        reduction=reduction,
        allow_y_true_none=False,
        allow_distance_measure_none=False,
    )


class FalseOmissionRateScorer(_ModelFairnessScorer):
    """
    Measures the disparity of a model's false omission rate between all subgroup pairs.

    For each subgroup, the disparity is measured by comparing the false
    omission rate on instances of a subgroup against the rest of the subgroups.

    False Omission Rate (also known as FOR) is calculated as
    FN / (FN + TN), where FN and TN are the number of false negatives and
    true negatives, respectively.

    Perfect score
        A perfect score for this metric means that the model does not make more
        mistakes on the negative class for any of the subgroups more often than it
        does for the rest of the subgroups. For example, if the protected
        attributes are race and sex, then a perfect false omission rate disparity
        would mean that all combinations of values for race and sex have identical
        false omission rates. Perfect values are:

        - 1 if using ``'ratio'`` as ``distance_measure``.
        - 0 if using ``'diff'`` as ``distance_measure``.

    Parameters
    ----------
    protected_attributes: pandas.Series, numpy.ndarray, list, str
        Array of attributes or single attribute that should be treated as
        protected. If an attribute is protected, then all of its unique
        values are considered as subgroups.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import FalseOmissionRateScorer
        scorer = FalseOmissionRateScorer(['race', 'sex'])
        scorer(model, X, y_true)
    """

    def __init__(
        self,
        protected_attributes: Union[pd.Series, np.ndarray, List, str],
        distance_measure: str = DEFAULT_DISTANCE,
        reduction: Optional[str] = DEFAULT_REDUCTION,
    ):
        super().__init__(
            protected_attributes=protected_attributes,
            metric=false_omission_rate,
            distance_measure=distance_measure,
            reduction=reduction,
            allow_distance_measure_none=False,
        )


def false_omission_rate(
    y_true: Union[pd.Series, np.ndarray, List],
    y_pred: Union[pd.Series, np.ndarray, List],
    subgroups: pd.DataFrame,
    distance_measure: str = DEFAULT_DISTANCE,
    reduction: Optional[str] = DEFAULT_REDUCTION,
):
    """
    Measures the disparity of a model's false omission rate between all subgroup pairs.

    For more details, refer to :class:`.FalseOmissionRateScorer`.

    Parameters
    ----------
    y_true : pandas.Series, numpy.ndarray, list
        Array of groundtruth labels.
    y_pred : pandas.Series, numpy.ndarray, list
        Array of model predictions.
    subgroups : pandas.DataFrame
        Dataframe containing protected attributes for each instance.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    Returns
    -------
    float, dict
        The computed metric value, with format according to `reduction`.


    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import false_omission_rate
        subgroups = X[['race', 'sex']]
        false_omission_rate(y_true, y_pred, subgroups)
    """
    return _model_metric(
        y_true,
        y_pred,
        subgroups,
        metric="false_omission_rate",
        distance_measure=distance_measure,
        reduction=reduction,
        allow_y_true_none=False,
        allow_distance_measure_none=False,
    )


class FalseDiscoveryRateScorer(_ModelFairnessScorer):
    """
    Measures the disparity of a model's false discovery rate between all subgroup pairs.

    For each subgroup, the disparity is measured by comparing the false
    discovery rate on instances of a subgroup against the rest of the
    subgroups.

    False Discovery Rate (also known as FDR) is calculated as
    FP / (FP + TP), where FP and TP are the number of false positives and
    true positives, respectively.

    Perfect score
        A perfect score for this metric means that the model does not make more
        mistakes on the positive class for any of the subgroups more often than it
        does for the rest of the subgroups. For example, if the protected
        attributes are race and sex, then a perfect false discovery rate disparity
        would mean that all combinations of values for race and sex have identical
        false discovery rates. Perfect values are:

        - 1 if using ``'ratio'`` as ``distance_measure``.
        - 0 if using ``'diff'`` as ``distance_measure``.

    Parameters
    ----------
    protected_attributes: pandas.Series, numpy.ndarray, list, str
        Array of attributes or single attribute that should be treated as
        protected. If an attribute is protected, then all of its unique
        values are considered as subgroups.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import FalseDiscoveryRateScorer
        scorer = FalseDiscoveryRateScorer(['race', 'sex'])
        scorer(model, X, y_true)
    """

    def __init__(
        self,
        protected_attributes: Union[pd.Series, np.ndarray, List, str],
        distance_measure: str = DEFAULT_DISTANCE,
        reduction: Optional[str] = DEFAULT_REDUCTION,
    ):
        super().__init__(
            protected_attributes=protected_attributes,
            metric=false_discovery_rate,
            distance_measure=distance_measure,
            reduction=reduction,
            allow_distance_measure_none=False,
        )


def false_discovery_rate(
    y_true: Union[pd.Series, np.ndarray, List],
    y_pred: Union[pd.Series, np.ndarray, List],
    subgroups: pd.DataFrame,
    distance_measure: str = DEFAULT_DISTANCE,
    reduction: Optional[str] = DEFAULT_REDUCTION,
):
    """
    Measures the disparity of a model's false discovery rate between all subgroup pairs.

    For more details, refer to :class:`.FalseDiscoveryRateScorer`.

    Parameters
    ----------
    y_true : pandas.Series, numpy.ndarray, list
        Array of groundtruth labels.
    y_pred : pandas.Series, numpy.ndarray, list
        Array of model predictions.
    subgroups : pandas.DataFrame
        Dataframe containing protected attributes for each instance.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    Returns
    -------
    float, dict
        The computed metric value, with format according to `reduction`.


    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import false_discovery_rate
        subgroups = X[['race', 'sex']]
        false_discovery_rate(y_true, y_pred, subgroups)
    """
    return _model_metric(
        y_true,
        y_pred,
        subgroups,
        metric="false_discovery_rate",
        distance_measure=distance_measure,
        reduction=reduction,
        allow_y_true_none=False,
        allow_distance_measure_none=False,
    )


class ErrorRateScorer(_ModelFairnessScorer):
    """
    Measures the disparity of a model's error rate between all subgroup pairs.

    For each subgroup, the disparity is measured by comparing the error rate on
    instances of a subgroup against the rest of the subgroups.

    Error Rate (also known as inaccuracy) is calculated as
    (FP + FN) / N, where FP and FN are the number of false positives and
    false negatives, respectively, while N is the total Number of
    instances.

    Perfect score
        A perfect score for this metric means that the model does not make more
        mistakes for any of the subgroups more often than it
        does for the rest of the subgroups. For example, if the protected
        attributes are race and sex, then a perfect error rate disparity would
        mean that all combinations of values for race and sex have identical
        error rates. Perfect values are:

        - 1 if using ``'ratio'`` as ``distance_measure``.
        - 0 if using ``'diff'`` as ``distance_measure``.

    Parameters
    ----------
    protected_attributes: pandas.Series, numpy.ndarray, list, str
        Array of attributes or single attribute that should be treated as
        protected. If an attribute is protected, then all of its unique
        values are considered as subgroups.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import ErrorRateScorer
        scorer = ErrorRateScorer(['race', 'sex'])
        scorer(model, X, y_true)
    """

    def __init__(
        self,
        protected_attributes: Union[pd.Series, np.ndarray, List, str],
        distance_measure: str = DEFAULT_DISTANCE,
        reduction: Optional[str] = DEFAULT_REDUCTION,
    ):
        super().__init__(
            protected_attributes=protected_attributes,
            metric=error_rate,
            distance_measure=distance_measure,
            reduction=reduction,
            allow_distance_measure_none=False,
        )


def error_rate(
    y_true: Union[pd.Series, np.ndarray, List],
    y_pred: Union[pd.Series, np.ndarray, List],
    subgroups: pd.DataFrame,
    distance_measure: str = DEFAULT_DISTANCE,
    reduction: Optional[str] = DEFAULT_REDUCTION,
):
    """
    Measures the disparity of a model's error rate between all subgroup pairs.

    For more details, refer to :class:`.ErrorRateScorer`.

    Parameters
    ----------
    y_true : pandas.Series, numpy.ndarray, list
        Array of groundtruth labels.
    y_pred : pandas.Series, numpy.ndarray, list
        Array of model predictions.
    subgroups : pandas.DataFrame
        Dataframe containing protected attributes for each instance.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    Returns
    -------
    float, dict
        The computed metric value, with format according to `reduction`.


    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import error_rate
        subgroups = X[['race', 'sex']]
        error_rate(y_true, y_pred, subgroups)
    """
    return _model_metric(
        y_true,
        y_pred,
        subgroups,
        metric="error_rate",
        distance_measure=distance_measure,
        reduction=reduction,
        allow_y_true_none=False,
        allow_distance_measure_none=False,
    )


class EqualizedOddsScorer(_ModelFairnessScorer):
    """
    Measures the disparity of a model's true positive and false positive rates
    between subgroups and the rest of the subgroups.

    The disparity is measured by comparing the true positive and false positive
    rates on instances of a subgroup against the rest of the subgroups.

    True Positive Rate (also known as TPR, recall, or sensitivity) is
    calculated as TP / (TP + FN), where TP and FN are the number of true
    positives and false negatives, respectively.

    False Positive Rate (also known as FPR or fall-out) is calculated as
    FP / (FP + TN), where FP and TN are the number of false positives and
    true negatives, respectively.

    Equalized Odds [1] is computed by taking the maximum distance between
    TPR and FPR for a subgroup against the rest of the subgroups.

    Perfect score
        A perfect score for this metric means that the model has the same TPR and
        FPR when comparing a subgroup to the rest of the subgroups. For example,
        if the protected attributes are race and sex, then a perfect
        Equalized Odds disparity would mean that all combinations of values for
        race and sex have identical TPR and FPR. Perfect values are:

        - 1 if using ``'ratio'`` as ``distance_measure``.
        - 0 if using ``'diff'`` as ``distance_measure``.

    Parameters
    ----------
    protected_attributes: pandas.Series, numpy.ndarray, list, str
        Array of attributes or single attribute that should be treated as
        protected. If an attribute is protected, then all of its unique
        values are considered as subgroups.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    References
    ----------
    [1] `Moritz Hardt et al. "Equality of Opportunity in Supervised Learning".
    Advances in Neural Information Processing Systems. 2016.
    <https://arxiv.org/pdf/1610.02413.pdf>`_

    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import EqualizedOddsScorer
        scorer = EqualizedOddsScorer(['race', 'sex'])
        scorer(model, X, y_true)
    """

    def __init__(
        self,
        protected_attributes: Union[pd.Series, np.ndarray, List, str],
        distance_measure: str = DEFAULT_DISTANCE,
        reduction: Optional[str] = DEFAULT_REDUCTION,
    ):
        super().__init__(
            protected_attributes=protected_attributes,
            metric=equalized_odds,
            distance_measure=distance_measure,
            reduction=reduction,
            allow_distance_measure_none=False,
        )


def equalized_odds(
    y_true: Union[pd.Series, np.ndarray, List],
    y_pred: Union[pd.Series, np.ndarray, List],
    subgroups: pd.DataFrame,
    distance_measure: str = DEFAULT_DISTANCE,
    reduction: Optional[str] = DEFAULT_REDUCTION,
):
    """
    Measures the disparity of a model's true positive and false positive rates
    between subgroups and the rest of the subgroups.

    For more details, refer to :class:`.EqualizedOddsScorer`.

    Parameters
    ----------
    y_true : pandas.Series, numpy.ndarray, list
        Array of groundtruth labels.
    y_pred : pandas.Series, numpy.ndarray, list
        Array of model predictions.
    subgroups : pandas.DataFrame
        Dataframe containing protected attributes for each instance.
    distance_measure : str, default='diff'
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    Returns
    -------
    float, dict
        The computed metric value, with format according to `reduction`.


    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import equalized_odds
        subgroups = X[['race', 'sex']]
        equalized_odds(y_true, y_pred, subgroups)
    """
    tpr = true_positive_rate(
        y_true,
        y_pred,
        subgroups,
        distance_measure=distance_measure,
        reduction=reduction,
    )

    fpr = false_positive_rate(
        y_true,
        y_pred,
        subgroups,
        distance_measure=distance_measure,
        reduction=reduction,
    )
    if isinstance(tpr, dict):
        eq_odds = {}
        for key in tpr:
            eq_odds[key] = np.nanmax([tpr[key], fpr[key]])
    else:
        eq_odds = np.nanmax([tpr, fpr])

    return eq_odds


class TheilIndexScorer(_ModelFairnessScorer):
    """
    Measures the disparity of a model's predictions according to groundtruth
    labels, as proposed by Speicher et al. [1].

    Intuitively, the Theil Index can be thought of as a measure of the
    divergence between a subgroup's different error distributions (i.e. false
    positives and false negatives) against the rest of the subgroups.

    Perfect score
        The perfect score for this metric is 0, meaning that the model does not
        have a different error distribution for any subgroup when compared to the
        rest of the subgroups. For example, if the protected attributes are
        race and sex, then a perfect Theil Index disparity would mean that all
        combinations of values for race and sex have identical error
        distributions.

    Parameters
    ----------
    protected_attributes: pandas.Series, numpy.ndarray, list, str
        Array of attributes or single attribute that should be treated as
        protected. If an attribute is protected, then all of its unique
        values are considered as subgroups.
    distance_measure : str or None, default=None
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.
    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    References
    ----------
    [1] `Speicher, Till, et al. "A unified approach to quantifying algorithmic
    unfairness: Measuring individual & group unfairness via inequality indices."
    Proceedings of the 24th ACM SIGKDD international conference on knowledge
    discovery & data mining. 2018. <https://arxiv.org/abs/1807.00787>`_

    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import TheilIndexScorer
        scorer = TheilIndexScorer(['race', 'sex'])
        scorer(model, X, y_true)
    """

    def __init__(
        self,
        protected_attributes: Union[pd.Series, np.ndarray, List, str],
        distance_measure: Optional[str] = None,
        reduction: Optional[str] = DEFAULT_REDUCTION,
    ):
        super().__init__(
            protected_attributes=protected_attributes,
            metric=theil_index,
            distance_measure=distance_measure,
            reduction=reduction,
            allow_distance_measure_none=True,
        )


def theil_index(
    y_true: Union[pd.Series, np.ndarray, List],
    y_pred: Union[pd.Series, np.ndarray, List],
    subgroups: pd.DataFrame,
    distance_measure: Optional[str] = None,
    reduction: Optional[str] = DEFAULT_REDUCTION,
):
    """
    Measures the disparity of a model's predictions according to groundtruth
    labels, as proposed by Speicher et al. [1].

    For more details, refer to :class:`.TheilIndexScorer`.

    Parameters
    ----------
    y_true : pandas.Series, numpy.ndarray, list
        Array of groundtruth labels.
    y_pred : pandas.Series, numpy.ndarray, list
        Array of model predictions.
    subgroups : pandas.DataFrame
        Dataframe containing protected attributes for each instance.
    distance_measure : str or None, default=None
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:

            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.

    reduction : str or None, default='mean'
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:

            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.

    Returns
    -------
    float, dict
        The computed metric value, with format according to `reduction`.

    Raises
    ------
    GuardianAIValueError
        If distance_measure values are given to Theil Index.

    References
    ----------
    [1]: `Speicher, Till, et al. "A unified approach to quantifying algorithmic
    unfairness: Measuring individual & group unfairness via inequality indices."
    Proceedings of the 24th ACM SIGKDD international conference on knowledge
    discovery & data mining. 2018. <https://arxiv.org/abs/1807.00787>`_

    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import theil_index
        subgroups = X[['race', 'sex']]
        theil_index(y_true, y_pred, subgroups)
    """

    if distance_measure is not None and not isinstance(
        distance_measure, _DistanceMetric
    ):
        raise GuardianAIValueError(
            "Theil Index does not accept distance_measure values. It should"
            "always be set to ``None``."
        )

    return _model_metric(
        y_true,
        y_pred,
        subgroups,
        metric="between_group_theil_index",
        distance_measure=None,
        reduction=reduction,
        allow_y_true_none=False,
        allow_distance_measure_none=True,
    )
