#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Evaluating the compliance of a dataset with specific fairness metrics"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Union

from guardian_ai.fairness.utils.lazy_loader import LazyLoader
from guardian_ai.fairness.metrics.utils import (
    DEFAULT_DISTANCE,
    DEFAULT_REDUCTION,
    _check_subgroups,
    _FairnessScorer,
    _get_attr_idx_mappings,
    _get_check_array,
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
    from aif360.metrics import BinaryLabelDatasetMetric
else:
    pd = LazyLoader("pandas")
    np = LazyLoader("numpy")
    BinaryLabelDatasetMetric = LazyLoader(
        "aif360.metrics", "BinaryLabelDatasetMetric", suppress_import_warnings=True
    )

BinaryLabelDatasetMetric = LazyLoader(
    "aif360.metrics", "BinaryLabelDatasetMetric", suppress_import_warnings=True
)


def _dataset_metric(
    y_true: Union[pd.Series, np.ndarray, List],
    subgroups: pd.DataFrame,
    metric: str,
    distance_measure: Optional[str],
    reduction: Optional[str],
    allow_distance_measure_none: bool,
):
    """
    Compute engine for all dataset metrics.

    This computes a given metric on all subgroup pairs for a specified ``subgroups`` input.

    Parameters
    ----------
    y_true : pandas.Series, numpy.ndarray, list
        Array of groundtruth labels
    subgroups : pandas.DataFrame
        Dataframe containing protected attributes for each instance.
    metric : str
        Name of the base metric to be called.
    distance_measure : str or None
        Determines the distance used to compare a subgroup's metric
        against the rest of the subgroups. Possible values are:
            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.
        - ``None``, to not use any distance metric. Only allowed if
            `allow_distance_measure_none` is set to True.
    reduction : str or None
        Determines how to reduce scores on all subgroups to
        a single output.
        Possible values are:
            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.
    allow_distance_measure_none : bool
        Whether or not to allow ``distance_measure`` to be set
        to ``None``.

    Returns
    -------
    float, dict
        The computed metric value, with format according to `reduction`.

    """
    y_true = _get_check_array(y_true, "y_true")
    (
        reduction,
        distance,
        attr_vals_to_idx,
        attr_idx_to_vals,
        subgroup_divisions,
    ) = _get_check_inputs(
        reduction, distance_measure, subgroups, allow_distance_measure_none
    )

    ds_true = _y_to_aifm_ds(y_true, subgroups, attr_vals_to_idx)

    groups = []
    scores = []
    for unpriv_group, priv_groups in subgroup_divisions:
        subgroup_metrics = BinaryLabelDatasetMetric(ds_true, unpriv_group, priv_groups)

        score, group_repr = _get_score_group_from_metrics(
            subgroup_metrics, distance, metric, unpriv_group, attr_idx_to_vals
        )

        scores.append(score)
        groups.append(group_repr)

    return reduction(groups, scores)


class _DatasetFairnessScorer(_FairnessScorer):
    """
    Common base object for all dataset metrics.

    This stores settings to pass on to the ``_dataset_metric``
    compute engine and does subgroups generation from a `protected_attributes`
    array on an input array of instances ``X``.

    Parameters
    ----------
    protected_attributes: pandas.Series, numpy.ndarray, list, str
        Array of attributes or single attribute that should be treated as
        protected. If an attribute is protected, then all of its unique
        values are considered as subgroups.
    metric : str or Callable
        Name of the base metric to be called.
    distance_measure : str or None
        Determines the distance used to compare a subgroup's metric against
        the rest of the subgroups. Possible values are:
            * ``'ratio'``: Uses ``(subgroup1_val / subgroup2_val)``. Inverted to always be >= 1 if needed.
            * ``'diff'``: Uses ``| subgroup1_val - subgroup2_val |``.
        - ``None``, to not use any distance metric. Only allowed if
        `allow_distance_measure_none` is set to True.
    reduction : str or None
        Determines how to reduce scores on all subgroups to a single output.
        Possible values are:
            * ``'max'``: Returns the maximal value among all subgroup metrics.
            * ``'mean'``: Returns the mean over all subgroup metrics.
            * ``None``: Returns a ``{subgroup: subgroup_metric, ...}`` dict.
    allow_distance_measure_none : bool
        Whether or not to allow ``distance_measure`` to be set to ``None``.
    """

    def __init__(
        self,
        protected_attributes: Union[pd.Series, np.ndarray, List, str],
        metric: Union[str, Callable],
        distance_measure: Optional[str],
        reduction: Optional[str],
        allow_distance_measure_none: bool,
    ):
        super().__init__(protected_attributes, metric)

        self.distance_measure = _get_check_distance(
            distance_measure, allow_distance_measure_none
        )
        self.reduction = _get_check_reduction(reduction)

    def __call__(
        self,
        model: Optional[object] = None,
        X: Optional[pd.DataFrame] = None,
        y_true: Optional[Union[pd.Series, np.ndarray, List]] = None,
        supplementary_features: Optional[pd.DataFrame] = None,
    ):
        """
        Compute the metric on a given array of instances ``X``.

        Parameters
        ----------
        model : object or None, default=None
            Object that implements a `predict(X)` function to collect
            categorical predictions.
        X : pandas.DataFrame or None, default=None
            Array of instances to compute the metric on.
        y_true : pandas.Series, numpy.ndarray, list or None, default=None
            Array of groundtruth labels.
        supplementary_features : pandas.DataFrame, or None, default=None
            Array of supplementary features for each instance. Used in case
            one attribute in ``self.protected_attributes`` is not contained by
            ``X`` (e.g. if the protected attribute is not used by the model).
            Raise an GuardianAIValueError if a feature is present in both ``X`` and
            ``supplementary_features``.

        Returns
        -------
        float, dict
            The computed metric value, with format according to ``self.reduction``.

        Raises
        ------
        GuardianAIValueError
            If a feature is present in both ``X`` and ``supplementary_features``.
        """
        # We use default values of None for the unused `model` and required
        # ``X`` and `y_true` arguments. This way model scorers can be called with
        # `model_scorer(X=X, y_true=y_true)`.
        if X is None or y_true is None:
            raise GuardianAIValueError(
                "Value of None was received for either ``X`` or ``y_true``. "
                "This may be due to calling the metric using only 2 positional "
                "arguments. If this is the case, either call the function by "
                "passing ``None`` as the first argument or use named arguments "
                "for ``X`` and ``y_true``."
            )

        subgroups = self._get_check_subgroups(X, supplementary_features)

        return self.metric(y_true, subgroups, self.distance_measure, self.reduction)

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


class DatasetStatisticalParityScorer(_DatasetFairnessScorer):
    """
    Measures the statistical parity [1] of a dataset. Statistical parity (also
    known as Base Rate or Disparate Impact) for a dataset states that a dataset
    is unbiased if the label is independent of the protected attribute.

    For each subgroup, statistical parity is computed as the ratio of positive
    labels in a subgroup.

    Statistical Parity (also known as Base Rate or Disparate Impact) is
    calculated as PL / N, where PL and N are the number of Positive Labels and
    total number of instances, respectively.

    Perfect score
        A perfect score for this metric means that the dataset does not have
        a different ratio of positive labels for a subgroup than it does for
        the rest of the subgroups. For example, if the protected attributes
        are race and sex, then a perfect statistical parity would mean that
        all combinations of values for race and sex have identical ratios of
        positive labels. Perfect values are:

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
    [1] `Cynthia Dwork et al. "Fairness Through Awareness". Innovations in
    Theoretical Computer Science. 2012. <https://arxiv.org/abs/1104.3913>`_

    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import DatasetStatisticalParityScorer
        scorer = DatasetStatisticalParityScorer(['race', 'sex'])
        scorer(X=X, y_true=y_true)
        scorer(None, X, y_true)
    """

    def __init__(
        self,
        protected_attributes: Union[pd.Series, np.ndarray, List, str],
        distance_measure: str = DEFAULT_DISTANCE,
        reduction: Optional[str] = DEFAULT_REDUCTION,
    ):
        super().__init__(
            protected_attributes=protected_attributes,
            metric=dataset_statistical_parity,
            distance_measure=distance_measure,
            reduction=reduction,
            allow_distance_measure_none=False,
        )


def dataset_statistical_parity(
    y_true: Union[pd.Series, np.ndarray, List],
    subgroups: pd.DataFrame,
    distance_measure: str = DEFAULT_DISTANCE,
    reduction: str = DEFAULT_REDUCTION,
):
    """
    Measures the statistical parity of a dataset.

    For more details, refer to :class:`.DatasetStatisticalParityScorer`.

    Parameters
    ----------
    y_true : pandas.Series, numpy.ndarray, list
        Array of groundtruth labels
    subgroups : pandas.DataFrame
        Dataframe containing protected attributes for each instance.
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

        from guardian_ai.fairness.metrics import dataset_statistical_parity
        subgroups = X[['race', 'sex']]
        dataset_statistical_parity(y_true, subgroups)
    """
    return _dataset_metric(
        y_true,
        subgroups,
        metric="base_rate",
        distance_measure=distance_measure,
        reduction=reduction,
        allow_distance_measure_none=False,
    )


def _simple_dataset_metric(
    y_true: Union[pd.Series, np.ndarray, List], subgroups: pd.DataFrame, metric: str
):
    """
    Compute engine for dataset metrics that do not require a distance
    measure or reduction function because they already return a float value.

    Parameters
    ----------
    y_true : pandas.Series, numpy.ndarray, list
        Array of groundtruth labels
    subgroups : pandas.DataFrame
        Dataframe containing protected attributes for each instance.
    metric : str
        Name of the base metric to be called.

    Returns
    -------
    float
        The computed metric value.
    """
    y_true = _get_check_array(y_true, "y_true")

    _check_subgroups(subgroups)
    attr_vals_to_idx, attr_idx_to_vals = _get_attr_idx_mappings(subgroups)

    ds_true = _y_to_aifm_ds(y_true, subgroups, attr_vals_to_idx)

    metrics_obj = BinaryLabelDatasetMetric(ds_true)
    metric_val = getattr(metrics_obj, metric)()

    return metric_val


class _SimpleDatasetFairnessScorer(_FairnessScorer):
    def __call__(
        self,
        model: Optional[object] = None,
        X: Optional[pd.DataFrame] = None,
        y_true: Optional[Union[pd.Series, np.ndarray, List]] = None,
        supplementary_features: Optional[pd.DataFrame] = None,
    ):
        # We use default values of None for the unused `model` and required
        # ``X`` and `y_true` arguments. This way model scorers can be called with
        # `model_scorer(X=X, y_true=y_true)`.
        if X is None or y_true is None:
            raise GuardianAIValueError(
                "Value of None was received for either ``X`` or `y_true`. "
                "This may be due to calling the metric using only 2 positional "
                "arguments. If this is the case, either call the function by "
                "passing ``None`` as the first argument or use named arguments "
                "for ``X`` and `y_true`."
            )

        subgroups = self._get_check_subgroups(X, supplementary_features)

        return self.metric(y_true, subgroups)


class ConsistencyScorer(_SimpleDatasetFairnessScorer):
    """
    Measures the consistency of a dataset.

    Consistency is measured as the number of ratio of instances that have a
    different label from the k=5 nearest neighbors.

    Perfect score
        A perfect score for this metric is 0, meaning that the dataset does
        not have different labels for instances that are similar to one another.

    Parameters
    ----------
    protected_attributes: pandas.Series, numpy.ndarray, list, str
        Array of attributes or single attribute that should be treated as
        protected. If an attribute is protected, then all of its unique
        values are considered as subgroups.

    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import ConsistencyScorer
        scorer = ConsistencyScorer(['race', 'sex'])
        scorer(X=X, y_true=y_true)
        scorer(None, X, y_true)
    """

    def __init__(self, protected_attributes: Union[pd.Series, np.ndarray, List, str]):
        super().__init__(protected_attributes=protected_attributes, metric=consistency)


def consistency(y_true: Union[pd.Series, np.ndarray, List], subgroups: pd.DataFrame):
    """
    Measures the consistency of a dataset.

    For more details, refer to :class:`.ConsistencyScorer`.

    Parameters
    ----------
    y_true : pandas.Series, numpy.ndarray, list
        Array of groundtruth labels
    subgroups : pandas.DataFrame
        Dataframe containing protected attributes for each instance.

    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import consistency
        subgroups = X[['race', 'sex']]
        consistency(y_true, subgroups)
    """
    # Need to read with [0] because consistency returns an array of size 1.
    return _simple_dataset_metric(y_true, subgroups, metric="consistency")[0]


class SmoothedEDFScorer(_SimpleDatasetFairnessScorer):
    """
    Measures the smoothed Empirical Differential Fairness (EDF) of a dataset, as
    proposed by Foulds et al. [1].

    Smoothed EDF returns the minimal exponential deviation of positive target
    ratios comparing a subgroup to the rest of the subgroups.

    This metric is related to :class:`.DatasetStatisticalParity` with
    `reduction='max'` and `distance_measure='ratio'`, with the only difference
    being that :class:`.SmoothedEDFScorer` returns a logarithmic value instead.

    Perfect score
        A perfect score for this metric is 0, meaning that the dataset does
        not have a different ratio of positive labels for a subgroup than
        it does for the rest of the subgroups. For example, if the
        protected attributes are race and sex, then a perfect smoothed EDF
        would mean that all combinations of values for race and sex have
        identical ratios of positive labels.

    Parameters
    ----------
    protected_attributes: pandas.Series, numpy.ndarray, list, str
        Array of attributes or single attribute that should be treated as
        protected. If an attribute is protected, then all of its unique
        values are considered as subgroups.

    References
    ----------
    [1] `Foulds, James R., et al. "An intersectional definition of fairness."
    2020 IEEE 36th International Conference on Data Engineering (ICDE).
    IEEE, 2020. <https://arxiv.org/abs/1807.08362>`_

    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import SmoothedEDFScorer
        scorer = SmoothedEDFScorer(['race', 'sex'])
        scorer(X=X, y_true=y_true)
        scorer(None, X, y_true)
    """

    def __init__(self, protected_attributes: Union[pd.Series, np.ndarray, List, str]):
        super().__init__(protected_attributes=protected_attributes, metric=smoothed_edf)


def smoothed_edf(y_true: Union[pd.Series, np.ndarray, List], subgroups: pd.DataFrame):
    """
    Measures the smoothed Empirical Differential Fairness (EDF) of a dataset, as
    proposed by Foulds et al. [1].

    For more details, refer to :class:`.SmoothedEDFScorer`.

    Parameters
    ----------
    y_true : pandas.Series, numpy.ndarray, list
        Array of groundtruth labels
    subgroups : pandas.DataFrame
        Dataframe containing protected attributes for each instance.

    References
    ----------
    [1] `Foulds, James R., et al. "An intersectional definition of fairness."
    2020 IEEE 36th International Conference on Data Engineering (ICDE).
    IEEE, 2020. <https://arxiv.org/abs/1807.08362>`_

    Examples
    --------
    .. code-block:: python

        from guardian_ai.fairness.metrics import smoothed_edf
        subgroups = X[['race', 'sex']]
        smoothed_edf(y_true, subgroups)
    """
    return _simple_dataset_metric(
        y_true, subgroups, metric="smoothed_empirical_differential_fairness"
    )
