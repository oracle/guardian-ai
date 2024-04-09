#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Sklearn API for bias mitigation"""
from __future__ import annotations
import copy

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

from guardian_ai.fairness.utils.lazy_loader import LazyLoader
from guardian_ai.fairness.metrics import _get_fairness_metric, fairness_metrics_dict
from guardian_ai.utils.exception import GuardianAITypeError, GuardianAIValueError
from guardian_ai.fairness.utils.util import dyn_docstring, _supported_score_metric
from guardian_ai.fairness.metrics.utils import (
    _positive_fairness_names,
    _automl_to_fairlearn_metric_names,
)
from guardian_ai.fairness.metrics.model import (
    _valid_regression_metrics,
)
from guardian_ai.fairness.metrics.utils import (
    _get_rate_scorer,
    _inhouse_metrics,
)

from fairlearn.postprocessing import ThresholdOptimizer

if TYPE_CHECKING:
    import numpy as np
    import optuna
    import pandas as pd
    import plotly.graph_objects as go
    import sklearn.metrics as skl_metrics
    from category_encoders.ordinal import OrdinalEncoder
    from sklearn.base import BaseEstimator
    from sklearn.model_selection import StratifiedShuffleSplit
else:
    np = LazyLoader("numpy")
    pd = LazyLoader("pandas")
    optuna = LazyLoader("optuna")
    BaseEstimator = LazyLoader("sklearn.base", "BaseEstimator")
    go = LazyLoader("plotly.graph_objects")
    skl_metrics = LazyLoader("sklearn.metrics")
    StratifiedShuffleSplit = LazyLoader(
        "sklearn.model_selection", "StratifiedShuffleSplit"
    )
    OrdinalEncoder = LazyLoader("category_encoders.ordinal", "OrdinalEncoder")


@dyn_docstring(
    "', '".join(fairness_metrics_dict), "', '".join(_supported_score_metric["binary"])
)
class ModelBiasMitigator:
    r"""
    Class to mitigate the bias of an already fitted machine learning model.

    The mitigation procedure works by multiplying the majority class label
    by a different scalar for every population subgroup and then rescaling
    the prediction probabilities, producing tweaked label probabilities.

    The different multiplying scalars are searched in order to find the
    best possible trade-offs between any fairness and accuracy metrics
    passed as input.

    This object produces a set of optimal fairness-accuracy trade-offs,
    which can be visualized using the `show_tradeoff` method.

    A default best multiplier is selected according to parametrizable input
    constraints. It is possible to select any other multiplier on the trade-off
    using the ``select_model`` method and inputting the index of the
    preferred multiplier, as shown when hovering over multipliers in
    ``show_tradeoff``.

    Parameters
    ----------
    base_estimator : model object
        The base estimator on which we want to mitigate bias.
    protected_attribute_names: str, List[str]
        The protected attribute names to use to compute fairness metrics.
        These should always be a part of any input dataset passed.
    fairness_metric : str, callable
        The fairness metric to mitigate bias for.

        - If str, it is the name of the scoring metric. Available metrics are:
          [%s]
        - If callable, it has to have the
            ``fairness_metric(y_true, y_pred, subgroups)`` signature.
    accuracy_metric : str, callable
        The accuracy metric to optimize for while mitigating bias.

        - If str, it is the name of the scoring metric. Available metrics are:
          [%s]
        - If callable, it has to have the
            ``accuracy_metric(y_true, y_pred)`` signature.
    higher_fairness_is_better : bool, 'auto', default='auto'
        Whether a higher fairness score with respect to `fairness_metric`
        is better. Needs to be set to "auto" if `fairness_metric` is a str,
        in which case it is set automatically.
    higher_accuracy_is_better : bool, 'auto', default='auto'
        Whether a higher accuracy score with respect to `accuracy_metric`
        is better. Needs to be set to "auto" if `accuracy_metric` is a str,
        in which case it is set automatically.
    fairness_metric_uses_probas: bool, 'auto', default='auto'
        Whether or not the fairness metric should be given label probabilities
        or actual labels as input. Needs to be set to "auto" if
        `fairness_metric` is a str, in which case it is set automatically.
    accuracy_metric_uses_probas: bool, 'auto', default='auto'
        Whether or not the accuracy metric should be given label probabilities
        or actual labels as input. Needs to be set to "auto" if
        `accuracy_metric` is a str, in which case it is set automatically.
    constraint_target: str, default='accuracy'
        On which metric should the constraint be applied for default
        model selection.
        Possible values are ``'fairness'`` and ``'accuracy'``.
    constraint_type: str, default='relative'
        Which type of constraint should be used to select the default
        model.
        Possible values are:

        - ``'relative'``: Apply a constraint relative to the best found
            models. A relative constraint on accuracy with F1 metric would
            look at the best F1 model found and tolerate a ``constraint_value``
            relative deviation to it at max, returning the model with
            the best fairness within that constraint.
        - ``'absolute'``: Apply an absolute constraint to best found
            models. An absolute constraint on fairness with Equalized Odds
            metric would only consider models with Equalized Odds below
            ``constraint_value``, returning the model with
            the best accuracy within that constraint.

    constraint_value: float, default=0.05
        What value to apply the constraint with when selecting the default
        model. Look at ``constraint_type``'s documentation for more
        details.
    base_estimator_uses_protected_attributes: bool, default=True
        Whether or not ``base_estimator`` uses the protected attributes for
        inference. If set to ``False``, protected attributes will be removed
        from any input dataset before being collecting predictions from
         ``base_estimator``.
    n_trials_per_group: int, default=100
        Number of different multiplying scalars to consider. Scales
        linearly with the number of groups in the data, i.e.
        ``n_trials = n_trials_per_group * n_groups``.
        When both ``n_trials_per_group`` and ``time_limit`` are specified,
        the first occurrence will stop the search procedure.
    time_limit: float or None, default=None
        Number of seconds to spend in search at maximum. ``None`` value
        means no time limit is set.
        When both ``n_trials_per_group`` and ``time_limit`` are specified,
        the first occurrence will stop the search procedure.
    subsampling: int, default=50000
        The number of rows to subsample the dataset to when tuning.
        This parameter drastically improves running time on large datasets
        with little decrease in overall performance. Can be deactivated
        by passing ``numpy.inf``.
    regularization_factor: float, default=0.001
        The amount of regularization to be applied when selecting multipliers.
    favorable_label_idx: int, default=1
        Index of the favorable label to use when computing metrics.
    random_seed: int, default=0
        Random seed to ensure reproducible outcome.
    third_objective: bool, default=True
        Uses Levelling Down if True

    Attributes
    ----------
    tradeoff_summary_: pd.DataFrame
        DataFrame containing the optimal fairness-accuracy trade-off
        models with only the most relevant information.
    selected_multipliers_idx_: int
        Index of the currently selected model for ``self._best_trials_detailed``.
    selected_multipliers_: pd.DataFrame
        DataFrame containing the multipliers for each sensitive group
        that are currently used for inference.
    constrained_metric_: str
        Name of the metric on which the constraint is applied.
    unconstrained_metric_: str
        Name of the metric on which no constraint is applied.
    constraint_criterion_value_: float
        Value of the constraint being currently applied.

    Raises
    ------
    GuardianAITypeError, GuardianAIValueError
        Will be raised when one input argument is invalid


    Examples
    --------

    .. code-block:: python

        from guardian_ai.fairness.bias_mitigation import ModelBiasMitigator

        bias_mitigated_model = ModelBiasMitigator(model,
                                               protected_attribute_names='sex',
                                               fairness_metric='equalized_odds',
                                               accuracy_metric='balanced_accuracy')

        # Scikit-learn like API supported
        bias_mitigated_model.fit(X_val, y_val)
        y_pred_proba = bias_mitigated_model.predict_proba(X_test)
        y_pred_labels = bias_mitigated_model.predict(X_test)

        # Use show_tradeoff() to display all available models
        bias_mitigated_model.show_tradeoff()

        # Can select a specific model manually
        bias_mitigated_model.select_model(1)

        # Predictions are now made with new model
        y_pred_proba = bias_mitigated_model.predict_proba(X_test)
        y_pred_labels = bias_mitigated_model.predict(X_test)
    """  # noqa D412

    def __init__(
        self,
        base_estimator: BaseEstimator,
        protected_attribute_names: Union[str, List[str]],
        fairness_metric: Union[str, Callable],
        accuracy_metric: Union[str, Callable],
        higher_fairness_is_better: Union[bool, str] = "auto",
        higher_accuracy_is_better: Union[bool, str] = "auto",
        fairness_metric_uses_probas: Union[bool, str] = "auto",
        accuracy_metric_uses_probas: Union[bool, str] = "auto",
        constraint_target: str = "accuracy",
        constraint_type: str = "relative",
        constraint_value: float = 0.05,
        base_estimator_uses_protected_attributes: bool = True,
        n_trials_per_group: int = 100,
        time_limit: Optional[float] = None,
        subsampling: int = 50000,
        regularization_factor: float = 1e-3,
        favorable_label_idx: int = 1,
        random_seed: int = 0,
        third_objective: bool = True,
    ):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self._base_estimator = base_estimator
        self._protected_attribute_names = protected_attribute_names
        self._higher_fairness_is_better = higher_fairness_is_better
        self._higher_accuracy_is_better = higher_accuracy_is_better
        self._fairness_metric_uses_probas = fairness_metric_uses_probas
        self._accuracy_metric_uses_probas = accuracy_metric_uses_probas
        self._constraint_target = constraint_target
        self._constraint_type = constraint_type
        self._constraint_value = constraint_value
        self._base_estimator_uses_protected_attributes = (
            base_estimator_uses_protected_attributes
        )
        self._n_trials_per_group = n_trials_per_group
        self._time_limit = time_limit
        self._subsampling = subsampling
        self._regularization_factor = regularization_factor
        self._favorable_label_idx = favorable_label_idx
        self._random_seed = random_seed
        self._warmstart_grid_resolution = 100
        self._warmstart = True

        self._set_metric_names_and_callables(fairness_metric, accuracy_metric)

        # Public attributes to be set by `fit`
        self.tradeoff_summary_: Optional[pd.DataFrame] = None
        self.selected_multipliers_idx_: Optional[int] = None
        self.constrained_metric_: Optional[str] = None
        self.unconstrained_metric_: Optional[str] = None
        self.constraint_criterion_value_: Optional[float] = None

        # Private attributes to be set by `fit`
        self._best_trials_detailed: Optional[pd.DataFrame] = None
        self._accuracy_base_: Optional[float] = None
        self._fairness_base_: Optional[float] = None
        self._unique_groups_: Optional[np.ndarray] = None
        self._unique_group_names_: Optional[list] = None
        self._multiplier_names_: Optional[List[str]] = None
        self._admissible_trials_mask_: Optional[pd.DataFrame] = None
        self._third_objective = third_objective
        self._third_objective_name = "Levelling Down"

        self._validate_current_state()

        # Setup the objective function
        def objective_fn(trial):
            probas = self._probas_predicted.copy()

            multipliers = {}
            for group_name, multiplier_name in zip(
                self._unique_group_names_, self._multiplier_names_
            ):
                min_val, max_val = self._group_ranges[group_name]
                multipliers[multiplier_name] = trial.suggest_float(
                    multiplier_name, min_val, max_val, log=True
                )

            penalty_acc, penalty_fairness = self._get_multiplier_penalty(
                multipliers, self._group_ranges, self._unique_group_names_
            )

            probas = _apply_multiplier(
                probas,
                self._groups,
                self._unique_groups_,
                self._unique_group_names_,
                multipliers,
                self._favorable_label_idx,
            )

            perf = self._get_accuracy_score(self._y, probas)
            fairness = self._get_fairness_score(self._y, probas, self._groups)
            if self.fairness_metric_name in _valid_regression_metrics:
                adjusted_fairness_regressions = []
                base_fairness_trial = self._regression_metric_trials_["base"]
                adjusted_fairness_trial = self._get_outcome_rates(
                    self._y, probas, self._groups
                )
                self._regression_metric_trials_[
                    trial._trial_id
                ] = adjusted_fairness_trial
                for (
                    group_fairness_name,
                    group_fairness_value,
                ) in adjusted_fairness_trial.items():
                    # Cur version: maximize avg_{groups} (min(TPR_group_adjusted - TPR_group_original, 0))
                    if self.fairness_metric_name in _positive_fairness_names:
                        adjusted_fairness_regressions.append(
                            min(
                                0,
                                group_fairness_value
                                - base_fairness_trial[group_fairness_name],
                            )
                        )
                    else:
                        adjusted_fairness_regressions.append(
                            min(
                                0,
                                base_fairness_trial[group_fairness_name]
                                - group_fairness_value,
                            )
                        )

                avg_adjusted_fairness_regression = sum(
                    adjusted_fairness_regressions
                ) / len(adjusted_fairness_regressions)
                self._avg_fairness_regression_[
                    trial._trial_id
                ] = avg_adjusted_fairness_regression
            if (
                self.fairness_metric_name in _valid_regression_metrics
                and self._third_objective
            ):
                return (
                    perf + penalty_acc,
                    fairness + penalty_fairness,
                    avg_adjusted_fairness_regression,
                )
            else:
                return perf + penalty_acc, fairness + penalty_fairness

        self._objective_fn = objective_fn

    def _validate_current_state(self):
        """
        Validate current attributes have valid values.

        Raises
        ------
        GuardianAITypeError
            Will be raised when one input argument has invalid type
        GuardianAIValueError
            Will be raised when one input argument has invalid value
        """
        if isinstance(self._protected_attribute_names, str):
            self._protected_attribute_names = [self._protected_attribute_names]

        if self._higher_accuracy_is_better != "auto":
            if not isinstance(self._higher_accuracy_is_better, bool):
                raise GuardianAIValueError(
                    "`higher_accuracy_is_better` should be a bool or 'auto'"
                    f", received {self._higher_accuracy_is_better} instead."
                )

        if self._higher_fairness_is_better != "auto":
            if not isinstance(self._higher_fairness_is_better, bool):
                raise GuardianAIValueError(
                    "`higher_fairness_is_better` should be a bool or 'auto'"
                    f", received {self._higher_fairness_is_better} instead."
                )

        if self._fairness_metric_uses_probas != "auto":
            if not isinstance(self._fairness_metric_uses_probas, bool):
                raise GuardianAIValueError(
                    "`_fairness_metric_uses_probas` should be a bool or 'auto'"
                    f", received {self._fairness_metric_uses_probas} instead."
                )

        if self._accuracy_metric_uses_probas != "auto":
            if not isinstance(self._accuracy_metric_uses_probas, bool):
                raise GuardianAIValueError(
                    "`_accuracy_metric_uses_probas` should be a bool or 'auto'"
                    f", received {self._accuracy_metric_uses_probas} instead."
                )

        supported_constraint_targets = ["accuracy", "fairness"]
        if self._constraint_target not in supported_constraint_targets:
            raise GuardianAIValueError(
                f"Received `{self._constraint_target}` for `constraint_target`. "
                f"Supported values are {supported_constraint_targets}"
            )

        supported_constraint_types = ["absolute", "relative"]
        if self._constraint_type not in supported_constraint_types:
            raise GuardianAIValueError(
                f"Received `{self._constraint_type}` for `constraint_type`. "
                f"Supported values are {supported_constraint_types}"
            )

        if not isinstance(self._constraint_value, (float, int)):
            raise GuardianAITypeError(
                "`constraint_value` should be a float, received "
                f"{self._constraint_type} instead."
            )

        if not isinstance(self._base_estimator_uses_protected_attributes, bool):
            raise GuardianAITypeError(
                "`base_estimator_uses_protected_attributes` should be a bool"
                f", received {self._base_estimator_uses_protected_attributes} instead."
            )

        if self._n_trials_per_group is not None:
            if (
                not isinstance(self._n_trials_per_group, int)
                or self._n_trials_per_group <= 0
            ):
                raise GuardianAIValueError(
                    "`n_trials_per_group` should be a positive integer or None, received "
                    f"{self._n_trials_per_group} instead."
                )

        if self._time_limit is not None:
            if (
                not isinstance(self._time_limit, (float, int))
                or self._time_limit <= 0.0
            ):
                raise GuardianAIValueError(
                    "`time_limit` should be a positive float or None, received "
                    f"{self._time_limit} instead."
                )

        if self._n_trials_per_group is None and self._time_limit is None:
            raise GuardianAIValueError(
                "`n_trials_per_group` and `time_limit` cannot both be None."
            )

        if not isinstance(self._subsampling, int) or self._subsampling <= 0:
            if not np.isinf(self._subsampling):
                raise GuardianAIValueError(
                    "`subsampling` should be a positive integer or `np.inf`, received "
                    f"{self._subsampling} instead."
                )

        if (
            not isinstance(self._regularization_factor, (float, int))
            or self._regularization_factor < 0
        ):
            raise GuardianAIValueError(
                "`regularization_factor` should be a non-negative float, received "
                f"{self._regularization_factor} instead."
            )

        if (
            not isinstance(self._favorable_label_idx, int)
            or self._favorable_label_idx < 0
        ):
            raise GuardianAIValueError(
                "`favorable_label_idx` should be a non-negative integer, received "
                f"{self._favorable_label_idx} instead."
            )

        if self._random_seed is not None:
            if not isinstance(self._random_seed, int) or self._random_seed < 0:
                raise GuardianAIValueError(
                    "`random_seed` should be a non-negative integer or None, received "
                    f"{self._random_seed} instead."
                )

    def _set_metric_names_and_callables(
        self,
        fairness_metric: Union[str, Callable],
        accuracy_metric: Union[str, Callable],
    ):
        """
        Grab fairness and accuracy metrics from input arguments.

        Set values to _callable and _name attributes for fairness and
        accuracy metrics.

        Arguments
        ---------
        fairness_metric: str, Callable
            The fairness metric to use.
        accuracy_metric: str, Callable
            The accuracy metric to use.

        Raises
        ------
        GuardianAITypeError
            Will be raised if one the metrics is not a str or callable
        GuardianAIValueError
            Will be raised if there is an invalid combination of a metric and
            its `higher_is_better` and `uses_probas` attributes.
        """
        self.accuracy_metric_callable: Callable
        self.accuracy_metric_name: str

        self.fairness_metric_callable: Callable
        self.fairness_metric_name: str

        if isinstance(accuracy_metric, str):
            if self._higher_accuracy_is_better != "auto":
                raise GuardianAIValueError(
                    '`higher_accuracy_is_better` should be set to "auto" when'
                    "`accuracy_metric` is a str."
                )

            if self._accuracy_metric_uses_probas != "auto":
                raise GuardianAIValueError(
                    '`accuracy_metric_uses_probas` should be set to "auto" when'
                    "`accuracy_metric` is a str."
                )

            metric_object = skl_metrics.get_scorer(accuracy_metric)

            self.accuracy_metric_callable = _PredictionScorer(metric_object)
            self.accuracy_metric_name = accuracy_metric
            # Always true because scores are inverted by sklearn when needed
            self._higher_accuracy_is_better = True
            self._accuracy_metric_uses_probas = isinstance(
                metric_object,
                (
                    skl_metrics._scorer._ProbaScorer,
                    skl_metrics._scorer._ThresholdScorer,
                ),
            )
        elif callable(accuracy_metric):
            if self._higher_accuracy_is_better == "auto":
                raise GuardianAIValueError(
                    "`higher_accuracy_is_better` should be manually set when"
                    "`accuracy_metric` is a callable."
                )

            if self._accuracy_metric_uses_probas == "auto":
                raise GuardianAIValueError(
                    "`accuracy_metric_uses_probas` should be manually set when"
                    "`accuracy_metric` is a callable."
                )

            self.accuracy_metric_callable = accuracy_metric
            self.accuracy_metric_name = accuracy_metric.__name__
        else:
            raise GuardianAITypeError(
                "`accuracy_metric` should be a `str` or callable. Received "
                f"{accuracy_metric} instead."
            )

        if isinstance(fairness_metric, str):
            if self._higher_fairness_is_better != "auto":
                raise GuardianAIValueError(
                    '`higher_fairness_is_better` should be set to "auto" when'
                    "`fairness_metric` is a str."
                )

            if self._fairness_metric_uses_probas != "auto":
                raise GuardianAIValueError(
                    '`fairness_metric_uses_probas` should be set to "auto" when'
                    "`fairness_metric` is a str."
                )

            self.fairness_metric_callable = _get_fairness_metric(fairness_metric)
            self.fairness_metric_name = fairness_metric
            self._higher_fairness_is_better = False
            self._fairness_metric_uses_probas = False
        elif callable(fairness_metric):
            if self._higher_fairness_is_better == "auto":
                raise GuardianAIValueError(
                    "`higher_fairness_is_better` should be manually set when"
                    "`fairness_metric` is a callable."
                )

            if self._fairness_metric_uses_probas == "auto":
                raise GuardianAIValueError(
                    "`fairness_metric_uses_probas` should be manually set when"
                    "`fairness_metric` is a callable."
                )

            self.fairness_metric_callable = fairness_metric
            self.fairness_metric_name = fairness_metric.__name__
        else:
            raise GuardianAITypeError(
                "`fairness_metric` should be a `str` or callable. Received "
                f"{fairness_metric} instead."
            )

    def _get_fairness_score(
        self, y_true: np.ndarray, y_probas: np.ndarray, groups: pd.DataFrame
    ) -> float:
        """
        Get fairness score.

        Arguments
        ---------
        y_true: np.ndarray
            True labels
        y_probas: np.ndarray
            Label probabilities
        groups: pd.DataFrame
            Protected attribute(s) value(s) for every sample.

        Returns
        -------
        float: score
            The fairness score
        """
        if self._fairness_metric_uses_probas:
            y_pred = y_probas[:, self._favorable_label_idx]
        else:
            y_pred = y_probas.argmax(-1)

        return self.fairness_metric_callable(y_true, y_pred, groups)

    def _get_accuracy_score(self, y_true: np.ndarray, y_probas: np.ndarray) -> float:
        """
        Get accuracy score.

        Arguments
        ---------
        y_true: np.ndarray
            True labels
        y_probas: np.ndarray
            Label probabilities

        Returns
        -------
        float: score
            The accuracy score
        """
        if self._accuracy_metric_uses_probas:
            y_pred = y_probas
        else:
            y_pred = y_probas.argmax(-1)

        return self.accuracy_metric_callable(y_true, y_pred)

    def _get_outcome_rates(
        self, y_true: np.ndarray, y_probas: np.ndarray, groups: pd.DataFrame
    ) -> Dict:
        """
        Get raw outcome rate scores.

        Arguments
        ---------
        y_true: np.ndarray
            True labels
        y_probas: np.ndarray
            Label probabilities
        groups: pd.DataFrame
            Protected attribute(s) value(s) for every sample.

        Returns
        -------
        Dict: score
            The fairness score
        """
        if self._fairness_metric_uses_probas:
            y_pred = y_probas[:, self._favorable_label_idx]
        else:
            y_pred = y_probas.argmax(-1)

        outcome_rates = {}
        for group_name in self._group_masks:
            mask = self._group_masks[group_name]
            outcome_rates[group_name] = self._rate_scorer(y_true[mask], y_pred[mask])

        return outcome_rates

    def fit(self, X: pd.DataFrame, y: Union[pd.DataFrame, pd.Series, np.ndarray]):
        """
        Apply bias mitigation to the base estimator given a dataset and labels.

        Note that it is highly recommended you use a validation set for this
        method, so as to have a more representative range of probabilities
        for the model instead of the potentially skewed probabilities on
        training samples.

        Parameters
        ----------
        X: pd.DataFrame
            The dataset on which to mitigate the estimator's bias.
        y: pd.DataFrame, pd.Series, np.ndarray
            The labels for which to mitigate the estimator's bias.

        Returns
        -------
        self: ModelBiasMitigator
            The fitted ModelBiasMitigator object.

        Raises
        ------
        GuardianAIValueError
            Raised when an invalid value is encountered.
        """
        groups, group_names = self._prepare_subgroups(X)

        X, y, group_names, groups = self._apply_subsampling(X, y, group_names, groups)

        self._probas_predicted = self._get_base_probas(X)
        self._groups = groups
        self._X = X
        self._y = y

        # Includes the original model and, depending on the fairness metrics, additional solutions
        # obtained by leveling up all groups and from the EO paper.
        default_multipliers, max_multiplier = self._warmstart_multipliers(
            X,
            y,
            groups,
        )
        self._default_multipliers = default_multipliers
        self._max_multiplier = max_multiplier

        self._group_ranges = self._get_group_ranges(
            self._probas_predicted,
            groups,
            max_multiplier,
        )

        self._regression_metric_trials_ = {}
        self._avg_fairness_regression_ = {}
        self._accuracy_base_ = self._get_accuracy_score(y, self._probas_predicted)
        self._fairness_base_ = self._get_fairness_score(
            y, self._probas_predicted, groups
        )

        if self.fairness_metric_name in _valid_regression_metrics:
            self._regression_metric_trials_["base"] = self._get_outcome_rates(
                y, self._probas_predicted, groups
            )

        sampler = optuna.samplers.NSGAIISampler(seed=self._random_seed)
        study = optuna.create_study(
            directions=self._get_optimization_directions(), sampler=sampler
        )

        if self._unique_group_names_ is None:
            raise GuardianAIValueError("_unique_group_names cannot be None!")

        for multipliers in default_multipliers:
            study.enqueue_trial(multipliers)

        # Finally, we allow Optuna to use these starting points to search for solutions
        # that trade-off between these three solutions/objectives.
        study.optimize(
            self._objective_fn,
            n_trials=self._n_trials_per_group * len(self._unique_group_names_),
            timeout=self._time_limit,
            show_progress_bar=True,
        )

        self._produce_best_trials_frame(study, self._group_ranges)

        self._select_best_model_from_constraints()

        # Otherwise the object cannot be pickled
        self._objective_fn = None

        return self

    def _prepare_subgroups(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Handle protected subgroups logic.

        Sets the `_unique_groups_`, `_unique_group_names_` and `_multiplier_names_`
        attributes.

        Arguments
        ---------
        X: pd.DataFrame
            Dataset to prepare subgroups for.

        Returns
        -------
        (pd.DataFrame, pd.Series)
        Tuple containing
            groups: pd.DataFrame
                DataFrame mapping every sample to its protected attribute(s) ]
                value(s).
            group_names: pd.Series
                Series mapping every sample to its unique group name.

        Raises
        ------
        GuardianAIValueError
            Raised when an invalid value is encountered.
        """
        groups = X[self._protected_attribute_names].astype("category")

        unique_groups_and_counts = groups.value_counts().reset_index(name="count")

        unique_groups = unique_groups_and_counts[self._protected_attribute_names]
        self._unique_groups_ = unique_groups.values

        unique_groups["name"] = unique_groups.apply(
            lambda x: "--".join(
                [f"{attr}={x[attr]}" for attr in self._protected_attribute_names]
            ),
            axis=1,
        )
        self._unique_group_names_ = unique_groups["name"].tolist()
        self._multiplier_names_ = [
            f"multiplier_{group_name}" for group_name in self._unique_group_names_  # type: ignore
        ]

        if self._unique_group_names_ is None:
            raise GuardianAIValueError("_unique_groupe_names cannot be None!")
        if self._multiplier_names_ is None:
            raise GuardianAIValueError("_multiplier_names cannot be None!")

        groups["name"] = ""
        for group, group_name in zip(
            self._unique_groups_, self._unique_group_names_  # type: ignore
        ):  # type: ignore
            mask = (groups.drop(columns="name") == group).all(1).to_numpy().squeeze()

            groups["name"][mask] = group_name

        return groups.drop(columns="name"), groups["name"]

    def _apply_subsampling(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        group_names: pd.Series,
        groups: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
        """
        Apply subsampling on the input dataset.

        The subsampling applied is stratified sampling from the groundtruth
        labels.

        Arguments
        ---------
        X: pd.DataFrame
            The dataset on which to apply subsampling.
        y: pd.Series
            The labels on which to apply subsampling.
        group_names: pd.Series
            Series mapping every sample to its unique group name. Used to
            apply subsampling and also to return subsampled version.
        groups: pd.DataFrame
            DataFrame mapping every sample to its protected attribute(s)
            value(s). Subsampling is applied to it.

        Returns
        -------
        pd.DataFrame, pd.Series, pd.Series, pd.DataFrame
        Tuple containing
            X_subsampled: pd.DataFrame
                Subsampled version of input X
            y_subsampled: pd.Series
                Subsampled version of input y
            group_names_subsampled: pd.Series
                Subsampled version of input group_names
            groups_subsampled: pd.DataFrame
                Subsampled version of input groups
        """
        if len(X) <= self._subsampling:
            return X, y, group_names, groups
        else:
            n_samples = min(len(X), self._subsampling)

            sss = StratifiedShuffleSplit(
                n_splits=1, test_size=n_samples / len(X), random_state=self._random_seed
            )

            stratas = pd.concat((groups, y), axis=1)
            stratas = OrdinalEncoder().fit_transform(stratas)

            _, idxs = next(
                iter(sss.split(np.arange(0, len(X)).reshape(-1, 1), y=stratas))
            )

            return X.iloc[idxs], y.iloc[idxs], group_names.iloc[idxs], groups.iloc[idxs]

    def _get_group_ranges(
        self,
        probas,
        groups: pd.DataFrame,
        max_multiplier,
    ) -> Dict:
        """
        Return the range for which to search multipliers for each sensitive
        group.

        The logic is that if probabilities are constrained to [0.45, 0.55]
        range, we should be looking at multipliers much closer to 1 than if
        the probabilities are in [0.05, 0.95] range. In the former case, a
        multiplier of 1.25 suffices to flip all predictions while in the latter
        a multiplier of 10 is not enough to flip all predictions.

        The returned ranges are set to ensure that total prediction flips are
        possible and that we constrain the search to relevant multipliers (e.g.
        it's pointless to test a value of 1.5 if 1.25 already suffices to flip
        all predictions).

        If there is already a large probability coverage (e.g. [0.05, 0.95]),
        we constraint the multipliers search range for this group to [0.1, 10]
        as a reasonable enough default.

        Arguments
        ---------
        probas: pd.DataFrame
            The probabilities used to collect group-specific probability ranges.
        groups: pd.DataFrame
            The groups used to separate samples.

        Returns
        -------
        Dict: group_ranges
            Dictionary mapping every group name to its (min, max) range
            to consider for multipliers.

        Raises
        ------
        GuardianAIValueError
            Raised when an invalid value is encountered.
        """
        group_ranges = {}

        for group, group_name in zip(self._unique_groups_, self._unique_group_names_):
            mask = (groups == group).all(1).to_numpy().squeeze()

            min_proba = probas[mask].min()
            max_proba = probas[mask].max()
            ratio = max_proba / (min_proba + 1e-6)

            ratio = min(ratio, max_multiplier)

            group_ranges[group_name] = (1 / ratio, ratio)

        return group_ranges

    def _get_multiplier_penalty(
        self, multipliers: Dict, group_ranges: Dict, unique_group_names: Optional[List]
    ) -> Tuple[float, float]:
        """
        Get the multiplier penalty for both the fairness and accuracy metrics.

        Returned values are already adjusted to be either negative or positive
        penalties so they can be added directly to the scores.

        Arguments
        ---------
        multipliers: Dict
            Mapping from multiplier name to multiplier value.
        group_ranges: Dict
            Mapping from group name to group range (min_val, max_val).
        unique_group_names: List or None
            Array of all unique group names.

        Returns
        -------
        (float, float)
        Tuple containing
            accuracy_penalty: float
                The penalty to be applied on the accuracy score.
            fairness_penalty: float
                The penalty to be applied on the fairness score.

        Raises
        ------
        GuardianAIValueError
            Raised when an invalid value is encountered.
        """
        if unique_group_names is None:
            raise GuardianAIValueError("unique_group_names cannot be None!")

        multiplier_reg_penalty = []
        for group_name in unique_group_names:
            multiplier = multipliers[f"multiplier_{group_name}"]
            _, max_val = group_ranges[group_name]

            penalty = np.abs(np.log(multiplier)) / np.log(max_val)
            multiplier_reg_penalty.append(penalty)

        penalty = self._regularization_factor * np.mean(multiplier_reg_penalty)

        penalty_direction_acc = -1 if self._higher_accuracy_is_better else 1
        penalty_direction_fairness = -1 if self._higher_fairness_is_better else 1

        return penalty * penalty_direction_acc, penalty * penalty_direction_fairness

    def _get_pareto_efficient_points(self, metrics):
        """
        Find the pareto-efficient points
        :param cometricssts: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        is_efficient = np.ones(metrics.shape[0], dtype=bool)
        for i, m in enumerate(metrics):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(
                    metrics[is_efficient] > m, axis=1
                )  # Keep any point with a higher cost
                is_efficient[i] = True  # And keep self
        return is_efficient

    def _produce_best_trials_frame(self, study: optuna.Study, group_ranges: Dict):
        """
        Produce _best_trials_detailed dataframe from optuna Study object.

        Arguments
        ---------
        study: optuna.Study
            The completed study object.
        group_ranges: Dict
            Mapping from group name to group range (min_val, max_val).

        Raises
        ------
        GuardianAIValueError
            Raised when an invalid value is encountered.
        """
        if self._multiplier_names_ is None:
            raise GuardianAIValueError("_multiplier_names_ cannot be None!")
        regularized_metric_names = [
            f"{metric}_regularized"
            for metric in [self.accuracy_metric_name, self.fairness_metric_name]
        ]
        metric_names = copy.deepcopy(regularized_metric_names)
        if (
            self.fairness_metric_name in _valid_regression_metrics
            and self._third_objective
        ):
            metric_names.append("Thrid Objective: " + self._third_objective_name)

        if self.fairness_metric_name in _valid_regression_metrics:
            fairness_trial_names = list(
                list(self._regression_metric_trials_.items())[0][1].keys()
            )
            df = pd.DataFrame(
                [
                    (
                        *trial.values,
                        *trial.params.values(),
                        *self._regression_metric_trials_[trial._trial_id].values(),
                    )
                    for trial in study.best_trials
                ],
                columns=metric_names + self._multiplier_names_ + fairness_trial_names,
            )
        else:
            df = pd.DataFrame(
                [
                    (*trial.values, *trial.params.values())
                    for trial in study.best_trials
                ],
                columns=regularized_metric_names + self._multiplier_names_,
            )

        # This is calculated for when objective_fn considers only two metrics:
        if self.fairness_metric_name in _valid_regression_metrics:
            avg_fairness_regression_best_trials = []
            for trial in study.best_trials:
                avg_fairness_regression_best_trials.append(
                    self._avg_fairness_regression_[trial._trial_id]
                )
            df[self._third_objective_name] = pd.Series(
                avg_fairness_regression_best_trials
            ).values

        # Unwrap regularization factors
        regularization_factors = np.array(
            [
                self._get_multiplier_penalty(
                    multipliers, group_ranges, self._unique_group_names_
                )
                for _, multipliers in df[self._multiplier_names_].iterrows()
            ],
            dtype=float,
        )
        df["regularization_accuracy"] = regularization_factors[:, 0]
        df["regularization_fairness"] = regularization_factors[:, 1]

        df[self.accuracy_metric_name] = (
            df[f"{self.accuracy_metric_name}_regularized"]
            - df["regularization_accuracy"]
        )
        df[self.fairness_metric_name] = (
            df[f"{self.fairness_metric_name}_regularized"]
            - df["regularization_fairness"]
        )

        # Remove possible multipliers duplicates
        df = df.drop_duplicates(self._multiplier_names_)

        # Filter pareto front solutions
        if (
            self.fairness_metric_name in _valid_regression_metrics
            and self._third_objective
        ):
            pareto_columns = [
                self.accuracy_metric_name,
                self.fairness_metric_name,
                self._third_objective_name,
            ]
        else:
            pareto_columns = [self.accuracy_metric_name, self.fairness_metric_name]
        datapoints = df[pareto_columns].copy()
        datapoints[self.fairness_metric_name] = -datapoints[self.fairness_metric_name]
        datapoints = datapoints.to_numpy()
        df = df.iloc[np.where(self._get_pareto_efficient_points(datapoints))]

        # Sort best trials by fairness
        df = df.sort_values(by=self.fairness_metric_name)

        # Need to reset index so that it's ordered by fairness
        df = df.reset_index(drop=True)

        self._best_trials_detailed = df
        self.tradeoff_summary_ = df.drop(
            [col for col in df.columns if "regulariz" in col], axis=1
        )
        self.tradeoff_summary_ = self.tradeoff_summary_[
            self.tradeoff_summary_.columns[::-1]
        ]

    def _get_optimization_directions(self) -> List[str]:
        """
        Return optimization direction list used by Optuna to optimize
        fairness-accuracy trade-off.

        Returns
        -------
        optimization_directions: List[str]
            List of str corresponding to optimization directions for the
            two metrics.
        """

        def _get_optimization_direction(higher_is_better: Union[bool, str]):
            return "maximize" if higher_is_better else "minimize"

        if (
            self.fairness_metric_name in _valid_regression_metrics
            and self._third_objective
        ):
            return [
                _get_optimization_direction(self._higher_accuracy_is_better),
                _get_optimization_direction(self._higher_fairness_is_better),
                _get_optimization_direction(
                    True
                ),  # higher outcome regression is better
            ]
        else:
            return [
                _get_optimization_direction(self._higher_accuracy_is_better),
                _get_optimization_direction(self._higher_fairness_is_better),
            ]

    def _get_base_probas(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get the probabilities from the base estimator on a dataset.

        Is in charge of removing the protected attributes if needed.

        Arguments
        ---------
        X: pd.DataFrame
            The dataset for which to collect label probabilities.

        Returns
        -------
        probas: np.ndarray
            Label probabilities for every row in X.
        """
        if self._base_estimator_uses_protected_attributes:
            return self._base_estimator.predict_proba(X)  # type: ignore
        else:
            return self._base_estimator.predict_proba(  # type: ignore
                X.drop(columns=self._protected_attribute_names)  # type: ignore
            )

    def _select_best_model_from_constraints(self):
        """
        Select best model from the available trade-offs according to
        constraint.

        Calls `select_best_model` with best found model.

        Raises
        ------
        GuardianAIValueError
            If ``constraint_target`` or ``constraint_type`` have invalid
            values.
        """
        if self._constraint_target == "fairness":
            self.constrained_metric_ = self.fairness_metric_name
            self.unconstrained_metric_ = self.accuracy_metric_name
            constrained_higher_is_better = self._higher_fairness_is_better
            unconstrained_higher_is_better = self._higher_accuracy_is_better
        elif self._constraint_target == "accuracy":
            self.constrained_metric_ = self.accuracy_metric_name
            self.unconstrained_metric_ = self.fairness_metric_name
            constrained_higher_is_better = self._higher_accuracy_is_better
            unconstrained_higher_is_better = self._higher_fairness_is_better
        else:
            raise GuardianAIValueError(
                "Only `accuracy` and `fairness` are supported for "
                f"`constraint_target`. Received {self._constraint_target}"
            )

        if self._constraint_type == "relative":
            if constrained_higher_is_better:
                ref_val = self._best_trials_detailed[self.constrained_metric_].max()
                relative_ratio = (
                    (1 - self._constraint_value)
                    if ref_val > 0
                    else (1 + self._constraint_value)
                )
                self.constraint_criterion_value_ = relative_ratio * ref_val
                self._admissible_trials_mask_ = (
                    self._best_trials_detailed[self.constrained_metric_]
                    >= self.constraint_criterion_value_
                )
            else:
                ref_val = self._best_trials_detailed[self.constrained_metric_].min()
                relative_ratio = (
                    (1 + self._constraint_value)
                    if ref_val > 0
                    else (1 - self._constraint_value)
                )
                self.constraint_criterion_value_ = relative_ratio * ref_val
                self._admissible_trials_mask_ = (
                    self._best_trials_detailed[self.constrained_metric_]
                    <= self.constraint_criterion_value_
                )
        elif self._constraint_type == "absolute":
            self.constraint_criterion_value_ = self._constraint_value
            if constrained_higher_is_better:
                self._admissible_trials_mask_ = (
                    self._best_trials_detailed[self.constrained_metric_]
                    >= self.constraint_criterion_value_
                )
            else:
                self._admissible_trials_mask_ = (
                    self._best_trials_detailed[self.constrained_metric_]
                    <= self.constraint_criterion_value_
                )
        else:
            raise GuardianAIValueError(
                "Only `relative` and `absolute` are supported for "
                f"`constraint_type`. Received {self._constraint_type}"
            )

        admissible = self._best_trials_detailed[self._admissible_trials_mask_]

        if unconstrained_higher_is_better:
            best_model = admissible[self.unconstrained_metric_].idxmax()
        else:
            best_model = admissible[self.unconstrained_metric_].idxmin()

        self.select_model(best_model)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class for input dataset X.

        Parameters
        ----------
        X: pd.DataFrame
            The dataset for which to collect labels.

        Returns
        -------
        labels: np.ndarray
            The labels for every sample.
        """
        return self.predict_proba(X).argmax(-1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for input dataset X.

        Parameters
        ----------
        X: pd.DataFrame
            The dataset for which to collect label probabilities.

        Returns
        -------
        probabilities: np.ndarray
            The label probabilities for every sample.
        """
        probas = self._get_base_probas(X)

        groups = X[self._protected_attribute_names].astype("category")

        self._unique_groups_ = cast(np.ndarray, self._unique_groups_)
        return _apply_multiplier(
            probas,
            groups,
            self._unique_groups_,
            self._unique_group_names_,
            self.selected_multipliers_,
            self._favorable_label_idx,
        )

    def show_tradeoff(self, hide_inadmissible: bool = False):
        """
        Show the models representing the best fairness-accuracy trade-off
        found.

        Arguments
        ---------
        hide_inadmissible: bool, default=False
            Whether or not to hide the models that don't satisfy the
            constraint.
        """
        metric_is_valid = self.fairness_metric_name in _valid_regression_metrics
        TPR_or_level_down = (
            self._third_objective_name if metric_is_valid else "TPR Difference"
        )
        df = self._best_trials_detailed

        if hide_inadmissible:
            df = df[self._admissible_trials_mask_]  # type: ignore

        df = df.reset_index()  # type: ignore

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df[self.fairness_metric_name],
                y=df[self.accuracy_metric_name],
                customdata=df[self._third_objective_name],
                text=df["index"],
                line_shape="vh" if self._higher_fairness_is_better else "hv",
                mode="markers" if metric_is_valid else "markers+lines",
                marker=(
                    dict(
                        color=df[self._third_objective_name],
                        colorscale="Bluered_r",
                        colorbar=dict(title=self._third_objective_name),
                        showscale=True,
                    )
                    if metric_is_valid
                    else None
                ),
                hovertemplate=f"{self.fairness_metric_name}"
                + ": %{x:.4f}"
                + f"<br>{self.accuracy_metric_name}"
                + ": %{y:.4f}</br>"
                + f"<br>{TPR_or_level_down}"
                + ": %{customdata:.4f}</br>"
                + "Index: %{text}",
                name="Multiplier Tuning (Best Models)",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[self._fairness_base_],
                y=[self._accuracy_base_],
                mode="markers",
                marker_symbol="cross",
                marker_color="green",
                marker_size=10,
                hovertemplate=f"{self.fairness_metric_name}"
                + ": %{x:.4f}"
                + f"<br>{self.accuracy_metric_name}"
                + ": %{y:.4f}</br>",
                name="Base Estimator",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    self._best_trials_detailed[self.fairness_metric_name].iloc[  # type: ignore
                        self.selected_multipliers_idx_
                    ]
                ],
                y=[
                    self._best_trials_detailed[self.accuracy_metric_name].iloc[  # type: ignore
                        self.selected_multipliers_idx_
                    ]
                ],
                mode="markers",
                marker_symbol="circle-open",
                marker_color="red",
                marker_line_width=3,
                marker_size=15,
                hoverinfo="skip",
                name="Currently Selected Model",
            )
        )

        # Constraint
        fig.add_trace(self._get_constraint_line(df))

        fig.update_xaxes(gridwidth=1, gridcolor="LightGrey", zerolinecolor="LightGrey")
        fig.update_yaxes(gridwidth=1, gridcolor="LightGrey", zerolinecolor="LightGrey")

        fig.update_layout(
            title="Bias Mitigation Best Models Found",
            xaxis_title=self.fairness_metric_name,
            yaxis_title=self.accuracy_metric_name,
            legend_title="Models",
            legend_orientation="h" if metric_is_valid else "v",
            plot_bgcolor="rgba(0,0,0,0)",
            width=None,
            height=600,
            margin={"t": 50, "b": 50},
        )

        fig.show()

    def _get_constraint_line(self, df: pd.DataFrame) -> Any:
        """
        Return the Plotly Line object that represents the constraint on
        the figure.

        Arguments
        ---------
        df: pd.DataFrame
            DataFrame of trials that will be plotted. Used to determine the
            range the constraint line has to cover.

        Returns
        -------
        Any: line
            The plotly line representing the constraint.
        """
        x_min = min(df[self.fairness_metric_name].min(), self._fairness_base_)
        x_max = max(df[self.fairness_metric_name].max(), self._fairness_base_)

        y_min = min(df[self.accuracy_metric_name].min(), self._accuracy_base_)
        y_max = max(df[self.accuracy_metric_name].max(), self._accuracy_base_)

        if self._constraint_target == "fairness":
            x = [self.constraint_criterion_value_] * 2
            y = [y_min, y_max]
        elif self._constraint_target == "accuracy":
            x = [x_min, x_max]
            y = [self.constraint_criterion_value_] * 2

        if self._constraint_type == "relative":
            name = f"{self._constraint_value * 100:.1f}% relative {self.constrained_metric_} drop"
        elif self._constraint_type == "absolute":
            name = f"{self.constrained_metric_} value of {self.constraint_criterion_value_:.2f}"

        return go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color="black", width=4, dash="dash"),
            name=name,
        )

    def select_model(self, model_idx: int):
        """
        Select the multipliers to use for inference.

        Arguments
        ---------
        model_idx: int
            The index of the multipliers in `self.best_trials_` to use for
            inference, as displayed by `show_tradeoff`.

        Raises
        ------
        GuardianAIValueError
            Raised when the passed model_idx is invalid.
        """
        if model_idx < 0 or model_idx >= len(self._best_trials_detailed):  # type: ignore
            raise GuardianAIValueError(f"Invalid `model_idx` received: {model_idx}")

        self.selected_multipliers_idx_ = model_idx

    @property
    def selected_multipliers_(self):  # noqa D102
        if self._best_trials_detailed is None or self.selected_multipliers_idx_ is None:
            return None
        else:
            return self._best_trials_detailed[self._multiplier_names_].iloc[
                self.selected_multipliers_idx_
            ]

    def _warmstart_multipliers(self, X, y, groups):
        default_multipliers = []

        # ----- Multipliers that re-create the original model -----
        # good accuracy, bad disparity, good regression
        default_multipliers.append(
            {
                f"multiplier_{group_name}": 1.0
                for group_name in self._unique_group_names_
            }
        )

        # Unless we calculate a better bound below, assume that multipliers should
        # never be larger than 10
        max_multiplier = 10

        # Get masks for filtering data by each group
        group_masks = {}
        for group, group_name in zip(self._unique_groups_, self._unique_group_names_):
            mask = (groups == group).all(1).to_numpy().squeeze()
            group_masks[group_name] = mask

        self._group_masks = group_masks

        if self.fairness_metric_name in _inhouse_metrics:
            # Get a rate calculator/scorer for the given constraint
            rate_scorer = _get_rate_scorer(self.fairness_metric_name)
            self._rate_scorer = rate_scorer

        # For some metrics, we can calculate additional defaults
        if self.fairness_metric_name in _inhouse_metrics and self._warmstart:
            # Check if metric is supported by fairlearn
            if self.fairness_metric_name in _automl_to_fairlearn_metric_names:
                # Fit the EO method as implemented in fairlearn
                adjusted_model = ThresholdOptimizer(
                    estimator=self._base_estimator,
                    constraints=_automl_to_fairlearn_metric_names[
                        self.fairness_metric_name
                    ],
                    objective=(
                        "accuracy_score"
                        if self.accuracy_metric_name == "accuracy"
                        else "balanced_accuracy_score"
                    ),
                    prefit=True,
                )
                adjusted_model.fit(X, y, sensitive_features=groups)

                # Estimate the target rate value for the given rate
                adjusted_predictions = adjusted_model.predict(
                    X, sensitive_features=groups
                )
                target_rate = rate_scorer(y, adjusted_predictions)

                # Find multipliers that produce the corresponding target rate as closely as possible for each group
                multipliers_eo = self._find_multipliers_for_rate(
                    y,
                    target_rate,
                    rate_scorer,
                    group_masks,
                )

                default_multipliers.append(multipliers_eo)

            # Do we want to minimize or maximize the given outcome rate?
            if self.fairness_metric_name in _positive_fairness_names:
                best = np.max
            else:
                best = np.min

            # Calculate the outcome rates for each group
            predictions = self._probas_predicted.argmax(-1)
            rates_by_group = []
            for group_name in self._unique_group_names_:
                mask = group_masks[group_name]
                rates_by_group.append(rate_scorer(y[mask], predictions[mask]))

            # Find the group with the best outcome rate
            target_rate = best(rates_by_group)

            # Find multipliers that produce the corresponding target rate as closely as possible for each group
            multipliers_or = self._find_multipliers_for_rate(
                y,
                target_rate,
                rate_scorer,
                group_masks,
            )

            default_multipliers.append(multipliers_or)

            max_multiplier = max(
                [m if m > 1 else 1 / m for m in multipliers_or.values()]
            )

        # Allow to increase/decrease probabilities by a factor of at most 10,
        # unless larger is required to allow the trivial solution above.
        max_multiplier = max(max_multiplier, 10)

        return default_multipliers, max_multiplier

    def _find_multipliers_for_rate(self, y, target_rate, rate_scorer, group_masks):
        multipliers = {}

        for group, group_name, multiplier_name in zip(
            self._unique_groups_, self._unique_group_names_, self._multiplier_names_
        ):
            # Get only the data for this group
            mask = group_masks[group_name]

            # Unique values of up to grid_resolution quantiles
            # Always include 0.5 -- the default threshold
            thresholds = np.unique(
                list(
                    np.quantile(
                        self._probas_predicted[mask],
                        np.linspace(0, 1, self._warmstart_grid_resolution + 1),
                        method="nearest",
                    )
                )
                + [0.5]
            )

            # Ensure the thresholds are in (0, 1) (exclusive)
            thresholds = thresholds[np.logical_and(thresholds > 0, thresholds < 1)]

            # Calcuate rate for each threshold
            rates = [
                rate_scorer(
                    y[mask],
                    self._probas_predicted[mask][:, self._favorable_label_idx]
                    > threshold,
                )
                for threshold in thresholds
            ]

            # Find threshold with closest rate to target rate, break ties towards
            # thresholds of 0.5 -- i.e., doing nothing
            ideal_index = np.argmin(
                (rates - target_rate) ** 2 + np.abs(thresholds - 0.5) * 10**-5
            )
            ideal_threshold = thresholds[ideal_index]

            # Closed form solution to convert thresholds to multipliers
            multipliers[multiplier_name] = (1 - ideal_threshold) / ideal_threshold

        return multipliers


def _apply_multiplier(
    probas: np.ndarray,
    groups: pd.DataFrame,
    unique_groups: np.ndarray,
    unique_group_names: pd.DataFrame,
    multipliers: pd.DataFrame,
    majority_class_idx: int,
) -> np.ndarray:
    """
    Apply multipliers over an input probas array.

    Arguments
    ---------
    probas: np.ndarray
        The class probabilities on which to apply multipliers.
    groups: pd.DataFrame
        DataFrame representing the different protected attributes' values
        for every sample.
    unique_groups: np.ndarray
        Array of all possible unique groups.
    unique_group_names: pd.DataFrame
        Name corresponding to a unique group. Used to map a group to its
        corresponding multiplier in ``multipliers``.
    multipliers: pd.DataFrame
        DataFrame mapping a group name to its multiplying scalar.
    majority_class_idx: int
        Which class label in ``probas`` to apply the multiplier on.

    Returns
    -------
    probas_multiplied: np.ndarray
        The multiplier class probabilities
    """
    for group, group_name in zip(unique_groups, unique_group_names):
        mask = (groups == group).all(1).to_numpy().squeeze()

        multiplier = multipliers[f"multiplier_{group_name}"]
        probas[:, majority_class_idx][mask] *= multiplier

    return probas / probas.sum(-1, keepdims=True)


class _PredictionReturner:
    _estimator_type = "classifier"
    classes_ = np.array([0, 1])

    def predict(self, y_pred):
        self.classes_ = np.unique(y_pred)
        return y_pred

    def predict_proba(self, y_pred):
        self.classes_ = np.arange(y_pred.shape[1])
        return y_pred


class _PredictionScorer:
    def __init__(self, scorer):
        self.scorer = scorer
        self.dummy_model = _PredictionReturner()

    def __call__(self, y_true, y_pred):
        return self.scorer(self.dummy_model, y_pred, y_true)
