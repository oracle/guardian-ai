#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Utils for computing fairness metrics"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from itertools import product
from collections import defaultdict
from typing import TYPE_CHECKING, Optional
from functools import partial

from guardian_ai.fairness.utils.lazy_loader import LazyLoader
from guardian_ai.utils.exception import GuardianAITypeError, GuardianAIValueError

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from aif360.datasets import BinaryLabelDataset
else:
    np = LazyLoader("numpy")
    pd = LazyLoader("pandas")
    BinaryLabelDataset = LazyLoader(
        "aif360.datasets", "BinaryLabelDataset", suppress_import_warnings=True
    )

DEFAULT_REDUCTION = "mean"
DEFAULT_DISTANCE = "diff"

capital_letters_regex = re.compile(r"[A-Z][^A-Z]*")


def _pandas_to_coltypes(X, ignore_time=False):
    i = 0
    col_types = ["numerical"] * X.shape[1]
    # Infer numerical/categorical/datetime columns for pandas dataframes
    inferdtypes = X.infer_objects().dtypes
    # User provided df: X may have (some or all) dtypes manually assigned, so we want to
    # use them if available. For object dtypes in X.dtypes, we overwrite with inferred
    # just in case by inferring we get the correct type. Inferring does not always work
    # for datetime columns, so we do a manual check (try/catch) at the bottom.
    for dtype, inferdtype in list(zip(X.dtypes, inferdtypes)):
        if dtype == object:
            # if the user provided dtype is object, perhaps the inferred is not
            dtype = inferdtype
        # First check if all nulls to avoid being type casted in to Timedelta or Datetime
        if pd.isnull(X.iloc[:, i]).all():
            # default to numerical and skip since column will be dropped by preprocessing
            pass
            # TODO

        if dtype == bool or dtype.name == "category":
            col_types[i] = "categorical"

        elif ignore_time:
            if "time" in dtype.name or dtype in [object]:
                col_types[i] = "categorical"

        elif "time" in dtype.name:
            col_types[i] = "timedelta" if "timedelta" in dtype.name else "datetime"

        elif dtype == object:
            try:
                # Try to force type of column to datetime, if no errors, type is datetime
                _ = pd.to_datetime(X.iloc[:, i], errors="raise")
                col_types[i] = "datetime"
            except Exception:
                # If there were errors, fall-back to categorical
                col_types[i] = "categorical"
        i += 1

    return col_types


def _get_check_array(arr, arr_name, allow_none=False):
    if arr is None:
        if allow_none:
            return arr
        else:
            raise GuardianAIValueError(f"'{arr_name}' cannot take None value.")

    if isinstance(arr, list) or isinstance(arr, np.ndarray):
        return pd.Series(arr)
    elif isinstance(arr, pd.Series):
        return arr
    else:
        raise GuardianAITypeError(
            f"Available input types for '{arr_name}' methods are 'list',"
            f"'numpy.ndarray' and 'pandas.Series', received '{type(arr)}' "
            "instead."
        )


def _get_check_arrays(y_true, y_pred, allow_y_true_none):
    y_true = _get_check_array(y_true, "y_true", allow_none=allow_y_true_none)
    y_pred = _get_check_array(y_pred, "y_pred", allow_none=False)

    n_instances = len(y_pred)

    if y_true is not None:
        if y_true.nunique() > 2:
            raise GuardianAIValueError(
                "Fairness metrics only support binary classification for the moment."
            )
        if len(y_true) != n_instances:
            raise GuardianAIValueError(
                f"y_true and y_pred should have the same number of instances. "
                f"Received {len(y_true)} and {len(y_pred)} instead."
            )

    return y_true, y_pred


class _Reduction(ABC):
    display_name = ""

    @abstractmethod
    def __call__(self, subgroup_pairs, metrics):
        pass


class _MaxReduction(_Reduction):
    display_name = "Maximum"

    def __call__(self, subgroup_pairs, metrics):
        return np.nanmax(metrics)


class _MeanReduction(_Reduction):
    display_name = "Mean"

    def __call__(self, subgroup_pairs, metrics):
        return np.nanmean(metrics)


class _RawReduction(_Reduction):
    display_name = "Raw"

    def __call__(self, subgroup_pairs, metrics):
        res = {}
        for subgroup_pair, metric in zip(subgroup_pairs, metrics):
            res[subgroup_pair] = metric
        return res


def _get_check_reduction(reduction):
    reduction_mappings = {
        "max": _MaxReduction(),
        "mean": _MeanReduction(),
        None: _RawReduction(),
    }

    if isinstance(reduction, _Reduction):
        return reduction
    else:
        if reduction in reduction_mappings:
            return reduction_mappings[reduction]
        else:
            raise GuardianAIValueError(
                f"Available reduction values are {list(reduction_mappings.keys())}, "
                f"received {reduction} instead."
            )


class _DistanceMetric(ABC):
    display_name = ""

    @abstractmethod
    def __call__(self, metrics_obj, metric):
        pass


class _DifferenceDistanceMetric(_DistanceMetric):
    display_name = "Difference"

    def __call__(self, metrics_obj, metric):
        score = metrics_obj.difference(metric)

        return np.abs(score)

    def from_raw_scores(self, score_priv, score_unpriv):
        return np.abs(score_priv - score_unpriv)


class _RatioDistanceMetric(_DistanceMetric):
    display_name = "Ratio"

    def __call__(self, metrics_obj, metric, eps=1e-6):
        score_priv = metric(privileged=True)
        score_unpriv = metric(privileged=False)

        return self.from_raw_scores(score_priv, score_unpriv, eps)

    def from_raw_scores(self, score_priv, score_unpriv, eps=1e-6):
        num = max(score_priv, score_unpriv)
        denum = min(score_priv, score_unpriv)

        num_is_zero = np.abs(num) <= eps
        denum_is_zero = np.abs(denum) <= eps

        # Handle zero-division by-hand
        if denum_is_zero:
            if num_is_zero:
                # Return 1 for 0/0 case
                return 1.0
            else:
                # Return infinity for x/0 case
                return np.inf

        return (num + eps) / (denum + eps)


class _VanillaDistanceMetric(_DistanceMetric):
    def __call__(self, metrics_obj, metric):
        return metric()


def _get_check_distance(distance_measure, allow_distance_measure_none):
    distance_mappings = {
        "diff": _DifferenceDistanceMetric(),
        "ratio": _RatioDistanceMetric(),
    }

    if isinstance(distance_measure, _DistanceMetric):
        return distance_measure
    elif distance_measure is None:
        if allow_distance_measure_none:
            return _VanillaDistanceMetric()
        else:
            raise GuardianAIValueError(
                "None is not supported as a distance measure for the "
                "chosen fairness metric."
            )
    else:
        if distance_measure in distance_mappings:
            return distance_mappings[distance_measure]
        else:
            raise GuardianAIValueError(
                f"Available distance_measure values are {list(distance_mappings.keys())}, "
                f"received {distance_measure} instead."
            )


def _check_subgroups(subgroups):
    protected_attributes = subgroups.columns

    attribute_types = _pandas_to_coltypes(subgroups)
    non_categorical_attributes = [
        (attr, attr_type)
        for attr, attr_type in zip(protected_attributes, attribute_types)
        if attr_type != "categorical"
    ]

    if len(non_categorical_attributes) > 0:
        error_msg = "Provided protected attributes should be of type 'category', 'bool', or 'object'."

        if len(non_categorical_attributes) <= 10:
            error_msg += (
                "The following attributes were not of required dtypes in X: "
                f"{non_categorical_attributes}"
            )
        raise GuardianAIValueError(error_msg)


def _get_attr_idx_mappings(subgroups):
    protected_attributes = subgroups.columns

    all_attr_vals = [np.unique(subgroups[attr]) for attr in protected_attributes]

    attr_vals_to_idx = {
        attr: {val: idx for idx, val in enumerate(attr_vals)}
        for attr, attr_vals in zip(protected_attributes, all_attr_vals)
    }
    attr_idx_to_vals = {
        attr: {idx: val for val, idx in vals_to_idx.items()}
        for attr, vals_to_idx in attr_vals_to_idx.items()
    }

    return attr_vals_to_idx, attr_idx_to_vals


def _get_subgroup_divisions(subgroups):
    protected_attributes = subgroups.columns

    all_attr_vals = [np.unique(subgroups[attr]) for attr in protected_attributes]
    all_attr_vals_tuples = [
        [(attr, idx) for idx, attr_val in enumerate(attr_vals)]
        for attr, attr_vals in zip(protected_attributes, all_attr_vals)
    ]

    subgroups = product(*all_attr_vals_tuples)
    subgroups = [
        {attr: attr_val for (attr, attr_val) in subgroup} for subgroup in subgroups
    ]
    divisions = [
        ([subgroup], [sg])
        for subgroup in subgroups
        for sg in subgroups
        if sg != subgroup
    ]

    return divisions


def _get_check_inputs(
    reduction: Optional[str],
    distance_measure: Optional[str],
    subgroups: pd.DataFrame,
    allow_distance_measure_none: bool,
):
    reduction, distance = _get_check_reduction_distance_subgroups(
        reduction,
        distance_measure,
        subgroups,
        allow_distance_measure_none,
    )

    attr_vals_to_idx, attr_idx_to_vals = _get_attr_idx_mappings(subgroups)

    subgroup_divisions = _get_subgroup_divisions(subgroups)

    return reduction, distance, attr_vals_to_idx, attr_idx_to_vals, subgroup_divisions


def _get_check_reduction_distance_subgroups(
    reduction: Optional[str],
    distance_measure: Optional[str],
    subgroups: pd.DataFrame,
    allow_distance_measure_none: bool,
):
    reduction = _get_check_reduction(reduction)
    distance = _get_check_distance(distance_measure, allow_distance_measure_none)

    _check_subgroups(subgroups)

    return reduction, distance


def _get_score_group_from_metrics(
    subgroup_metrics, distance, metric, unpriv_group, priv_group, attr_idx_to_vals
):
    metric_fn = getattr(subgroup_metrics, metric)
    score = distance(subgroup_metrics, metric_fn)

    group_repr = tuple()
    for group in [unpriv_group, priv_group]:
        cur_group_repr = tuple(
            attr_idx_to_vals[attr][idx] for attr, idx in group[0].items()
        )
        if len(cur_group_repr) == 1:
            cur_group_repr = cur_group_repr[0]
        group_repr += (cur_group_repr,)

    return score, group_repr


def _y_to_aifm_ds(y, subgroups, attr_vals_to_idx):
    # AIF360 does not allow for NA values in input dataframes
    # so we only send it protected attributes and drop rows where
    # any of the protected attribute is NA.
    df = subgroups.copy()
    df.dropna(inplace=True)

    protected_attributes = subgroups.columns

    # AIF360 requires all columns to be numerical
    for col, vals_to_idx in attr_vals_to_idx.items():
        df[col].replace(
            list(vals_to_idx.keys()), list(vals_to_idx.values()), inplace=True
        )

    df["y"] = y.to_numpy()
    ds = BinaryLabelDataset(
        df=df, label_names=["y"], protected_attribute_names=protected_attributes
    )

    return ds


class _FairnessScorer(ABC):
    def __init__(self, protected_attributes, metric):
        if isinstance(protected_attributes, str):
            protected_attributes = [protected_attributes]
        self.protected_attributes = protected_attributes
        self.metric = metric

    def _get_check_subgroups(self, X, supplementary_features):
        if supplementary_features is None:
            supplementary_features = pd.DataFrame()
        elif not isinstance(supplementary_features, pd.DataFrame):
            raise GuardianAIValueError(
                "``supplementary_features`` should be a Pandas DataFrame. Received "
                f"{type(supplementary_features)} instead."
            )

        duplicate_features = set(X.columns).intersection(
            set(supplementary_features.columns)
        )
        if len(duplicate_features) > 0:
            raise GuardianAIValueError(
                "The following feature were found in both ``X`` and"
                f"``supplementary_features``: {list(duplicate_features)}. No "
                "feature should be present in both to avoid ambiguity."
            )

        features_avail = X.columns.append(supplementary_features.columns)
        missing_features = set(self.protected_attributes) - set(features_avail)

        if len(missing_features) > 0:
            error_msg = f"The following protected attributes were not found in X: {missing_features}."

            if len(features_avail) <= 10:
                error_msg += f" Available features are: {features_avail}."

            raise GuardianAIValueError(error_msg)

        subgroups_x = X[
            [attr for attr in self.protected_attributes if attr in X.columns]
        ]
        subgroups_sf = supplementary_features[
            [
                attr
                for attr in self.protected_attributes
                if attr in supplementary_features.columns
            ]
        ]
        subgroups = pd.concat([subgroups_x, subgroups_sf], axis=1)

        # Reorder sensitive features in the order received
        subgroups = subgroups[self.protected_attributes]

        return subgroups

    @abstractmethod
    def __call__(self, model, X, y_true=None, supplementary_features=None):
        pass

    @property
    def display_name(self):
        class_name = self.__class__.__name__

        cleaned_class_name = class_name.replace("Model", "")
        cleaned_class_name = cleaned_class_name.replace("Dataset", "")
        cleaned_class_name = cleaned_class_name.replace("Scorer", "")

        return cleaned_class_name

    @property
    def _display_name_protected_attributes(self):
        base = "for "

        # Fairness metrics override protected_attributes with __copy__ appended
        # copies, so we need to clean them back here before display.
        subword_to_remove = "__copy__"
        n_chars_subword = len(subword_to_remove)

        prot_attrs = []
        for attr in self.protected_attributes:
            if attr[-n_chars_subword:] == subword_to_remove:
                prot_attrs.append(attr[:-n_chars_subword])
            else:
                prot_attrs.append(attr)
        prot_attr_display_names = [f"'{attr}'" for attr in prot_attrs]

        if len(prot_attr_display_names) <= 2:
            return base + " and ".join(prot_attr_display_names)
        else:
            first_attrs = prot_attr_display_names[:-1]
            last_attr = prot_attr_display_names[-1]

            return base + ", ".join(first_attrs) + ", and " + last_attr


def _place_space_before_capital_letters(input_str):
    capital_letter_words = capital_letters_regex.findall(input_str)

    capital_letter_words = [word.strip() for word in capital_letter_words]

    return " ".join(capital_letter_words)


def _TP(y_true, y_pred):
    return np.sum(np.logical_and(y_pred == 1, y_true == 1))


def _FN(y_true, y_pred):
    return np.sum(np.logical_and(y_pred == 0, y_true == 1))


def _FP(y_true, y_pred):
    return np.sum(np.logical_and(y_pred == 1, y_true == 0))


def _TN(y_true, y_pred):
    return np.sum(np.logical_and(y_pred == 0, y_true == 0))


def _get_rate(y_true, y_pred, rate):
    if rate == "statistical_parity":
        # Positive prediction rate
        return np.mean(y_pred)
    elif rate == "error_rate":
        # Rate of all error types
        return np.mean(np.array(y_pred) != np.array(y_true))
    elif rate == "TPR":
        # Sensitivity, hit rate, recall, or true positive rate
        TP = _TP(y_true, y_pred)
        FN = _FN(y_true, y_pred)

        return TP / (TP + FN)
    elif rate == "TNR":
        # Specificity or true negative rate
        FP = _FP(y_true, y_pred)
        TN = _TN(y_true, y_pred)

        return TN / (TN + FP)
    elif rate == "PPV":
        # Precision or positive predictive value
        TP = _TP(y_true, y_pred)
        FP = _FP(y_true, y_pred)

        return TP / (TP + FP)
    elif rate == "NPV":
        # Negative predictive value
        FN = _FN(y_true, y_pred)
        TN = _TN(y_true, y_pred)

        return TN / (TN + FN)
    elif rate == "FPR":
        # Fall out or false positive rate
        FP = _FP(y_true, y_pred)
        TN = _TN(y_true, y_pred)

        return FP / (FP + TN)
    elif rate == "FNR":
        # False negative rate
        TP = _TP(y_true, y_pred)
        FN = _FN(y_true, y_pred)

        return FN / (TP + FN)
    elif rate == "FDR":
        # False discovery rate
        TP = _TP(y_true, y_pred)
        FP = _FP(y_true, y_pred)

        return FP / (TP + FP)
    elif rate == "FOR":
        # False ommission rate
        FN = _FN(y_true, y_pred)
        TN = _TN(y_true, y_pred)

        return FN / (FN + TN)
    else:
        raise GuardianAIValueError(f"Undefined rate {rate}")


def _get_rate_scorer(fairness_metric_name):
    return partial(_get_rate, rate=fairness_metric_name)


_positive_fairness_names = ["TPR", "statistical_parity"]

_automl_to_aif360_metric_names = {
    "statistical_parity": "selection_rate",
    "TPR": "true_positive_rate",
    "FPR": "false_positive_rate",
    "FNR": "false_negative_rate",
    "FOR": "false_omission_rate",
    "FDR": "false_discovery_rate",
    "error_rate": "error_rate",
    "theil_index": "between_group_theil_index",
}

_aif360_to_automl_metric_names = dict(
    (v, k) for k, v in _automl_to_aif360_metric_names.items()
)

_automl_to_fairlearn_metric_names = {
    "statistical_parity": "demographic_parity",
    "TPR": "true_positive_rate_parity",
    "FPR": "false_positive_rate_parity",
    "FNR": "false_negative_rate_parity",
}

_inhouse_metrics = [
    "statistical_parity",
    "error_rate",
    "TPR",
    "FPR",
    "FNR",
    "FOR",
    "FDR",
]
