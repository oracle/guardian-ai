#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import math
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

from guardian_ai.fairness.bias_mitigation import ModelBiasMitigator
from guardian_ai.fairness.metrics import model_statistical_parity
from guardian_ai.utils.exception import GuardianAITypeError, GuardianAIValueError
from tests.utils import get_dummy_dataset

# Constants used when any metric is needed
A_FAIRNESS_METRIC = ["equalized_odds", "TPR"]
AN_ACCURACY_METRIC = "accuracy"

RANDOM_SEED = 12345


@pytest.fixture(scope="module", autouse=True)
def init():
    np.random.seed(RANDOM_SEED)


def is_close(a, b):
    return math.isclose(a, b, rel_tol=1e-5)


def approx_dict(d):
    return pytest.approx(d, rel=1e-5)


class DummyBinaryStochasticModel:
    def predict(self, X):
        return np.random.randint(0, 2, size=X.shape[0])


def create_concat_sensitive_attrs(dataset, n_classes):
    if not isinstance(n_classes, list):
        n_classes = list(n_classes)

    sensitive_dataset = dataset.copy()
    sensitive_attrs_names = []
    for i, n_classes_i in enumerate(n_classes):
        sensitive_vals = np.array(
            [f"sensitive_val_{idx}" for idx in range(n_classes_i)]
        )
        attr_name = f"sensitive_attr_{i}"
        sensitive_dataset = concat_sensitive_attr_column(
            sensitive_vals, sensitive_dataset, attr_name
        )
        sensitive_attrs_names.append(attr_name)

    return sensitive_dataset, sensitive_attrs_names


def concat_sensitive_attr_column(vals, dataset, attr_name):
    sensitive_vals = np.random.choice(vals, size=len(dataset))
    sensitive_feats = pd.DataFrame(np.transpose(sensitive_vals), columns=[attr_name])
    return pd.concat([dataset, sensitive_feats], axis=1)


@pytest.fixture(scope="module")
def model_type():
    return "LogisticRegression"


@pytest.fixture(scope="module")
def base_dataset():
    return get_dummy_dataset(n_samples=500, n_features=5, n_classes=2)


# By default, all tests are ran with (1 protected attr with 2 groups),
# (1 protected attr with more than 2 groups), and (more than 2 protected attr)
SENSITIVE_FEATURES_VARIATIONS = {
    "one_attr_two_classes": {"n_classes": (2,)},
    "one_attr_n_classes": {"n_classes": (4,)},
    "n_attrs": {"n_classes": (3, 4)},
}


@pytest.fixture(
    scope="module",
    params=SENSITIVE_FEATURES_VARIATIONS.values(),
    ids=SENSITIVE_FEATURES_VARIATIONS.keys(),
)
def sensitive_dataset_and_model(model_type, base_dataset, request):
    dataset, target = base_dataset
    dataset, sensitive_attr_names = create_concat_sensitive_attrs(
        dataset, **request.param
    )
    model = Pipeline(
        steps=[
            ("preprocessor", OneHotEncoder(handle_unknown="ignore")),
            ("classifier", LogisticRegression(random_state=0)),
        ]
    )
    model.fit(dataset, target)

    return dataset, target, model, sensitive_attr_names


# (metric_name, callable, higher_is_better, requires_proba) dict
FAIRNESS_METRICS = {
    "statistical_parity": (
        "statistical_parity",
        model_statistical_parity,
        False,
        False,
    ),
}


def neg_log_loss_score(y_true, y_pred, **kwargs):
    return -log_loss(y_true, y_pred, **kwargs)


# (metric_name, callable, higher_is_better, requires_proba) dict
ACCURACY_METRICS = {
    "roc_auc": ("roc_auc", roc_auc_score, True, True),
    "balanced_accuracy": ("balanced_accuracy", balanced_accuracy_score, True, False),
    "neg_log_loss": ("neg_log_loss", neg_log_loss_score, False, True),
}

METRIC_COMBOS = {
    f"{fair_name}--{acc_name}": (fair_metric, acc_metric)
    for fair_name, fair_metric in FAIRNESS_METRICS.items()
    for acc_name, acc_metric in ACCURACY_METRICS.items()
}


@pytest.fixture(scope="module", params=METRIC_COMBOS.values(), ids=METRIC_COMBOS.keys())
def responsible_model_and_metrics(sensitive_dataset_and_model, request):
    X, y, model, sensitive_attr_names = sensitive_dataset_and_model

    fairness_metric, accuracy_metric = request.param
    (
        fairness_name,
        fairness_callable,
        fairness_hib,
        fairness_uses_probas,
    ) = fairness_metric
    (
        accuracy_name,
        accuracy_callable,
        accuracy_hib,
        accuracy_uses_probas,
    ) = accuracy_metric

    resp_model = ModelBiasMitigator(
        model,
        sensitive_attr_names,
        fairness_metric=fairness_name,
        accuracy_metric=accuracy_name,
        n_trials_per_group=5,
        random_seed=RANDOM_SEED,
    )  # limit number of trials for faster tests

    resp_model.fit(X, y)

    return X, y, sensitive_attr_names, resp_model, fairness_metric, accuracy_metric


def test_sanity_checks(responsible_model_and_metrics):
    (
        X,
        y,
        sensitive_attr_names,
        resp_model,
        fairness_metric,
        accuracy_metric,
    ) = responsible_model_and_metrics

    assert len(resp_model.predict(X)) == len(X)
    assert len(resp_model.predict_proba(X)) == len(X)

    assert resp_model._best_trials_detailed is not None


def test_display(responsible_model_and_metrics):
    (
        X,
        y,
        sensitive_attr_names,
        resp_model,
        fairness_metric,
        accuracy_metric,
    ) = responsible_model_and_metrics

    resp_model.show_tradeoff()

    # Assert that displays worked correctly (best we can do automatically currently)
    assert True


@pytest.mark.parametrize("a_fairness_metric", A_FAIRNESS_METRIC)
def test_group_ranges(a_fairness_metric, sensitive_dataset_and_model):
    X, y, model, sensitive_attr_names = sensitive_dataset_and_model

    group_small_range = np.array([[0.4, 0.6], [0.6, 0.4]])
    group_big_range = np.array([[0.05, 0.95], [0.95, 0.05]])

    probas = np.vstack((group_small_range, group_big_range))
    groups = ["small"] * len(group_small_range) + ["big"] * len(group_big_range)

    groups = pd.DataFrame(groups, columns=["group_val"])

    unique_groups = groups["group_val"].unique()
    unique_group_names = groups["group_val"].unique().tolist()

    resp_model = ModelBiasMitigator(
        model,
        sensitive_attr_names,
        fairness_metric=a_fairness_metric,
        accuracy_metric=AN_ACCURACY_METRIC,
        random_seed=RANDOM_SEED,
    )
    resp_model._unique_groups_ = unique_groups
    resp_model._unique_group_names_ = unique_group_names

    group_ranges = resp_model._get_group_ranges(probas, groups, 10)

    small_ratio = 0.6 / (0.4 + 1e-6)
    expected_small = (1 / small_ratio, small_ratio)

    expected_big = (0.1, 10.0)

    for received, expected in zip(group_ranges["small"], expected_small):
        assert is_close(received, expected)
    for received, expected in zip(group_ranges["big"], expected_big):
        assert is_close(received, expected)


@pytest.mark.parametrize("a_fairness_metric", A_FAIRNESS_METRIC)
def test_accepted_inputs(a_fairness_metric, sensitive_dataset_and_model):
    X, y, model, sensitive_attr_names = sensitive_dataset_and_model

    ### Bool or 'auto' attributes
    # Sanity checks
    ModelBiasMitigator(
        model,
        sensitive_attr_names,
        fairness_metric=a_fairness_metric,
        accuracy_metric=AN_ACCURACY_METRIC,
        higher_accuracy_is_better="auto",
        higher_fairness_is_better="auto",
        fairness_metric_uses_probas="auto",
        accuracy_metric_uses_probas="auto",
    )

    def test_bool_auto_attr(attr_name):
        # Only 'auto' supported str
        with pytest.raises(GuardianAIValueError):
            ModelBiasMitigator(
                model,
                sensitive_attr_names,
                fairness_metric=a_fairness_metric,
                accuracy_metric=AN_ACCURACY_METRIC,
                **{attr_name: "any_other_str"},
            )

        # No support for other input types
        with pytest.raises(GuardianAIValueError):
            ModelBiasMitigator(
                model,
                sensitive_attr_names,
                fairness_metric=a_fairness_metric,
                accuracy_metric=AN_ACCURACY_METRIC,
                **{attr_name: 4},
            )

    for attr in [
        "higher_accuracy_is_better",
        "higher_fairness_is_better",
        "fairness_metric_uses_probas",
        "accuracy_metric_uses_probas",
    ]:
        test_bool_auto_attr(attr)

    ### Bool attributes
    # Sanity checks
    ModelBiasMitigator(
        model,
        sensitive_attr_names,
        fairness_metric=a_fairness_metric,
        accuracy_metric=AN_ACCURACY_METRIC,
        base_estimator_uses_protected_attributes=True,
    )

    # No support for other input types
    with pytest.raises(GuardianAITypeError):
        ModelBiasMitigator(
            model,
            sensitive_attr_names,
            fairness_metric=a_fairness_metric,
            accuracy_metric=AN_ACCURACY_METRIC,
            base_estimator_uses_protected_attributes="other_type",
        )


def test_select_model(responsible_model_and_metrics):
    (
        X,
        y,
        sensitive_attr_names,
        resp_model,
        fairness_metric,
        accuracy_metric,
    ) = responsible_model_and_metrics
    (
        fairness_name,
        fairness_callable,
        fairness_hib,
        fairness_uses_probas,
    ) = fairness_metric
    (
        accuracy_name,
        accuracy_callable,
        accuracy_hib,
        accuracy_uses_probas,
    ) = accuracy_metric

    # Don't accept negative values
    with pytest.raises(GuardianAIValueError):
        resp_model.select_model(-1)

    # Only accept up to len(_best_trials_detailed) - 1
    with pytest.raises(GuardianAIValueError):
        resp_model.select_model(len(resp_model._best_trials_detailed))

    for idx in range(
        min(len(resp_model._best_trials_detailed), 3)
    ):  # test at max 2 other idxs
        row = resp_model._best_trials_detailed.iloc[idx]
        expected_fairness_score = row[fairness_name]
        expected_accuracy_score = row[accuracy_name]
        expected_multipliers = row[resp_model._multiplier_names_]

        resp_model.select_model(idx)

        assert resp_model.selected_multipliers_idx_ == idx
        assert (resp_model.selected_multipliers_ == expected_multipliers).all()

        y_pred = resp_model.predict(X)
        y_proba = resp_model.predict_proba(X)[:, 1]
        subgroups = X[sensitive_attr_names]

        if fairness_uses_probas:
            actual_fairness = fairness_callable(y, y_proba, subgroups)
        else:
            actual_fairness = fairness_callable(y, y_pred, subgroups)

        if accuracy_uses_probas:
            actual_accuracy = accuracy_callable(y, y_proba)
        else:
            actual_accuracy = accuracy_callable(y, y_pred)

        assert is_close(expected_accuracy_score, actual_accuracy)
        assert is_close(expected_fairness_score, actual_fairness)


def test_pickle(responsible_model_and_metrics):
    fname = tempfile.NamedTemporaryFile().name
    try:
        X, _, _, resp_model, _, _ = responsible_model_and_metrics

        with open(fname, "wb") as f:
            pickle.dump(resp_model, f)

        with open(fname, "rb") as f:
            unpickled_resp_model = pickle.load(f)

        # Assert that reloaded model predicts same thing as saved model
        assert (
            resp_model.predict_proba(X) == unpickled_resp_model.predict_proba(X)
        ).all()
    finally:
        if os.path.isfile(fname):
            os.remove(fname)


def test_subsampling():
    dataset, target = get_dummy_dataset(n_samples=1000, n_features=5, n_classes=2)
    target = target.astype("bool")
    model = Pipeline(
        steps=[
            ("preprocessor", OneHotEncoder(handle_unknown="ignore")),
            ("classifier", LogisticRegression(random_state=0)),
        ]
    )
    model.fit(dataset, target)

    resp_model = ModelBiasMitigator(
        model,
        ["1"],
        fairness_metric="statistical_parity",
        accuracy_metric="accuracy",
        n_trials_per_group=5,
        random_seed=RANDOM_SEED,
        subsampling=50,
    )  # limit number of trials for faster tests
    resp_model.fit(dataset, target)
