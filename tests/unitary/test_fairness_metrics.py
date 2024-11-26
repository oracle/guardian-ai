#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/
import math

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from guardian_ai.fairness.metrics.dataset import (
    ConsistencyScorer,
    DatasetStatisticalParityScorer,
    SmoothedEDFScorer,
    consistency,
    dataset_statistical_parity,
    smoothed_edf,
)
from guardian_ai.fairness.metrics.model import (
    EqualizedOddsScorer,
    ErrorRateScorer,
    FalseDiscoveryRateScorer,
    FalseNegativeRateScorer,
    FalseOmissionRateScorer,
    FalsePositiveRateScorer,
    ModelStatisticalParityScorer,
    TheilIndexScorer,
    TruePositiveRateScorer,
    equalized_odds,
    error_rate,
    false_discovery_rate,
    false_negative_rate,
    false_omission_rate,
    false_positive_rate,
    model_statistical_parity,
    theil_index,
    true_positive_rate,
)
from guardian_ai.utils.exception import GuardianAITypeError, GuardianAIValueError
from tests.utils import get_dummy_dataset


@pytest.fixture(scope="module", autouse=True)
def init():
    np.random.seed(12345)


def is_close(a, b):
    return math.isclose(a, b, rel_tol=1e-5)


def approx_dict(d):
    return pytest.approx(d, rel=1e-5)


MODEL_X_Y_SCORERS = {
    "model_statistical_parity_scorer": ModelStatisticalParityScorer,
    "true_positive_rate_scorer": TruePositiveRateScorer,
    "false_positive_rate_scorer": FalsePositiveRateScorer,
    "false_negative_rate_scorer": FalseNegativeRateScorer,
    "false_omission_rate_scorer": FalseOmissionRateScorer,
    "false_discovery_rate_scorer": FalseDiscoveryRateScorer,
    "error_rate_scorer": ErrorRateScorer,
    "equalized_odds_scorer": EqualizedOddsScorer,
    "theil_index_scorer": TheilIndexScorer,
}

MODEL_SUBGROUPS_SCORERS = {
    "model_statistical_parity_scorer": model_statistical_parity,
    "true_positive_rate_scorer": true_positive_rate,
    "false_positive_rate_scorer": false_positive_rate,
    "false_negative_rate_scorer": false_negative_rate,
    "false_omission_rate_scorer": false_omission_rate,
    "false_discovery_rate_scorer": false_discovery_rate,
    "error_rate_scorer": error_rate,
    "equalized_odds_scorer": equalized_odds,
    "theil_index_scorer": theil_index,
}

MODEL_SCORERS_ALLOWING_REDUCTION = list(MODEL_X_Y_SCORERS.keys())
MODEL_SCORERS_USING_DISTANCE = [
    scorer for scorer in MODEL_X_Y_SCORERS if scorer != "theil_index_scorer"
]
MODEL_SCORERS_ALLOWING_Y_TRUE_NONE = ["model_statistical_parity_scorer"]

DATASET_X_Y_SCORERS = {
    "dataset_statistical_parity_scorer": DatasetStatisticalParityScorer,
    "consistency_scorer": ConsistencyScorer,
    "smoothed_edf_scorer": SmoothedEDFScorer,
}

DATASET_SUBGROUPS_SCORERS = {
    "dataset_statistical_parity_scorer": dataset_statistical_parity,
    "consistency_scorer": consistency,
    "smoothed_edf_scorer": smoothed_edf,
}

DATASET_SCORERS_ALLOWING_REDUCTION = ["dataset_statistical_parity_scorer"]
DATASET_SCORERS_USING_DISTANCE = ["dataset_statistical_parity_scorer"]

ALL_X_Y_SCORERS = {**MODEL_X_Y_SCORERS, **DATASET_X_Y_SCORERS}

SENSITIVE_FEATURES_VARIATIONS = {
    "one_attr_two_classes": {"n_classes": (2,)},
    "one_attr_n_classes": {"n_classes": (4,)},
    "n_attrs": {"n_classes": (3, 4)},
}


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
            ("classifier", RandomForestClassifier()),
        ]
    )
    model.fit(dataset, target)

    return dataset, target, model, sensitive_attr_names


@pytest.mark.parametrize("scorer", DATASET_X_Y_SCORERS.keys())
def test_dataset_X_y_scorer_signature(sensitive_dataset_and_model, scorer):
    dataset, target, model, sensitive_attr_names = sensitive_dataset_and_model
    scorer = DATASET_X_Y_SCORERS[scorer](sensitive_attr_names)

    # Validate call signatures
    assert isinstance(scorer(X=dataset, y_true=target), float)
    assert isinstance(scorer(None, dataset, target), float)
    with pytest.raises(GuardianAIValueError):
        scorer(dataset, target)

    # Two ways to call metric are equivalent
    assert is_close(scorer(X=dataset, y_true=target), scorer(None, dataset, target))


@pytest.mark.parametrize("scorer", DATASET_SUBGROUPS_SCORERS.keys())
def test_dataset_subgroups_scorer_signature(sensitive_dataset_and_model, scorer):
    dataset, target, model, sensitive_attr_names = sensitive_dataset_and_model
    scorer = DATASET_SUBGROUPS_SCORERS[scorer]
    subgroups = dataset[sensitive_attr_names]

    # Validate call signatures
    assert isinstance(scorer(target, subgroups), float)


@pytest.mark.parametrize("scorer", DATASET_SUBGROUPS_SCORERS.keys())
def test_dataset_scorers_equivalence(sensitive_dataset_and_model, scorer):
    dataset, target, model, sensitive_attr_names = sensitive_dataset_and_model
    X_y_scorer = DATASET_X_Y_SCORERS[scorer](sensitive_attr_names)
    subgroup_scorer = DATASET_SUBGROUPS_SCORERS[scorer]
    subgroups = dataset[sensitive_attr_names]

    # Validate same value
    assert is_close(
        subgroup_scorer(target, subgroups), X_y_scorer(X=dataset, y_true=target)
    )


@pytest.mark.parametrize("scorer", MODEL_X_Y_SCORERS.keys())
def test_model_X_y_scorer_signature(sensitive_dataset_and_model, scorer):
    dataset, target, model, sensitive_attr_names = sensitive_dataset_and_model
    scorer = MODEL_X_Y_SCORERS[scorer](sensitive_attr_names)

    # Validate call signature
    assert isinstance(scorer(model, dataset, target), float)


# Utility that can ignore some input columns. Useful for testing when a
# column is moved from X to supp_features and wanting identical model
# predictions
class ModelIgnoreOtherFeatures(sklearn.base.BaseEstimator):
    def __init__(self, model, features_to_keep):
        self.model = model
        self.features_to_keep = features_to_keep

    def _trim_X(self, X):
        features_to_drop = [
            col for col in X.columns if col not in self.features_to_keep
        ]
        return X.drop(columns=features_to_drop)

    def predict(self, X):
        return self.model.predict(self._trim_X(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._trim_X(X))


@pytest.mark.parametrize("scorer", ALL_X_Y_SCORERS.keys())
def test_X_y_scorer_supp_features(base_dataset, model_type, scorer):
    # Need to create our own dataset and model to test variations of supp_features
    dataset_no_sf, target = base_dataset
    dataset, sensitive_attr_names = create_concat_sensitive_attrs(
        dataset_no_sf, n_classes=(3, 4)
    )

    if scorer in MODEL_X_Y_SCORERS:
        scorer = MODEL_X_Y_SCORERS[scorer](sensitive_attr_names)
        model = Pipeline(
            steps=[
                ("preprocessor", OneHotEncoder(handle_unknown="ignore")),
                ("classifier", RandomForestClassifier()),
            ]
        )
        model.fit(dataset_no_sf, target)
        model = ModelIgnoreOtherFeatures(model, dataset_no_sf.columns)
    else:
        scorer = DATASET_X_Y_SCORERS[scorer](sensitive_attr_names)
        model = None

    correct_score = scorer(model, dataset, target)

    # All sensitive features are in X (default)
    assert is_close(scorer(model, dataset, target), correct_score)

    # All sensitive features are in supplementary_features
    dataset_no_sf = dataset.drop(columns=sensitive_attr_names)
    all_supp_features = dataset[sensitive_attr_names]

    assert is_close(
        scorer(model, dataset_no_sf, target, supplementary_features=all_supp_features),
        correct_score,
    )

    # Features are split across X and supplementary_features
    some_sf = [sensitive_attr_names[0]]
    dataset_some_sf = dataset.drop(columns=some_sf)
    supp_features = dataset[some_sf]

    assert is_close(
        scorer(model, dataset_some_sf, target, supplementary_features=supp_features),
        correct_score,
    )

    # supplementary_features invalid type
    with pytest.raises(GuardianAIValueError):
        scorer(model, dataset, target, supplementary_features=[])

    # Duplicate features between X and supplementary_features
    with pytest.raises(GuardianAIValueError):
        scorer(model, dataset, target, supplementary_features=all_supp_features)


@pytest.mark.parametrize("scorer", MODEL_SUBGROUPS_SCORERS.keys())
def test_model_subgroups_scorer_signature(sensitive_dataset_and_model, scorer):
    dataset, target, model, sensitive_attr_names = sensitive_dataset_and_model
    scorer = MODEL_SUBGROUPS_SCORERS[scorer]
    subgroups = dataset[sensitive_attr_names]
    y_pred = model.predict(dataset)

    # Validate call signatures
    assert isinstance(scorer(target, y_pred, subgroups), float)


@pytest.mark.parametrize("scorer", MODEL_SUBGROUPS_SCORERS.keys())
def test_model_scorers_equivalence(sensitive_dataset_and_model, scorer):
    dataset, target, model, sensitive_attr_names = sensitive_dataset_and_model
    X_y_scorer = MODEL_X_Y_SCORERS[scorer](sensitive_attr_names)
    subgroup_scorer = MODEL_SUBGROUPS_SCORERS[scorer]
    subgroups = dataset[sensitive_attr_names]
    y_pred = model.predict(dataset)

    # Validate same value
    assert is_close(
        subgroup_scorer(target, y_pred, subgroups), X_y_scorer(model, dataset, target)
    )


@pytest.mark.parametrize("scorer", ALL_X_Y_SCORERS.keys())
def test_X_y_scorer_sensitive_attr_formats(base_dataset, scorer):
    dataset, target = base_dataset
    sensitive_attrs_name = "sensitive_attr"

    if scorer in MODEL_X_Y_SCORERS:
        scorer = MODEL_X_Y_SCORERS[scorer]([sensitive_attrs_name])
        model = DummyBinaryStochasticModel()
    else:
        scorer = DATASET_X_Y_SCORERS[scorer]([sensitive_attrs_name])
        model = None

    # Accept str vals
    vals = [f"val_{i}" for i in range(5)]
    sensitive_dataset = concat_sensitive_attr_column(
        vals, dataset, sensitive_attrs_name
    )
    assert isinstance(scorer(model, sensitive_dataset, target), float)

    # Accept categorical vals
    vals = list(range(5))
    sensitive_vals = np.random.choice(vals, size=len(dataset))
    sensitive_feats = pd.Series(
        np.transpose(sensitive_vals), dtype="category", name=sensitive_attrs_name
    )
    sensitive_dataset = pd.concat([dataset, sensitive_feats], axis=1)
    assert isinstance(scorer(model, sensitive_dataset, target), float)

    # Accept bool vals
    vals = [True, False]
    sensitive_dataset = concat_sensitive_attr_column(
        vals, dataset, sensitive_attrs_name
    )
    assert isinstance(scorer(model, sensitive_dataset, target), float)

    # Reject (non-categoricalized) integer vals
    vals = list(range(5))
    sensitive_dataset = concat_sensitive_attr_column(
        vals, dataset, sensitive_attrs_name
    )
    with pytest.raises(GuardianAIValueError):
        scorer(model, sensitive_dataset, target)

    # Reject float vals
    vals = np.random.rand(5)
    sensitive_dataset = concat_sensitive_attr_column(
        vals, dataset, sensitive_attrs_name
    )
    with pytest.raises(GuardianAIValueError):
        scorer(model, sensitive_dataset, target)


@pytest.mark.parametrize("scorer", MODEL_X_Y_SCORERS.keys())
def test_model_metrics_y_true_None(sensitive_dataset_and_model, scorer):
    dataset, target, model, sensitive_attr_names = sensitive_dataset_and_model
    X_y_scorer = MODEL_X_Y_SCORERS[scorer](sensitive_attr_names)
    subgroup_scorer = MODEL_SUBGROUPS_SCORERS[scorer]
    subgroups = dataset[sensitive_attr_names]
    y_pred = model.predict(dataset)

    if scorer in MODEL_SCORERS_ALLOWING_Y_TRUE_NONE:
        # Can pass y_true=None
        assert isinstance(X_y_scorer(model, dataset, None), float)
        assert isinstance(subgroup_scorer(None, y_pred, subgroups), float)

        # Cannot pass only two arguments
        with pytest.raises(GuardianAIValueError):
            subgroup_scorer(y_pred, subgroups)
    else:
        with pytest.raises(GuardianAIValueError):
            X_y_scorer(model, dataset, None)

        with pytest.raises(GuardianAIValueError):
            subgroup_scorer(None, y_pred, subgroups)


@pytest.mark.parametrize("scorer", ALL_X_Y_SCORERS.keys())
def test_X_y_scorer_feature_not_in_dataset(sensitive_dataset_and_model, scorer):
    dataset, target, model, sensitive_attr_names = sensitive_dataset_and_model

    if scorer in MODEL_X_Y_SCORERS:
        scorer_maker = MODEL_X_Y_SCORERS[scorer]
    else:
        scorer_maker = DATASET_X_Y_SCORERS[scorer]
        model = None

    # Correct output if features present
    scorer_obj = scorer_maker(sensitive_attr_names)
    assert isinstance(scorer_obj(model, dataset, target), float)

    # Error if one missing feature
    one_missing_feature = sensitive_attr_names + ["missing_feature"]
    scorer_obj = scorer_maker(one_missing_feature)
    with pytest.raises(GuardianAIValueError):
        scorer_obj(model, dataset, target)

    # Error if all missing features
    all_missing_features = [f"missing_feature_{i}" for i in range(3)]
    scorer_obj = scorer_maker(all_missing_features)
    with pytest.raises(GuardianAIValueError):
        scorer_obj(model, dataset, target)


@pytest.mark.parametrize("scorer", MODEL_SCORERS_ALLOWING_REDUCTION)
def test_model_scorer_reduction(sensitive_dataset_and_model, scorer):
    dataset, target, model, sensitive_attr_names = sensitive_dataset_and_model
    X_y_scorer_fn = MODEL_X_Y_SCORERS[scorer]
    subgroups_scorer = MODEL_SUBGROUPS_SCORERS[scorer]
    y_pred = model.predict(dataset)
    subgroups = dataset[sensitive_attr_names]

    # Mean reduction
    X_y_scorer = X_y_scorer_fn(sensitive_attr_names, reduction="mean")
    assert isinstance(X_y_scorer(model, dataset, target), float)
    assert isinstance(
        subgroups_scorer(target, y_pred, subgroups, reduction="mean"), float
    )
    assert is_close(
        X_y_scorer(model, dataset, target),
        subgroups_scorer(target, y_pred, subgroups, reduction="mean"),
    )

    # Max reduction
    X_y_scorer = X_y_scorer_fn(sensitive_attr_names, reduction="max")
    assert isinstance(X_y_scorer(model, dataset, target), float)
    assert isinstance(
        subgroups_scorer(target, y_pred, subgroups, reduction="max"), float
    )
    assert is_close(
        X_y_scorer(model, dataset, target),
        subgroups_scorer(target, y_pred, subgroups, reduction="max"),
    )

    # None reduction
    X_y_scorer = X_y_scorer_fn(sensitive_attr_names, reduction=None)
    X_y_result = X_y_scorer(model, dataset, target)
    assert isinstance(X_y_result, dict)

    subgroups_result = subgroups_scorer(target, y_pred, subgroups, reduction=None)
    assert isinstance(subgroups_result, dict)
    assert X_y_result == approx_dict(subgroups_result)

    # Other value
    with pytest.raises(GuardianAIValueError):
        X_y_scorer = X_y_scorer_fn(sensitive_attr_names, reduction="other")
    with pytest.raises(GuardianAIValueError):
        subgroups_scorer(target, y_pred, subgroups, reduction="other")


@pytest.mark.parametrize("scorer", DATASET_SCORERS_ALLOWING_REDUCTION)
def test_dataset_scorer_reduction(sensitive_dataset_and_model, scorer):
    dataset, target, model, sensitive_attr_names = sensitive_dataset_and_model
    X_y_scorer_fn = DATASET_X_Y_SCORERS[scorer]
    subgroups_scorer = DATASET_SUBGROUPS_SCORERS[scorer]
    subgroups = dataset[sensitive_attr_names]

    # Mean reduction
    X_y_scorer = X_y_scorer_fn(sensitive_attr_names, reduction="mean")
    assert isinstance(X_y_scorer(None, dataset, target), float)
    assert isinstance(subgroups_scorer(target, subgroups, reduction="mean"), float)
    assert is_close(
        X_y_scorer(None, dataset, target),
        subgroups_scorer(target, subgroups, reduction="mean"),
    )

    # Max reduction
    X_y_scorer = X_y_scorer_fn(sensitive_attr_names, reduction="max")
    assert isinstance(X_y_scorer(None, dataset, target), float)
    assert isinstance(subgroups_scorer(target, subgroups, reduction="max"), float)
    assert is_close(
        X_y_scorer(None, dataset, target),
        subgroups_scorer(target, subgroups, reduction="max"),
    )

    # None reduction
    X_y_scorer = X_y_scorer_fn(sensitive_attr_names, reduction=None)
    X_y_result = X_y_scorer(None, dataset, target)
    assert isinstance(X_y_result, dict)

    subgroups_result = subgroups_scorer(target, subgroups, reduction=None)
    assert isinstance(subgroups_result, dict)
    assert X_y_result == approx_dict(subgroups_result)

    # Other value
    with pytest.raises(GuardianAIValueError):
        X_y_scorer = X_y_scorer_fn(sensitive_attr_names, reduction="other")
    with pytest.raises(GuardianAIValueError):
        subgroups_scorer(target, subgroups, reduction="other")


@pytest.mark.parametrize("scorer", MODEL_X_Y_SCORERS.keys())
def test_model_scorer_distance(sensitive_dataset_and_model, scorer):
    dataset, target, model, sensitive_attr_names = sensitive_dataset_and_model
    X_y_scorer_fn = MODEL_X_Y_SCORERS[scorer]
    subgroup_scorer = MODEL_SUBGROUPS_SCORERS[scorer]
    subgroups = dataset[sensitive_attr_names]
    y_pred = model.predict(dataset)

    if scorer in MODEL_SCORERS_USING_DISTANCE:
        # Validate for ratio distance
        X_y_scorer = X_y_scorer_fn(sensitive_attr_names, distance_measure="ratio")
        assert isinstance(X_y_scorer(model, dataset, target), float)
        assert isinstance(
            subgroup_scorer(target, y_pred, subgroups, distance_measure="ratio"), float
        )
        assert is_close(
            X_y_scorer(model, dataset, target),
            subgroup_scorer(target, y_pred, subgroups, distance_measure="ratio"),
        )

        # Validate for diff distance
        X_y_scorer = X_y_scorer_fn(sensitive_attr_names, distance_measure="diff")
        assert isinstance(X_y_scorer(model, dataset, target), float)
        assert isinstance(
            subgroup_scorer(target, y_pred, subgroups, distance_measure="diff"), float
        )
        assert is_close(
            X_y_scorer(model, dataset, target),
            subgroup_scorer(target, y_pred, subgroups, distance_measure="diff"),
        )

        # Do not accept other distances
        with pytest.raises(GuardianAIValueError):
            X_y_scorer = X_y_scorer_fn(
                sensitive_attr_names, distance_measure="something"
            )

        with pytest.raises(GuardianAIValueError):
            subgroup_scorer(target, y_pred, subgroups, distance_measure="something")

        # Do not accept None distances
        with pytest.raises(GuardianAIValueError):
            X_y_scorer = X_y_scorer_fn(sensitive_attr_names, distance_measure=None)

        with pytest.raises(GuardianAIValueError):
            subgroup_scorer(target, y_pred, subgroups, distance_measure=None)
    else:
        # Accepts only None as distance_measure
        X_y_scorer = X_y_scorer_fn(sensitive_attr_names, distance_measure=None)
        assert isinstance(X_y_scorer(model, dataset, target), float)
        assert isinstance(
            subgroup_scorer(target, y_pred, subgroups, distance_measure=None), float
        )

        with pytest.raises(GuardianAIValueError):
            X_y_scorer = X_y_scorer_fn(
                sensitive_attr_names, distance_measure="something"
            )

        with pytest.raises(GuardianAIValueError):
            subgroup_scorer(target, y_pred, subgroups, distance_measure="something")


@pytest.mark.parametrize("scorer", DATASET_SCORERS_USING_DISTANCE)
def test_dataset_scorer_distance(sensitive_dataset_and_model, scorer):
    dataset, target, model, sensitive_attr_names = sensitive_dataset_and_model
    X_y_scorer_fn = DATASET_X_Y_SCORERS[scorer]
    subgroup_scorer = DATASET_SUBGROUPS_SCORERS[scorer]
    subgroups = dataset[sensitive_attr_names]

    # Validate for ratio distance
    X_y_scorer = X_y_scorer_fn(sensitive_attr_names, distance_measure="ratio")
    assert isinstance(X_y_scorer(None, dataset, target), float)
    assert isinstance(
        subgroup_scorer(target, subgroups, distance_measure="ratio"), float
    )
    assert is_close(
        X_y_scorer(None, dataset, target),
        subgroup_scorer(target, subgroups, distance_measure="ratio"),
    )

    # Validate for diff distance
    X_y_scorer = X_y_scorer_fn(sensitive_attr_names, distance_measure="diff")
    assert isinstance(X_y_scorer(None, dataset, target), float)
    assert isinstance(
        subgroup_scorer(target, subgroups, distance_measure="diff"), float
    )
    assert is_close(
        X_y_scorer(None, dataset, target),
        subgroup_scorer(target, subgroups, distance_measure="diff"),
    )

    # Do not accept other distances
    with pytest.raises(GuardianAIValueError):
        X_y_scorer = X_y_scorer_fn(sensitive_attr_names, distance_measure="something")

    with pytest.raises(GuardianAIValueError):
        subgroup_scorer(target, subgroups, distance_measure="something")


@pytest.mark.parametrize("scorer", MODEL_SUBGROUPS_SCORERS)
def test_model_scorers_y_format(sensitive_dataset_and_model, scorer):
    dataset, target, model, sensitive_attr_names = sensitive_dataset_and_model
    run_y_true_tests = scorer not in MODEL_SCORERS_ALLOWING_Y_TRUE_NONE
    scorer = MODEL_SUBGROUPS_SCORERS[scorer]
    subgroups = dataset[sensitive_attr_names]

    # Accept same number of instances
    y_pred = target.copy()
    assert isinstance(scorer(target, y_pred, subgroups), float)

    # Do not accept different number of instances
    if run_y_true_tests:
        y_pred = target[:-1].copy()
        with pytest.raises(GuardianAIValueError):
            scorer(target, y_pred, subgroups)

        # Do not accept multiclass classification
        multiclass_y = target.copy()
        multiclass_y[:3] = 2  # Change a few labels
        assert multiclass_y.nunique() == 3  # Sanity check

        with pytest.raises(GuardianAIValueError):
            scorer(multiclass_y, y_pred, subgroups)

    # Do not accept non-array inputs
    if run_y_true_tests:
        y_pred = target.copy()
        bad_target = {i: itm for i, itm in enumerate(target)}
        with pytest.raises(GuardianAITypeError):
            scorer(bad_target, y_pred, subgroups)

    y_pred = {i: itm for i, itm in enumerate(target)}
    with pytest.raises(GuardianAITypeError):
        scorer(target, y_pred, subgroups)


@pytest.fixture(scope="module")
def calibration_dataset_two_classes():
    data = np.array(
        [
            ["male", 0, 0],
            ["male", 0, 1],
            ["male", 1, 0],
            ["male", 1, 1],
            ["male", 1, 0],
            ["male", 1, 1],
            ["female", 0, 1],
            ["female", 0, 1],
            ["female", 1, 0],
            ["female", 1, 1],
        ]
    )

    dataset = pd.DataFrame(data[:, 0], columns=["sex"])
    target = pd.Series(data[:, 1], name="label")

    predictions = data[:, 2]

    return dataset, target, predictions


model_scorers_two_classes_calibration_exps = {
    "model_statistical_parity_scorer": {  # male=0.5, female=0.75
        (("distance_measure", "diff"), ("reduction", None)): {("female", "male"): 0.25},
        (("distance_measure", "diff"), ("reduction", "max")): 0.25,
        (("distance_measure", "diff"), ("reduction", "mean")): 0.25,
        (("distance_measure", "ratio"), ("reduction", None)): {("female", "male"): 1.5},
        (("distance_measure", "ratio"), ("reduction", "max")): 1.5,
        (("distance_measure", "ratio"), ("reduction", "mean")): 1.5,
    },
    "true_positive_rate_scorer": {  # male=0.5, female=0.5
        (("distance_measure", "diff"), ("reduction", None)): {("female", "male"): 0.0},
        (("distance_measure", "diff"), ("reduction", "max")): 0.0,
        (("distance_measure", "diff"), ("reduction", "mean")): 0.0,
        (("distance_measure", "ratio"), ("reduction", None)): {("female", "male"): 1.0},
        (("distance_measure", "ratio"), ("reduction", "max")): 1.0,
        (("distance_measure", "ratio"), ("reduction", "mean")): 1.0,
    },
    "false_positive_rate_scorer": {  # male=0.5, female=1.0
        (("distance_measure", "diff"), ("reduction", None)): {("female", "male"): 0.5},
        (("distance_measure", "diff"), ("reduction", "max")): 0.5,
        (("distance_measure", "diff"), ("reduction", "mean")): 0.5,
        (("distance_measure", "ratio"), ("reduction", None)): {("female", "male"): 2.0},
        (("distance_measure", "ratio"), ("reduction", "max")): 2.0,
        (("distance_measure", "ratio"), ("reduction", "mean")): 2.0,
    },
    "false_negative_rate_scorer": {  # male=0.5, female=0.5
        (("distance_measure", "diff"), ("reduction", None)): {("female", "male"): 0.0},
        (("distance_measure", "diff"), ("reduction", "max")): 0.0,
        (("distance_measure", "diff"), ("reduction", "mean")): 0.0,
        (("distance_measure", "ratio"), ("reduction", None)): {("female", "male"): 1.0},
        (("distance_measure", "ratio"), ("reduction", "max")): 1.0,
        (("distance_measure", "ratio"), ("reduction", "mean")): 1.0,
    },
    "false_omission_rate_scorer": {  # male=0.67, female=1.0
        (("distance_measure", "diff"), ("reduction", None)): {
            ("female", "male"): 1 / 3
        },
        (("distance_measure", "diff"), ("reduction", "max")): 1 / 3,
        (("distance_measure", "diff"), ("reduction", "mean")): 1 / 3,
        (("distance_measure", "ratio"), ("reduction", None)): {("female", "male"): 1.5},
        (("distance_measure", "ratio"), ("reduction", "max")): 1.5,
        (("distance_measure", "ratio"), ("reduction", "mean")): 1.5,
    },
    "false_discovery_rate_scorer": {  # male=0.33, female=0.67
        (("distance_measure", "diff"), ("reduction", None)): {
            ("female", "male"): 1 / 3
        },
        (("distance_measure", "diff"), ("reduction", "max")): 1 / 3,
        (("distance_measure", "diff"), ("reduction", "mean")): 1 / 3,
        (("distance_measure", "ratio"), ("reduction", None)): {("female", "male"): 2.0},
        (("distance_measure", "ratio"), ("reduction", "max")): 2.0,
        (("distance_measure", "ratio"), ("reduction", "mean")): 2.0,
    },
    "error_rate_scorer": {  # male=0.5, female=0.75
        (("distance_measure", "diff"), ("reduction", None)): {("female", "male"): 0.25},
        (("distance_measure", "diff"), ("reduction", "max")): 0.25,
        (("distance_measure", "diff"), ("reduction", "mean")): 0.25,
        (("distance_measure", "ratio"), ("reduction", None)): {("female", "male"): 1.5},
        (("distance_measure", "ratio"), ("reduction", "max")): 1.5,
        (("distance_measure", "ratio"), ("reduction", "mean")): 1.5,
    },
}


@pytest.mark.parametrize("scorer", model_scorers_two_classes_calibration_exps.keys())
def test_calibration_model_scorers_two_classes(calibration_dataset_two_classes, scorer):
    dataset, target, y_pred = calibration_dataset_two_classes
    calibration_exps = model_scorers_two_classes_calibration_exps[scorer]
    scorer = MODEL_SUBGROUPS_SCORERS[scorer]
    subgroups = dataset[["sex"]]

    for params, expected_result in calibration_exps.items():
        kwargs = {key: val for key, val in params}
        actual_result = scorer(target, y_pred, subgroups, **kwargs)
        if isinstance(expected_result, float):
            assert is_close(actual_result, expected_result)
        else:  # dict
            assert actual_result == approx_dict(expected_result)


dataset_scorers_two_classes_calibration_exps = {
    "dataset_statistical_parity_scorer": {  # male=0.66, female=0.5
        (("distance_measure", "diff"), ("reduction", None)): {
            ("female", "male"): 1 / 6
        },
        (("distance_measure", "diff"), ("reduction", "max")): 1 / 6,
        (("distance_measure", "diff"), ("reduction", "mean")): 1 / 6,
        (("distance_measure", "ratio"), ("reduction", None)): {
            ("female", "male"): 4 / 3
        },
        (("distance_measure", "ratio"), ("reduction", "max")): 4 / 3,
        (("distance_measure", "ratio"), ("reduction", "mean")): 4 / 3,
    },
    # The other two dataset metrics are computed using intermediate objects
    # like a kNN regression so testing is likely to fail too often to implement
}


@pytest.mark.parametrize("scorer", dataset_scorers_two_classes_calibration_exps.keys())
def test_calibration_dataset_scorers_two_classes(
    calibration_dataset_two_classes, scorer
):
    dataset, target, y_pred = calibration_dataset_two_classes
    calibration_exps = dataset_scorers_two_classes_calibration_exps[scorer]
    scorer = DATASET_SUBGROUPS_SCORERS[scorer]
    subgroups = dataset[["sex"]]

    for params, expected_result in calibration_exps.items():
        kwargs = {key: val for key, val in params}
        actual_result = scorer(target, subgroups, **kwargs)
        if isinstance(expected_result, float):
            assert is_close(actual_result, expected_result)
        else:  # dict
            assert actual_result == approx_dict(expected_result)


@pytest.mark.parametrize("scorer", ALL_X_Y_SCORERS.keys())
def test_scorers_have_display_name(scorer):
    # Test 1 protected attribute
    scorer_obj = ALL_X_Y_SCORERS[scorer]("any_protected_attribute")

    display_name = scorer_obj.display_name

    assert isinstance(display_name, str)

    # Test more protected attributes
    scorer_obj = ALL_X_Y_SCORERS[scorer](["prot_attr_1", "prot_attr_2"])

    display_name = scorer_obj.display_name

    assert isinstance(display_name, str)


def test_scorers_correct_display_name():
    # Single prot attr w/ reduction and distance measure
    scorer = ModelStatisticalParityScorer(
        "race", distance_measure="diff", reduction="mean"
    )

    assert scorer.display_name == "Mean Statistical Parity Difference for 'race'"

    # Two prot attrs w/ other reduction and distance measure
    scorer = TruePositiveRateScorer(
        ["race", "sex"], distance_measure="ratio", reduction="max"
    )

    assert (
        scorer.display_name == "Maximum True Positive Rate Ratio for 'race' and 'sex'"
    )

    # Metric w/o reduction or distance measure and 3 prot attrs
    scorer = TheilIndexScorer(["race", "sex", "age"], reduction=None)

    assert scorer.display_name == "Raw Theil Index for 'race', 'sex', and 'age'"


def test_new_fairness_metrics_aggregation():
    df = pd.DataFrame(
        {
            "group": ["a", "a", "b", "b", "b", "c", "c", "c"],
            "label": [0, 1, 0, 0, 1, 0, 1, 1],
        }
    )
    bias = dataset_statistical_parity(
        df["label"], df[["group"]], distance_measure="diff", reduction="max"
    )
    assert abs(bias - 0.3333333333333333) < 1e-6
