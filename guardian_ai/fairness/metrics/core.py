#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Core for fairness metrics"""

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
from guardian_ai.utils.exception import GuardianAIValueError

fairness_scorers_dict = {  # noqa N816
    "statistical_parity": ModelStatisticalParityScorer,
    "TPR": TruePositiveRateScorer,
    "FPR": FalsePositiveRateScorer,
    "FNR": FalseNegativeRateScorer,
    "FOR": FalseOmissionRateScorer,
    "FDR": FalseDiscoveryRateScorer,
    "error_rate": ErrorRateScorer,
    "equalized_odds": EqualizedOddsScorer,
    "theil_index": TheilIndexScorer,
}


def _get_fairness_scorer(metric, protected_attributes, **kwargs):  # noqa N802
    if metric not in fairness_scorers_dict:
        raise GuardianAIValueError(
            f"{metric} is not a supported model fairness metric. Supported "
            f"metrics are: {list(fairness_scorers_dict)}."
        )

    return fairness_scorers_dict[metric](protected_attributes, **kwargs)


fairness_metrics_dict = {
    "statistical_parity": model_statistical_parity,
    "TPR": true_positive_rate,
    "FPR": false_positive_rate,
    "FNR": false_negative_rate,
    "FOR": false_omission_rate,
    "FDR": false_discovery_rate,
    "error_rate": error_rate,
    "equalized_odds": equalized_odds,
    "theil_index": theil_index,
}


def _get_fairness_metric(metric):
    if metric not in fairness_metrics_dict:
        raise GuardianAIValueError(
            f"{metric} is not a supported model fairness metric. Supported "
            f"metrics are: {list(fairness_metrics_dict)}."
        )

    return fairness_metrics_dict[metric]
