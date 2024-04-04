#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from guardian_ai.fairness.metrics.core import (
    _get_fairness_metric,
    _get_fairness_scorer,
    fairness_metrics_dict,
    fairness_scorers_dict,
)
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
from guardian_ai.fairness.metrics.utils import _FairnessScorer, _positive_fairness_names

__all__ = [
    "_get_fairness_scorer",
    "fairness_scorers_dict",
    "_get_fairness_metric",
    "fairness_metrics_dict",
    "_positive_fairness_names",
    "FairnessMetric",
    "_FairnessScorer",
    "DatasetStatisticalParityScorer",
    "dataset_statistical_parity",
    "ConsistencyScorer",
    "consistency",
    "SmoothedEDFScorer",
    "smoothed_edf",
    "ModelStatisticalParityScorer",
    "model_statistical_parity",
    "TruePositiveRateScorer",
    "true_positive_rate",
    "FalsePositiveRateScorer",
    "false_positive_rate",
    "FalseNegativeRateScorer",
    "false_negative_rate",
    "FalseOmissionRateScorer",
    "false_omission_rate",
    "FalseDiscoveryRateScorer",
    "false_discovery_rate",
    "ErrorRateScorer",
    "error_rate",
    "EqualizedOddsScorer",
    "equalized_odds",
    "TheilIndexScorer",
    "theil_index",
]
