#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Module containing generic helper classes and functions."""
from typing import Dict, List

_supported_score_metric: Dict[str, List[str]] = (
    {  # the first value entry will be default scoring
        "binary": [
            "neg_log_loss",
            "roc_auc",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "f1_micro",
            "f1_macro",
            "f1_weighted",
            "f1_samples",
            "recall_micro",
            "recall_macro",
            "recall_weighted",
            "recall_samples",
            "precision_micro",
            "precision_macro",
            "precision_weighted",
            "precision_samples",
        ],
        # f1/precision/roc_auc is not supported in multiclass
        "multiclass": [
            "neg_log_loss",
            "accuracy",
            "f1_micro",
            "f1_macro",
            "f1_weighted",
            "f1_samples",
            "recall_macro",
            "recall_micro",
            "recall_weighted",
            "recall_samples",
            "precision_micro",
            "precision_macro",
            "precision_weighted",
            "precision_samples",
        ],
        "continuous": [
            "neg_mean_squared_error",
            "r2",
            "neg_mean_absolute_error",
            "neg_mean_squared_log_error",
            "neg_median_absolute_error",
        ],
        "continuous_forecast": [
            "neg_sym_mean_abs_percent_error",
            "neg_root_mean_squared_percent_error",
            "neg_mean_abs_scaled_error",
            "neg_root_mean_squared_error",
            "neg_mean_squared_error",
            "neg_max_absolute_error",
            "neg_mean_absolute_error",
            "neg_max_abs_error",
            "neg_mean_abs_error",
        ],
        # metrics starting with 'unsupervised' do not require contamination factor to be provided.
        "unsupervised": [
            # "unsupervised_n-1_experts",
            "unsupervised_unify95",
            "unsupervised_unify95_log_loss",
        ],
    }
)


def dyn_docstring(*args):  # noqa
    """Decorate a method to replace placeholders in the docstring with
    the decorator args.

    Parameters
    ----------
    *args
        Values to fill in the placeholders

    Returns
    -------
    A decorator for the method
    """

    def dec(obj):
        obj.__doc__ = obj.__doc__ % args
        return obj

    return dec
