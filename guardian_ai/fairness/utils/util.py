#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

"""Module containing generic helper classes and functions."""
from typing import Dict, List

_supported_score_metric: Dict[
    str, List[str]
] = {  # the first value entry will be default scoring
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
    ]
}


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
