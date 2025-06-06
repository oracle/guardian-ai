#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np


def log_loss(y_true, y_pred, labels=None):
    """
    Calculates the standard log loss function.

    Parameters
    ----------
    y_true : array-like list with correct labels for n_samples samples
    y_pred : array-like of float with shape (n_samples, n_classes) or (n_samples,). These
        are the predicted probabilities
    labels : array-like, default=None
        If None, the labels are inferred from ``y_true``

    Returns
    -------
    loss: float
        The log loss value
    """

    return np.average(loss_vector(y_true, y_pred, labels))


def log_loss_vector(y_true, y_pred, labels=None):
    """
    Return the loss vector that is used to compute log loss. The negative sign from the
    standard log loss function is distributed through the vector. To get the log loss value
    use the `log_loss` function.

    This function is used in place of ``sklearn.metrics.log_loss`` because calculations
    need access the loss vector itself and not just the final log loss value.

    Parameters
    ----------
    y_true : array-like list with correct labels for n_samples samples
    y_pred : array-like of float with shape (n_samples, n_classes) or (n_samples,). These
        are the predicted probabilities
    labels : array-like, default=None
        If None, the labels are inferred from ``y_true``

    Returns
    -------
    loss vector: np.array
        The cross entropy loss for each sample.
    """

    n_samples = len(y_true)

    # Preliminary checks
    if labels is not None:
        if set(y_true) != set(labels):
            raise ValueError("Label mismatch between y_true and labels")
    else:
        labels = sorted(list(set(y_true)))

    if np.shape(y_pred) != (n_samples, len(labels)):
        raise ValueError("y_pred is not well formed")

    spos_dict = dict(zip(labels, range(len(labels))))

    # Calculate loss vector
    loss_vector = []
    for i, sample in enumerate(y_true):
        sample_loss = np.sum(
            [-int(j == spos_dict[sample]) * np.log(y_pred[i][j]) for j in range(len(labels))]
        )
        loss_vector.append(sample_loss)

    return np.array(loss_vector)
