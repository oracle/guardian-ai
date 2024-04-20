#!/usr/bin/env python
# -*- coding: utf-8 -*--
from typing import List

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from guardian_ai.privacy_estimation.attack import (
    BlackBoxAttack,
    LossBasedBlackBoxAttack,
    ThresholdClassifier,
    AttackType,
)
from guardian_ai.privacy_estimation.merlin_attack import MerlinAttack
from guardian_ai.privacy_estimation.model import TargetModel
from guardian_ai.privacy_estimation.utils import log_loss_vector


class MorganClassifier(ThresholdClassifier):
    """
    Implements the Morgan Attack as described in the paper: Revisiting Membership Inference
    Under Realistic Assumptions by Jayaraman et al.
    The main idea is to combine the merlin ratio and per instance loss using multiple
    thresholds. This classifier goes along with the Morgan Attack, which implements a
    custom decision function that combines the three thresholds.
    """

    def __init__(
        self,
        loss_lower_threshold: float,
        merlin_threshold: float,
        threshold: float = 0.5,
    ):
        """
        Morgan attack uses three thresholds, of which, two are given and one is tuned.

        Parameters
        ----------
        loss_lower_threshold: float
            Lower threshold on the per instance loss.
        merlin_threshold: float
            Threshold on the merlin ration.
        threshold: float
            Upper threshold on the per instance loss.

        """
        super(MorganClassifier, self).__init__(threshold)
        self.parameters["loss_lower_threshold"] = loss_lower_threshold
        # I'm doing it this way, since the attack tuner calls a clone object,
        # which messes up this constructor
        self.parameters["merlin_threshold"] = merlin_threshold

    def predict(self, X):
        """
        Calls the custom decision function that is required for the Morgan attack

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Input features of the attack datapoints, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the membership labels for each attack point.
        """
        d = self.decision_function(X)
        return self.classes_[np.argmax(d, axis=1)]

    def decision_function(self, X):
        """
        Custom decision function that applies the three thresholds of the Morgan attack

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Input features of the attack datapoints, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        Binary decision ndarray of shape (n_samples,) or (n_samples, n_classes)
            The feature value over a certain threshold.

        """
        check_is_fitted(self)

        threshold = self.parameters["threshold"]
        if hasattr(self, "threshold"):
            threshold = self.threshold
        assert X.shape[1] == 2

        d_true = (
            (self.parameters["loss_lower_threshold"] <= X[:, 0])
            & (X[:, 0] <= threshold)
            & (X[:, 1] >= self.parameters["merlin_threshold"])
        )

        # create the decision vector
        index_of_true = np.where(self.classes_ == 1)
        if index_of_true == 0:
            d = np.column_stack((d_true, np.zeros((X.shape[0], 1))))
        else:
            d = np.column_stack((np.zeros((X.shape[0], 1)), d_true))
        return d


class MorganAttack(BlackBoxAttack):
    """
    Implements the Morgan Attack as described in the paper: Revisiting Membership Inference
    Under Realistic Assumptions by Jayaraman et al.
    The main idea is to combine the merlin ratio and per instance loss using multiple thresholds.
    """

    def __init__(
        self,
        attack_model: BaseEstimator,
        loss_attack: LossBasedBlackBoxAttack,
        merlin_attack: MerlinAttack,
    ):
        """
        Initialize MorganAttack.

        Parameters
        ----------
        attack_model: sklearn.base.BaseEstimator
            Base attack model. Usually the Morgan Classifier.
        loss_attack: guardian_ai.privacy_estimation.attack.LossBasedBlackBoxAttack
            Loss attack object.
        merlin_attack: guardian_ai.privacy_estimation.merlin_attack.MerlinAttack
            Merlin attack object.

        """
        self.loss_attack = loss_attack
        self.merlin_attack = merlin_attack
        super(MorganAttack, self).__init__(
            attack_model, name=AttackType.MorganAttack.name
        )
    
    def transform_attack_data(
            self,
            target_model: TargetModel,
            X_attack,
            y_attack,
            y_membership,
            split_type: str = None,
            use_cache: bool = False,
            features: List[List[float]] = None,
    ):
        """
        This is the central method in designing the attack, and captures the attacker's
        hypothesis about the membership of a data point in the training dataset of the target
        model. Its job is to derive signals from the original data that might be relevant to
        determining membership. Takes a dataset in the original format and converts it to the
        input variable for the attack. Think of it as feature engineering for building
        the attack model, which is essentially a binary classifier.

        Parameters
        ----------
        target_model: guardian_ai.privacy_estimation.model.TargetModel
            Target model being attacked.
        X_attack: {array-like, sparse matrix} of shape (n_samples, n_features)
            Input features of the attack datapoints, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        y_attack: ndarray of shape (n_samples, )
            Vector containing the output labels of the attack data points (not membership label).
        y_membership: array of shape (n_samples, 1)
            Vector containing the membership labels.
        split_type: str
            Use information cached from running the loss based and merlin attacks.
        use_cache: bool
            Using the cache or not.
        features: List[List[float]]
            Feature vectors of the items - required when the collaborative filtering model
            is being attacked
        Returns
        -------
        X_membership:  {array-like, sparse matrix} of shape (n_samples, n_features)
            Input features for the attack model, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.


        """
        if use_cache:
            if split_type == "train":
                my_per_instance_loss = self.loss_attack.X_membership_train
                merlin_ratio = self.merlin_attack.X_membership_train
            elif split_type == "test":
                my_per_instance_loss = self.loss_attack.X_membership_test
                merlin_ratio = self.merlin_attack.X_membership_test
            else:
                raise Exception("split type specified is not cached")
        else:
            labels = target_model.model.classes_
            pred_y = target_model.get_prediction_probs(X_attack)
            my_per_instance_loss = -log_loss_vector(y_attack, pred_y, labels=labels)
            merlin_ratio = self.merlin_attack.get_merlin_ratio(
                target_model, X_attack, y_attack
            )
        X_membership = np.column_stack((my_per_instance_loss, merlin_ratio))
        return X_membership
