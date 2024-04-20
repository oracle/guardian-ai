#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
from sklearn.base import BaseEstimator

from guardian_ai.privacy_estimation.attack import (
    BlackBoxAttack,
    ConfidenceBasedBlackBoxAttack,
    LossBasedBlackBoxAttack,
    AttackType,
)
from guardian_ai.privacy_estimation.merlin_attack import MerlinAttack
from guardian_ai.privacy_estimation.model import TargetModel
from guardian_ai.privacy_estimation.utils import log_loss_vector
from typing import List

class CombinedBlackBoxAttack(BlackBoxAttack):
    """
    Similar in spirit to the Morgan attack, which combines loss and the merlin ratio.
    In this attack, we combine loss, and confidence values and instead of tuning the
    thresholds, we combine them using a trained classifier, like stacking.
    """

    def __init__(
        self,
        attack_model: BaseEstimator,
        loss_attack: LossBasedBlackBoxAttack = None,
        confidence_attack: ConfidenceBasedBlackBoxAttack = None,
    ):
        """
        Initialize CombinedBlackBoxAttack.

        Parameters
        ----------
        attack_model: sklearn.base.BaseEstimator
        loss_attack: guardian_ai.privacy_estimation.attack.LossBasedBlackBoxAttack
        confidence_attack: guardian_ai.privacy_estimation.attack.ConfidenceBasedBlackBoxAttack

        """
        self.loss_attack = loss_attack
        self.confidence_attack = confidence_attack
        super(CombinedBlackBoxAttack, self).__init__(
            attack_model, name=AttackType.CombinedBlackBoxAttack.name
        )

    def transform_attack_data(
        self,
        target_model: TargetModel,
        X_attack,
        y_attack,
        y_membership,
        features=None,
        split_type: str = None,
        use_cache=False,
    ):
        """
        Overriding the method transform_attack_data from the base class.
        Calculates the  per instance loss and confidence.

        Parameters
        ----------
        target_model: guardian_ai.privacy_estimation.model.TargetModel
            Target model being attacked.
        X_attack: {array-like, sparse matrix} of shape (n_samples, n_features)
            Input features of the attack datapoints, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        y_attack: ndarray of shape (n_samples,)
            Vector containing the  output labels of the attack data points (not membership label).
        split_type: str
            Use information cached from running the loss based and merlin attacks
        use_cache: bool
            Using the cache or not

        Returns
        -------
        X_membership:  {array-like, sparse matrix} of shape (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is
            the number of features.
            Input feature for the attack model - in this case,
            per-instance loss and confidence values
            @param features:
            @param y_membership:

        """
        if use_cache:
            if split_type == "train":
                my_per_instance_loss = self.loss_attack.X_membership_train
                my_confidence = self.confidence_attack.X_membership_train
            elif split_type == "test":
                my_per_instance_loss = self.loss_attack.X_membership_test
                my_confidence = self.confidence_attack.X_membership_test
            else:
                raise Exception("split type specified is not cached")
        else:
            labels = target_model.model.classes_
            probs = target_model.get_prediction_probs(X_attack)
            my_per_instance_loss = -log_loss_vector(y_attack, probs, labels=labels)
            my_confidence = np.max(probs, 1)
        X_membership = np.column_stack((my_per_instance_loss, my_confidence))
        return X_membership


class CombinedWithMerlinBlackBoxAttack(BlackBoxAttack):
    """
    Similar in spirit to the Morgan attack, which combines loss and the merlin ratio.
    In this attack, we combine loss,  confidence values and merlin ratio,
    and instead of tuning the thresholds, we combine them using
    a trained classifier, like stacking.
    """

    def __init__(
        self,
        attack_model: BaseEstimator,
        merlin_attack: MerlinAttack,  # this must be passed
        loss_attack: LossBasedBlackBoxAttack = None,
        confidence_attack: ConfidenceBasedBlackBoxAttack = None,
    ):
        self.merlin_attack = merlin_attack
        self.loss_attack = loss_attack
        self.confidence_attack = confidence_attack
        super(CombinedWithMerlinBlackBoxAttack, self).__init__(
            attack_model, name=AttackType.CombinedWithMerlinBlackBoxAttack.name
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
                my_confidence = self.confidence_attack.X_membership_train
                merlin_ratio = self.merlin_attack.X_membership_train
            elif split_type == "test":
                my_per_instance_loss = self.loss_attack.X_membership_test
                my_confidence = self.confidence_attack.X_membership_test
                merlin_ratio = self.merlin_attack.X_membership_test
            else:
                raise Exception("split type specified is not cached")
        else:
            labels = target_model.model.classes_
            probs = target_model.get_prediction_probs(X_attack)
            my_per_instance_loss = -log_loss_vector(y_attack, probs, labels=labels)
            my_confidence = np.max(probs, 1)
            merlin_ratio = self.merlin_attack.get_merlin_ratio(
                target_model, X_attack, y_attack
            )
        X_membership = np.column_stack(
            (my_per_instance_loss, my_confidence, merlin_ratio)
        )
        return X_membership
