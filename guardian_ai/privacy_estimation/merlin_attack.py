#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator

from guardian_ai.privacy_estimation.attack import BlackBoxAttack, AttackType
from guardian_ai.privacy_estimation.model import TargetModel
from guardian_ai.privacy_estimation.utils import log_loss_vector


class MerlinAttack(BlackBoxAttack):
    """
    Implements the Merlin Attack as described in the paper: Revisiting Membership Inference
    Under Realistic Assumptions by Jayaraman et al.
    The main idea is to perturb a data point, and calculate noise on all the data points in
    this neighborhood. If the loss of large fraction of these points is above the target point,
    it might imply that the target point is in a local minima, and therefore the model might
    have fitted around it, implying it might have seen it at training time.
    """

    def __init__(
        self,
        attack_model: BaseEstimator,
        noise_type: str = "gaussian",
        noise_coverage: str = "full",
        noise_magnitude: float = 0.01,
        max_t: int = 50,
    ):
        """
        These default values are mostly taken from the original implementation of this attack.

        Parameters
        ----------
        attack_model: sklearn.base.BaseEstimator
            The type of attack model to be used.
            Typically, it's ThresholdClassifier.
        noise_type: str
            Choose the type of noise to add based on the data.
            Supports uniform and gaussian.
        noise_coverage: str
            Add noise to all attributes ("full") or only a subset.
        noise_magnitude: float
            Size of the noise.
        max_t: int
            The number of noisy points to generate to calculate the Merlin Ratio.

        """
        self.noise_type = noise_type
        self.noise_coverage = noise_coverage
        self.noise_magnitude = noise_magnitude
        self.max_t = max_t
        super(MerlinAttack, self).__init__(
            attack_model, name=AttackType.MerlinAttack.name
        )

    def generate_noise(self, shape: np.shape, dtype):
        """
        Generate noise to be added to the target data point.

        Parameters
        ----------
        shape: : np.shape
            Shape of the target data point
        dtype: np.dtype
            Datatype of the target data point

        Returns
        -------
        {array-like}
            Noise generated according to the parameters to match the shape of the target.

        """
        noise = np.zeros(shape, dtype=dtype)
        if self.noise_coverage == "full":
            if self.noise_type == "uniform":
                noise = np.array(
                    np.random.uniform(0, self.noise_magnitude, size=shape), dtype=dtype
                )
            else:
                noise = np.array(
                    np.random.normal(0, self.noise_magnitude, size=shape), dtype=dtype
                )
        else:
            attr = np.random.randint(shape[1])
            if self.noise_type == "uniform":
                noise[:, attr] = np.array(
                    np.random.uniform(0, self.noise_magnitude, size=shape[0]),
                    dtype=dtype,
                )
            else:
                noise[:, attr] = np.array(
                    np.random.normal(0, self.noise_magnitude, size=shape[0]),
                    dtype=dtype,
                )
        return noise

    def get_merlin_ratio(self, target_model: TargetModel, X_attack, y_attack):
        """
        Returns the merlin-ratio for the Merlin attack.

        Parameters
        ----------
        target_model: guardian_ai.privacy_estimation.model.TargetModel
            Model that is being targeted by the attack.
        X_attack: {array-like, sparse matrix} of shape (n_samples, n_features)
            Input features of the attack datapoints, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        y_attack: ndarray of shape (n_samples,)
            Vector containing the  output labels of the attack data points (not membership label).

        Returns
        -------
        float
            Merlin Ratio. Value between 0 and 1.

        """

        labels = target_model.model.classes_
        pred_y = target_model.get_prediction_probs(X_attack)
        my_per_instance_loss = log_loss_vector(y_attack, pred_y, labels=labels)
        counts = np.zeros((X_attack).shape[0])
        for _t in range(self.max_t):
            noise = self.generate_noise(X_attack.shape, X_attack.dtype)
            if sp.issparse(X_attack):
                noise = sp.csr_matrix(noise)
            noisy_x = X_attack + noise
            predictions = target_model.get_prediction_probs(noisy_x)
            my_noisy_per_instance_loss = log_loss_vector(
                y_attack, predictions, labels=labels
            )
            counts += np.where(my_noisy_per_instance_loss > my_per_instance_loss, 1, 0)
        return counts / self.max_t

    def transform_attack_data(
        self,
        target_model: TargetModel,
        X_attack,
        y_attack,
        split_type: str = None,
        use_cache=False,
    ):
        """
        Overriding the method transform_attack_data from the base class.
        Calculates the  merlin ratio.

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
            Use information cached from running the loss based and merlin attacks.
        use_cache: bool
            Using the cache or not.

        Returns
        -------
        X_membership:  {array-like, sparse matrix} of shape (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is
            the number of features.
            Input feature for the attack model - in this case, the Merlin
            ratio.

        """
        X_membership = self.get_merlin_ratio(target_model, X_attack, y_attack)
        return X_membership
