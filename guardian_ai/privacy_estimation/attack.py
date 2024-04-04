#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from abc import abstractmethod
import enum

import numpy as np
import sklearn.metrics
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import classification_report, get_scorer, roc_curve
from sklearn.utils.validation import check_is_fitted
from typing import List

from guardian_ai.privacy_estimation.attack_tuner import AttackTuner
from guardian_ai.privacy_estimation.model import TargetModel
from guardian_ai.privacy_estimation.utils import log_loss_vector


class AttackType(enum.Enum):
    """
    All the attack types currently supported by this tool.
    """

    LossBasedBlackBoxAttack = 0
    ExpectedLossBasedBlackBoxAttack = 1
    ConfidenceBasedBlackBoxAttack = 2
    ExpectedConfidenceBasedBlackBoxAttack = 3
    MerlinAttack = 4
    CombinedBlackBoxAttack = 5
    CombinedWithMerlinBlackBoxAttack = 6
    MorganAttack = 7


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """
    Base Classifier for all threshold based attacks. For a given attack point with just
    a single feature, a threshold based classifier predicts if that feature value is over
    a threshold value.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Instantiate the classifier

        Parameters
        ----------
        threshold: float, Default value is 0.5.
            This threshold is usually tuned.

        """
        self.parameters = {}
        self.classes_ = None
        self.parameters["threshold"] = threshold

    def fit(self, X, y):
        """
        Fit the data to the classifier, but because this is a simple threshold classifier, fit
        doesn't really do much, except record the data and the domain of the class labels.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input features of the attack model, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        y: array-like of shape (n_samples,)
            Output label of the attack model (usually 0/1).

        Returns
        -------
        ThresholdClassifier
            The trained classifier.

        """
        self.classes_, y = np.unique(y, return_inverse=True)
        self.X_ = X
        self.y = y
        return self

    def predict(self, X):
        """
        Make prediction using the decision function of the classifier.

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
        For a given attack point with just a single feature, a threshold based classifier
        predicts if that feature value is over a threshold value.

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features)
            Input features of the attack datapoints, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features. For ThresholdClassifier, it's usually just
            a single feature, but can be more.

        Returns
        -------
        Binary decision ndarray of shape (n_samples,) or (n_samples, n_classes)
            The feature value over a certain threshold.

        """
        check_is_fitted(self)

        threshold = self.parameters["threshold"]
        if hasattr(self, "threshold"):
            threshold = self.threshold

        d_true = X >= threshold

        index_of_true = np.where(self.classes_ == 1)
        if index_of_true == 0:
            d = np.column_stack((d_true, np.zeros((X.shape[0], 1))))
        else:
            d = np.column_stack((np.zeros((X.shape[0], 1)), d_true))
        return d

    def get_params(self, deep: bool = True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep: bool, default is True.
            If True, will return the parameters for this estimator and contained
            subobjects that are estimators.

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        return self.parameters

    def set_params(self, **parameters):
        """
        Set estimator parametes.

        Parameters
        ----------
        parameters: dict
            Estimator parameters.

        Returns
        -------
        Estimator instance.

        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


class BlackBoxAttack:
    """
    This is the base class for all black box attacks. It has a base estimator, which could be a
    threshold based, or learning based classifier - typically a binary classifier that decides
    whether an attack data point was part of the original training data for the target model
    or not. It's black box because this type of attack can only access the prediction API of
    the target model and does not have access to the model parameters.
    """

    def __init__(
        self,
        attack_model: BaseEstimator,
        name: str = "generic_black_box_attack",
    ):
        """
        Initialize the attack.

        Parameters
        ----------
        attack_model: sklearn.base.BaseEstimator
        name: str
            Name of this attack for reporting purposes.

        """
        self.name = name
        self.attack_model = attack_model
        self.X_membership_train = None  # Useful for caching the feature values for the attack (e.g. Morgan attack)
        self.X_membership_test = None

    @abstractmethod
    def transform_attack_data(
        self,
        target_model: TargetModel,
        X_attack,
        y_attack,
        split_type: str = None,
        use_cache: bool = False,
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
        y_attack: ndarray of shape (n_samples,)
            Vector containing the  output labels of the attack data points (not membership label).
        split_type: str
            Whether this is "train" set or "test" set, which is used for Morgan attack.
        use_cache: bool
            Whether to use the cache or not.

        Returns
        -------
        X_membership:  {array-like, sparse matrix} of shape (n_samples, n_features)
            Input features for the attack model, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        """

    def train_attack_model(
        self,
        target_model: TargetModel,
        X_attack_train,
        y_attack_train,
        y_membership_train,
        threshold_grid: List[float] = None,
        cache_input: bool = False,
        use_cache: bool = False,
    ):
        """
        Takes the attack data points, transforms them into attack features and then trains the
        attack model using membership labels for those points. If a threshold grid is provided,
        it will simply tune the threshold using that grid, otherwise, it will train the model.

        Parameters
        ----------
        target_model: guardian_ai.privacy_estimation.model.TargetModel
            Target model that is being attacked.
        X_attack_train: {array-like, sparse matrix} of shape (n_samples, n_features),
            where `n_samples` is the number of samples and `n_features` is the
            number of features.
            Input variables for the dataset on which we want to train
            the attack model. These are the original features (not attack/membership features).
        y_attack_train: ndarray of shape (n_samples,)
            Output labels for the dataset on which we want to train
            the attack model. These are the original labels (not membership labels).
        y_membership_train: ndarray of shape (n_samples,)
            Membership labels for the dataset on which we want to train
            the attack model. These are binary and indicate whether the data
            point was included in the training dataset of the target model.
        threshold_grid: List[float]
            Threshold grid to use for tuning this model.
        cache_input: bool
            Should we cache the input values - useful for expensive feature
            calculations like the merlin ratio.
        use_cache: bool
            Should we use the feature values from the cache - useful for Morgan
            and Combined attacks.

        Returns
        -------
        Trained attack model, usually a binary classifier.

        """
        if isinstance(
            self.attack_model, ThresholdClassifier
        ):  # We only need this for threshold based attacks
            self.attack_model.fit(X=None, y=y_membership_train)

        X_membership_train = self.transform_attack_data(
            target_model,
            X_attack_train,
            y_attack_train,
            split_type="train",
            use_cache=use_cache,
        )
        if cache_input:
            self.X_membership_train = X_membership_train
        if threshold_grid is not None:
            model_tuner = AttackTuner()
            best_params = model_tuner.tune_attack(
                self.attack_model,
                X_membership_train,
                y_membership_train,
                threshold_grid,
            )
            self.attack_model.threshold = best_params["threshold"]
        else:
            self.attack_model = self.attack_model.fit(
                X_membership_train, y_membership_train
            )

    def perform_attack(self, target_model: TargetModel, X_attack, y_attack):
        """
        Perform the actual attack. For now, this method would only be used in settings where
        the attacks themselves are being audited.
        Usually, we only call the evaluate_attack method.

        Parameters
        ----------
        target_model: guardian_ai.privacy_estimation.model.TargetModel
            Target model being attacked.
        X_attack: {array-like, sparse matrix} of shape (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is the
            number of features.
            Input variables for the attack points. These are the
            original features (not attack/membership features).
        y_attack: ndarray of shape (n_samples,)
            Output labels for the attack points. These are the
            original labels (not membership labels).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the Binary predictions on whether the attack points
            were part of the dataset used to train the target model.

        """
        X_membership = self.transform_attack_data(
            target_model, X_attack, y_attack, split_type="runtime"
        )
        return self.attack_model.predict(X_membership)

    def evaluate_attack(
        self,
        target_model: TargetModel,
        X_attack_test,
        y_attack_test,
        y_membership_test,
        metric_functions: List[str],
        print_roc_curve: bool = False,
        cache_input: bool = False,
        use_cache: bool = False,
    ):
        """
        Runs the attack against the target model, evaluates its accuracy and provides the
        metrics of interest on the success of the attack.

        Parameters
        ----------
        target_model: guardian_ai.privacy_estimation.model.TargetModel
            Target model being attacked.
        X_attack_test: {array-like, sparse matrix} of shape (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is the
            number of features.
            Input variables for the dataset on which to run the attack model.
            These are the original features (not attack/membership features).
        y_attack_test: ndarray of shape (n_samples,)
            Output labels for the dataset on which to run the attack model.
            These are the original labels (not membership labels).
        y_membership_test: ndarray of shape (n_samples,)
            Membership labels for the dataset on which we want to run
            the attack model. These are binary and indicate whether the data
            point was included in the training dataset of the target model,
            and helps us evaluate the attack model's accuracy.
        metric_functions: List[str]
            List of metric functions that we care about for evaluating the
            success of these attacks. Supports all sklearn.metrics that are
            relevant to binary classification, since the attack model is almost
            always a binary classifier.
        print_roc_curve: bool, Defaults to False.
            Print out the values of the tpr and fpr. Only works for
            trained attack classifiers for now.
        cache_input: bool, Defaults to False.
            Should we cache the input values - useful for expensive feature
            calculations like the merlin ratio.
        use_cache: bool, Defaults to False.
            Should we use the feature values from the cache - useful for Morgan
            attack, which uses merlin ratio and loss values.

        Returns
        -------
        List[float]
            Success metrics for the attack.

        """

        X_membership_test = self.transform_attack_data(
            target_model,
            X_attack_test,
            y_attack_test,
            split_type="test",
            use_cache=use_cache,
        )
        if cache_input:
            self.X_membership_test = X_membership_test
        predictions = self.attack_model.predict(X_membership_test)
        print(classification_report(y_membership_test, predictions))

        if print_roc_curve and not isinstance(self.attack_model, ThresholdClassifier):
            predictions_prob = self.attack_model.predict_proba(X_membership_test)
            fpr, tpr, thresholds = roc_curve(
                y_membership_test, predictions_prob[:, 1], pos_label=1
            )
            print(fpr)
            print(tpr)
            print(thresholds)

        metrics = []
        for metric_function_name in metric_functions:
            scorer = get_scorer(
                metric_function_name
            )  # converts the string of the scorer name into the actual metric function
            metric_value = scorer._score_func(y_membership_test, predictions)
            metrics.append(metric_value)
            print(metric_function_name + " = " + str(metric_value))
        return metrics


class LossBasedBlackBoxAttack(BlackBoxAttack):
    """
    One of the simplest, but fairly effective attack - which looks at the loss value of the
    attack point. Attacker hypothesis is that lower loss indicates that the target model has
    seen this data point at training time.
    """

    def __init__(
        self,
        attack_model: BaseEstimator,
    ):
        """
        Instantiate the Loss based attack.

        Parameters
        -------
        attack_model: sklearn.base.BaseEstimator
            Typically Threshold classifier, but could also
            be a single feature logistic regression.

        """
        super(LossBasedBlackBoxAttack, self).__init__(
            attack_model, name=AttackType.LossBasedBlackBoxAttack.name
        )

    def transform_attack_data(
        self,
        target_model: TargetModel,
        X_attack,
        y_attack,
        split_type: str = None,
        use_cache: bool = False,
    ):
        """
        Takes the input attack points, and calculates loss values on them.

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
            Whether this is "train" set or "test" set, which is used for Morgan
            attack, which uses cached values of loss and merlin ratios for efficiency.
        use_cache: bool
            Using the cache or not.

        Returns
        -------
        X_membership:  {array-like, sparse matrix} of shape (n_samples, n_features)
            Input loss value features for the attack model, where ``n_samples`` is
            the number of samples and ``n_features`` is the number of features.

        """
        labels = target_model.model.classes_
        probs = target_model.get_prediction_probs(X_attack)
        X_membership = -log_loss_vector(
            y_attack, probs, labels=labels
        )  # lower is better
        return X_membership


class ExpectedLossBasedBlackBoxAttack(BlackBoxAttack):
    """
    Same as Loss based attack, but the difference is that we're going to use a logistic
    regression classifier. The only reason we need a separate attack for this is because the
    shape of the attack feature needs to be different.
    """

    def __init__(self, attack_model: BaseEstimator):
        """
        Instantiate the Expected Loss based attack.

        Parameters
        ----------
        attack_model: sklearn.base.BaseEstimator
            Typically a single feature logistic regression.

        """

        super(ExpectedLossBasedBlackBoxAttack, self).__init__(
            attack_model, name=AttackType.ExpectedLossBasedBlackBoxAttack.name
        )

    def transform_attack_data(
        self,
        target_model: TargetModel,
        X_attack,
        y_attack,
        split_type: str = None,
        use_cache: bool = False,
    ):
        """
        Takes the input attack points, and calculates loss values on them.

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
            Whether this is "train" set or "test" set, which is used for Morgan
            attack, which uses cached values of loss and merlin ratios for efficiency
        use_cache: bool
            Using the cache or not.

        Returns
        -------
        X_membership:  {array-like, sparse matrix} of shape (n_samples, n_features)
            Input loss value features for the attack model, where `n_samples` is
            the number of samples and `n_features` is the number of features.

        """
        labels = target_model.model.classes_
        probs = target_model.get_prediction_probs(X_attack)
        X_membership = -log_loss_vector(
            y_attack, probs, labels=labels
        )  # lower is better
        # Note that this is the main difference.
        # We're using the right shape to be used with a classifier with a single feature
        return np.column_stack((X_membership, np.zeros((X_membership.shape[0], 1))))


class ConfidenceBasedBlackBoxAttack(BlackBoxAttack):
    """
    One of the simplest, but fairly effective attack - which looks at the confidence of the
    attack point. Attacker hypothesis is that higher confidence indicates that the target
    model has seen this data point at training time.
    """

    def __init__(self, attack_model: BaseEstimator):
        """
        Instantiate the Confidence based attack
        Parameters
        ----------
        attack_model:  sklearn.base.BaseEstimator
            Typically Threshold classifier, but could also
            be a single feature logistic regression.
        """
        super(ConfidenceBasedBlackBoxAttack, self).__init__(
            attack_model, name=AttackType.ConfidenceBasedBlackBoxAttack.name
        )

    def transform_attack_data(
        self,
        target_model: TargetModel,
        X_attack,
        y_attack,
        split_type: str = None,
        use_cache: bool = False,
    ):
        """
        Takes the input attack points, and calculates confidence values on them.

        Parameters
        ----------
        target_model: guardian_ai.privacy_estimation.model.TargetModel
            Target model being attacked.
        X_attack: {array-like, sparse matrix} of shape (n_samples, n_features)
            Input features of the attack datapoints, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
        y_attack: ndarray of shape (n_samples,)
            Vector containing the  output labels of the attack data points (not membership label)
        split_type: str
            Whether this is "train" set or "test" set, which is used for Morgan
            attack, which uses cached values of loss and merlin ratios for efficiency
        use_cache: bool
            Using the cache or not

        Returns
        -------
        X_membership:  {array-like, sparse matrix} of shape (n_samples, n_features)
            Input confidence value features for the attack model, where ``n_samples`` is
            the number of samples and ``n_features`` is the number of features.

        """
        probs = target_model.get_prediction_probs(X_attack)
        X_membership = np.max(probs, 1)
        return X_membership


class ExpectedConfidenceBasedBlackBoxAttack(BlackBoxAttack):
    """
    Classification based version of the Confidence based attack
    """

    def __init__(self, attack_model: BaseEstimator):
        """
        Instantiate the Expected Confidence based attack

        Parameters
        ----------
        attack_model: sklearn.base.BaseEstimator
            Typically a single feature logistic regression.

        """
        super(ExpectedConfidenceBasedBlackBoxAttack, self).__init__(
            attack_model, name=AttackType.ExpectedConfidenceBasedBlackBoxAttack.name
        )

    def transform_attack_data(
        self,
        target_model: TargetModel,
        X_attack,
        y_attack,
        split_type: str = None,
        use_cache: bool = False,
    ):
        """
        Takes the input attack points, and calculates loss values on them.

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
            Whether this is "train" set or "test" set, which is used for Morgan
            attack, which uses cached values of loss and merlin ratios for efficiency
        use_cache: bool
            Using the cache or not

        Returns
        -------
        X_membership:  {array-like, sparse matrix} of shape (n_samples, n_features)
            Input confidence value features for the attack model, where ``n_samples`` is
            the number of samples and ``n_features`` is the number of features.
        """
        probs = target_model.get_prediction_probs(X_attack)
        X_membership = np.max(probs, 1)
        return np.column_stack((X_membership, np.zeros((X_membership.shape[0], 1))))
