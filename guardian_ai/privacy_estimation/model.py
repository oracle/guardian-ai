#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import pickle
from abc import abstractmethod
import pandas as pd
import sklearn.base as base
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.neural_network import MLPClassifier


class TargetModel:
    """
    Wrapper for the target model that is being attacked.
    For now, we're only supporting sklearn classifiers that implement .predict_proba
    """

    def __init__(self):
        """
        Create the target model that is being attacked, and check that it's a classifier
        """
        self.model = self.get_model()
        assert base.is_classifier(self.model)

    @abstractmethod
    def get_model(self):
        """
        Create the target model that is being attacked.

        Returns
        -------
        Model that is not yet trained.
        """
        pass

    def train_model(self, x_train, y_train):
        """
        Train the model that is being attacked.

        Parameters
        ----------
        x_train: {array-like, sparse matrix} of shape (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
            Input variables of the training set for the target model.
        y_train: ndarray of shape (n_samples,)
            Output labels of the training set for the target model.

        Returns
        -------
        Trained model

        """
        return self.model.fit(x_train, y_train)

    def test_model(self, x_test, y_test):
        """
        Test the model that is being attacked.

        Parameters
        ----------
        x_test: {array-like, sparse matrix} of shape (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
            Input variables of the test set for the target model.
        y_test: ndarray of shape (n_samples,)
            Output labels of the test set for the target model.

        Returns
        -------
        None

        """
        predictions = self.model.predict(x_test)
        print(classification_report(y_test, predictions))

    def get_f1(self, x_test, y_test):
        """
        Gets f1 score.

        Parameters
        ----------
        x_test: {array-like, sparse matrix} of shape (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.
        y_test: ndarray of shape (n_samples,)

        """
        predictions = self.model.predict(x_test)
        return f1_score(y_test, predictions, average="macro")

    def get_predictions(self, X):
        """
        Gets model prediction.

        Parameters
        ----------
        {array-like, sparse matrix} of shape (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        """
        return self.model.predict(X)

    def get_prediction_probs(self, X):
        """
        Gets model proba.

        Parameters
        ----------
        X: {array-like, sparse matrix} of shape (n_samples, n_features),
            where ``n_samples`` is the number of samples and ``n_features`` is the number of features.

        """
        probs = []
        try:
            probs = self.model.predict_proba(X)
        except NotImplementedError:
            print("This classifier doesn't output probabilities")
        return probs

    def save_model(self, filename):
        """
        Save model.

        Parameters
        ----------
        filename: FileDescriptorOrPath

        """
        pickle.dump(self.model, open(filename, "wb"))

    def load_model(self, filename):
        """
        Load model.

        Parameters
        ----------
        filename: FileDescriptorOrPath

        """
        self.model = pickle.load(open(filename, "rb"))

    
    def get_model_name(self):
        """Get default model name."""
        return "default_target_model"


class GradientBoostingTargetModel(TargetModel):
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        super(GradientBoostingTargetModel, self).__init__()

    def get_model(self):
        return GradientBoostingClassifier(n_estimators=self.n_estimators)

    def get_model_name(self):
        return "gradient_boosting_n_estimators_" + str(self.n_estimators)


class RandomForestTargetModel(TargetModel):
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        super(RandomForestTargetModel, self).__init__()

    def get_model(self):
        return RandomForestClassifier(n_estimators=self.n_estimators)

    def get_model_name(self):
        return "random_forest_n_estimators_" + str(self.n_estimators)


class LogisticRegressionTargetModel(TargetModel):
    def __init__(self):
        super(LogisticRegressionTargetModel, self).__init__()

    def get_model(self):
        return LogisticRegression(max_iter=1000)

    def get_model_name(self):
        return "logistic_regression_max_iter_1000"


class SGDTargetModel(TargetModel):
    def __init__(self):
        super(SGDTargetModel, self).__init__()

    def get_model(self):
        return SGDClassifier(loss="log_loss", max_iter=1000)

    def get_model_name(self):
        return "sgd_max_iter_1000"


class MLPTargetModel(TargetModel):
    def __init__(self, hidden_layer_sizes=(100,)):
        self.hidden_layer_sizes = hidden_layer_sizes
        super(MLPTargetModel, self).__init__()

    def get_model(self):
        return MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes)

    def get_model_name(self):
        return "mlp_" + str(self.hidden_layer_sizes)
        