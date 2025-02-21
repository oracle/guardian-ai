#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

from typing import List

import pandas as pd
from sklearn.model_selection import GridSearchCV


class AttackTuner:
    def __init__(self):
        pass

    def print_dataframe(self, filtered_cv_results):
        """
        Pretty print for filtered dataframe

        Parameters
        ----------
        filtered_cv_results: dict
            Dictionary record filtered results.

        Returns
        -------
        None

        """
        for (
            mean_precision,
            std_precision,
            mean_recall,
            std_recall,
            mean_f1,
            std_f1,
            params,
        ) in zip(
            filtered_cv_results["mean_test_precision"],
            filtered_cv_results["std_test_precision"],
            filtered_cv_results["mean_test_recall"],
            filtered_cv_results["std_test_recall"],
            filtered_cv_results["mean_test_f1"],
            filtered_cv_results["std_test_f1"],
            filtered_cv_results["params"],
        ):
            print(
                f"precision: {mean_precision:0.3f} (±{std_precision:0.03f}),"
                f" recall: {mean_recall:0.3f} (±{std_recall:0.03f}),"
                f" f1: {mean_f1:0.3f} (±{std_f1:0.03f}),"
                f" for {params}"
            )

    def refit_strategy_f1(self, cv_results):
        """Define the strategy to select the best estimator.

        The strategy defined here is to filter-out all results below a precision threshold
        of 0.5, rank the remaining by f1, and get the model with best f1

        Parameters
        ----------
        cv_results : dict of numpy (masked) ndarrays
            CV results as returned by the `GridSearchCV`.

        Returns
        -------
        best_index : int
            The index of the best estimator as it appears in `cv_results`.

        """
        # print the info about the grid-search for the different scores
        precision_threshold = 0.50

        cv_results_ = pd.DataFrame(cv_results)
        print("All grid-search results:")
        self.print_dataframe(cv_results_)

        # Filter-out all results below the threshold
        high_precision_cv_results = cv_results_[
            cv_results_["mean_test_precision"] >= precision_threshold
        ]

        print(f"Models with a precision higher than {precision_threshold}:")
        self.print_dataframe(high_precision_cv_results)

        high_precision_cv_results = high_precision_cv_results[
            [
                "mean_score_time",
                "mean_test_recall",
                "std_test_recall",
                "mean_test_precision",
                "std_test_precision",
                "rank_test_recall",
                "rank_test_precision",
                "mean_test_f1",
                "std_test_f1",
                "params",
            ]
        ]

        # From the best candidates, select the model with the best f1
        best_f1_high_precision_index = 0
        try:
            best_f1_high_precision_index = high_precision_cv_results["mean_test_f1"].idxmax()
            print(
                "\nThe selected final model with the best f1:\n"
                f"{high_precision_cv_results.loc[best_f1_high_precision_index]}"
            )
        except:
            print("Couldn't find optimal model")

        return best_f1_high_precision_index

    def refit_strategy(self, cv_results):
        """Define the strategy to select the best estimator.

        The strategy defined here is to filter-out all results below a precision threshold
        of 0.98, rank the remaining by recall and keep all models with one standard
        deviation of the best by recall. Once these models are selected, we can select the
        fastest model to predict.

        Parameters
        ----------
        cv_results : dict of numpy (masked) ndarrays
            CV results as returned by the `GridSearchCV`.

        Returns
        -------
        best_index : int
            The index of the best estimator as it appears in `cv_results`.

        """
        # print the info about the grid-search for the different scores
        precision_threshold = 0.5

        cv_results_ = pd.DataFrame(cv_results)
        print("All grid-search results:")
        self.print_dataframe(cv_results_)

        # Filter-out all results below the threshold
        high_precision_cv_results = cv_results_[
            cv_results_["mean_test_precision"] > precision_threshold
        ]

        print(f"Models with a precision higher than {precision_threshold}:")
        self.print_dataframe(high_precision_cv_results)

        high_precision_cv_results = high_precision_cv_results[
            [
                "mean_score_time",
                "mean_test_recall",
                "std_test_recall",
                "mean_test_precision",
                "std_test_precision",
                "rank_test_recall",
                "rank_test_precision",
                "params",
            ]
        ]

        # Select the most performant models in terms of recall
        # (within 1 sigma from the best)
        best_recall_std = high_precision_cv_results["mean_test_recall"].std()
        best_recall = high_precision_cv_results["mean_test_recall"].max()
        best_recall_threshold = best_recall - best_recall_std

        high_recall_cv_results = high_precision_cv_results[
            high_precision_cv_results["mean_test_recall"] > best_recall_threshold
        ]
        print(
            "Out of the previously selected high precision models, we keep all the\n"
            "the models within one standard deviation of the highest recall model:"
        )
        self.print_dataframe(high_recall_cv_results)

        # From the best candidates, select the fastest model to predict
        fastest_top_recall_high_precision_index = high_recall_cv_results["mean_score_time"].idxmin()

        print(
            "\nThe selected final model is the fastest to predict out of the previously\n"
            "selected subset of best models based on precision and recall.\n"
            "Its scoring time is:\n\n"
            f"{high_recall_cv_results.loc[fastest_top_recall_high_precision_index]}"
        )

        return fastest_top_recall_high_precision_index

    def tune_attack(self, classifier, X_train, y_train, threshold_grid: List[float]):
        """
        Tune a threshold based attack over a given grid.

        Parameters
        ----------
        classifier: ThresholdClassifier
            Threshold based classifier.
        X_train:  {array-like, sparse matrix} of shape (n_samples, n_features),
            where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.
            Input features for the set on which the attack is trained.
        y_train: ndarray of shape (n_samples,)
            Output labels for the set on which the attack is trained.
        threshold_grid: List[float]
            Grid to search over

        Returns
        -------
        float
            Best parameters (in this case, threshold).

        """
        tuned_parameters = [
            {"threshold": threshold_grid},
        ]

        scores = ["precision", "recall", "f1"]

        grid_search = GridSearchCV(
            classifier, tuned_parameters, scoring=scores, refit=self.refit_strategy_f1
        )
        grid_search.fit(X_train, y_train)

        return grid_search.best_params_
