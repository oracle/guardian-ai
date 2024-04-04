#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2023, 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import pandas as pd
import os
import matplotlib.pyplot as plt


class ResultPlot:
    @staticmethod
    def print_best_attack(
        dataset_name: str,
        result_filename: str,
        graphs_dir: str,
        metric_to_sort_on: str = "attack_accuracy",
    ):
        """
        Given a result file, sort attack performance by the given metric and print out the
        best attacks for each dataset for each model.

        Parameters
        ----------
        dataset_name: str
            Name of the dataset.
        result_filename: str
            File in which all the attack results are stored.
        graphs_dir: str
            Directory to store the plotted graph (a table in this case).
        metric_to_sort_on: str
            Which metric to sort on. Assumes higher is better.

        Returns
        -------
        None
        """
        print("Plotting dataset: " + dataset_name)
        plt.figure()

        df = pd.read_csv(result_filename, sep="\t")

        rows_with_max = df.loc[
            df.groupby(["dataset", "target_model"])[metric_to_sort_on].idxmax()
        ]
        selected_cols = [
            "target_model",
            "train_f1",
            "test_f1",
            "attack_type",
            "attack_precision",
            "attack_recall",
            "attack_f1",
            "attack_accuracy",
        ]
        rows_with_max = rows_with_max[selected_cols]

        rows_with_max = rows_with_max.round(decimals=2)
        rows_with_max = rows_with_max.replace(regex=["_attack"], value="")
        rows_with_max = rows_with_max.replace(regex=["_black_box"], value="")
        rows_with_max = rows_with_max.replace(regex=["_with_merlin"], value="")

        cell_text = []
        for row in range(len(rows_with_max)):
            cell_text.append(rows_with_max.iloc[row])

        colColors = []
        for col in range(len(rows_with_max.columns)):
            colColors.append("lightgrey")

        colors = []
        for row in range(len(rows_with_max)):
            row_colors = []
            for col in range(len(rows_with_max.columns) - 1):
                row_colors.append("white")
            accuracy = rows_with_max.iloc[row][metric_to_sort_on]
            if accuracy < 0.55:
                accuracy_color = "white"
            elif accuracy < 0.70:
                accuracy_color = "yellow"
            else:
                accuracy_color = "red"
            row_colors.append(accuracy_color)
            colors.append(row_colors)

        table = plt.table(
            cellText=cell_text,
            cellColours=colors,
            colColours=colColors,
            colLabels=rows_with_max.columns,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        table.auto_set_column_width(col=list(range(len(rows_with_max.columns))))
        plt.axis("off")
        plt.title(dataset_name)

        plt.savefig(
            os.path.join(graphs_dir, str(dataset_name) + ".png"), bbox_inches="tight"
        )
        plt.clf()
