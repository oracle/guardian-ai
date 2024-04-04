#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import copy
import pickle
from dash_scatter_bias_mitigation import *
from guardian_ai.fairness.metrics import _positive_fairness_names

# Settings for plots
plt.rcParams["figure.figsize"] = [10, 7]
plt.rcParams["font.size"] = 15
sns.set(color_codes=True)
sns.set(font_scale=1.5)
sns.set_palette("bright")
sns.set_style("whitegrid")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container(
    [
        dbc.Row([html.H2("Dash is Awesome!")]),
    ]
)

possible_protected_attributes = {
    "adult": ["sex"],  #'race',
    "german": ["sex"],
    "acs_income": ["sex"],
    "acs_public_coverage": ["sex"],
    "acs_mobility": ["sex"],
    "acs_employment": ["sex"],
    "acs_travel_time": ["sex"],
}

fairness_metrics = [
    "TPR"
]  # ["TPR","statistical_parity","FPR","FNR","FOR","FDR","error_rate",] # "equalized_odds","theil_index"

datasets = [
    "adult"
]  # ['adult', 'german', 'acs_mobility', 'acs_income', 'acs_public_coverage', 'acs_employment', 'acs_travel_time']
dfm = {}
fmt = {}
base_trial = {}
metric_names = {}
initial_bars = {}
bars = {}
n_trials = 250
n_objs = ["2obj", "3obj"]
method_names = {"2obj": "Multiploer 2obj", "3obj": "Multiplier 3obj"}
model_names = {"2obj": "Two Objectives", "3obj": "Three Objectives"}
color_col = "Outcome Regression"
symbol_col = "symbol"
n_obj_col = "n_obj"
method_col = "method"

# These two optins are valid for bar plot type:
BAR_PLOT_TYPE = "change"  # 'outcome_rate'


def bar_plot(hover_data, cid):
    x_bar = [i for i in list(dfm[cid].columns) if "GroupFairness_" in i]
    # X axis values:
    x_bar_updated = [i.replace("GroupFairness_", "") for i in x_bar]

    if hover_data:
        # Y values for original model (before) and adjusted model (after)
        y_bar_before = [base_trial[cid][i] for i in x_bar]
        if "customdata" in hover_data["points"][0]:
            # Use customdata available in plotly figure when we called scatter plot
            # The customdata with index 2 is regarding the category type (n_obj_col) (hover_data['points'][0]['customdata'])
            # The customdata with index 1 is regarding index in dataframe (hover_data['points'][0]['customdata'][1])
            y_bar_after = [
                dfm[cid].loc[hover_data["points"][0]["customdata"][1], i] for i in x_bar
            ]
        else:
            y_bar_after = copy.deepcopy(y_bar_before)

        # Plot bar based on its type:
        if BAR_PLOT_TYPE == "change":
            y_sgn = 1 if (metric_names[cid] in _positive_fairness_names) else -1
            y_diff = [
                (y_bar_after[i] - y_bar_before[i]) * y_sgn
                for i in range(len(y_bar_before))
            ]
            bars[cid] = {
                "data": [
                    {
                        "x": x_bar_updated,
                        "y": y_diff,
                        "type": "bar",
                        "marker": {
                            "color": ["#B9503D" if i < 0 else "#7AB98D" for i in y_diff]
                        },
                    },
                ],
                "layout": {
                    "yaxis": {
                        "title": "Change in Outcomes",
                        "range": (-0.8, 0.8),
                    }  # 'range': (-0.5, 0.5)
                },
            }
        else:
            bars[cid] = {
                "data": [
                    {
                        "x": x_bar_updated,
                        "y": y_bar_before,
                        "type": "bar",
                        "name": "Original",
                    },
                    {
                        "x": x_bar_updated,
                        "y": y_bar_after,
                        "type": "bar",
                        "name": "Adjusted",
                    },
                ],
                "layout": {"yaxis": {"title": metric_names[cid], "range": (0, 1)}},
            }
        return bars[cid]
    else:
        if BAR_PLOT_TYPE == "change":
            initial_bars[cid] = {
                "data": [
                    {"x": x_bar_updated, "y": [0] * len(x_bar_updated), "type": "bar"},
                ],
                "layout": {"yaxis": {"title": "Change in Outcomes", "range": (-1, 1)}},
            }
        else:
            initial_bars[cid] = {
                "data": [
                    {
                        "x": x_bar_updated,
                        "y": [0] * len(x_bar_updated),
                        "type": "bar",
                        "name": "Original",
                    },
                    {
                        "x": x_bar_updated,
                        "y": [0] * len(x_bar_updated),
                        "type": "bar",
                        "name": "Adjusted",
                    },
                ],
                "layout": {"yaxis": {"title": metric_names[cid], "range": (-1, 1)}},
            }
    return initial_bars[cid]


scatter_fns = {"2obj": scatter_2obj, "3obj": scatter_3obj}
MULTIPLIER_VERSION = "3obj"  # '2obj'


def visualize_regression_dash(dataset_name, fairness_metric, bias_mitigated_model):
    cur_id = "graph_" + dataset_name + "_" + fairness_metric
    cur_sub_id = "sub_graph_" + dataset_name + "_" + fairness_metric

    metric_names[cur_id] = copy.deepcopy(fairness_metric)
    metric_name = metric_names[cur_id]
    accuracy_name = "balanced_accuracy"

    df = copy.deepcopy(bias_mitigated_model._best_trials_detailed)
    dfm[cur_id] = df
    dfm[cur_id][method_col] = method_names[MULTIPLIER_VERSION]
    base_trial[cur_id] = bias_mitigated_model._fairness_metric_trials_["base"]

    # True outcome is zero, False outcome is not:
    symbol_map = {
        "2obj": {True: "star-diamond", False: "square"},
        "3obj": {True: "star", False: "circle"},
    }
    outcome_is_zero = dfm[cur_id][color_col] == 0
    dfm[cur_id].loc[outcome_is_zero, symbol_col] = symbol_map[MULTIPLIER_VERSION][True]
    dfm[cur_id].loc[~outcome_is_zero, symbol_col] = symbol_map[MULTIPLIER_VERSION][
        False
    ]
    dfm[cur_id][n_obj_col] = MULTIPLIER_VERSION

    min_dict = {}
    max_dict = {}
    margin_dict = {}
    for i in [metric_name, accuracy_name, color_col]:
        min_dict[i] = min(dfm[cur_id][i])
        max_dict[i] = max(dfm[cur_id][i])
        margin_dict[i] = (max_dict[i] - min_dict[i]) / 10
    max_dict[color_col] = 0
    if min_dict[color_col] == 0:
        min_dict[color_col] = -1

    fig = go.Figure()

    # Color Bar
    marker_colorbar_dict = dict(
        colorbar=dict(
            thickness=15,
            title_font_size=12,
            title_text="Outcome<br>Regression<br>&nbsp;",
        ),
        cmin=min_dict[color_col],
        cmax=0,
        showscale=True,
    )

    fig.add_trace(
        scatter_fns[MULTIPLIER_VERSION](
            df=dfm[cur_id],
            fairness_metric_name=metric_name,
            accuracy_metric_name=accuracy_name,
            metric_col=metric_name,
            accuracy_col=accuracy_name,
            color_col=color_col,
            symbol_col=symbol_col,
            n_obj_col=n_obj_col,
            method_col=method_col,
        )
    )

    # fig.add_trace(
    #     scatter_fns[[MULTIPLIER_VERSION]](
    #         df=dfm[cur_id],
    #         bias_mitigated_model=bias_mitigated_model,
    #         metric_col=metric_name,
    #         accuracy_col=accuracy_name,
    #         color_col=color_col,
    #         symbol_col=symbol_col,
    #         n_obj_col=n_obj_col,
    #         model_name=model_names[MULTIPLIER_VERSION]
    #         )
    # )

    # Update colorbar & Original solution
    fig["data"][0]["marker"].update(marker_colorbar_dict)

    # Plot Original Solution
    fig.add_trace(scatter_original_solution(bias_mitigated_model))

    # Update layout & setting
    app.layout.children.append(
        dbc.Row([html.H3("Dataset: " + dataset_name + ", Metric: " + fairness_metric)]),
    )
    fig.update_traces(marker=dict(size=10))
    fig.update_coloraxes(
        colorbar_thickness=15,
        colorbar_title_font_size=14,
        colorbar_title_text="Outcome<br>Regression",
    )

    fig.update_layout(
        title=dict(
            text="Dataset: "
            + dataset_name
            + ", Model: "
            + model_names[MULTIPLIER_VERSION],
            y=0.9,
            x=0.5,
            xanchor="center",
            yanchor="top",
            font_size=18,
        ),
        xaxis_title=metric_name + " Disparity",
        yaxis_title="Accuracy",  # accuracy_name,
        legend_title=color_col,
        xaxis_range=[
            min_dict[metric_name] - margin_dict[metric_name],
            max_dict[metric_name] + margin_dict[metric_name],
        ],
        yaxis_range=[
            min_dict[accuracy_name] - margin_dict[accuracy_name],
            max_dict[accuracy_name] + margin_dict[accuracy_name],
        ],
        # autosize=False,
        # width=600,
        # height=500,
        font_size=14,
    )
    app.layout.children.append(
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id=cur_id, figure=fig), width=8),
                dbc.Col(
                    dcc.Graph(
                        id=cur_sub_id,
                    ),
                    width=4,
                ),
            ],
            justify="center",
        ),
    )

    @app.callback(
        Output(cur_sub_id, "figure"),
        Input(cur_id, "hoverData"),
        Input(cur_id, "id"),
    )
    def toggle_modal(hover_data, cid):
        return bar_plot(hover_data, cid)


if __name__ == "__main__":
    for metric in fairness_metrics:
        for dataset in datasets:
            print(dataset, metric)
            with open(
                "saved/250trials/"
                + MULTIPLIER_VERSION
                + "_bias_mitigated_"
                + dataset
                + "_prot"
                + str(len(possible_protected_attributes[dataset]))
                + "_"
                + metric
                + "_trial"
                + str(n_trials)
                + ".pkl",
                "rb",
            ) as f:
                bias_mitigated = pickle.load(f)
            visualize_regression_dash(dataset, metric, bias_mitigated)
    app.run_server(debug=True, use_reloader=False, port=8012)
