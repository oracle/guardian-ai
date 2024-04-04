#!/usr/bin/env python
# -*- coding: utf-8 -*--

# Copyright (c) 2024 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v 1.0 as shown at https://oss.oracle.com/licenses/upl/

import plotly.graph_objects as go
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def scatter_base(
    df,
    fairness_metric_name,
    accuracy_metric_name,
    metric_col,
    accuracy_col,
    color_col,
    symbol_col,
    n_obj_col,
    method_col,
    **kwargs,
):
    return go.Scatter(
        x=df[metric_col],
        y=df[accuracy_col],
        mode="markers",
        marker=dict(color=df[color_col], symbol=df[symbol_col], colorscale="Bluered_r"),
        customdata=np.stack(
            (df[color_col], df.index, df[n_obj_col], df[symbol_col], df[method_col]),
            axis=-1,
        ),
        hovertemplate="Symbol=%{customdata[3]}"
        + f"<br>{fairness_metric_name}"
        + " Disparity=%{x}"
        + f"<br>{accuracy_metric_name}"
        + "=%{y}"
        + "</br>Outcome Regression=%{customdata[0]}"
        + "<br>index=%{customdata[1]}"
        + "<br>model=%{customdata[4]}",
        showlegend=False,
    )


def scatter_2obj(
    df,
    fairness_metric_name,
    accuracy_metric_name,
    metric_col,
    accuracy_col,
    color_col,
    symbol_col,
    n_obj_col,
    method_col,
    **kwargs,
):
    fig = scatter_base(
        df,
        fairness_metric_name,
        accuracy_metric_name,
        metric_col,
        accuracy_col,
        color_col,
        symbol_col,
        n_obj_col,
        method_col,
        **kwargs,
    )
    fig["mode"] = "markers+lines"
    fig["line"] = dict(color="gray", dash="dash", width=1)
    fig["line_shape"] = (
        "vh" if False else "hv"
    )  # bias_mitigated_model._higher_fairness_is_better
    fig["marker"]["size"] = 10
    return fig


def scatter_3obj(
    df,
    fairness_metric_name,
    accuracy_metric_name,
    metric_col,
    accuracy_col,
    color_col,
    symbol_col,
    n_obj_col,
    method_col,
    **kwargs,
):
    fig = scatter_base(
        df,
        fairness_metric_name,
        accuracy_metric_name,
        metric_col,
        accuracy_col,
        color_col,
        symbol_col,
        n_obj_col,
        method_col,
        **kwargs,
    )
    fig["marker"]["line"] = dict(width=1, color="white")
    fig["marker"]["size"] = 10
    return fig


def scatter_original_solution(bias_mitigated_model):
    print(bias_mitigated_model._fairness_base_, bias_mitigated_model._accuracy_base_)
    return go.Scatter(
        x=[bias_mitigated_model._fairness_base_],
        y=[bias_mitigated_model._accuracy_base_],
        mode="markers",
        marker_symbol="cross",
        marker_color="green",
        marker_size=10,
        hovertemplate=f"{bias_mitigated_model.fairness_metric_name}"
        + " Disparity"
        + "=%{x:.4f}"
        + f"<br>{bias_mitigated_model.accuracy_metric_name}"
        + "=%{y:.4f}</br>",
        name="Base Estimator",
        showlegend=False,
    )
