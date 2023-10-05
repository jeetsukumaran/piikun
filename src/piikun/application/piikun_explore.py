#! /usr/bin/env python
# -*- coding: utf-8 -*-

##############################################################################
## Copyright (c) 2023 Jeet Sukumaran.
## All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
##     * Redistributions of source code must retain the above copyright
##       notice, this list of conditions and the following disclaimer.
##     * Redistributions in binary form must reproduce the above copyright
##       notice, this list of conditions and the following disclaimer in the
##       documentation and/or other materials provided with the distribution.
##     * The names of its contributors may not be used to endorse or promote
##       products derived from this software without specific prior written
##       permission.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
## ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
## WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL JEET SUKUMARAN BE LIABLE FOR ANY DIRECT,
## INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
## BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
## DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
## LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
## OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
## ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##
##############################################################################

import os
import pathlib
import sys
import argparse
import json
import math
import functools

from piikun import utility

import numpy as np
import pandas as pd
import plotly.express as px


import plotly.graph_objects as go

import pandas as pd
import numpy as np
import plotly.graph_objects as go

def visualize_distances_on_regionalized_support_space_plotly(
    df,
    support_quantiles=None,
    distance_quantiles=None,
    gradient_calibration="shared",
    background_palette="Viridis",
    scatterplot_palette="Viridis",
    is_log_scale=True,
):
    if not support_quantiles:
        support_quantiles = [0.25, 0.5, 0.75]
    if not distance_quantiles:
        distance_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    df = df.copy()
    if is_log_scale:
        df["ptn1_support"] = np.log2(df["ptn1_support"])
        df["ptn2_support"] = np.log2(df["ptn2_support"])

    padding = (df["ptn1_support"].max() - df["ptn1_support"].min()) * 0.05
    internal_thresholds = df["ptn1_support"].quantile(support_quantiles).tolist()
    thresholds = [df["ptn1_support"].min() - padding]
    thresholds.extend(internal_thresholds)
    thresholds.append(df["ptn1_support"].max() + padding)
    bounds = np.array(thresholds)

    range_fns = []
    for idx, threshold in enumerate(thresholds[:-1]):
        next_threshold = thresholds[idx + 1]
        range_fns.append(
            lambda x, t1=threshold, t2=next_threshold: (x >= t1) & (x < t2)
        )

    n_ranges = len(range_fns)
    mean_values = np.zeros((n_ranges, n_ranges))

    bgdf = df[df["vi_distance"] > 1e-8]
    for i, ptn1_condition in enumerate(range_fns):
        for j, ptn2_condition in enumerate(range_fns):
            subset = bgdf[
                ptn1_condition(bgdf["ptn1_support"]) & ptn2_condition(bgdf["ptn2_support"])
            ]
            # mean_values[n_ranges - 1 - i, j] = subset["vi_distance"].mean()
            mean_values[i, j] = subset["vi_distance"].mean()

    # Create the plotly plot
    fig = go.Figure()

    # Add a heatmap for the background
    fig.add_trace(
        go.Heatmap(
            z=mean_values,
            x=bounds,
            y=bounds,
            colorscale=background_palette,
            colorbar=dict(title="vi_distance"),
            opacity=0.8,
        )
    )

    # def scatter_annotation_fn(row):
    #     content = ""
    #     for col in df.columns():
    #         content.

    df['scatter_hovertext'] = df.apply(
        lambda row: f'ptn1: {row.ptn1}<br>ptn2: {row.ptn2}<br>ptn1_support: {row.ptn1_support}<br>ptn2_support: {row.ptn2_support}<br>vi_distance: {row.vi_distance}',
        axis=1
    )

    fig.add_trace(
        go.Scatter(
            x=df["ptn1_support"],
            y=df["ptn2_support"],
            mode="markers",
            marker=dict(
                color=df["vi_distance"],
                colorscale=scatterplot_palette,
                size=6
            ),
            text=df['scatter_hovertext'],
            hoverinfo='text'
        )
    )

    # Set axis labels and title
    fig.update_layout(
        xaxis_title="log(ptn1_support)",
        yaxis_title="log(ptn2_support)",
        title="Visualize Distances on Regionalized Support Space"
    )

    fig.show()
    return fig


def visualize_scatter(
    df,
    support_quantiles=None,
    distance_quantiles=None,
    gradient_calibration="shared",
    is_log_scale=True,
):
    if not support_quantiles:
        support_quantiles = [0.25, 0.5, 0.75]
    if not distance_quantiles:
        distance_quantiles = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
        ]
    df = df.copy()
    if is_log_scale:
        df["ptn1_support"] = np.log2(df["ptn1_support"])
        df["ptn2_support"] = np.log2(df["ptn2_support"])

    fig = px.scatter(
        df,
        x='ptn1_support',
        y='ptn2_support',
        color='vi_distance',
        labels={"ptn1_support": "log(ptn1_support)", "ptn2_support": "log(ptn2_support)"},
        hover_data=['vi_distance'],
        title="Scatterplot with Hover Annotations"
    )

    fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))

    return fig


def main(args=None):
    parent_parser = argparse.ArgumentParser()
    input_options = parent_parser.add_argument_group("Input Options")
    input_options.add_argument(
        "src_path",
        action="store",
        metavar="FILE",
        nargs="+",
        help="Path to data source file.",
    )
    args = parent_parser.parse_args(args)
    df = utility.read_files_to_dataframe(filepaths=args.src_path)
    fig = visualize_distances_on_regionalized_support_space_plotly(
        df=df,
        # background_palette="Plasma",
        # scatterplot_palette="Plasma",
        # background_palette="Turbo",
        # scatterplot_palette="Turbo",
        # background_palette="Electric",
        # scatterplot_palette="electric",
         background_palette="Portland",
        scatterplot_palette="Portland",
        )
    visualization_name = "distance-vs-support-quantiles"
    format = "html"
    output_filepath = f"{pathlib.Path(args.src_path[0]).stem}_{visualization_name}.{format}"
    fig.write_html(output_filepath)

if __name__ == "__main__":
    main()


