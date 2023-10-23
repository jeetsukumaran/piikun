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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from piikun import plot
from piikun import utility


def visualize_support_cdf(
    distance_df,
    profile_df,
    palette="Portland",
):
    x = profile_df["label"].astype(str)
    y1 = profile_df["support"]
    y2 = profile_df["support"].cumsum()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=y1,
        yaxis="y1",
        marker=dict(color=y1, colorscale=palette + "_r")
    ))
    fig.add_trace(go.Scatter(
        x=x,
        y=y2,
        mode='lines+markers',
        yaxis="y2",
    ))
    # fig.add_trace(go.Scatter(
    #     x=x,
    #     y=y2,
    #     mode='lines+markers'
    # ))
    fig.update_layout(title="Support and Cumulative Support",
                  xaxis_title="Labels",
                  yaxis_title="Support",
                  yaxis2=dict(title="Cumulative Support", overlaying="y", side="right"))
    return fig


def visualize_distances_on_regionalized_support_space(
    distance_df,
    profile_df,
    palette="Geyser",
    support_quantiles=None,
    distance_quantiles=None,
    gradient_calibration="shared",
    is_log_scale=True,
):
    background_palette = palette
    scatterplot_palette = palette
    if not support_quantiles:
        support_quantiles = [0.25, 0.5, 0.75]
    if not distance_quantiles:
        distance_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    distance_df = distance_df.copy()
    if is_log_scale:
        distance_df["ptn1_support"] = np.log2(distance_df["ptn1_support"])
        distance_df["ptn2_support"] = np.log2(distance_df["ptn2_support"])

    padding = (distance_df["ptn1_support"].max() - distance_df["ptn1_support"].min()) * 0.05
    internal_thresholds = distance_df["ptn1_support"].quantile(support_quantiles).tolist()
    thresholds = [distance_df["ptn1_support"].min() - padding]
    thresholds.extend(internal_thresholds)
    thresholds.append(distance_df["ptn1_support"].max() + padding)
    bounds = np.array(thresholds)

    range_fns = []
    for idx, threshold in enumerate(thresholds[:-1]):
        next_threshold = thresholds[idx + 1]
        range_fns.append(
            lambda x, t1=threshold, t2=next_threshold: (x >= t1) & (x < t2)
        )

    n_ranges = len(range_fns)
    mean_values = np.zeros((n_ranges, n_ranges))

    bgdf = distance_df[distance_df["vi_distance"] > 1e-8]
    for i, ptn1_condition in enumerate(range_fns):
        for j, ptn2_condition in enumerate(range_fns):
            subset = bgdf[
                ptn1_condition(bgdf["ptn1_support"])
                & ptn2_condition(bgdf["ptn2_support"])
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
    #     for col in distance_df.columns():
    #         content.

    distance_df["scatter_hovertext"] = distance_df.apply(
        lambda row: f"ptn1: {row.ptn1}<br>ptn2: {row.ptn2}<br>ptn1_support: {row.ptn1_support}<br>ptn2_support: {row.ptn2_support}<br>vi_distance: {row.vi_distance}",
        axis=1,
    )

    # hover_data = list(distance_df.columns)
    fig.add_trace(
        go.Scattergl(
            x=distance_df["ptn1_support"],
            y=distance_df["ptn2_support"],
            mode="markers",
            marker=dict(
                color=distance_df["vi_distance"], colorscale=scatterplot_palette, size=6
            ),
            text=distance_df["scatter_hovertext"],
            hoverinfo="text",
        )
    )

    # Set axis labels and title
    fig.update_layout(
        xaxis_title="log(ptn1_support)",
        yaxis_title="log(ptn2_support)",
        title="Distances on Regionalized Support Space",
    )

    return fig


def visualize_scatter(
    df,
    x_key,
    y_key,
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
        x="ptn1_support",
        y="ptn2_support",
        color="vi_distance",
        labels={
            "ptn1_support": "log(ptn1_support)",
            "ptn2_support": "log(ptn2_support)",
        },
        hover_data=["vi_distance"],
        title="Scatterplot with Hover Annotations",
    )

    fig.update_traces(marker=dict(size=5), selector=dict(mode="markers"))

    return fig

def _log_info(*args, **kwargs):
    sys.stderr.write("[piikun-visualize] ")
    sys.stderr.write(*args, **kwargs)
    sys.stderr.write("\n")

class PlotGenerator:

    def __init__(
        self,
        output_directory=None,
        output_name_stem=None,
        output_formats=None,
        is_show_plot=False,
        is_save_plot=True,
    ):
        if output_directory:
            self.output_directory = pathlib.Path(output_directory)
        else:
            self.output_directory = pathlib.Path.cwd()
        if output_name_stem:
            self.output_name_stem = output_name_stem
        else:
            self.output_name_stem = "piikun"
        self.output_name_stem = output_name_stem
        if output_formats is None:
            self.output_formats = ["html", "jpg"]
        else:
            self.output_formats = output_formats
        self.is_show_plot = is_show_plot
        self.is_save_plot = is_save_plot


    def generate_plot(
        self,
        plot_name,
        plot_fn,
        plot_kwargs=None,
        post_plot_fn=None,
        plot_system="plotly",
    ):
        if plot_kwargs is None:
            plot_kwargs = {}
        fig = plot_fn(**plot_kwargs)
        if post_plot_fn:
            post_plot_fn(fig)
        if self.is_save_plot:
            for format_type in self.output_formats:
                if format_type == "html" and plot_system == "matplotlib":
                    continue
                if format_type.startswith("."):
                    ext = format_type
                else:
                    ext = f".{format_type}"
                output_filepath = self.output_directory / f"{self.output_name_stem}_{plot_name}{ext}"
                _log_info(f"- Saving to: '{output_filepath}'")
                if plot_system == "plotly":
                    if format_type == "html":
                        fig.write_html(output_filepath)
                    else:
                        fig.write_image(output_filepath)
                elif plot_system == "matplotlib":
                    if format_type == "html":
                        pass
                    else:
                        plt.savefig(
                            output_filepath,
                            format=format_type,
                        )
                else:
                    raise ValueError(plot_system)

        if self.is_show_plot:
            if plot_system == "plotly":
                fig.show()
            elif plot_system == "matplotlib":
                plt.show()
            else:
                raise ValueError(plot_system)

def main(args=None):
    visualization_types = {
        # "partition-support-profile": {
        #     "plot_fn": visualize_scatter,
        # }
        "distance-vs-support-quantiles": {
            "plot_fn": plot.visualize_distances_on_quantized_support_space,
            "plot_system": "matplotlib",
        },
        "distance-vs-support-regions": {
            "plot_fn": visualize_distances_on_regionalized_support_space,
        },
        "cumulative-support": {
            "plot_fn": visualize_support_cdf,
        },
    }
    visualization_types_str = ", ".join(f"'{vkey}'" for vkey in visualization_types)

    parent_parser = argparse.ArgumentParser()
    data_source_options = parent_parser.add_argument_group("Data Source Options")
    data_source_options.add_argument(
        "src_path",
        action="append",
        metavar="DATAFILE",
        nargs="+",
        help="Path to data source file(s).",
    )
    visualization_options = parent_parser.add_argument_group("Visualization Types")
    visualization_options.add_argument(
        "-v", "--visualize",
        metavar="<NAME>",
        dest="selected_visualizations",
        action="append",
        help=(f"One or more of the following visualization types: {visualization_types_str}. Default is 'all'."),
    )
    plot_options = parent_parser.add_argument_group("Plot Options")
    plot_options.add_argument(
        "--palette",
        action="store",
        default="Geyser",
        help=("Palette or color scheme." " [default='%(default)s']."),
    )
    output_options = parent_parser.add_argument_group("Output Options")
    output_options.add_argument(
        "-o",
        "--output-title",
        action="store",
        default=None,
        help="Custom prefix for output filenames (defaults to first source filepath stem).",
    )
    output_options.add_argument(
        "-O",
        "--output-directory",
        action="store",
        default=os.curdir,
        help="Directory for output files [default='%(default)s'].",
    )
    output_options.add_argument(
        "-F",
        "--output-format",
        action="append",
        nargs="+",
        default=None,
        help="Output format (multiple formats may be specified; defaults to 'html' and 'jpg').",
    )
    output_options.add_argument(
        "--show-plot",
        action=argparse.BooleanOptionalAction,
        dest="is_show_plot",
        default=False,
        help="Show / do not show plots in interactive viewer",
    )
    output_options.add_argument(
        "--save-plot",
        action=argparse.BooleanOptionalAction,
        dest="is_save_plot",
        default=True,
        help="Save / do not save plots in interactive viewer",
    )
    args = parent_parser.parse_args(args)
    src_paths = [i for sublist in args.src_path for i in sublist]
    distance_df = utility.read_files_to_dataframe(filepaths=src_paths, format_type="json")
    output_format = []
    if args.output_format:
        for fspecs in args.output_format:
            for fspec in fspecs:
                output_format.append(fspec)
    else:
        output_format.append("html")
        output_format.append("jpg")
    plotter = PlotGenerator(
        is_show_plot = args.is_show_plot,
        is_save_plot = args.is_save_plot,
        output_directory = args.output_directory,
        output_name_stem = pathlib.Path(src_paths[0]).stem,
        output_formats = output_format,
    )
    if args.selected_visualizations:
        visualizations = list(args.selected_visualizations)
    else:
        visualizations = list(visualization_types)
    profile_df = utility.extract_profile(
        df=distance_df,
        key_col="ptn1",
        prop_col_filter_fn=lambda x: x.startswith("ptn1"),
    )
    common_plot_kwargs = {
        "profile_df": profile_df,
        "distance_df": distance_df,
        "palette": args.palette,
    }

    for visualization_type_key in visualizations:
        visualization_name = visualization_type_key
        visualization_d = visualization_types[visualization_type_key]
        _log_info(f"Visualization: {visualization_name}")
        plot_kwargs = dict(common_plot_kwargs)
        if "plot_kwargs" in visualization_d:
            plot_kwargs.update(visualization_d["plot_kwargs"])
        plotter.generate_plot(
            plot_name = visualization_name,
            plot_fn = visualization_d["plot_fn"],
            plot_kwargs = plot_kwargs,
            plot_system = visualization_d.get("plot_system", "plotly")
        )


if __name__ == "__main__":
    main()
