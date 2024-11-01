#! /usr/bin/env python
# -*- coding: utf-8 -*-

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
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from piikun import plot
from piikun import utility
from piikun import runtime

import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_distance_correlations(
    distance_df,
    profile_df,
    palette="RdYlBu",
):
    """Create correlation matrix heatmap between different distance metrics"""
    distance_cols = [
        'vi_mi', 'vi_joint_entropy', 'vi_distance',
        'vi_normalized_kraskov', 'mirkin_metric', 'ahrens_match_ratio'
    ]
    corr_matrix = distance_df[distance_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = sns.heatmap(
        corr_matrix,
        annot=True,
        # cmap=palette, # <== k
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax
    )
    plt.title('Correlations Between Distance Metrics')
    return fig

def visualize_distance_distributions(
    distance_df,
    profile_df,
    palette="Geyser",
):
    """Create violin plots showing distribution of each distance metric"""
    distance_cols = [
        'vi_mi', 'vi_joint_entropy', 'vi_distance',
        'vi_normalized_kraskov', 'mirkin_metric', 'ahrens_match_ratio'
    ]

    # Create plotly figure
    fig = go.Figure()

    for metric in distance_cols:
        fig.add_trace(go.Violin(
            y=distance_df[metric],
            name=metric,
            box_visible=True,
            meanline_visible=True,
        ))

    fig.update_layout(
        title="Distribution of Distance Metrics",
        xaxis_title="Distance Metric",
        yaxis_title="Value",
        showlegend=False,
    )
    return fig

def visualize_distance_matrices(
    distance_df,
    profile_df,
    palette="Viridis",
):
    """Create heatmaps showing pairwise distances for each metric"""
    distance_cols = [
        'vi_mi', 'vi_joint_entropy', 'vi_distance',
        'vi_normalized_kraskov', 'mirkin_metric', 'ahrens_match_ratio'
    ]

    fig = go.Figure()

    # Create subplot grid
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=distance_cols
    )

    for idx, metric in enumerate(distance_cols):
        row = idx // 3 + 1
        col = idx % 3 + 1

        # Reshape data into matrix form
        matrix = distance_df.pivot(
            index='ptn1',
            columns='ptn2',
            values=metric
        )

        fig.add_trace(
            go.Heatmap(
                z=matrix,
                colorscale=palette,
                showscale=True,
                name=metric
            ),
            row=row, col=col
        )

    fig.update_layout(
        title="Distance Matrices by Metric",
        height=800,
        showlegend=False,
    )

    return fig


def visualize_score_cdf(
    distance_df,
    profile_df,
    palette="Portland",
):
    x = profile_df["label"].astype(str)
    if "score" not in profile_df.columns:
        raise utility.UnavailableFieldException("ptn1_score")
    y1 = profile_df["score"]
    y2 = profile_df["score"].cumsum()
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


def visualize_distances_on_regionalized_score_space(
    distance_df,
    profile_df,
    palette="Geyser",
    score_quantiles=None,
    distance_quantiles=None,
    gradient_calibration="shared",
    is_log_scale=True,
):
    background_palette = palette
    scatterplot_palette = palette
    if not score_quantiles:
        score_quantiles = [0.25, 0.5, 0.75]
    if not distance_quantiles:
        distance_quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    distance_df = distance_df.copy()
    if "ptn1_score" not in distance_df.columns:
        raise utility.UnavailableFieldException("ptn1_score")
    if is_log_scale:
        distance_df["ptn1_score"] = distance_df["ptn1_score"].clip(lower=1e-9)
        distance_df["ptn2_score"] = distance_df["ptn2_score"].clip(lower=1e-9)
        distance_df["ptn1_score"] = np.log2(distance_df["ptn1_score"])
        distance_df["ptn2_score"] = np.log2(distance_df["ptn2_score"])

    padding = (distance_df["ptn1_score"].max() - distance_df["ptn1_score"].min()) * 0.05
    internal_thresholds = distance_df["ptn1_score"].quantile(score_quantiles).tolist()
    thresholds = [distance_df["ptn1_score"].min() - padding]
    thresholds.extend(internal_thresholds)
    thresholds.append(distance_df["ptn1_score"].max() + padding)
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
                ptn1_condition(bgdf["ptn1_score"])
                & ptn2_condition(bgdf["ptn2_score"])
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
        lambda row: f"ptn1: {row.ptn1}<br>ptn2: {row.ptn2}<br>ptn1_score: {row.ptn1_score}<br>ptn2_score: {row.ptn2_score}<br>vi_distance: {row.vi_distance}",
        axis=1,
    )

    # hover_data = list(distance_df.columns)
    fig.add_trace(
        go.Scattergl(
            x=distance_df["ptn1_score"],
            y=distance_df["ptn2_score"],
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
        xaxis_title="log(ptn1_score)",
        yaxis_title="log(ptn2_score)",
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
    if not score_quantiles:
        score_quantiles = [0.25, 0.5, 0.75]
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
        df["ptn1_score"] = np.log2(df["ptn1_score"])
        df["ptn2_score"] = np.log2(df["ptn2_score"])

    fig = px.scatter(
        df,
        x="ptn1_score",
        y="ptn2_score",
        color="vi_distance",
        labels={
            "ptn1_score": "log(ptn1_score)",
            "ptn2_score": "log(ptn2_score)",
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
        output_formats=None,
        is_show_plot=False,
        is_save_plot=True,
        runtime_context=None,
    ):
        if output_formats is None:
            self.output_formats = ["html", "jpg"]
        else:
            self.output_formats = output_formats
        self.is_show_plot = is_show_plot
        self.is_save_plot = is_save_plot
        self.runtime_context = runtime_context

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
                # output_filepath = self.output_directory / f"{self.output_name_stem}_{plot_name}{ext}"
                output_filepath = self.runtime_context.compose_output_path(subtitle=plot_name, ext=format_type)
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
        # "partition-score-profile": {
        #     "plot_fn": visualize_scatter,
        # }
        "distance-vs-score-quantiles": {
            "plot_fn": plot.visualize_value_on_quantized_space,
            "plot_system": "matplotlib",
        },
        "distance-vs-score-regions": {
            "plot_fn": visualize_distances_on_regionalized_score_space,
        },
        "cumulative-score": {
            "plot_fn": visualize_score_cdf,
        },
        "distance-correlations": {
            "plot_fn": visualize_distance_correlations,
            "plot_system": "matplotlib",
        },
        "distance-distributions": {
            "plot_fn": visualize_distance_distributions,
            "plot_system": "plotly",
        },
        "distance-matrices": {
            "plot_fn": visualize_distance_matrices,
            "plot_system": "plotly",
        },
    }
    visualization_types_str = ", ".join(f"'{vkey}'" for vkey in visualization_types)

    parent_parser = argparse.ArgumentParser()
    data_source_options = parent_parser.add_argument_group("Data Source Options")
    data_source_options.add_argument(
        "source_paths",
        action="store",
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
    runtime_context = runtime.RuntimeContext()
    runtime_context.logger.info("Starting: [b]piikun-visualize[/b]")
    source_paths = args.source_paths
    runtime_context.output_directory = args.output_directory
    runtime_context.compose_output_title(
        output_title=args.output_title,
        source_paths=source_paths,
        is_merge_output=True,
    )
    distance_df = utility.read_files_to_dataframe(
        filepaths=source_paths,
        format_type="json",
    )
    output_format = []
    if args.output_format:
        for fspecs in args.output_format:
            for fspec in fspecs:
                output_format.append(fspec)
    else:
        output_format.append("html")
        output_format.append("jpg")

    # plotter_old = plot.Plotter(runtime_context=runtime_context)
    # plotter_old.plot_partition_score_cdf()
    # plotter_old.plot_partition_profile_comparison()
    # plotter_old.plot_size_entropy_score()
    # plotter_old.plot_size_vs_score()
    # plotter_old.plot_entropy_vs_score()
    # plotter.plot_clustermaps()

    plotter = PlotGenerator(
        is_show_plot = args.is_show_plot,
        is_save_plot = args.is_save_plot,
        runtime_context=runtime_context,
        # output_directory = args.output_directory,
        # output_name_stem = pathlib.Path(src_paths[0]).stem,
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
    distance_df["score"] = distance_df["vi_distance"]
    for visualization_type_key in visualizations:
        visualization_name = visualization_type_key
        visualization_d = visualization_types[visualization_type_key]
        _log_info(f"Visualization: {visualization_name}")
        plot_kwargs = dict(common_plot_kwargs)
        if "plot_kwargs" in visualization_d:
            plot_kwargs.update(visualization_d["plot_kwargs"])
        try:
            plotter.generate_plot(
                plot_name = visualization_name,
                plot_fn = visualization_d["plot_fn"],
                plot_kwargs = plot_kwargs,
                plot_system = visualization_d.get("plot_system", "plotly")
            )
        except utility.UnavailableFieldException as e:
            _log_info(f"- Required data field unavailable: {e}")


if __name__ == "__main__":
    main()
