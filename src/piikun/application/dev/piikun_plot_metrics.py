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
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from matplotlib.colors import LogNorm, Normalize
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull, Delaunay
from matplotlib.path import Path
from scipy.stats import binned_statistic_2d
from matplotlib import cm
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap

import yakherd
from piikun import partitionmodel
from piikun import utility
from piikun import plot

# def mirror_dataframe(df, col1, col2, col3):
#     """
#     Create a mirrored dataframe based on the input columns.

#     Parameters:
#     df (pd.DataFrame): Input dataframe.
#     col1 (str): The first column to be mirrored.
#     col2 (str): The second column to be mirrored.
#     col3 (str): The third column to be included.

#     Returns:
#     pd.DataFrame: The resulting mirrored dataframe.
#     """

#     df_1 = df[[col1, col2, col3]].copy()
#     df_2 = df_1.copy()
#     df_2[[col1, col2]] = df_2[[col2, col1]]

#     # Append df_mirror to the relevant columns of df
#     df_final = pd.concat([df_1, df_2], ignore_index=True)
#     return df_final

# def scatter_plot(X, Y, Z, xscale='log', yscale='log', zscale='linear'):
#     # Create a DataFrame from the vectors
#     import pandas as pd
#     df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})

#     # If zscale is log, transform Z values
#     if zscale == 'log':
#         df['Z'] = np.log1p(df['Z'])

#     # Create the scatter plot using Seaborn
#     plt.figure(figsize=(10, 8))
#     sns.scatterplot(x='X', y='Y', size='Z', sizes=(20, 200), data=df)

#     # Set the scales for X and Y axes
#     plt.xscale(xscale)
#     plt.yscale(yscale)

# def x_heatmap_plot(
#     df,
#     config_d,
# ):

#     pf1_key = "ptn1_support"
#     pf2_key = "ptn2_support"
#     dist_key = "vi_distance"
#     X = df[pf1_key]
#     Y = df[pf2_key]
#     Z = df[dist_key]

#     xscale = config_d.get("xscale", "log")
#     yscale = config_d.get("yscale", "log")
#     colormap = config_d.get("colormap", "coolwarm")
#     zscale = config_d.get("zscale", "linear")
#     zero_heat_values = config_d.get("zero_heat_values", 'mask')
#     separate_zero_heat_color = config_d.get("separate_zero_heat_color")
#     num_bins = config_d.get("num_bins", 100)
#     if zero_heat_values == 'clip':
#         if zscale == 'log':
#             Z = Z.clip(lower=1e-10)
#     elif zero_heat_values == 'mask':
#         mask_zero = Z == 0
#         Z[mask_zero] = -1

#     stat, x_edge, y_edge, binnumber = binned_statistic_2d(
#         X,
#         Y,
#         Z,
#         statistic='mean',
#         bins=num_bins
#     )

#     stat = np.nan_to_num(stat, nan=1e-10)

#     if zero_heat_values == 'mask':
#         stat[stat == -1] = -1

#     plt.figure(figsize=(10, 8))

#     if zero_heat_values == 'mask':
#         base_cmap = cm.get_cmap(colormap)
#         custom_colors = [base_cmap(i) for i in range(base_cmap.N)]
#         custom_colors.insert(0, separate_zero_heat_color if separate_zero_heat_color else (1, 1, 1, 0))
#         custom_cmap = ListedColormap(custom_colors)
#         cmap = custom_cmap
#         if zscale == 'log':
#             norm = BoundaryNorm([stat.min()] + np.logspace(np.log10(1e-10), np.log10(stat.max()), 100), custom_cmap.N)
#         else:
#             norm = BoundaryNorm([stat.min()] + np.linspace(stat.min(), stat.max(), 100), custom_cmap.N)
#     else:
#         cmap = cm.get_cmap(colormap)
#         norm = LogNorm(vmin=stat.min(), vmax=stat.max()) if zscale == 'log' else plt.Normalize(stat.min(), stat.max())

#     plt.imshow(stat.T, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap, norm=norm, aspect='auto')

#     plt.xscale(xscale)
#     plt.yscale(yscale)

#     plt.xlim([X.min(), X.max()])
#     plt.ylim([Y.min(), Y.max()])

#     plt.xlabel(X.name, fontsize=12)
#     plt.ylabel(Y.name, fontsize=12)

#     m = cm.ScalarMappable(cmap=cmap, norm=norm)
#     m.set_array(stat)
#     plt.colorbar(m, shrink=0.5, aspect=5)


# # Updated heatmap_plot_by_support function working with copies to avoid warnings

# def x_heatmap_plot(df, config_d):
#     pf1_key = "ptn1_support"
#     pf2_key = "ptn2_support"
#     dist_key = "vi_normalized_kraskov"

#     # Work with copies of the series to avoid modifying the original DataFrame
#     X = df[pf1_key].copy()
#     Y = df[pf2_key].copy()
#     Z = df[dist_key].copy()

#     xscale = config_d.get("xscale", "log")
#     yscale = config_d.get("yscale", "log")
#     colormap = config_d.get("colormap", "coolwarm")
#     zscale = config_d.get("zscale", "linear")
#     zero_heat_values = config_d.get("zero_heat_values", 'mask')
#     separate_zero_heat_color = config_d.get("separate_zero_heat_color")
#     num_bins = config_d.get("num_bins", 100)
#     if zero_heat_values == 'clip':
#         if zscale == 'log':
#             Z = Z.clip(lower=1e-10)
#     elif zero_heat_values == 'mask':
#         mask_zero = Z == 0
#         Z[mask_zero] = -1

#     stat, x_edge, y_edge, binnumber = binned_statistic_2d(
#         X,
#         Y,
#         Z,
#         statistic='mean',
#         bins=num_bins
#     )

#     stat = np.nan_to_num(stat, nan=1e-10)

#     if zero_heat_values == 'mask':
#         stat[stat == -1] = -1

#     plt.figure(figsize=(10, 8))

#     if zero_heat_values == 'mask':
#         base_cmap = cm.get_cmap(colormap)
#         custom_colors = [base_cmap(i) for i in range(base_cmap.N)]
#         custom_colors.insert(0, separate_zero_heat_color if separate_zero_heat_color else (1, 1, 1, 0))
#         custom_cmap = ListedColormap(custom_colors)
#         cmap = custom_cmap
#         if zscale == 'log':
#             norm = BoundaryNorm([stat.min()] + np.logspace(np.log10(1e-10), np.log10(stat.max()), 100), custom_cmap.N)
#         else:
#             norm = BoundaryNorm([stat.min()] + np.linspace(stat.min(), stat.max(), 100), custom_cmap.N)
#     else:
#         cmap = cm.get_cmap(colormap)
#         norm = LogNorm(vmin=stat.min(), vmax=stat.max()) if zscale == 'log' else plt.Normalize(stat.min(), stat.max())

#     plt.imshow(stat.T, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', cmap=cmap, norm=norm, aspect='auto')

#     plt.xscale(xscale)
#     plt.yscale(yscale)

#     plt.xlim([X.min(), X.max()])
#     plt.ylim([Y.min(), Y.max()])

#     plt.xlabel(X.name, fontsize=12)
#     plt.ylabel(Y.name, fontsize=12)

#     m = cm.ScalarMappable(cmap=cmap, norm=norm)
#     m.set_array(stat)
#     plt.colorbar(m, shrink=0.5, aspect=5)

# def x_heatmap_plot(df, config_d):
#     # Define the keys for the pivot table
#     pf1_key = "ptn1"
#     pf2_key = "ptn2"
#     dist_key = "vi_distance"

#     # Filter out self-comparison
#     df = df[df[pf1_key] != df[pf2_key]]

#     # Pivot the data to create a matrix of VI distances
#     pivot_table = df.pivot(index=pf1_key, columns=pf2_key, values=dist_key)

#     # Create a symmetrical matrix by combining the pivot table with its transpose
#     symmetrical_matrix = pivot_table.combine_first(pivot_table.T)

#     # Plot configuration
#     colormap = config_d.get("colormap", "coolwarm")
#     cmap = cm.get_cmap(colormap)
#     norm = plt.Normalize(symmetrical_matrix.min().min(), symmetrical_matrix.max().max())

#     # Plot the heatmap
#     plt.imshow(symmetrical_matrix, cmap=cmap, norm=norm, aspect='auto')

#     # Iterate through the cells and annotate the plot
#     for i in range(symmetrical_matrix.shape[0]):
#         for j in range(symmetrical_matrix.shape[1]):
#             heat_value = symmetrical_matrix.iloc[i, j]

#             # Skip annotating if ptn1 equals ptn2, the heat value is 0, or the cell is null
#             if i == j or heat_value == 0 or np.isnan(heat_value):
#                 continue

#             # Annotate the plot with ptn1, ptn2, and heat value, with a transparent background
#             annotation_text = f'VI({symmetrical_matrix.index[i]}, {symmetrical_matrix.columns[j]})=\n{heat_value:.2f}'
#             plt.annotate(annotation_text, (i, j), fontsize=6, va='center', ha='center',
#                          bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="none"))

#     # Axis labels and colorbar
#     plt.xticks(range(len(symmetrical_matrix.columns)), symmetrical_matrix.columns)
#     plt.yticks(range(len(symmetrical_matrix.index)), symmetrical_matrix.index)
#     plt.xlabel(pf1_key, fontsize=5)
#     plt.ylabel(pf2_key, fontsize=5)
#     plt.colorbar(cmap=cmap, norm=norm, shrink=0.5, aspect=5)

# def x_contour_plot(
#     X,
#     Y,
#     Z,
#     num_points=100,
#     xscale='linear',
#     yscale='linear',
#     colormap='coolwarm',
#     overlay_hexbin=False
# ):
#     xi = np.linspace(X.min(), X.max(), num_points)
#     yi = np.linspace(Y.min(), Y.max(), num_points)
#     zi = griddata((X, Y), Z, (xi[None,:], yi[:,None]), method='linear')

#     # Generate a mask for the interpolated grid
#     tri = Delaunay(np.vstack((X, Y)).T)
#     X_grid, Y_grid = np.meshgrid(xi, yi)
#     mask = tri.find_simplex(np.vstack((X_grid.flatten(), Y_grid.flatten())).T) < 0
#     mask = mask.reshape(X_grid.shape)
#     zi_masked = np.ma.array(zi, mask=mask)

#     plt.figure()
#     plt.contour(xi, yi, zi_masked, levels=14, linewidths=0.5, colors='k')
#     cntr = plt.contourf(xi, yi, zi_masked, levels=14, cmap=colormap)
#     # Overlay hexbin plot if overlay_hexbin is True
#     if overlay_hexbin:
#         plt.hexbin(X, Y, C=Z, gridsize=50, cmap=colormap, reduce_C_function=np.mean, mincnt=1, xscale=xscale, yscale=yscale)
#     plt.xscale(xscale)
#     plt.yscale(yscale)


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
    plot_options = parent_parser.add_argument_group("Plot Options")
    plot_options.add_argument(
            "--num-support-vs-distance-hue-bins",
            dest="num_support_vs_distance_bins",
            action="store",
            type=int,
            default=25,
            # help="Number of (probability) bins; if '0' or 'None', number of bins set to number of partitions [default=%(default)s].")
            help="Number of bins for distance color gradient.")
    plot_options.add_argument(
            "--add-jitter",
            dest="is_jitter_support",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Add / do not add small noise to separate identical support value when plotting.",
        )
    output_options = parent_parser.add_argument_group("Output Options")
    output_options.add_argument(
        "-o",
        "--output-title",
        action="store",
        default="piikun",
        help="Prefix for output filenames [default='%(default)s'].",
    )
    output_options.add_argument(
        "-O",
        "--output-directory",
        action="store",
        default=os.curdir,
        help="Directory for output files [default='%(default)s'].",
    )
    output_options.add_argument(
            "-F", "--output-format",
            action="append",
            default=None,
            help="Output format [default='jpg'].")
    output_options.add_argument(
            "--dpi",
            action="append",
            default=300,
            help="DPI [default=%(default)s].")
    # cluster_plot_options = parent_parser.add_argument_group("Cluster Plot Options")
    # cluster_plot_options.add_argument(
    #     "--cluster-rows",
    #     action=argparse.BooleanOptionalAction,
    #     dest="is_cluster_rows",
    #     default=False,
    #     help="Reorder / do not reorder partition rows to show clusters clearly",
    # )
    # cluster_plot_options.add_argument(
    #     "--cluster-cols",
    #     action=argparse.BooleanOptionalAction,
    #     dest="is_cluster_cols",
    #     default=False,
    #     help="Reorder / do not reorder partition colums to show clusters clearly",
    # )

    logger_configuration_parser = yakherd.LoggerConfigurationParser(name="piikun")
    logger_configuration_parser.attach(parent_parser)
    logger_configuration_parser.console_logging_parser_group.add_argument(
        "--progress-report-frequency",
        type=int,
        action="store",
        help="Frequency of progress reporting.",
    )
    args = parent_parser.parse_args(args)
    if not args.output_format:
        args.output_format = ["jpg"]
    config_d = dict(vars(args))
    logger = logger_configuration_parser.get_logger(args_d=config_d)
    runtime_context = utility.RuntimeContext(
        logger=logger,
        random_seed=None,
        output_directory=args.output_directory,
        output_title=args.output_title,
        output_configuration=config_d,
    )

    df = utility.read_files_to_dataframe(filepaths=args.src_path)
    plotter = plot.Plotter(
        runtime_context=runtime_context,
        config_d=config_d,
    )
    plotter.load_data(df)
    plotter.plot_data()


if __name__ == "__main__":
    main()

