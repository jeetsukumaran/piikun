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


if __name__ == "__main__":
    main()


