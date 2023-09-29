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
import itertools
import time
import datetime

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

import time
import datetime

def pivot_to_nested_dict(df):
    data = {}
    for _, row in df.iterrows():
        ptn1, ptn2, distance = row["ptn1"], row["ptn2"], row["vi_distance"]
        if ptn1 not in data:
            data[ptn1] = {}
        data[ptn1][ptn2] = distance
    return data

def format_time(seconds):
    """Convert a time duration in seconds to a string in HH:MM:ss format."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def calculate_time_metrics(start_time, n_iterations, current_iteration_idx):
    """
    Calculate and return the elapsed time, estimated time remaining, and ETA
    for a loop.

    Parameters:
    - start_time (float): The time at which the loop started (seconds since the epoch).
    - n_iterations (int): The total number of iterations for the loop.
    - current_iteration_idx (int): The current iteration index (0-based).

    Returns:
    - dict: A dictionary containing the formatted time strings.
    """
    # Calculate elapsed and remaining time
    elapsed_time = time.time() - start_time
    remaining_time = elapsed_time * (n_iterations - (current_iteration_idx + 1)) / (current_iteration_idx + 1)

    # Calculate expected time of completion (ETA)
    eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining_time)

    # Convert times to formatted strings
    elapsed_str = format_time(elapsed_time)
    remaining_str = format_time(remaining_time)
    eta_str = eta.strftime('%Y-%m-%dT%H:%M:%S')
    projected_total_time = elapsed_time + remaining_time
    projected_total_str = format_time(projected_total_time)

    return {
        'elapsed_str': elapsed_str,
        'remaining_str': remaining_str,
        'eta_str': eta_str,
        'projected_total_str': projected_total_str,
    }


def validate_comparisons(
    data_d,
    partition_keys,
    config_d,
    logger,
):
    logger.log_info("Validating all distinct pairwise comparisons are present")
    missing = 0
    for pk1 in partition_keys:
        if pk1 not in data_d:
            logger.log_error(f"Partition '{pk1}' not found as 'ptn1'")
            missing += 1
            continue
        for pk2 in partition_keys:
            if pk2 not in data_d[pk1]:
                missing += 1
                logger.log_error(f"Comparison '{pk1}' (present) vs. '{pk2}' (missing): not found")
    if missing:
        sys.exit("Exiting due to missing comparisons")

def validate_triangle_inequality(
    data_d,
    partition_keys,
    config_d,
    logger,
):
    logger.log_info("Validating triangle inequality")
    n_combinations = math.comb(len(partition_keys), 3)
    logger.log_info(f"{n_combinations} unique triplets")
    start_time = time.time()
    log_freq = config_d["progress_report_frequency"]
    report_freq = config_d["progress_report_frequency"]
    is_single_triangle_inequality_condition = config_d["is_single_triangle_inequality_condition"]
    for comb_idx, ptn_ijk in enumerate(itertools.combinations(partition_keys, 3)):
        if comb_idx and comb_idx % log_freq == 0:
            tm = calculate_time_metrics(
                start_time=start_time,
                n_iterations=n_combinations,
                current_iteration_idx=comb_idx,
            )
            logger.log_info([
                f"Comparison {comb_idx+1} of {n_combinations}:",
                f" - Elapsed time: {tm['elapsed_str']}",
                f" - Remaining time: {tm['remaining_str']}",
                f" - Total time: {tm['projected_total_str']}",
                f" - Completion time: {tm['eta_str']}",
            ])

        d_ijk = [
            # data[ptn_ijk[0]][ptn_ijk[1]][distance_idx],
            # data[ptn_ijk[0]][ptn_ijk[2]][distance_idx],
            # data[ptn_ijk[1]][ptn_ijk[2]][distance_idx],
            data_d[ptn_ijk[0]][ptn_ijk[1]],
            data_d[ptn_ijk[0]][ptn_ijk[2]],
            data_d[ptn_ijk[1]][ptn_ijk[2]],
        ]
        distances_labels = sorted(
            list(
                zip(
                    d_ijk,
                    (
                        (ptn_ijk[0], ptn_ijk[1]),
                        (ptn_ijk[0], ptn_ijk[2]),
                        (ptn_ijk[1], ptn_ijk[2]),
                    ),
                )
            ),
            key=lambda x: x[0],
        )
        [d_a, d_b, d_c] = [dl[0] for dl in distances_labels]
        if report_freq and comb_idx % report_freq == 0:
            assert d_a <= d_b <= d_c
            [pair_a, pair_b, pair_c] = [dl[1] for dl in distances_labels]
            logger.log_info(
                [
                    f"d_a=VI{pair_a}={d_a}",
                    f"d_b=VI{pair_b}={d_b}",
                    f"d_c=VI{pair_c}={d_c}",
                ]
            )
        if not is_single_triangle_inequality_condition:
            assert False
            assert (
                ((d_b + d_c) - d_a) + 1e-12 >= 0
            ), f"{d_b + d_c} >= {d_a} ({(d_b+d_c-d_a)})"
            assert (
                ((d_a + d_c) - d_b) + 1e-12 >= 0
            ), f"{d_a + d_c} >= {d_b} ({(d_a+d_c-d_b)})"
        assert (
            ((d_a + d_b) - d_c) + 1e-12 >= 0
        ), f"{d_a + d_b} >= {d_c} ({(d_a+d_b-d_c)})"

def main():
    parent_parser = argparse.ArgumentParser()
    src_options = parent_parser.add_argument_group("Partition Distance Sources")
    src_options.add_argument(
        "src_path",
        action="store",
        metavar="FILE",
        nargs="+",
        help="Path to distances ('<title>-distances.tsv') data file(s).",
    )
    performance_options = parent_parser.add_argument_group("Performance Options")
    performance_options.add_argument(
        "--triangle-equality-single",
        action=argparse.BooleanOptionalAction,
        dest="is_single_triangle_inequality_condition",
        default=True,
        help="Restrict / do not restrict triangle inequality check to a single (the maximum) case for efficiency.",
    )
    # output_options = parent_parser.add_argument_group("Partition Dis")
    # output_options.add_argument(
    #     "-o",
    #     "--output-title",
    #     action="store",
    #     default="piikun",
    #     help="Prefix for output filenames [default='%(default)s'].",
    # )
    # output_options.add_argument(
    #     "-O",
    #     "--output-directory",
    #     action="store",
    #     default=os.curdir,
    #     help="Directory for output files [default='%(default)s'].",
    # )
    logger_configuration_parser = yakherd.LoggerConfigurationParser(name="piikun")
    logger_configuration_parser.attach(parent_parser)
    logger_configuration_parser.console_logging_parser_group.add_argument(
        "--progress-report-frequency",
        type=int,
        action="store",
        default=1000,
        help="Frequency of progress reporting.",
    )
    args = parent_parser.parse_args()
    config_d = dict(vars(args))
    logger = logger_configuration_parser.get_logger(args_d=config_d)
    logger.log_info("Reading sources")
    df = utility.read_files_to_dataframe(filepaths=args.src_path)
    logger.log_info(f"{len(df)} entries read from distances file")
    partition_keys = set(df["ptn1"].unique()) | set(df["ptn2"].unique())
    logger.log_info(f"{len(partition_keys)} unique partition labels")
    data_d = pivot_to_nested_dict(df)
    validate_comparisons(
        data_d=data_d,
        partition_keys=partition_keys,
        config_d=config_d,
        logger=logger,
    )
    validate_triangle_inequality(
        data_d=data_d,
        partition_keys=partition_keys,
        config_d=config_d,
        logger=logger,
    )




if __name__ == "__main__":
    main()
