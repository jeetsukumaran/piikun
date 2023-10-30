#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import itertools
import os
import sys
import random
import functools
from dataclasses import dataclass
import datetime
import pandas as pd
import numpy as np
from yakherd import container
import yakherd

import csv

class UnavailableFieldException(Exception):
    pass

def extract_profile(df, key_col, prop_col_filter_fn):
    prop_cols = [
            col for col in df.columns if prop_col_filter_fn(col)
    ]
    grouped_by_key = df.groupby(key_col)
    result_d = {}
    # result_d["ptn1"] = list(grouped_by_key.groups.keys())
    result_d["ptn1"] = [str(k) for k in grouped_by_key.groups.keys()]
    for prop in prop_cols:
        result_d[prop] = []
        for key, group_df in grouped_by_key:
            unique_values = group_df[prop].unique()
            # Assertion to ensure all values are the same for each key
            assert (
                len(unique_values) == 1
                ), f"All values for key {key} in column {prop} must be the same."
            result_d[prop].append(unique_values[0])
    profile_df = pd.DataFrame(result_d).rename(
        lambda x: x.replace("ptn1_", "") if "ptn1_" in x else "label", axis="columns"
        )
    # profile_df["label"] = profile_df["label"].map(str)
    # print(profile_df)
    return profile_df


def bin_indices(X, Y, num_bins):
    from scipy.stats import binned_statistic_2d
    _, x_edge, y_edge, binnumber = binned_statistic_2d(X, Y, None, statistic='count', bins=num_bins)

    bin_dict = {}
    for i, bin_idx in enumerate(binnumber):
        x_bin = (bin_idx - 1) // num_bins
        y_bin = (bin_idx - 1) % num_bins
        bin_key = (x_bin, y_bin)
        if bin_key not in bin_dict:
            bin_dict[bin_key] = []
        bin_dict[bin_key].append(i)
    return bin_dict

def get_binning_info(df):
    binning_info = {
        "ptn1_score": {
            "bins": list(df["ptn1_score_bin"].cat.categories),
            "data_points": {
                bin_label: df[df["ptn1_score_bin"] == bin_label][
                    ["ptn1_score", "ptn2_score", "vi_distance"]
                ]
                for bin_label in df["ptn1_score_bin"].cat.categories
            },
        },
        "ptn2_score": {
            "bins": list(df["ptn2_score_bin"].cat.categories),
            "data_points": {
                bin_label: df[df["ptn2_score_bin"] == bin_label][
                    ["ptn1_score", "ptn2_score", "vi_distance"]
                ]
                for bin_label in df["ptn2_score_bin"].cat.categories
            },
        },
    }
    return binning_info

def create_compact_markdown_table(table):
    header = "| " + " | ".join([str(col) for col in table.columns]) + " |"
    separator = "| " + " | ".join(["---" for _ in table.columns]) + " |"
    rows = [header, separator]
    for idx, row in table.iterrows():
        formatted_row = "| "
        for value in row:
            if pd.isna(value):
                formatted_row += "  | "
            else:
                formatted_cell = f"{value:.2f}".rstrip('0').rstrip('.') if '.' in f"{value:.2f}" else f"{value:.2f}"
                formatted_row += formatted_cell + " | "
        rows.append(formatted_row)
    markdown_table = "\n".join(rows)
    return markdown_table

def plot3d(x, y, z):
    # from: https://www.geeksforgeeks.org/3d-heatmap-in-python/
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np
    import pylab
    # x = np.random.randint(low=100, high=500, size=(1000,))
    # y = np.random.randint(low=300, high=500, size=(1000,))
    # z = np.random.randint(low=200, high=500, size=(1000,))
    colo = [x + y + z]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    color_map = pylab.cm.ScalarMappable(cmap=pylab.cm.Greens_r)
    color_map.set_array(colo)
    img = ax.scatter(x, y, z, marker='s',
                    s=200, color='green')
    plt.colorbar(color_map)
    # ax.set_title("3D Heatmap")
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')

def random_nonsymmetric_matrix(n):
    df = pd.DataFrame(np.random.rand(n, n))
    np.fill_diagonal(df.values, 0)
    return df

def force_df_symmetric_if_within_tolerance(df, tol=1e-8):
    # Calculate absolute difference between df and its transpose
    diff = df.subtract(df.transpose()).abs()
    # Identify elements with difference less than tolerance
    mask = diff < tol
    # Replace these elements in df with the corresponding elements from the transpose
    df[mask] = df.transpose()[mask]
    return df

def force_symmetric_if_within_tolerance(A, tol=1e-8):
    # Calculate absolute difference between A and its transpose
    diff = np.abs(A - A.T)
    # Identify elements with difference less than tolerance
    mask = diff < tol
    # Replace these elements in A with the corresponding elements from the transpose
    A[mask] = A.T[mask]
    return A

def find_nonsymmetric(A):
    # Assuming A is your 2D NumPy array
    diff = np.where(A != A.T)
    # Note: diff is a tuple of two 1-D arrays containing the row and column indices.
    # We can convert these to pairs of indices for easier interpretation.
    indices = list(zip(diff[0], diff[1]))
    # Since A[i, j] = A[j, i], each pair of non-symmetric indices will appear twice in the list.
    # We can eliminate the duplicates by keeping only pairs where i < j.
    nonsymmetric_indices = [(i, j) for i, j in indices if i < j]
    nonsymmetric_values = [(A[i, j], A[j, i]) for i, j in nonsymmetric_indices]
    return nonsymmetric_indices, nonsymmetric_values


def force_diagonals_to_zero(arr, threshold=1e-12):
    threshold = None
    if threshold is None:
        np.fill_diagonal(arr, 0.0)
    else:
        diag_indices = np.diag_indices(arr.shape[0])
        arr[diag_indices] = np.where(np.abs(arr[diag_indices]) < threshold, 0, arr[diag_indices])
    return arr

def mirror_upper_half(
    df,
    diagonal_zero_threshold=1e-12,
):
    arr = df.to_numpy()
    upper = np.triu(arr)
    lower = upper.T
    symmetric_arr = upper + lower
    if diagonal_zero_threshold:
        symmetric_arr = force_diagonals_to_zero(symmetric_arr, threshold=diagonal_zero_threshold)
    symmetric_df = pd.DataFrame(symmetric_arr, index=df.index, columns=df.columns)
    return symmetric_df

def mirror_lower_half(
    df,
    diagonal_zero_threshold=1e-12,
):
    arr = df.to_numpy()
    lower = np.tril(arr)
    upper = lower.T
    symmetric_arr = upper + lower
    if diagonal_zero_threshold:
        symmetric_arr = force_diagonals_to_zero(symmetric_arr, threshold=diagonal_zero_threshold)
    symmetric_df = pd.DataFrame(symmetric_arr, index=df.index, columns=df.columns)
    return symmetric_df


def extract_unique_keys_from_nested_dicts(dict_list, axis):
    for _ in range(axis):
        dict_list = list(itertools.chain.from_iterable([d.values() for d in dict_list]))
    unique_keys = sorted(
        list(set(itertools.chain.from_iterable([d.keys() for d in dict_list])))
    )
    return unique_keys

def read_files_to_dataframe(
    filepaths,
    delimiter="\t",
    is_record_source_filepath=True,
    df=None,
    format_type="json",
):
    dfs = []
    if df is not None:
        dfs.append(df)
    for fpath in filepaths:
        fp = open(fpath)
        if format_type == "json":
            d = pd.read_json(fp)
        elif format_type == "csv":
            d = pd.read_csv(fp, delimiter=delimiter)
        else:
            raise ValueError(format_type)
        if is_record_source_filepath:
            d["file"] = str(pathlib.Path(fpath).absolute())
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    return df


def build_dataframe(
    source_paths,
    signal_type=None,
    include_patterns=None,
    exclude_patterns=None,
    logger=None,
):
    main_df = read_files_to_dataframe(source_paths)
    if logger:
        logger.log_info(["Field: "] + main_df.columns)
    if include_patterns:
        dfs = []
        for field_name, patterns in include_patterns.items():
            for pattern in patterns:
                df = main_df[main_df[field_name].str.fullmatch(pattern)]
                dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = main_df
    if signal_type:
        df = df[df["signal_type"] == signal_type]
    if exclude_patterns:
        for field_name, patterns in exclude_patterns.items():
            for pattern in patterns:
                df = df[~df[field_name].str.fullmatch(pattern)]
    return df

