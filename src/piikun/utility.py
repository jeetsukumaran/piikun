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
        "ptn1_support": {
            "bins": list(df["ptn1_support_bin"].cat.categories),
            "data_points": {
                bin_label: df[df["ptn1_support_bin"] == bin_label][
                    ["ptn1_support", "ptn2_support", "vi_distance"]
                ]
                for bin_label in df["ptn1_support_bin"].cat.categories
            },
        },
        "ptn2_support": {
            "bins": list(df["ptn2_support_bin"].cat.categories),
            "data_points": {
                bin_label: df[df["ptn2_support_bin"] == bin_label][
                    ["ptn1_support", "ptn2_support", "vi_distance"]
                ]
                for bin_label in df["ptn2_support_bin"].cat.categories
            },
        },
    }
    return binning_info

def create_full_profile_distance_df(
    profiles_df=None,
    distances_df=None,
    export_profile_columns=None,
    export_distance_columns=None,
    profiles_path=None,
    distances_path=None,
    merged_path=None,
    delimiter="\t",
    logger=None,
):
    if not profiles_df:
        assert profiles_path
        profiles_df = pd.read_csv(profiles_path, delimiter=delimiter,)
    if not distances_df:
        assert distances_path
        distances_df = pd.read_csv(distances_path, delimiter=delimiter,)
    if export_profile_columns:
        profile_columns = list(export_profile_columns)
    else:
        profile_columns = [column for column in profiles_df if column not in set([
            "partition_id",
            "label",
        ])]
    if export_distance_columns:
        distance_columns = list(export_distance_columns)
    else:
        distances_columns = [
            "vi_distance",
            "vi_normalized_kraskov",
        ]
    partition_keys = list(profiles_df["label"])
    new_dataset = []
    for pkd_idx, pk1 in enumerate(partition_keys):
        if logger:
            logger.log_info(f"Exporting partition {pkd_idx+1} of {len(partition_keys)}: '{pk1}'")
        seen_comparisons = set()
        # pk1_ptn1_df = distances_df[ distances_df["ptn1"] == pk1 ]
        # pk1_ptn2_df = distances_df[ distances_df["ptn2"] == pk1 ]
        # pk1_ptns_df = pd.concat([pk1_ptn1_df, pk1_ptn2_df])
        for pk2 in partition_keys:
            key = (pk1, pk2)
            if key in seen_comparisons:
                continue
            # print(f"---- {key}: PK1 = {pk1}, PK2 = {pk2} ")
            seen_comparisons.add(key)
            condition = ((distances_df["ptn1"] == pk1) & (distances_df["ptn2"] == pk2)) | ((distances_df["ptn1"] == pk2) & (distances_df["ptn2"] == pk1))
            dists_sdf = distances_df[condition]
            if len(dists_sdf) == 0 and pk1 != pk2:
                raise ValueError(f"Missing non-self comparison: {pk1}, {pk2}")
            elif len(dists_sdf) == 1:
                row_d = {}
                for ptn_idx, ptn_key in zip((1,2), (pk1, pk2)):
                    row_d[f"ptn{ptn_idx}"] = ptn_key
                for ptn_idx, ptn_key in zip((1,2), (pk1, pk2)):
                    for column in profile_columns:
                        values = profiles_df[ profiles_df["label"] == ptn_key ][ column ].values.tolist()
                        assert len(values) == 1
                        row_d[f"ptn{ptn_idx}_{column}"] = values[0]
                for dist_column in distances_columns:
                    dists = dists_sdf[dist_column].values.tolist()
                    assert len(dists) == 1
                    row_d[dist_column] = dists[0]
                new_dataset.append(row_d)
            else:
                raise NotImplementedError()
    df = pd.DataFrame.from_records(new_dataset)
    if merged_path:
        df.to_csv(merged_path, sep=delimiter)
    return df

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

def read_files_to_dataframe(
    filepaths,
    delimiter="\t",
    is_record_source_filepath=True,
    df=None,
):
    dfs = []
    if df is not None:
        dfs.append(df)
    for fp in filepaths:
        d = pd.read_csv(fp, delimiter=delimiter)
        if is_record_source_filepath:
            d["file"] = fp
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    return df

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


def get_logger():
    return yakherd.Logger(
        name="linsang",
        max_allowed_message_noise_level=0,
        is_enable_console=True,
        is_colorize=True,
        is_enable_log_file=False,
    )


def read_files_to_dataframe(
    filepaths,
    delimiter="\t",
    is_record_source_filepath=True,
    df=None,
):
    dfs = []
    if df is not None:
        dfs.append(df)
    for fp in filepaths:
        d = pd.read_csv(fp, delimiter=delimiter)
        if is_record_source_filepath:
            d["file"] = fp
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


@dataclass
class KeyValuePair:
    key: str
    value: str


# try:
#     import importlib.resources.files as importlib_resources_files
# except ImportError:
#     from importlib_resources import files as importlib_resources_files


class FileHandler:
    def __init__(self, output_root, output_title):
        self.output_root = (
            os.path.expanduser(output_root) if output_root else os.getcwd()
        )
        self.output_title = output_title
        self.file_handles = {}
        self.dict_writers = {}

    def __getitem__(self, key):
        if key not in self.file_handles:
            key = key.replace("${title}", self.output_title)
            file_path = os.path.join(self.output_root, key)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.file_handles[key] = open(file_path, "w", newline="")
        return self.file_handles[key]

    def write_tsv(self, key, row_dict):
        key += ".tsv"
        if key not in self.dict_writers:
            self.dict_writers[key] = csv.DictWriter(
                self[key], fieldnames=row_dict.keys(), delimiter="\t"
            )
            self.dict_writers[key].writeheader()
        self.dict_writers[key].writerow(row_dict)

    def write(self, key, data):
        self[key].write(data)

    def close_all(self):
        for file_handle in self.file_handles.values():
            file_handle.close()


class RuntimeContext:
    @staticmethod
    def ensure_random_seed(random_seed=None):
        if random_seed is None:
            rtmp = random.Random()
            random_seed = rtmp.getrandbits(32)
        return random_seed

    @staticmethod
    def get_rng(random_seed=None):
        random_seed = RuntimeContext.ensure_random_seed(random_seed)
        rng = random.Random(random_seed)
        rng._random_seed = random_seed
        return rng


    @property
    def logger(self):
        if (
            not hasattr(self, "_logger")
            or self._logger is None
        ):
            if self._logger is None:
                self._logger = yakherd.Logger(
                    name="piikun",
                    max_allowed_message_noise_level=0,
                    is_enable_console=True,
                    console_logging_level="ERROR",
                    is_colorize=True,
                    is_enable_log_file=False,
                    logfile_path=None,
                    )
        return self._logger
    @logger.setter
    def logger(self, value):
        self._logger = value

    def __init__(
        self,
        random_seed=None,
        logger=None,
        output_directory=".",
        output_title="piikun",
        output_configuration=None,
    ):
        self.logger = logger
            # from yakherd import Logger
        self.logger.log_info(
            f"Initializing system runtime context at: {datetime.datetime.now()}"
        )
        if random_seed is not False:
            self.rng = self.get_rng(random_seed=random_seed)
            self.logger.log_info(
                f"Seeding random number sampler with seed: {self.rng._random_seed}"
            )
        else:
            self.rng = None
        self.file_handler = FileHandler(
            output_root=output_directory,
            output_title=output_title,
        )
        if output_directory and output_title:
            self.output_directory = pathlib.Path(output_directory).resolve()
            self.output_title = output_title
            self.logger.log_info(f"Output directory: '{self.output_directory}'")
            self.logger.log_info(f"Output title: '{self.output_title}'")
        self.output_configuration = output_configuration or {}
        self.signal_stores = container.AttributeMap()

    def ensure_file(
        self,
        key,
        name_parts=None,
        extension=None,
        **kwargs,
    ):
        if key not in self.signal_stores:
            if not name_parts:
                raise ValueError(
                    f"'{key}' not in store: `name_parts` required to build file handle"
                )
            file_handle = self.create_file_handle(
                name_parts=name_parts, extension=extension, **kwargs
            )
            self.signal_stores[key] = file_handle
        return self.signal_stores[key]

    def ensure_store(
        self,
        key,
        name_parts=None,
        extension="tsv",
        **kwargs,
    ):
        if key not in self.signal_stores:
            if not name_parts:
                raise ValueError(
                    f"'{key}' not in store: `name_parts` required to build file handle"
                )
            file_handle = self.create_file_handle(
                name_parts=name_parts, extension=extension, **kwargs
            )
            self.signal_stores[key] = SignalStore(
                file_handle=file_handle,
                config_d=self.output_configuration,
            )
        return self.signal_stores[key]

    @property
    def output_prefix(self):
        if not hasattr(self, "_output_prefix") or self._output_prefix is None:
            self._output_prefix = self.output_directory / self.output_title
        return self._output_prefix

    def compose_output_path(
        self,
        name_parts,
        separator="-",
        extension="",
        subdir_paths=None,
        is_ensure_parent_dir=False,
    ):
        if extension and not extension.startswith("."):
            extension = f".{extension}"
        else:
            extension = ""
        parts = separator.join(str(i) for i in name_parts)
        if parts and self.output_title:
            parts = f"{separator}{parts}"
        fname = pathlib.Path(f"{self.output_title}{parts}{extension}")
        parent_dir = self.output_directory
        if subdir_paths:
            for sd in subdir_paths:
                parent_dir = self.output_directory / sd
        if is_ensure_parent_dir:
            parent_dir.mkdir(parents=True, exist_ok=True)
        return parent_dir / fname

    def ensure_parent_dir(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)

    def create_file_handle(
        self,
        *args,
        **kwargs,
    ):
        # self.output_directory.mkdir(parents=True, exist_ok=True)
        path = self.compose_output_path(*args, **kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            f = open(path, kwargs.get("mode", "w"))
        except FileNotFoundError as e:
            sys.exit(f"Failed to create file: '{path}'\n{e}")
        f._bound_path = path
        return f

    def terminate(self, msg=None, exit_code=0):
        if msg:
            self.logger.log_info(msg)
        self.logger.log_info(f"Terminating normally at: {datetime.datetime.now()}")
        sys.exit(exit_code)

    def abort(self, msg=None, exit_code=1):
        if msg:
            self.logger.log_error(msg)
        self.logger.log_critical(f"Aborted due to error at: {datetime.datetime.now()}")
        sys.exit(exit_code)


class RuntimeClient:
    def __init__(self, runtime_context=None, **kwargs):
        if runtime_context is None:
            # if not kwargs:
            #     raise TypeError(
            #         "RuntimeClient cannot be instantiated: runtime_context object or arguments to create runtime_context required"
            #     )
            runtime_context = RuntimeContext(**kwargs)
        self.runtime_context = runtime_context

    @property
    def logger(self):
        return self.runtime_context.logger

    @property
    def rng(self):
        return self.runtime_context.rng

    def set_attributes_from_dict(
        self,
        keys,
        config_d,
        is_overwrite_if_null=True,
        min_required=None,
        max_allowed=None,
    ):
        set_keys = {}
        for key in keys:
            value = config_d.get(key, None)
            if value is not None or is_overwrite_if_null:
                setattr(self, key, value)
            if value is not None:
                set_keys[key] = value
        if min_required and len(set_keys) < min_required:
            self.runtime_context.abort(
                f"At least {min_required} of the following parameters must be set: {keys}"
            )
        if max_allowed and len(set_keys) > max_allowed:
            self.runtime_context.abort(
                f"No more than {max_allowed} of the following parameters must be set: {keys}"
            )
        return set_keys


class ColorProvider(object):
    """
    Colors that contrast well over various color perception regimes.
    """

    contrast_pairs = [
        ["#ffc20a", "#0c7bdc"],
        ["#1aff1a", "#4b0092"],
        ["#994f00", "#006cd1"],
        ["#fefe62", "#d35fb7"],
        ["#e1be6a", "#40b0a6"],
        ["#005ab5", "#dc3220"],
        ["#e66100", "#5d3a9b"],
        ["#1a85ff", "#d41159"],
    ]

    def __init__(self):
        self.label_color_map = {}
        self.color_label_map = {}
        self.colors = [
            "#000000",
            "#e69f00",
            "#56b4e9",
            "#009e73",
            "#f0e442",
            "#0072b2",
            "#d55e00",
            "#cc79a7",
        ]
        self.available_colors = list(self.colors)

    def __getitem__(self, label):
        try:
            return self.label_color_map[label]
        except KeyError:
            new_color = self.available_colors.pop()
            self.label_color_map[label] = new_color
            self.color_label_map[new_color] = label
        return new_color


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def parse_fieldname_and_value(labels):
    if not labels:
        return collections.OrderedDict()
    fieldname_value_map = collections.OrderedDict()
    for label in labels:
        match = re.match(r"\s*(.*?)\s*:\s*(.*)\s*", label)
        if not match:
            raise ValueError(
                "Cannot parse fieldname and label (format required: fieldname:value): {}".format(
                    label
                )
            )
        fieldname, value = match.groups(0)
        fieldname_value_map[fieldname] = value
    return fieldname_value_map


class SignalStore:
    def __init__(self, file_handle, config_d):
        self.file_handle = file_handle
        self.is_preamble_written = False
        if config_d:
            self.configure(config_d)

    def configure(self, config_d):
        self.delimiter = config_d.get("column_delimiter", "\t")
        self.is_write_preamble = config_d.get("header_row", True)

    @property
    def path(self):
        if not hasattr(self, "_path") or self._path is None:
            self._path = pathlib.Path(self.file_handle.name)
        return self._path

    def write_preamble(self, preamble):
        self.file_handle.write(preamble)
        self.file_handle.write("\n")
        self.is_preamble_written = True

    def write_v(self, data_v):
        data_str = self.delimiter.join(f"{v}" for v in data_v)
        self.file_handle.write(data_str)
        self.file_handle.write("\n")

    def write_d(self, data_d):
        if not self.is_preamble_written and self.is_write_preamble:
            header = self.delimiter.join(f"{k}" for k in data_d.keys())
            self.write_preamble(preamble=header)
        self.write_v(data_d.values())

    # def write_d(self, *args):
    #     line = "\t".join(map(str, args)) + "\n"
    #     self.file_handle.write(line)

    def close(self):
        self.file_handle.close()

