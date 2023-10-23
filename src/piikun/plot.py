import numpy as np
import math
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
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import BoundaryNorm

from piikun import utility


def matrix_plot(data):
    selected_data = data[["ptn1_support", "ptn2_support", "vi_distance"]]
    sns.pairplot(selected_data)
    plt.show()


def add_jitter(
    df,
    columns,
    epsilon=1e-10,
):
    """
    Add a small jitter to equal values in specified columns of a DataFrame.

    Parameters:
    - df: Pandas DataFrame
    - columns: List of column names where jitter is to be added
    - epsilon: A small float to be used as the base jitter magnitude

    Returns:
    - Pandas DataFrame with jitter added to specified columns
    """
    for col in columns:
        unique_vals = df[col].unique()
        for val in unique_vals:
            mask = df[col] == val
            jitter = np.random.uniform(-epsilon, epsilon, size=mask.sum())
            # jitter = np.random.uniform(1, 3, size=mask.sum())
            df.loc[mask, col] += jitter
    return df


def df_to_dict_by_key(df, key_col, prop_cols):
    result_dict = {}
    for key, group_df in df.groupby(key_col):
        property_dict = {}
        for prop in prop_cols:
            property_dict[prop] = group_df[prop].tolist()
        result_dict[key] = property_dict
    return result_dict


def df_to_dict_by_key_single_value(df, key_col, prop_cols):
    result_dict = {}
    for key, group_df in df.groupby(key_col):
        property_dict = {}
        for prop in prop_cols:
            unique_values = group_df[prop].unique()
            # Assertion to ensure all values are the same for each key
            assert (
                len(unique_values) == 1
            ), f"All values for key {key} in column {prop} must be the same."
            property_dict[prop] = unique_values[0]
        result_dict[key] = property_dict
    return result_dict


# def extract_profile(df, key_col, prop_cols):
#     grouped_by_key = df.groupby(key_col)
#     result_d = {}
#     result_d["label"] = [k for k in grouped_by_key.groups.keys()]
#     for prop in prop_cols:
#         result_d[prop] = []
#         for key, group_df in grouped_by_key:
#             unique_values = group_df[prop].unique()
#             # Assertion to ensure all values are the same for each key
#             assert (
#                 len(unique_values) == 1
#             ), f"All values for key {key} in column {prop} must be the same."
#             result_d[prop].append(unique_values[0])
#     # profile_df = pd.DataFrame(result_d).rename(lambda x: x.replace("ptn1_", "") if "ptn1_" in x else "label", axis="columns")
#     profile_df = pd.DataFrame(result_d).rename(lambda x: x.replace("ptn1_", ""))
#     return profile_df

def visualize_distances_on_quantized_support_space(
    profile_df=None,
    distance_df=None,
    support_quantiles=None,
    distance_quantiles=None,
    gradient_calibration="shared",
    background_palette="coolwarm",
    scatterplot_palette="coolwarm",
    is_log_scale=True,
    palette=None,
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
    df = distance_df.copy()
    # df = df[ df["vi_distance"] > 1e-8 ]
    if "ptn1_support" not in df.columns:
        raise utility.UnavailableFieldException("ptn1_support")
    if is_log_scale:
        df["ptn1_support"] = np.log2(df["ptn1_support"])
        df["ptn2_support"] = np.log2(df["ptn2_support"])

    padding = (df["ptn1_support"].max() - df["ptn1_support"].min()) * 0.05
    internal_thresholds = df["ptn1_support"].quantile(support_quantiles).tolist()
    thresholds = [df["ptn1_support"].min() - padding]
    thresholds.extend(internal_thresholds)
    thresholds.append(df["ptn1_support"].max() + padding)
    bounds = np.array(thresholds)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_xlim(bounds[0], bounds[-1])
    ax.set_ylim(bounds[0], bounds[-1])

    range_fns = []
    for idx, threshold in enumerate(thresholds[:-1]):
        next_threshold = thresholds[idx + 1]
        range_fns.append(
            lambda x, t1=threshold, t2=next_threshold: (x >= t1) & (x < t2)
        )

    n_ranges = len(range_fns)
    mean_values = np.zeros((n_ranges, n_ranges))
    quantile_data = {
        "quantile_ptn1_support_idx": [],
        "quantile_ptn2_support_idx": [],
        "quantile_ptn1_support_max": [],
        "quantile_ptn1_support_min": [],
        "quantile_ptn2_support_max": [],
        "quantile_ptn2_support_min": [],
        "quantile_support_mean": [],
        "quantile_metric_mean": [],
        "quantile_metric_max": [],
        "quantile_metric_min": [],
    }
    quantile_data_stacked = {
        "quantile_support": [],
        "metric_value": [],
    }

    bgdf = df[df["vi_distance"] > 1e-8]
    for i, ptn1_condition in enumerate(range_fns):
        for j, ptn2_condition in enumerate(range_fns):
            subset = bgdf[
                ptn1_condition(bgdf["ptn1_support"])
                & ptn2_condition(bgdf["ptn2_support"])
            ]
            mean_values[n_ranges - 1 - i, j] = subset["vi_distance"].mean()

            mean_support = (
                sum(
                    [
                        thresholds[i],
                        thresholds[j],
                        thresholds[i + 1],
                        thresholds[j + 1],
                    ]
                )
                / 4
            )

            quantile_data["quantile_ptn1_support_idx"].append(i)
            quantile_data["quantile_ptn2_support_idx"].append(j)
            quantile_data["quantile_ptn1_support_min"].append(thresholds[i])
            quantile_data["quantile_ptn2_support_min"].append(thresholds[j])
            quantile_data["quantile_ptn1_support_max"].append(thresholds[i + 1])
            quantile_data["quantile_ptn2_support_max"].append(thresholds[j + 1])

            quantile_data["quantile_support_mean"].append(mean_support),
            quantile_data["quantile_metric_mean"].append(subset["vi_distance"].mean()),
            quantile_data["quantile_metric_max"].append(subset["vi_distance"].max()),
            quantile_data["quantile_metric_min"].append(subset["vi_distance"].min()),

            for v in subset["vi_distance"]:
                quantile_data_stacked["quantile_support"].append(mean_support)
                quantile_data_stacked["metric_value"].append(v)

    # Define quartile boundaries for the color scale
    background_cmap = plt.get_cmap(background_palette)
    scatterplot_cmap = plt.get_cmap(scatterplot_palette)
    if gradient_calibration == "shared":
        if not distance_quantiles:
            distance_quantiles = support_quantiles
        distance_quantiles.insert(0, 0)
        distance_quantiles.append(1)
        color_bounds = df["vi_distance"].quantile(distance_quantiles).tolist()
        background_norm = BoundaryNorm(color_bounds, background_cmap.N)
        scatterplot_norm = BoundaryNorm(color_bounds, scatterplot_cmap.N)
        scatterplot_legend = False
    elif gradient_calibration == "independent":
        background_norm = None
        scatterplot_norm = None
        scatterplot_legend = "brief"

    for bound in bounds[1:-1]:
        ax.axvline(x=bound, color="#000000", alpha=0.3, linestyle=":")
        ax.axhline(y=bound, color="#000000", alpha=0.3, linestyle=":")

    for (quantile, threshold) in zip(support_quantiles, internal_thresholds):
        txt = ax.annotate(
            f"{quantile} support quantile",
            xy=(threshold, 1),
            xycoords=("data", "axes fraction"),
            xytext=(threshold, 1.05),
            textcoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="->"),
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=8,
            wrap=True,
        )
        txt._get_wrap_line_width = lambda: 60
        txt = ax.annotate(
            f"{quantile} support quantile",
            xy=(0, threshold),
            xycoords=(
                "axes fraction",
                "data",
            ),
            xytext=(-0.125, threshold),
            textcoords=(
                "axes fraction",
                "data",
            ),
            arrowprops=dict(arrowstyle="->"),
            horizontalalignment="center",
            verticalalignment="bottom",
            rotation=90,
            fontsize=8,
            wrap=True,
        )
        txt._get_wrap_line_width = lambda: 60

    c = ax.pcolormesh(
        bounds,
        bounds[::-1],
        mean_values,
        shading="auto",
        cmap=background_cmap,
        norm=background_norm,
        alpha=0.5,
    )

    sns.scatterplot(
        x=df["ptn1_support"],
        y=df["ptn2_support"],
        hue="vi_distance",
        data=df,
        palette=scatterplot_cmap,
        hue_norm=scatterplot_norm,
        edgecolor=None,
        # edgecolor='grey',
        # edgecolor="#888888cc",
        # linewidth=0.2,
        legend=scatterplot_legend,
        ax=ax,
    )

    plt.xlabel("log(ptn1_support)")
    plt.ylabel("log(ptn2_support)")
    if scatterplot_legend is not False:
        plt.legend(title="vi_distance", loc="upper left")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(c, cax=cbar_ax, label="vi_distance")
    # fig.tight_layout()
    fig.subplots_adjust(right=0.88)
    return {
        "quantile_df": pd.DataFrame(quantile_data),
        "quantile_stacked_df": pd.DataFrame(quantile_data_stacked),
    }




def visualize_distances_on_regionalized_support_space(
    df,
    support_quantiles=None,
    distance_quantiles=None,
    gradient_calibration="shared",
    background_palette="coolwarm",
    scatterplot_palette="coolwarm",
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
    # df = df[ df["vi_distance"] > 1e-8 ]
    if is_log_scale:
        df["ptn1_support"] = np.log2(df["ptn1_support"])
        df["ptn2_support"] = np.log2(df["ptn2_support"])

    padding = (df["ptn1_support"].max() - df["ptn1_support"].min()) * 0.05
    internal_thresholds = df["ptn1_support"].quantile(support_quantiles).tolist()
    thresholds = [df["ptn1_support"].min() - padding]
    thresholds.extend(internal_thresholds)
    thresholds.append(df["ptn1_support"].max() + padding)
    bounds = np.array(thresholds)

    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_xlim(bounds[0], bounds[-1])
    ax.set_ylim(bounds[0], bounds[-1])

    range_fns = []
    for idx, threshold in enumerate(thresholds[:-1]):
        next_threshold = thresholds[idx + 1]
        range_fns.append(
            lambda x, t1=threshold, t2=next_threshold: (x >= t1) & (x < t2)
        )

    n_ranges = len(range_fns)
    mean_values = np.zeros((n_ranges, n_ranges))
    quantile_data = {
        "quantile_ptn1_support_idx": [],
        "quantile_ptn2_support_idx": [],
        "quantile_ptn1_support_max": [],
        "quantile_ptn1_support_min": [],
        "quantile_ptn2_support_max": [],
        "quantile_ptn2_support_min": [],
        "quantile_support_mean": [],
        "quantile_metric_mean": [],
        "quantile_metric_max": [],
        "quantile_metric_min": [],
    }
    quantile_data_stacked = {
        "quantile_support": [],
        "metric_value": [],
    }

    bgdf = df[df["vi_distance"] > 1e-8]
    for i, ptn1_condition in enumerate(range_fns):
        for j, ptn2_condition in enumerate(range_fns):
            subset = bgdf[
                ptn1_condition(bgdf["ptn1_support"])
                & ptn2_condition(bgdf["ptn2_support"])
            ]
            mean_values[n_ranges - 1 - i, j] = subset["vi_distance"].mean()

            mean_support = (
                sum(
                    [
                        thresholds[i],
                        thresholds[j],
                        thresholds[i + 1],
                        thresholds[j + 1],
                    ]
                )
                / 4
            )

            quantile_data["quantile_ptn1_support_idx"].append(i)
            quantile_data["quantile_ptn2_support_idx"].append(j)
            quantile_data["quantile_ptn1_support_min"].append(thresholds[i])
            quantile_data["quantile_ptn2_support_min"].append(thresholds[j])
            quantile_data["quantile_ptn1_support_max"].append(thresholds[i + 1])
            quantile_data["quantile_ptn2_support_max"].append(thresholds[j + 1])

            quantile_data["quantile_support_mean"].append(mean_support),
            quantile_data["quantile_metric_mean"].append(subset["vi_distance"].mean()),
            quantile_data["quantile_metric_max"].append(subset["vi_distance"].max()),
            quantile_data["quantile_metric_min"].append(subset["vi_distance"].min()),

            for v in subset["vi_distance"]:
                quantile_data_stacked["quantile_support"].append(mean_support)
                quantile_data_stacked["metric_value"].append(v)

    # Define quartile boundaries for the color scale
    background_cmap = plt.get_cmap(background_palette)
    scatterplot_cmap = plt.get_cmap(scatterplot_palette)
    if gradient_calibration == "shared":
        if not distance_quantiles:
            distance_quantiles = support_quantiles
        distance_quantiles.insert(0, 0)
        distance_quantiles.append(1)
        color_bounds = df["vi_distance"].quantile(distance_quantiles).tolist()
        background_norm = BoundaryNorm(color_bounds, background_cmap.N)
        scatterplot_norm = BoundaryNorm(color_bounds, scatterplot_cmap.N)
        scatterplot_legend = False
    elif gradient_calibration == "independent":
        background_norm = None
        scatterplot_norm = None
        scatterplot_legend = "brief"

    for bound in bounds[1:-1]:
        ax.axvline(x=bound, color="#000000", alpha=0.3, linestyle=":")
        ax.axhline(y=bound, color="#000000", alpha=0.3, linestyle=":")

    for (quantile, threshold) in zip(support_quantiles, internal_thresholds):
        txt = ax.annotate(
            f"{quantile} support quantile",
            xy=(threshold, 1),
            xycoords=("data", "axes fraction"),
            xytext=(threshold, 1.05),
            textcoords=("data", "axes fraction"),
            arrowprops=dict(arrowstyle="->"),
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=8,
            wrap=True,
        )
        txt._get_wrap_line_width = lambda: 60
        txt = ax.annotate(
            f"{quantile} support quantile",
            xy=(0, threshold),
            xycoords=(
                "axes fraction",
                "data",
            ),
            xytext=(-0.125, threshold),
            textcoords=(
                "axes fraction",
                "data",
            ),
            arrowprops=dict(arrowstyle="->"),
            horizontalalignment="center",
            verticalalignment="bottom",
            rotation=90,
            fontsize=8,
            wrap=True,
        )
        txt._get_wrap_line_width = lambda: 60

    c = ax.pcolormesh(
        bounds,
        bounds[::-1],
        mean_values,
        shading="auto",
        cmap=background_cmap,
        norm=background_norm,
        alpha=0.5,
    )

    sns.scatterplot(
        x=df["ptn1_support"],
        y=df["ptn2_support"],
        hue="vi_distance",
        data=df,
        palette=scatterplot_cmap,
        hue_norm=scatterplot_norm,
        edgecolor=None,
        # edgecolor='grey',
        # edgecolor="#888888cc",
        # linewidth=0.2,
        legend=scatterplot_legend,
        ax=ax,
    )

    plt.xlabel("log(ptn1_support)")
    plt.ylabel("log(ptn2_support)")
    if scatterplot_legend is not False:
        plt.legend(title="vi_distance", loc="upper left")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plt.colorbar(c, cax=cbar_ax, label="vi_distance")
    # fig.tight_layout()
    fig.subplots_adjust(right=0.88)
    return {
        "quantile_df": pd.DataFrame(quantile_data),
        "quantile_stacked_df": pd.DataFrame(quantile_data_stacked),
    }


# class Plotter(utility.RuntimeContext):
#     def __init__(
#         self,
#         config_d=None,
#         runtime_context=None,
#     ):
#         super().__init__(runtime_context=runtime_context)
#         self.distance_keys = [
#             "vi_distance",
#             "vi_normalized_kraskov",
#         ]
#         self.configure(config_d=config_d)

#     @property
#     def data(self):
#         if not hasattr(self, "_data") or self._data is None:
#             raise ValueError("Data not set")
#         return self._data

#     def load_data(self, data):
#         self._data = data.copy()
#         self._profile_df = utility.extract_profile(
#             df=self._data,
#             key_col="ptn1",
#             prop_col_filter_fn=lambda x: x.startswith("ptn1"),
#         )
#         if f"ptn1_{self.support_key}" in data.columns:
#             self._data[f"ptn1_{self.support_key}_original"] = self._data[
#                 f"ptn1_{self.support_key}"
#             ]
#             self._data[f"ptn2_{self.support_key}_original"] = self._data[
#                 f"ptn2_{self.support_key}"
#             ]
#         return self._data

#     def configure(self, config_d):
#         self.config_d = config_d or {}
#         self.support_key = "support"

#     def plot_data(self):
#         # sns.displot(x=self._profile_df["support"])
#         # plt.show()
#         self.plot_partition_profiles()
#         self.plot_partition_distances()
#         self.plot_support_vs_distances()

#     def plot_partition_profiles(self):
#         self.plot_partition_support_cdf()
#         self.plot_partition_profile_comparison()
#         self.plot_size_entropy_support()
#         self.plot_size_vs_support()
#         self.plot_entropy_vs_support()

#     def plot_partition_distances(self):
#         self.plot_clustermaps()

#     def plot_support_vs_distances(self):
#         # self._execute_plot(
#         #     plot_fn=self.plot_support_vs_distance_scatterheat,
#         #     kwargs={
#         #     },
#         #     name_parts=[f"distance-vs-support-s1"],
#         # )
#         rv = self._execute_plot(
#             plot_fn=visualize_distances_on_regionalized_support_space,
#             kwargs={
#                 "df": self._data,
#                 # "support_quantiles": [0.25, 0.5, 0.75],
#                 # "distance_quantiles": [
#                 #     # 0.01,
#                 #     # 0.05,
#                 #     0.1,
#                 #     0.2,
#                 #     0.3,
#                 #     0.4,
#                 #     0.5,
#                 #     0.6,
#                 #     0.7,
#                 #     0.8,
#                 # ],
#                 # "gradient_calibration": "shared",
#                 # "background_palette":  "coolwarm",
#                 # "scatterplot_palette": "coolwarm",
#             },
#             name_parts=[f"distance-vs-support-quantiles"],
#         )
#         qdf = rv["quantile_df"]
#         print(qdf)
#         print(qdf.mean())
#         print(qdf.max())
#         for metric_summary in ("max", "mean", "min", "range"):
#             kwargs = {"qdf": qdf}
#             kwargs["is_add_lines"] = True
#             kwargs["is_show_metric_max"] = (
#                 metric_summary == "max" or metric_summary == "range"
#             )
#             kwargs["is_show_metric_mean"] = (
#                 metric_summary == "mean" or metric_summary == "range"
#             )
#             kwargs["is_show_metric_min"] = (
#                 metric_summary == "min" or metric_summary == "range"
#             )
#             self._execute_plot(
#                 plot_fn=self.plot_quantile_data,
#                 kwargs=kwargs,
#                 name_parts=[f"quantized-distance-{metric_summary}-vs-support-mean"],
#             )

#     def plot_quantile_data(
#         self,
#         qdf,
#         is_add_lines=True,
#         is_show_metric_max=True,
#         is_show_metric_mean=True,
#         is_show_metric_min=True,
#     ):
#         # ptn1_s = (qdf["quantile_ptn1_support_max"] + qdf["quantile_ptn1_support_min"]) / 2
#         # ptn2_s = (qdf["quantile_ptn2_support_max"] + qdf["quantile_ptn2_support_min"]) / 2
#         # qdf["quantile_support_mean"] = (ptn1_s + ptn2_s)/2
#         qdf.sort_values("quantile_support_mean", ascending=True)
#         x = qdf["quantile_support_mean"]
#         y1 = qdf["quantile_metric_max"]
#         y2 = qdf["quantile_metric_mean"]
#         y3 = qdf["quantile_metric_min"]
#         fig, ax = plt.subplots(figsize=(10, 6))
#         if is_add_lines:
#             for xi, y1i, y2i, y3i in zip(x, y1, y2, y3):
#                 if is_show_metric_mean:
#                     if is_show_metric_max:
#                         plt.plot(
#                             [xi, xi],
#                             [y1i, y2i],
#                             alpha=0.1,
#                         )
#                     if is_show_metric_min:
#                         plt.plot(
#                             [xi, xi],
#                             [y2i, y3i],
#                             alpha=0.1,
#                         )
#                 elif is_show_metric_max:
#                     if is_show_metric_min:
#                         plt.plot(
#                             [xi, xi],
#                             [y1i, y3i],
#                             alpha=0.1,
#                         )
#         if is_show_metric_max:
#             plt.scatter(
#                 x,
#                 y1,
#                 label="max",
#                 color="blue",
#                 edgecolor="grey",
#                 linewidth=0.5,
#             )
#         if is_show_metric_mean:
#             plt.scatter(
#                 x,
#                 y2,
#                 label="mean",
#                 color="red",
#                 edgecolor="grey",
#                 linewidth=0.5,
#             )
#         if is_show_metric_min:
#             plt.scatter(
#                 x,
#                 y3,
#                 label="min",
#                 color="green",
#                 edgecolor="grey",
#                 linewidth=0.5,
#             )
#         plt.xlabel("support")
#         # plt.ylabel("n_subsets / 2^vi_entropy")
#         legend = plt.legend(
#             loc="center left",
#             bbox_to_anchor=(-0.15, 0.5),
#             frameon=True,
#             # prop={"size": 8},
#             fontsize=8,
#             handlelength=0.5,
#             # labelspacing=0,
#         )

#     def plot_partition_profile_comparison(self):
#         df = self._profile_df
#         fields = {
#             "log(support)": np.log2(df["support"]),
#             "vi_entropy": df["vi_entropy"],
#             "n_subsets": df["n_subsets"],
#         }
#         # columns = set(df.columns)
#         # for key in (
#         #             "constrained_probability",
#         #             "unconstrained_probability",

#         # ):
#         #     if key in columns:
#         #         fields[f"log({key})"] = np.log2(df[key])

#         data = pd.DataFrame(fields)
#         self._execute_plot(
#             plot_fn=sns.pairplot,
#             kwargs={
#                 "data": data,  # data
#             },
#             name_parts=[f"partition-profile-comparison"],
#         )

#     def plot_partition_support_cdf(self):
#         # self._profile_df.sort_values("support", ascending=True)
#         self._execute_plot(
#             plot_fn=sns.scatterplot,
#             kwargs={
#                 "x": self._profile_df["label"],
#                 "y": self._profile_df[self.support_key].cumsum(),
#                 # "size": self._profile_df["label"],
#             },
#             name_parts=[f"cumulative-support"],
#         )

#     def plot_size_entropy_support(self):
#         if self.config_d.get("is_add_jitter", True):
#             df = self._profile_df.copy()
#             add_jitter(
#                 df=self._profile_df,
#                 columns=[self.support_key],
#             )
#         else:
#             df = self._profile_df

#     def plot_size_vs_support(self):
#         self._execute_plot(
#             plot_fn=self._plot_size_vs_support,
#             kwargs={"is_add_lines": True},
#             name_parts=[f"size-vs-support-v1"],
#         )
#         self._execute_plot(
#             plot_fn=self._plot_size_vs_support,
#             kwargs={"is_add_lines": False},
#             name_parts=[f"size-vs-support-v0"],
#         )

#     # x: log-likelihood
#     # y: entropy
#     def plot_entropy_vs_support(self):
#         self._execute_plot(
#             plot_fn=sns.scatterplot,
#             kwargs={
#                 "x": np.log2(self._profile_df["support"]),
#                 "y": self._profile_df["vi_entropy"],
#                 # "size": self._profile_df["label"],
#             },
#             name_parts=[f"entropy-vs-support"],
#         )

#     # x: log-likelihood
#     # y1: number of clusters
#     # y2: 2^entropy
#     def _plot_size_vs_support(
#         self,
#         is_add_lines=True,
#     ):
#         if self.config_d.get("is_add_jitter", True):
#             df = self._profile_df.copy()
#             add_jitter(
#                 df=self._profile_df,
#                 columns=[self.support_key],
#             )
#         else:
#             df = self._profile_df
#         x = np.log2(df["support"])
#         y1 = df["n_subsets"]
#         y2 = np.power(2, df["vi_entropy"])

#         # self.plot_size_vs_support_plotly( x=x, y1=y1, y2=y2,)
#         fig, ax = plt.subplots(figsize=(10, 6))
#         if is_add_lines:
#             for xi, y1i, y2i in zip(x, y1, y2):
#                 plt.plot(
#                     [xi, xi],
#                     [y1i, y2i],
#                     alpha=0.1,
#                 )
#         plt.scatter(
#             x,
#             y1,
#             label="# subsets",
#             color="blue",
#             edgecolor="grey",
#             linewidth=0.5,
#         )
#         plt.scatter(
#             x,
#             y2,
#             label="$2^{Entropy}$",
#             color="red",
#             edgecolor="grey",
#             linewidth=0.5,
#         )
#         plt.xlabel("log2(Support)")
#         # plt.ylabel("n_subsets / 2^vi_entropy")
#         legend = plt.legend(
#             loc="center left",
#             bbox_to_anchor=(-0.15, 0.5),
#             frameon=True,
#             # prop={"size": 8},
#             fontsize=8,
#             handlelength=0.5,
#             # labelspacing=0,
#         )
#         # plt.setp(legend.get_texts(), rotation=90)

#     def plot_clustermaps(
#         self,
#     ):
#         for value_fieldname in (
#             "vi_distance",
#             "vi_normalized_kraskov",
#         ):
#             name_parts = [
#                 f"{value_fieldname.replace('_','-')}-clustermap",
#             ]
#             kwargs = {
#                 "value_fieldname": value_fieldname,
#             }
#             self._execute_plot(
#                 plot_fn=self.plot_clustermap,
#                 kwargs=kwargs,
#                 name_parts=name_parts,
#             )

#     def plot_clustermap(
#         self,
#         value_fieldname,
#         is_store_data=True,
#         is_store_plot=True,
#         is_show_plot=False,
#         is_index_first_partition=False,
#         is_cluster_rows=False,
#         is_cluster_cols=False,
#         plot_kwargs=None,
#     ):
#         c_df = self.prep_df_for_heatmap(
#             value_fieldname=value_fieldname,
#             is_index_first_partition=is_index_first_partition,
#         )
#         if not plot_kwargs:
#             plot_kwargs = {}
#         try:
#             # check to see if it can be compressed
#             dist_array = squareform(c_df)
#             dist_linkage = hierarchy.linkage(dist_array)
#             plot_kwargs["row_linkage"] = dist_linkage
#             plot_kwargs["col_linkage"] = dist_linkage
#         except ValueError:
#             pass

#         # plot_kwargs["row_cluster"] = is_cluster_rows
#         # plot_kwargs["col_cluster"] = is_cluster_cols

#         g = sns.clustermap(
#             c_df,
#             **plot_kwargs,
#         )

#     def plot_support_vs_distance_scatterheat(
#         self,
#         num_bins=8,
#     ):
#         df = self._data
#         fig_size = 10
#         n_parts = len(df["ptn1"].unique())
#         plt.figure(figsize=(fig_size, fig_size))
#         num_bins = self.config_d.get("num_support_vs_distance_bins", 8)
#         if self.config_d.get("is_drop_autocomparisons", True):
#             df = df[df["ptn1"] != df["ptn2"]].copy()
#         df["heat"] = pd.cut(df["vi_distance"], bins=num_bins, labels=False)
#         # df["euclidean"] = np.sqrt(
#         #     np.power(np.log2(df["ptn1_support"]) - np.log2(df["ptn2_support"]), 2)
#         # )
#         # df["power_scaled"] = (
#         #     np.power(df["ptn1_support"], 2) + np.power(df["ptn2_support"], 2)
#         # ) / 2
#         sz = max(80000 / (n_parts * n_parts), 1)
#         g = sns.scatterplot(
#             data=df,
#             x=np.log2(df["ptn1_support"]),
#             y=np.log2(df["ptn2_support"]),
#             hue="heat",
#             s=sz,
#             palette="coolwarm",
#             edgecolor=".7",
#         )
#         plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

#     def prep_df_for_heatmap(
#         self,
#         value_fieldname,
#         is_index_first_partition=False,
#     ):
#         if is_index_first_partition:
#             c_df = self.data.pivot(
#                 index="ptn1",
#                 columns="ptn2",
#                 values=value_fieldname,
#             )
#             # c_df = utility.mirror_upper_half(c_df)
#         else:
#             c_df = self.data.pivot(
#                 index="ptn2",
#                 columns="ptn1",
#                 values=value_fieldname,
#             )
#             # c_df = utility.mirror_lower_half(c_df)
#         np.fill_diagonal(c_df.values, 0.0)
#         c_df = c_df.round(12)
#         return c_df

#     def _execute_plot(
#         self,
#         plot_fn,
#         kwargs,
#         name_parts,
#         postprocess_fn=None,
#     ):
#         rv = plot_fn(**kwargs)
#         if postprocess_fn:
#             postprocess_fn(rv)
#         output_title = self.runtime_context.compose_output_path(
#             name_parts=name_parts,
#             extension=None,
#         )
#         self._finish_plot(output_title=output_title)
#         return rv

#     def _finish_plot(
#         self,
#         output_title,
#     ):
#         if self.config_d.get("is_store_plot", True):
#             savefig_kwargs = {}
#             savefig_kwargs["dpi"] = self.config_d.get("dpi", 300)
#             for plot_format in self.config_d.get(
#                 "output_format",
#                 [
#                     "jpg",
#                 ],
#             ):
#                 plt.savefig(
#                     f"{output_title}.{plot_format}",
#                     format=plot_format,
#                     **savefig_kwargs,
#                 )
#         if self.config_d.get("is_show_plot", False):
#             plt.show()
#         plt.close()


class Other:

    # def plot_size_vs_support_plotly(self, x, y1, y2, is_add_lines=True):
    #     import plotly.graph_objects as go
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=x, y=y1, mode='markers', marker=dict(color='blue')))
    #     fig.add_trace(go.Scatter(x=x, y=y2, mode='markers', marker=dict(color='red')))
    #     if is_add_lines:
    #         for xi, y1i, y2i in zip(x, y1, y2):
    #             fig.add_trace(go.Scatter(x=[xi, xi], y=[y1i, y2i], mode='lines', line=dict(color='grey', width=0.5)))
    #     fig.update_layout(
    #         xaxis_title="log2(Support)",
    #         yaxis_title="Value",
    #         showlegend=False,
    #         xaxis=dict(showline=True, linewidth=2, linecolor='black'),
    #         yaxis=dict(showline=True, linewidth=2, linecolor='black')
    #     )
    #     fig.show()

    def plot_support_distance_analyses(self):
        for distance_key in self.distance_keys:
            self._execute_plot(
                plot_fn=sns.pairplot,
                kwargs={
                    "data": self.data[["ptn1_support", "ptn2_support", distance_key]]
                },
                name_parts=[f"support-vs-{distance_key}-matrix"],
            )
        self._execute_plot(
            # plot_fn=self.plot_support_vs_distance_scatter,
            plot_fn=self.plot_support_vs_distance_scatterheat,
            kwargs={},
            name_parts=["support-vs-distance-scatter"],
        )

    def plot_comparisons(
        self,
    ):
        # self.plot_profiles(self)
        self._execute_plot(
            plot_fn=sns.pairplot,
            kwargs={"data": self.data[self.distance_keys]},
            name_parts=[f"distance-types-matrix"],
        )
        self.plot_clustermaps()

    def plot_support_vs_distance_scatter(
        self,
    ):
        plt.figure(figsize=(10, 10))
        g = sns.scatterplot(
            x=self.data["ptn1_support"],
            y=self.data["ptn2_support"],
            hue="vi_distance",
            palette="coolwarm",
            marker="o",
            s=20,
            data=self.data,
        )
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
