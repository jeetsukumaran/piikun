
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import pandas as pd
from collections import namedtuple

class Regionalizer(utility.RuntimeContext):

    def __init__(
        self,
        df,
        config_d,
        runtime_context=None,
    ):
        super().__init__(runtime_context=runtime_context)
        self.configure(config_d=config_d)

    def configure(self, config_d):
        self.config_d = config_d or {}

    @property
    def ptn1_low_threshold_quantile(self):
        if (
            not hasattr(self, "_ptn1_low_threshold_quantile")
            or self._ptn1_low_threshold_quantile is None
        ):
            self._ptn1_low_threshold_quantile = None
        return self._ptn1_low_threshold_quantile
    @ptn1_low_threshold_quantile.setter
    def ptn1_low_threshold_quantile(self, value):
        self._ptn1_low_threshold_quantile = value

    @property
    def ptn2_high_threshold_quantile(self):
        if (
            not hasattr(self, "_ptn2_high_threshold_quantile")
            or self._ptn2_high_threshold_quantile is None
        ):
            self._ptn2_high_threshold_quantile = None
        return self._ptn2_high_threshold_quantile
    @ptn2_high_threshold_quantile.setter
    def ptn2_high_threshold_quantile(self, value):
        self._ptn2_high_threshold_quantile = value

    @property
    def ptn1_low_threshold_value(self):
        if (
            not hasattr(self, "_ptn1_low_threshold_value")
            or self._ptn1_low_threshold_value is None
        ):
            return self._ptn1_low_threshold_value
        return self._value_for_quantile(df, self._ptn1_low_threshold_quantile)
    @ptn1_low_threshold_value.setter
    def ptn1_low_threshold_value(self, value):
        self._ptn1_low_threshold_value = value

    @property
    def ptn2_low_threshold_value(self):
        if (
            not hasattr(self, "_ptn2_low_threshold_value")
            or self._ptn2_low_threshold_value is None
        ):
            return self._ptn2_low_threshold_value
        return self._value_for_quantile(df, self._ptn2_low_threshold_quantile)
    @ptn2_low_threshold_value.setter
    def ptn2_low_threshold_value(self, value):
        self._ptn2_low_threshold_value = value

def visualize_grid_stats_heatmap(df, low_threshold_quantile, high_threshold_quantile, statistic="mean"):
    # Determine thresholds from quantiles
    low_threshold_ptn1 = df['ptn1_score'].quantile(low_threshold_quantile)
    high_threshold_ptn1 = df['ptn1_score'].quantile(high_threshold_quantile)
    low_threshold_ptn2 = df['ptn2_score'].quantile(low_threshold_quantile)
    high_threshold_ptn2 = df['ptn2_score'].quantile(high_threshold_quantile)

    # Define ranges for the 3x3 grid
    ranges = [
        ('low', lambda x: x < low_threshold_ptn1),
        ('mid', lambda x: (x >= low_threshold_ptn1) & (x <= high_threshold_ptn1)),
        ('high', lambda x: x > high_threshold_ptn1),
    ]

    # Create a matrix to represent the grid
    matrix = np.zeros((3, 3))

    # Fill the matrix with the values of the specified statistic
    for i, (_, ptn1_condition) in enumerate(ranges[::-1]): # Reverse the order of the y-axis labels
        for j, (_, ptn2_condition) in enumerate(ranges):
            subset = df[ptn1_condition(df['ptn1_score']) & ptn2_condition(df['ptn2_score'])]
            matrix[i, j] = getattr(subset['vi_distance'], statistic)()

    # Create the heatmap using Seaborn
    sns.heatmap(matrix, annot=True, xticklabels=["low", "mid", "high"], yticklabels=["high", "mid", "low"])
    plt.title(f"Heatmap of {statistic} vi_distance")
    plt.xlabel("ptn1_score")
    plt.ylabel("ptn2_score")
    plt.show()

def log_log_scatterplot(df):
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='ptn1_score', y='ptn2_score', hue='vi_distance', data=df, palette="viridis", edgecolor=None)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Log-Log Scatter Plot of ptn1_score vs ptn2_score')
    plt.xlabel('ptn1_score (log scale)')
    plt.ylabel('ptn2_score (log scale)')
    plt.legend(title='vi_distance', loc='upper left')
    plt.show()


def regionalize_data(
    df,
    low_threshold_quantile=0.2,
    high_threshold_quantile=0.8,
    distance_key="vi_distance",
    stats_suffix=None,
):
    # Determine thresholds from quantiles
    low_threshold_ptn1 = df['ptn1_score'].quantile(low_threshold_quantile)
    high_threshold_ptn1 = df['ptn1_score'].quantile(high_threshold_quantile)
    low_threshold_ptn2 = df['ptn2_score'].quantile(low_threshold_quantile)
    high_threshold_ptn2 = df['ptn2_score'].quantile(high_threshold_quantile)

    # Define regions based on thresholds
    ranges1 = [
        ('low', lambda x: x < low_threshold_ptn1),
        ('mid', lambda x: (x >= low_threshold_ptn1) & (x <= high_threshold_ptn1)),
        ('high', lambda x: x > high_threshold_ptn1),
    ]
    ranges2 = [
        ('low', lambda x: x < low_threshold_ptn2),
        ('mid', lambda x: (x >= low_threshold_ptn2) & (x <= high_threshold_ptn2)),
        ('high', lambda x: x > high_threshold_ptn2),
    ]

    # Copy original DataFrame to add "region_key" column
    regionalized_df = df.copy()
    regionalized_df['region_key'] = ''

    # Assign region_key based on ptn1_score and ptn2_score values
    for ptn1_label, ptn1_condition in ranges1:
        for ptn2_label, ptn2_condition in ranges2:
            condition = ptn1_condition(regionalized_df['ptn1_score']) & ptn2_condition(regionalized_df['ptn2_score'])
            regionalized_df.loc[condition, 'region_key'] = f"{ptn1_label}-{ptn2_label}"

    # Defining the statistics namedtuple
    if stats_suffix is True:
        stats_suffix = distance_key
    if stats_suffix:
        label = f"_{stats_suffix}"
    else:
        label = ""
    Stats = namedtuple("Stats", [
        f"count",
        f"mean{label}",
        f"median{label}",
        f"sum{label}",
        f"relative_sum{label}"
    ])

    # Function to calculate the required statistics for a given subset
    def calculate_stats(subset):
        count = len(subset)
        mean_vi = subset['vi_distance'].mean()
        median_vi = subset['vi_distance'].median()
        sum_vi = subset['vi_distance'].sum()
        relative_sum_vi = sum_vi / total_vi_sum
        return Stats(count, mean_vi, median_vi, sum_vi, relative_sum_vi)

    # Calculating the total sum of vi_distances for relative calculation
    total_vi_sum = df['vi_distance'].sum()

    # Dictionary to store the grid statistics
    grid_stats = {}
    for region_key in regionalized_df['region_key'].unique():
        subset = regionalized_df[regionalized_df['region_key'] == region_key]
        grid_stats[region_key] = calculate_stats(subset)

    # Convert grid_stats to DataFrame
    regionalized_stats_df = pd.DataFrame.from_dict(grid_stats, orient='index', columns=Stats._fields)

    return regionalized_df, regionalized_stats_df

# Example usage
# regionalized_df, regionalized_stats_df = regionalize_data(df, low_threshold_quantile=0.2, high_threshold_quantile=0.8)

