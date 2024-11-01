#! /usr/bin/env python
# -*- coding: utf-8 -*-

#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pathlib
import sys
import argparse
import json
import pandas as pd
import resource
from rich import progress
from piikun import runtime
from piikun import partitionmodel
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class OutputFormatConfig:
    """Configuration for output formats"""
    format_name: str
    file_extension: str
    delimiter: Optional[str] = None

    @property
    def is_delimited(self):
        return bool(self.delimiter)

class OutputFormatRegistry:
    """Registry of supported output formats"""

    def __init__(self):
        self._formats = {}
        # Register default formats
        self.register_format(OutputFormatConfig("json", "json"))
        self.register_format(OutputFormatConfig("csv", "csv", delimiter=","))
        self.register_format(OutputFormatConfig("tsv", "tsv", delimiter="\t"))

    def register_format(self, format_config: OutputFormatConfig):
        """Register a new output format"""
        self._formats[format_config.format_name] = format_config

    def get_format(self, format_name: str) -> OutputFormatConfig:
        """Get format configuration by name"""
        if format_name not in self._formats:
            raise ValueError(f"Unsupported format: {format_name}")
        return self._formats[format_name]

    @property
    def supported_formats(self) -> List[str]:
        """List of supported format names"""
        return list(self._formats.keys())

class OutputManager:
    """Manages multiple output formats and file handles"""

    def __init__(self, runtime_context, format_registry):
        self.runtime_context = runtime_context
        self.format_registry = format_registry
        self.active_formats = set()
        self.writers = {}

    def add_format(self, format_name: str):
        """Add an output format to the active set"""
        if format_name not in self.format_registry.supported_formats:
            raise ValueError(f"Unsupported format: {format_name}")
        self.active_formats.add(format_name)

    def open_writers(self, subtitle: str) -> Dict:
        """Open writers for all active formats"""
        writers = {}
        for fmt in self.active_formats:
            format_config = self.format_registry.get_format(fmt)
            writer = self.runtime_context.open_data_writer(
                subtitle=f"{subtitle}.{fmt}",
                format=fmt,
            )
            writers[fmt] = writer
        return writers

    def write_data(self, writers: Dict, data: Dict):
        """Write data to all active format writers"""
        for fmt, writer in writers.items():
            if self.format_registry.get_format(fmt).is_delimited:
                writer.write_d(data)
            else:
                writer.write(data)

    def close_writers(self, writers: Dict):
        """Close all writers"""
        for writer in writers.values():
            writer.close()

def read_data_file(
    filepath: Union[str, pathlib.Path],
    format: str,
    format_registry: OutputFormatRegistry
) -> pd.DataFrame:
    """
    Read data from a file in the specified format into a pandas DataFrame.

    Parameters
    ----------
    filepath : str or pathlib.Path
        Path to the data file
    format : str
        Format of the data file
    format_registry : OutputFormatRegistry
        Registry containing format configurations

    Returns
    -------
    pandas.DataFrame
        Data from the file as a DataFrame
    """
    format_config = format_registry.get_format(format)

    if format == "json":
        return pd.read_json(filepath)
    elif format_config.is_delimited:
        return pd.read_csv(filepath, sep=format_config.delimiter)
    else:
        raise ValueError(f"Unsupported format: {format}")

def create_full_profile_distance_df(
    profiles_df: Optional[pd.DataFrame] = None,
    distances_df: Optional[pd.DataFrame] = None,
    export_profile_columns: Optional[List[str]] = None,
    export_distance_columns: Optional[List[str]] = None,
    profiles_path: Optional[Union[str, pathlib.Path]] = None,
    distances_path: Optional[Union[str, pathlib.Path]] = None,
    merged_path: Optional[Union[str, pathlib.Path]] = None,
    output_format: str = "json",
    format_registry: Optional[OutputFormatRegistry] = None,
    runtime_context=None,
) -> pd.DataFrame:
    """
    Create a full profile distance DataFrame by combining profile and distance data.

    Parameters
    ----------
    profiles_df : pandas.DataFrame, optional
        DataFrame containing profile data
    distances_df : pandas.DataFrame, optional
        DataFrame containing distance data
    export_profile_columns : list of str, optional
        Specific profile columns to export
    export_distance_columns : list of str, optional
        Specific distance columns to export
    profiles_path : str or pathlib.Path, optional
        Path to profiles data file if profiles_df not provided
    distances_path : str or pathlib.Path, optional
        Path to distances data file if distances_df not provided
    merged_path : str or pathlib.Path, optional
        Path to write merged data
    output_format : str, default="json"
        Format for output file
    format_registry : OutputFormatRegistry, optional
        Registry of format configurations
    runtime_context : RuntimeContext, optional
        Runtime context for logging and progress tracking

    Returns
    -------
    pandas.DataFrame
        Combined profile and distance data
    """
    if format_registry is None:
        format_registry = OutputFormatRegistry()

    if not profiles_df and profiles_path:
        profiles_df = read_data_file(profiles_path, output_format, format_registry)
    if not distances_df and distances_path:
        distances_df = read_data_file(distances_path, output_format, format_registry)

    if export_profile_columns:
        profile_columns = list(export_profile_columns)
    else:
        profile_columns = [
            column
            for column in profiles_df
            if column not in set(["partition_id", "label"])
        ]

    if export_distance_columns:
        distances_columns = list(export_distance_columns)
    else:
        distances_columns = [
            key for key in distances_df.columns if not key.startswith("ptn")
        ]

    partition_keys = list(profiles_df["partition_id"])
    new_dataset = []

    with progress.Progress(
        progress.SpinnerColumn(),
        progress.TextColumn("Exporting:"),
        progress.MofNCompleteColumn(),
        progress.BarColumn(),
        progress.TaskProgressColumn(),
        progress.TimeElapsedColumn(),
        progress.TimeRemainingColumn(),
        transient=True,
        console=runtime_context.console if runtime_context else None,
    ) as progress_bar:
        n_expected_cmps = len(partition_keys) * len(partition_keys)
        task1 = progress_bar.add_task(
            "Comparing ...", total=n_expected_cmps, memory_usage=0
        )

        for pkd_idx, pk1 in enumerate(partition_keys):
            seen_comparisons = set()
            for pk2 in partition_keys:
                key = (pk1, pk2)
                if key in seen_comparisons:
                    continue
                seen_comparisons.add(key)
                progress_bar.update(task1, advance=1)
                progress_bar.refresh()

                condition = (
                    (distances_df["ptn1"] == pk1) & (distances_df["ptn2"] == pk2)
                ) | ((distances_df["ptn1"] == pk2) & (distances_df["ptn2"] == pk1))
                dists_sdf = distances_df[condition]

                if len(dists_sdf) == 0 and pk1 != pk2:
                    raise ValueError(f"Missing non-self comparison: {pk1}, {pk2}")
                elif len(dists_sdf) == 1:
                    row_d = {}
                    for ptn_idx, ptn_key in zip((1, 2), (pk1, pk2)):
                        row_d[f"ptn{ptn_idx}"] = ptn_key
                    for ptn_idx, ptn_key in zip((1, 2), (pk1, pk2)):
                        for column in profile_columns:
                            values = profiles_df[
                                profiles_df["partition_id"] == ptn_key
                            ][column].values.tolist()
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
        format_config = format_registry.get_format(output_format)
        if output_format == "json":
            df.to_json(merged_path, orient="records")
        elif format_config.is_delimited:
            df.to_csv(merged_path, sep=format_config.delimiter, index=False)
        if runtime_context:
            runtime_context.logger.info(f"Exported distances to: '{merged_path}'")

    return df

def compare_partitions(
    runtime_context,
    partitions,
    output_formats: List[str],
):
    """
    Compare partitions and output results in specified formats.

    Parameters
    ----------
    runtime_context : RuntimeContext
        Runtime context for logging and output management
    partitions : PartitionCollection
        Collection of partitions to compare
    output_formats : list of str
        List of output formats to generate
    """
    format_registry = OutputFormatRegistry()
    output_manager = OutputManager(runtime_context, format_registry)

    # Add requested formats
    for fmt in output_formats:
        output_manager.add_format(fmt)

    partitions.validate(runtime_context=runtime_context)
    n_expected_cmps = int(len(partitions) * len(partitions) / 2) + int(
        len(partitions) / 2
    )
    n_comparisons = 0
    seen_compares = set()

    # Open writers for all formats
    profile_writers = output_manager.open_writers("profiles")
    oneway_distance_writers = output_manager.open_writers("1d")

    progress_bar = progress.Progress(
        progress.TextColumn("({task.fields[memory_usage]} MB)"),
        progress.SpinnerColumn(),
        progress.TextColumn("Comparison:"),
        progress.MofNCompleteColumn(),
        progress.BarColumn(),
        progress.TaskProgressColumn(),
        progress.TimeElapsedColumn(),
        progress.TimeRemainingColumn(),
        transient=True,
    )

    with progress_bar:
        comparison_evaluation_fns = {
            "vi_mi": lambda ptn1, ptn2: ptn1.vi_mutual_information(ptn2),
            "vi_joint_entropy": lambda ptn1, ptn2: ptn1.vi_joint_entropy(ptn2),
            "vi_distance": lambda ptn1, ptn2: ptn1.vi_distance(ptn2),
            "vi_normalized_kraskov": lambda ptn1, ptn2: ptn1.vi_normalized_kraskov(ptn2),
            "hamming_loss": lambda ptn1, ptn2: ptn1.hamming_loss(ptn2),
            "ahrens_match_ratio": lambda ptn1, ptn2: ptn1.ahrens_match_ratio(ptn2),
        }

        task1 = progress_bar.add_task(
            "Comparing ...", total=n_expected_cmps, memory_usage=0
        )

        for pkey1, ptn1 in partitions._partitions.items():
            profile_d = {
                "partition_id": pkey1,
                "n_elements": ptn1.n_elements,
                "n_subsets": ptn1.n_subsets,
                "vi_entropy": ptn1.vi_entropy(),
            }
            if ptn1.metadata_d:
                profile_d.update(ptn1.metadata_d)
            output_manager.write_data(profile_writers, profile_d)

            ptn1_metadata = {}
            for k, v in ptn1.metadata_d.items():
                ptn1_metadata[f"ptn1_{k}"] = v

            for pkey2, ptn2 in partitions._partitions.items():
                cmp_key = frozenset([pkey1, pkey2])
                if cmp_key in seen_compares:
                    continue

                message_kwargs = {
                    "memory_usage": int(
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    )
                }
                progress_bar.update(task1, advance=1, **message_kwargs)
                progress_bar.refresh()

                seen_compares.add(cmp_key)
                n_comparisons += 1

                comparison_d = {
                    "ptn1": pkey1,
                    "ptn2": pkey2,
                }
                comparison_d.update(ptn1_metadata)

                for k, v in ptn2.metadata_d.items():
                    comparison_d[f"ptn2_{k}"] = v

                comparison_d["vi_entropy_ptn1"] = ptn1.vi_entropy()
                comparison_d["vi_entropy_ptn2"] = ptn2.vi_entropy()

                for value_fieldname, value_fn in comparison_evaluation_fns.items():
                    comparison_d[value_fieldname] = value_fn(ptn1, ptn2)

                output_manager.write_data(oneway_distance_writers, comparison_d)

    runtime_context.logger.info("Comparison completed")
    output_manager.close_writers(profile_writers)
    output_manager.close_writers(oneway_distance_writers)

    # Handle two-way distances for each format
    twoway_distance_writers = output_manager.open_writers("distances")

    for fmt in output_formats:
        format_config = format_registry.get_format(fmt)
        df = create_full_profile_distance_df(
            profiles_path=profile_writers[fmt].path,
            distances_path=oneway_distance_writers[fmt].path,
            export_distance_columns=list(comparison_evaluation_fns.keys()),
            runtime_context=runtime_context,
            output_format=fmt,
            format_registry=format_registry,
        )

        if format_config.is_delimited:
            for record in df.to_dict('records'):
                twoway_distance_writers[fmt].write_d(record)
        else:
            df.to_json(twoway_distance_writers[fmt].file_handle, orient="records")
    output_manager.close_writers(twoway_distance_writers)

def main():
    """Main entry point for the piikun-evaluate script."""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare partitions with flexible output format support"
    )

    source_options = parser.add_argument_group("Source Options")
    source_options.add_argument(
        "source_paths",
        action="store",
        metavar="SOURCE-FILE",
        nargs="+",
        help="Path to data source file.",
    )
    source_options.add_argument(
        "--limit-partitions",
        action="store",
        default=None,
        type=int,
        help="Limit data to this number of partitions.",
    )

    output_options = parser.add_argument_group("Output Options")
    output_options.add_argument(
        "--format",
        action="append",
        choices=OutputFormatRegistry().supported_formats,
        default=None,
        help="Output format(s) for results. Can be specified multiple times for multiple formats. Default: json",
    )
    output_options.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        dest="is_validate",
        default=True,
        help="Check each partition definition ensuring that every lineage is represented exactly once.",
    )
    output_options.add_argument(
        "-o",
        "--output-title",
        action="store",
        default=None,
        help="Prefix for output filenames (specify '-' for standard output).",
    )
    output_options.add_argument(
        "-O",
        "--output-directory",
        action="store",
        default=os.curdir,
        help="Directory for output files [default='%(default)s'].",
    )
    output_options.add_argument(
        "--print-artifact-paths",
        dest="is_print_output_paths",
        action="store_true",
        default=None,
        help="Print a (JSON) dictionary of output files: ``{ '<output-type>': '<output-filepath>' }``.",
    )

    args = parser.parse_args()

    # Set default format if none specified
    if not args.format:
        args.format = ["json"]

    runtime_context = runtime.RuntimeContext()
    runtime_context.logger.info("Starting: [b]piikun-evaluate[/b]")

    if not args.source_paths:
        runtime_context.terminate_error(
            "Standard input piping is under development", exit_code=1
        )

    runtime_context.output_directory = args.output_directory
    runtime_context.compose_output_title(
        output_title=args.output_title,
        source_paths=args.source_paths,
        title_from_source_stem_fn=lambda x: x.split("__")[0],
        is_merge_output=True,
    )

    partitions = partitionmodel.PartitionCollection()
    for sidx, source_path in enumerate(args.source_paths):
        partitions.read(
            source_path=source_path,
            source_format="piikun",  # hardcoded as per original
            limit_partitions=args.limit_partitions,
            runtime_context=runtime_context,
        )

    compare_partitions(
        partitions=partitions,
        runtime_context=runtime_context,
        output_formats=args.format,
    )

    if args.is_print_output_paths:
        sys.stdout.write(json.dumps(runtime_context.output_tracker) + "\n")

if __name__ == "__main__":
    main()
