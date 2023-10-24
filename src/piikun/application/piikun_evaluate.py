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
import pandas as pd
import resource
from rich import progress
from piikun import runtime
from piikun import partitionmodel

def create_full_profile_distance_df(
    profiles_df=None,
    distances_df=None,
    export_profile_columns=None,
    export_distance_columns=None,
    profiles_path=None,
    distances_path=None,
    merged_path=None,
    delimiter="\t",
    runtime_context=None,
):
    if not profiles_df:
        assert profiles_path
        profiles_df = pd.read_json(profiles_path)
    if not distances_df:
        assert distances_path
        distances_df = pd.read_json(distances_path)
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
        # progress.TextColumn("(Mem: {task.fields[memory_usage]} MB)"),
        transient=True,
    ) as progress_bar:
        n_expected_cmps = len(partition_keys) * len(partition_keys)
        task1 = progress_bar.add_task("Comparing ...", total=n_expected_cmps, memory_usage=0)
        for pkd_idx, pk1 in enumerate(partition_keys):
            # runtime_context and runtime_context.logger.info(f"Exporting partition {pkd_idx+1} of {len(partition_keys)}: '{pk1}'")
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
                # message_kwargs = {}
                # message_kwargs["memory_usage"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
                # progress_bar.update(task1, advance=1, **message_kwargs)
                progress_bar.update(task1, advance=1)
                progress_bar.refresh()
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
                            values = profiles_df[ profiles_df["partition_id"] == ptn_key ][ column ].values.tolist()
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
        runtime_context.logger.info(f"Exported distances to: '{merged_path}'")
        df.to_json(merged_path, orient="records")
    return df


def compare_partitions(
    runtime_context,
    partitions,
):
    partitions.validate(runtime_context=runtime_context)
    n_expected_cmps = int(len(partitions) * len(partitions) / 2) + int(len(partitions)/2)
    n_comparisons = 0
    seen_compares = set()
    partition_profile_store = runtime_context.open_json_list_writer(
        subtitle="profiles",
    )
    partition_oneway_distances = runtime_context.open_json_list_writer(
        subtitle="1d",
    )
    # f"[ {int(n_comparisons * 100/n_expected_cmps): 4d} % ] Comparison {n_comparisons} of {n_expected_cmps}: Partition {ptn1.label} vs. partition {ptn2.label}"
    # with runtime.get_progress_bar(text="Memory usage: {task.fields[memory_usage]}") as progress_bar:
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
    with (
        partition_profile_store,
        partition_oneway_distances,
        progress_bar,
    ):
        task1 = progress_bar.add_task("Comparing ...", total=n_expected_cmps, memory_usage=0)
        for pkey1, ptn1 in partitions._partitions.items():
            profile_d = {
                "partition_id": pkey1,
                "n_elements": ptn1.n_elements,
                "n_subsets": ptn1.n_subsets,
                "vi_entropy": ptn1.vi_entropy(),
            }
            if ptn1.metadata_d:
                profile_d.update(ptn1.metadata_d)
            partition_profile_store.write(profile_d)
            ptn1_metadata = {}
            for k, v in ptn1.metadata_d.items():
                ptn1_metadata[f"ptn1_{k}"] = v
            for pkey2, ptn2 in partitions._partitions.items():
                cmp_key = frozenset([pkey1, pkey2])
                if cmp_key in seen_compares:
                    continue
                message_kwargs = {}
                message_kwargs["memory_usage"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
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
                for value_fieldname, value_fn in (
                    ("vi_mi", ptn1.vi_mutual_information),
                    ("vi_joint_entropy", ptn1.vi_joint_entropy),
                    ("vi_distance", ptn1.vi_distance),
                    ("vi_normalized_kraskov", ptn1.vi_normalized_kraskov),
                ):
                    comparison_d[value_fieldname] = value_fn(ptn2)
                partition_oneway_distances.write(comparison_d)
    runtime_context.logger.info("Comparison completed")
    partition_profile_store.close()
    partition_oneway_distances.close()
    partition_twoway_distances = runtime_context.open_json_list_writer(
        subtitle="distances",
    )
    create_full_profile_distance_df(
        profiles_path=partition_profile_store.path,
        distances_path=partition_oneway_distances.path,
        merged_path=partition_twoway_distances.path,
        runtime_context=runtime_context,
    )
    partition_twoway_distances.close()

# print("Memory usage: {} MB".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
# if n_comparisons == 0 or (n_comparisons % progress_step) == 0:
#     self.logger.log_info(
#         f"[ {int(n_comparisons * 100/n_expected_cmps): 4d} % ] Comparison {n_comparisons} of {n_expected_cmps}: Partition {ptn1.label} vs. partition {ptn2.label}"
#     )
def main():
    parser = argparse.ArgumentParser(description=None)
    source_options = parser.add_argument_group("Source Options")
    source_options.add_argument(
        "source_paths",
        action="store",
        metavar="SOURCE-FILE",
        nargs="+",
        help="Path to data source file.",
    )
    # source_options.add_argument(
    #     "-f",
    #     "--format",
    #     action="store",
    #     dest="source_format",
    #     default="piikun",
    #     choices=[
    #         "piikun",
    #         "delineate",
    #         # "bpp-a10",
    #         "bpp-a11",
    #         "json-lists",
    #         "spart-xml",
    #     ],
    #     help="Format of source species delimitation model data [default='%(default)s'].",
    # )
    source_options.add_argument(
        "--limit-partitions",
        action="store",
        default=None,
        type=int,
        help="Limit data to this number of partitions.",
    )
    output_options = parser.add_argument_group("Output Options")
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
    # run_options = parser.add_argument_group("Run Options")
    # parser.add_argument(
    #         "--verbosity",
    #         action="store",
    #         default=3,
    #         help="Run noise level [default=%(default)s].")
    args = parser.parse_args()
    runtime_context = runtime.RuntimeContext()
    runtime_context.logger.info("Starting: [b]piikun-evaluate[/b]")
    args.source_format = "piikun"

    if not args.source_paths:
        runtime_context.terminate_error("Standard input piping is under development", exit_code=1)

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
            source_format=args.source_format,
            limit_partitions=args.limit_partitions,
            runtime_context=runtime_context,
        )
    compare_partitions(
        partitions=partitions,
        runtime_context=runtime_context,
    )

if __name__ == "__main__":
    main()
