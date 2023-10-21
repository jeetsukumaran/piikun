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
from rich import progress
from piikun import runtime
from piikun import partitionmodel

def compare_partitions(
    rc,
    partitions,
    is_mirror=False,
):
    if is_mirror:
        n_expected_cmps = len(partitions) * len(partitions)
    else:
        n_expected_cmps = int(len(partitions) * len(partitions) / 2)
    n_comparisons = 0
    seen_compares = set()
    progress_bar = progress.Progress(
        # console=rc.console,
    )
    task1 = progress_bar.add_task("Comparing partitions ...", total=n_expected_cmps)
    partition_oneway_distances = rc.open_output_datastore(
        subtitle="1d",
        ext="tsv",
    )
    # f"[ {int(n_comparisons * 100/n_expected_cmps): 4d} % ] Comparison {n_comparisons} of {n_expected_cmps}: Partition {ptn1.label} vs. partition {ptn2.label}"
    with progress_bar:
        for pkey1, ptn1 in partitions._partitions.items():
            ptn1_metadata = {}
            for k, v in ptn1.metadata_d.items():
                ptn1_metadata[f"ptn1_{k}"] = v
            for pkey2, ptn2 in partitions._partitions.items():
                cmp_key = frozenset([pkey1, pkey2])
                if not is_mirror and cmp_key in seen_compares:
                    continue
                seen_compares.add(cmp_key)
                n_comparisons += 1
                progress_bar.update(task1)
                progress_bar.refresh()
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
                partition_oneway_distances.write_d(comparison_d)
        self.partition_profile_store.close()
        partition_oneway_distances.close()
        utility.create_full_profile_distance_df(
            profiles_path=self.partition_profile_store.path,
            distances_path=self.partition_oneway_distances.path,
            merged_path=self.partition_twoway_distances.path,
            logger=self.logger,
        )
        self.partition_twoway_distances.close()

# print("Memory usage: {} MB".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
# if n_comparisons == 0 or (n_comparisons % progress_step) == 0:
#     self.logger.log_info(
#         f"[ {int(n_comparisons * 100/n_expected_cmps): 4d} % ] Comparison {n_comparisons} of {n_expected_cmps}: Partition {ptn1.label} vs. partition {ptn2.label}"
#     )
def main():
    rc = runtime.RuntimeClient()
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
    args.source_format = "piikun"

    rc.logger.info("Starting: [b]piikun-evaluate[/b]")

    if not args.source_paths:
        rc.terminate_error("Standard input piping is under development", exit_code=1)

    rc.output_directory = args.output_directory
    rc.compose_output_title(
        output_title=args.output_title,
        source_paths=args.source_paths,
    )
    partitions = partitionmodel.PartitionCollection()
    for sidx, source_path in enumerate(args.source_paths):
        partitions.read(
            source_path=source_path,
            source_format=args.source_format,
            limit_partitions=args.limit_partitions,
            rc=rc,
        )
    compare_partitions(
        partitions=partitions,
        rc=rc,
        is_mirror=False,
    )

if __name__ == "__main__":
    main()
