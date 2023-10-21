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
from piikun import runtime
from piikun import parse
from piikun import partitionmodel

def main():
    parser = argparse.ArgumentParser(description=None)
    source_options = parser.add_argument_group("Source Options")
    source_options.add_argument(
        "src_paths",
        action="store",
        metavar="FILE",
        nargs="*",
        help="Path to data source file(s); if not specified, defaults to standard input.",
    )
    source_options.add_argument(
        "-f",
        "--format",
        action="store",
        dest="source_format",
        default=None,
        choices=[
            "delineate",
            # "bpp-a10",
            "bpp-a11",
            "json-lists",
            "spart-xml",
            "piikun",
        ],
        help="Format of source species delimitation model data: [default='delineate'].",
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
        "--merge",
        action=argparse.BooleanOptionalAction,
        dest="is_merge_output",
        default=True,
        help="Merge partitions into single output file ('merge') or otherwise keep separate ('--no-merge') [default='merge'].",
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
    # run_options = parser.add_argument_group("Run Options")
    # parser.add_argument(
    #         "--verbosity",
    #         action="store",
    #         default=3,
    #         help="Run noise level [default=%(default)s].")
    args = parser.parse_args()
    output_title = None
    if args.output_title:
        output_title = args.output_title.strip()
    elif args.src_paths:
        if len(args.src_paths) == 1:
            output_title = runtime.compose_output_title_from_source(args.src_paths[0])
        elif args.is_merge_output and len(args.src_paths) > 1:
            output_title = (
                runtime.compose_output_title_from_source(args.src_paths[0]) + "+others"
            )
    rc = runtime.RuntimeClient(
        output_title=output_title,
        output_directory=args.output_directory,
    )
    rc.logger.info("Starting: [b]piikun-compile[/b]")
    if not args.src_paths:
        rc.terminate_error("Standard input piping is under development", exit_code=1)
    if not args.source_format:
        args.source_format = "delineate"
    src_paths = args.src_paths
    parser = parse.Parser(
        source_format=args.source_format,
    )
    rc.logger.info(f"{len(src_paths)} sources to parse")
    partitions = None

    for src_idx, src_path in enumerate(src_paths):
        rc.logger.info(
            f"Reading source {src_idx+1} of {len(src_paths)}: '{src_path}'"
        )
        if not partitions or not args.is_merge_output:
            partitions = partitionmodel.PartitionCollection()
            parser.partition_factory = partitions.new_partition
        start_len = len(partitions)
        for pidx, ptn in enumerate(parser.read_path(src_path)):
            rc.logger.info(
                # f"Partition {pidx+1:>5d} of {len(src_partitions)} ({len(subsets)} subsets)"
                f"Partition {pidx+1:>5d}: {ptn.n_elements} lineages organized into {ptn.n_subsets} species"
            )
            # if rc.output_title and rc.output_title != "-":
            if args.is_validate:
                partitions.validate(logger=rc.logger)
            # -1 as we need to anticipate limit being reached in the next loop
            if args.limit_partitions and (pidx >= args.limit_partitions - 1):
                rc.logger.info(
                    f"Number of partitions read is at limit ({args.limit_partitions}): skipping remaining"
                )
                break
        end_len = len(partitions)
        rc.logger.info(
            f"{end_len - start_len} partitions read from source ({len(partitions)} read in total)"
        )
        if not args.is_merge_output:
            rc.output_title = runtime.compose_output_title_from_source(src_path)
        if not args.is_merge_output or src_idx == len(src_paths)-1:
            if rc.output_title:
                out = rc.open_output(subtitle="partitions", ext="json")
                rc.logger.info(f"Writing {len(partitions)} partitions to file: '{out.name}'")
            else:
                out = sys.stdout
                rc.logger.info("(Writing to standard output)")
            partition_source_data = partitions.export_source_data()
            out.write(json.dumps(partition_source_data))
            out.write("\n")
            out.close()
    # if args.is_merge_output:
    #     _store_partitions(
    #         partitions=partitions,
    #         rc=rc,
    #     )


if __name__ == "__main__":
    main()
