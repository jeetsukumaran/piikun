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
from piikun import partitionmodel

def main():
    rc = runtime.RuntimeClient()
    parser = argparse.ArgumentParser(description=None)
    source_options = parser.add_argument_group("Source Options")
    source_options.add_argument(
        "source_paths",
        action="store",
        metavar="FILE",
        nargs="+",
        help="Path to data source file(s).",
    )
    source_options.add_argument(
        "-f",
        "--format",
        action="store",
        dest="source_format",
        default="piikun",
        choices=[
            "piikun",
            "delineate",
            # "bpp-a10",
            "bpp-a11",
            "json-lists",
            "spart-xml",
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
    args = parser.parse_args()

    rc.logger.info("Starting: [b]piikun-compile[/b]")

    if not args.source_paths:
        rc.terminate_error("Standard input piping is under development", exit_code=1)
    source_paths = args.source_paths
    rc.logger.info(f"{len(source_paths)} sources to parse")

    rc.output_directory = args.output_directory
    rc.compose_output_title(
        output_title=args.output_title,
        source_paths=source_paths,
        is_merge_output=args.is_merge_output,
    )

    partitions = None
    for src_idx, source_path in enumerate(source_paths):
        # rc.console.rule()
        rc.logger.info(
            f"Reading source {src_idx+1} of {len(source_paths)}: '{source_path}'"
        )
        if not args.is_merge_output:
            rc.output_title = runtime.compose_output_title(source_paths=[source_path])
        if not partitions or not args.is_merge_output:
            partitions = partitionmodel.PartitionCollection()
        partitions.read(
            source_path=source_path,
            source_format=args.source_format,
            limit_partitions=args.limit_partitions,
            rc=rc,
        )
        if not args.is_merge_output or src_idx == len(source_paths)-1:
            # rc.console.rule()
            if args.is_validate:
                partitions.validate(rc=rc)
            if rc.output_title:
                out = rc.open_output(subtitle="partitions", ext="json")
                rc.logger.info(f"Writing {len(partitions)} partitions to file: '{out.name}'")
            else:
                out = sys.stdout
                rc.logger.info("(Writing to standard output)")
            partition_definition_d = partitions.export_definition_d()
            out.write(json.dumps(partition_definition_d))
            out.write("\n")
            out.close()

if __name__ == "__main__":
    main()
