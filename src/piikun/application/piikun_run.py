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
import subprocess

def main():
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
            "bpp-a10",
            "bpp-a11",
            "spart-xml",
            "nested-lists",
        ],
        help="Format of source species delimitation model data [default='%(default)s'].",
    )

    data_options = parser.add_argument_group("Data Options")
    data_options.add_argument(
        "--store-source-path",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="is_store_source_path",
        help="Store/do not add a field with the data filepath as value inthe exported data.",
    )
    data_options.add_argument(
        "--add-metadata",
        **runtime.field_name_value_argument_kwargs(),
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

    runtime_context = runtime.RuntimeContext()
    runtime_context.logger.info("Starting: [b]piikun-run[/b]")

    runtime_context.output_directory = args.output_directory
    runtime_context.compose_output_title(
        output_title=args.output_title,
        source_paths=args.source_paths,
    )

    is_run_compiler = True
    is_run_evaluator = True
    is_run_visualizer = True
    command_sets = []

    if is_run_compiler:
        cmd = [ "piikun-compile" ]
        if args.source_format:
            cmd.append("--format")
            cmd.append(args.source_format)
        if args.is_store_source_path:
            cmd.append("--store-source-path")
        else:
            cmd.append("--no-store-source-path")
        if args.add_metadata:
            cmd.append("--add-metadata")
            for d1 in args.add_metadata:
                for d2 in d1:
                    cmd.append(d2)
        if args.limit_partitions:
            cmd.append("--limit-partitions")
            cmd.append(str(args.limit_partitions))
        if args.output_directory:
            cmd.append("--output-directory")
            cmd.append(args.output_directory)
        cmd.extend(args.source_paths)
        runtime_context.logger.info(f"Executing command:\n  [bold][italic]{' '.join(cmd)}[/italic][/bold]")
        cp = subprocess.run(cmd)







if __name__ == "__main__":
    main()

