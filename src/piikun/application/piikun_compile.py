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

    def _field_name_value_type(field_spec):
        try:
            field_name, field_value = field_spec.split("=")
        except IndexError:
            sys.exit(f"Specification not in '<name>=<value>' format: '{field_spec}'")
        d = {field_name: field_value}
        return d

    data_options.add_argument(
        "--add-metadata",
        dest="add_metadata",
        action="append",
        default=None,
        nargs="+",
        type=_field_name_value_type,
        help=(
            "Add data field/values to the exported data using the syntax"
            " '<field_name>=<field_value>'. Multiple field/values"
            " can be specified."
            " For e.g., '--add-metadata n_genes=65 guide_tree=starbeast-20231023.04'."
            " This can be useful in pipelines or analyses to track workflow "
            " metadata or provenance."
        ),
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
    output_options.add_argument(
        "--print-artifact-paths",
        dest="is_print_artifact_paths",
        action="store_true",
        default=None,
        help="Print a (JSON) dictionary of output files: ``{ '<source-filepath>': '<output-filepath>' }``.",
    )
    args = parser.parse_args()

    runtime_context = runtime.RuntimeContext()
    runtime_context.logger.info("Starting: [b]piikun-compile[/b]")

    if not args.source_paths:
        runtime_context.terminate_error("Standard input piping is under development", exit_code=1)
    source_paths = args.source_paths
    runtime_context.logger.info(f"{len(source_paths)} sources to parse")

    runtime_context.output_directory = args.output_directory
    runtime_context.compose_output_title(
        output_title=args.output_title,
        source_paths=source_paths,
        is_merge_output=args.is_merge_output,
    )

    update_metadata = {}
    if args.add_metadata:
        for a1 in args.add_metadata:
            for a2 in a1:
                update_metadata.update(a2)

    partitions = None
    output_paths = {}
    for src_idx, source_path in enumerate(source_paths):
        # runtime_context.console.rule()
        # runtime_context.logger.info(
        #     f"Reading source {src_idx+1} of {len(source_paths)}: '{source_path}'"
        # )
        if not args.is_merge_output:
            runtime_context.output_title = runtime.compose_output_title(
                source_paths=[source_path],
            )
        if not partitions or not args.is_merge_output:
            partitions = partitionmodel.PartitionCollection()
        partitions.read(
            source_path=source_path,
            source_format=args.source_format,
            limit_partitions=args.limit_partitions,
            is_store_source_path=args.is_store_source_path,
            update_metadata=update_metadata,
            runtime_context=runtime_context,
        )
        # partitions.update_metadata()
        if not args.is_merge_output or src_idx == len(source_paths) - 1:
            # runtime_context.console.rule()
            if args.is_validate:
                partitions.validate(runtime_context=runtime_context)
            if runtime_context.output_title:
                out = runtime_context.open_output(subtitle="partitions", ext="json")
                runtime_context.logger.info(
                    f"Writing {len(partitions)} partitions to file: '{out.name}'"
                )
                output_path = pathlib.Path(out.name).absolute()
            else:
                out = sys.stdout
                output_path = "<stdin>"
                runtime_context.logger.info("(Writing to standard output)")
            output_paths[str(pathlib.Path(source_path).absolute())] = str(output_path)
            partition_definition_d = partitions.export_definition_d()
            out.write(json.dumps(partition_definition_d))
            out.write("\n")
            out.close()
    if args.is_print_artifact_paths:
        sys.stdout.write(json.dumps(output_paths) + "\n")

if __name__ == "__main__":
    main()
