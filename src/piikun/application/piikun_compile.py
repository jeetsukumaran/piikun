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
from piikun.runtime import console, logger
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
            "json-list",
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
        "-o",
        "--output-title",
        action="store",
        default=None,
        help="Prefix for output filenames.",
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

    logger.info("Starting: [b]piikun-compile[/b]")
    if not args.source_format:
        args.source_format = "delineate"
    parser = parse.Parser(
        source_format=args.source_format,
    )
    if not args.src_paths:
        partitions = partitionmodel.PartitionCollection()
        parser.partition_factory = partitions.new_partition
        logger.info("(Reading from standard input)")
        for pidx, ptn in enumerate(parser.read_stream(sys.stdin)):
            pass
        partitions.store_source_data(sys.stdout)
    else:
        src_data = None
        src_paths = args.src_paths
        partitions = partitionmodel.PartitionCollection()
        for src_idx, src_path in enumerate(src_paths):
            for pidx, ptn in enumerate(parser.read_path(src_path)):
                pass

if __name__ == '__main__':
    main()
