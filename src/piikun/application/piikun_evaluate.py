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
    rc = runtime.RuntimeClient()
    parser = argparse.ArgumentParser(description=None)
    source_options = parser.add_argument_group("Source Options")
    source_options.add_argument(
        "src_paths",
        action="store",
        metavar="SOURCE-FILE",
        nargs="+",
        help="Path to data source file. if not specified, defaults to standard input.",
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

    rc.output_directory = args.output_directory
    if args.output_title:
        rc.output_title = args.output_title.strip()
    elif args.src_paths:
        if len(args.src_paths) == 1:
            rc.output_title = runtime.compose_output_title_from_source(args.src_paths[0])
        else:
            rc.terminate_error(f"Multiple sources are not supported at this time", 1)
    rc.logger.info("Starting: [b]piikun-evaluate[/b]")
    if not args.src_paths:
        rc.terminate_error("Standard input piping is under development", exit_code=1)

if __name__ == "__main__":
    main()
