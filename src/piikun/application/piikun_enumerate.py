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
import math
import xml.etree.ElementTree as ET

import yakherd
from piikun import partitionmodel
from piikun import utility

def execute_enumerate_partitions(collection, config_d):
    d = []
    for ptn in partitionmodel.iterate_partitions(
        collection=collection,
    ):
        d.append(ptn)
    # if config_d["format"] == "json":
    json_lists = [json.dumps(list_item) for list_item in d]
    result = '[\n  ' + ',\n  '.join(json_lists) + '\n]'
    sys.stdout.write(f"{result}\n")

def read_and_process_lines(source):
    return [label for line in source for label in line.strip().split()]

def handle_input(args):
    labels = []
    if args.from_file:
        with open(args.from_file, 'r') as source:
            labels.extend(read_and_process_lines(source))
    for label in args.labels:
        if label == "-":
            labels.extend(read_and_process_lines(sys.stdin))
        else:
            labels.append(label)
    labels = sorted(set(labels))
    return labels

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
        'labels',
        type=str,
        nargs='*',
        help='List of labels or "-" to read from standard input.',
    )
    parser.add_argument(
        '--from-file',
        type=str,
        help='Path to a file containing a list of labels.',
    )
    # parser.add_argument(
    #         "-f", "--output-format",
    #         action="store",
    #         default="json",
    #         help="Format for output.")
    # parser.add_argument(
    #         "-n", "--num-elements",
    #         action="store",
    #         type=int,
    #         default=None,
    #         help="Number of elements.")
    args = parser.parse_args()
    config_d = dict(vars(args))
    labels = handle_input(args)
    if not labels:
        sys.exit("No elements defined.")
    execute_enumerate_partitions(
        collection=labels,
        config_d=config_d,
    )

if __name__ == "__main__":
    main()


if __name__ == '__main__':
    main()



