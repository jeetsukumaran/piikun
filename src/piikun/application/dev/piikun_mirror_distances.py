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
from piikun import utility

def create_ds(
    profiles_path,
    distances_path,
    merged_path=None,
    ):
    dfm = utility.create_profile_distance_df(
        profiles_path=profiles_path,
        distances_path=distances_path,
        merged_path=merged_path,
        delimiter="\t",
    )

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument(
            "-p", "--profiles",
            required=True,
            action="store",
            help="Path to partition profiles.")
    parser.add_argument(
            "-d", "--distances",
            required=True,
            action="store",
            help="Path to partition distances.")
    parser.add_argument(
        "-o",
        "--output-title",
        action="store",
        default="piikun-full-profile-distances",
        help="Prefix for output filenames [default='%(default)s'].",
    )
    parser.add_argument(
        "-O",
        "--output-directory",
        action="store",
        default=os.curdir,
        help="Directory for output files [default='%(default)s'].",
    )
    args = parser.parse_args()
    merged_path = pathlib.Path(args.output_directory) / pathlib.Path(args.output_title + ".tsv")
    print(merged_path)
    create_ds(
        profiles_path=args.profiles,
        distances_path=args.distances,
        merged_path=merged_path,
    )

if __name__ == '__main__':
    main()
