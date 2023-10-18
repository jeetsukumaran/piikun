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

import json
from piikun import runtime
from . import partitionmodel

def parse_piikun_json(
    partition_factory,
    *,
    source_stream=None,
    source_data=None,
):
    runtime.logger.info("Parsing 'pikkun-json' format")
    if not source_data:
        source_data = source_stream.read()
    data_d = json.loads(source_data)
    partition_ds = data_d["partitions"]
    for ptn_idx, partition_d in enumerate(src_partitions):
        partition = partition_factory(
            subsets=partition_d["subsets"],
            metadata_d=partition_d["metadata"],
        )
        yield partition

def parse_delineate(
    partition_factory,
    *,
    source_stream=None,
    source_data=None,
):
    runtime.logger.info("Parsing 'delineate' format")
    if not source_data:
        source_data = source_stream.read()
    delineate_results = json.loads(source_data)
    src_partitions = delineate_results["partitions"]
    runtime.logger.info(f"{len(src_partitions)} partitions in source")
    for spart_idx, src_partition in enumerate(src_partitions):
        partition_data = src_partition["species_leafsets"]
        if not isinstance(partition_data, dict):
            # delineate legacy format!
            subsets = partition_data
        else:
            subsets = partition_data.values()
        runtime.logger.info(
            f"Partition {spart_idx+1:>5d} of {len(src_partitions)} ({len(subsets)} subsets)"
        )
        metadata_d = {
            "constrained_probability": src_partition.get(
                "constrained_probability", 0
            ),
            "unconstrained_probability": src_partition.get(
                "unconstrained_probability", 0
            ),
            "support": src_partition.get("unconstrained_probability", 0),
        }
        kwargs = {
            # "label": spart_idx + 1,
            "metadata_d": metadata_d,
            "subsets": subsets,
        }
        partition = partition_factory(**kwargs)
        yield partition

class Parser:

    format_parser_map = {
        "piikun-json": parse_piikun_json,
        "delineate": parse_delineate,
    }

    def __init__(
        self,
        source_format,
        partition_factory=None,
    ):
        self.source_format = source_format
        self.partition_factory = partition_factory

    @property
    def parse_fn(self):
        if (
            not hasattr(self, "_parse_fn")
            or self._parse_fn is None
        ):
            try:
                self._parse_fn = self.format_parser_map[self.source_format]
            except KeyError:
                runtime.terminate_error(
                    message=f"Unrecognized source format: '{self.source_format}'\nSupported formats: {list(self.format_parser_map.keys())}",
                    exit_code=1,
                )
        return self._parse_fn

    @property
    def partition_factory(self):
        if (
            not hasattr(self, "_partition_factory")
            or self._partition_factory is None
        ):
            # runtime.logger.error(f"Partition factory not defined")
            runtime.terminate_error(
                message="Partition factory not defined",
                exit_code=1,
            )
        return self._partition_factory
    @partition_factory.setter
    def partition_factory(self, value):
        self._partition_factory = value

    def read_path(
        self,
        source,
    ):
        source_stream = open(source)
        return self.read_stream(source_stream)

    def read_stream(
        self,
        source,
    ):
        return self.parse_fn(
            source_stream=source,
            partition_factory=self.partition_factory
        )
