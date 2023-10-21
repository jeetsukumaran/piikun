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

import pathlib
import json
import xml.etree.ElementTree as ET
from piikun import runtime
from piikun import parsebpp
from . import partitionmodel

def parse_piikun_json(
    source_stream,
    partition_factory,
):
    runtime.logger.info("Parsing 'pikkun-json' format")
    source_data = source_stream.read()
    data_d = json.loads(source_data)
    partition_ds = data_d["partitions"]
    runtime.logger.info(f"{len(partition_ds)} partitions in source")
    for ptn_idx, (partition_key, partition_d) in enumerate(partition_ds.items()):
        subsets = partition_d["subsets"]
        metadata_d = partition_d["metadata"]
        runtime.logger.info(
            f"Partition {ptn_idx+1:>5d} of {len(partition_ds)} ({len(subsets)} subsets)"
        )
        partition = partition_factory(
            subsets=subsets,
            metadata_d=metadata_d,
        )
        partition._origin_size = len(partition_ds)
        partition._origin_offset = ptn_idx
        yield partition

def parse_delineate(
    source_stream,
    partition_factory,
    # runtime_client,
):
    # import time
    source_data = source_stream.read()
    delineate_results = json.loads(source_data)
    src_partitions = delineate_results["partitions"]
    for ptn_idx, src_partition in enumerate(src_partitions):
        try:
            partition_data = src_partition["species_leafsets"]
        except TypeError as e:
            runtime.terminate_error(
                message=f"Invalid 'delineate' format:\nPartition {ptn_idx+1}: partitions dictionary 'species_leafsets' element is not a list",
                exit_code=1,
            )
        except KeyError as e:
            runtime.terminate_error(
                message=f"Invalid 'delineate' format:\nPartition {ptn_idx+1}: key 'species_leafsets' not found",
                exit_code=1,
            )
        if not isinstance(partition_data, dict):
            # delineate legacy format!
            subsets = partition_data
        else:
            subsets = partition_data.values()
        metadata_d = {}
        exclude_keys = set([
            "species_leafsets",
        ])
        for k, v in src_partition.items():
            if k not in exclude_keys:
                metadata_d[k] = v
        if "constrained_probability" in metadata_d:
            metadata_d["support"] = metadata_d["unconstrained_probability"]
        kwargs = {
            # "label": ptn_idx + 1,
            "metadata_d": metadata_d,
            "subsets": subsets,
        }
        partition = partition_factory(**kwargs)
        partition._origin_size = len(src_partitions)
        partition._origin_offset = ptn_idx
        yield partition

def parse_bpp_a11(
    source_stream,
    partition_factory,
):
    current_section = "pre"
    lineage_labels = []
    partition_info = []
    line = None
    line_idx = 0
    n_partitions_expected = None
    for line_idx, line in enumerate(source_stream):
        line = line.strip()
        if not line:
            continue
        if line.startswith("(A) List of best models"):
            assert current_section == "pre"
            current_section = "(A)"
        # elif line.startswith("(B) "):
        elif (m := parsebpp.a11_section_b.match(line)):
            assert current_section == "(A)"
            current_section = "(B)"
            n_partitions_expected = int(m[1])
        elif line.startswith("(C"):
            current_section = "post"
        if current_section == "pre":
            pass
        elif current_section == "(A)":
            if line.startswith("(A)"):
                continue
            elif not lineage_labels:
                m = parsebpp.a11_treemodel_entry.match(line)
                if not m:
                    runtime.logger.log_warning(f"Expecting species tree model data: line {line_idx}: '{line}'")
                else:
                    lineage_labels = [label for label in parsebpp.strip_tree_tokens.split(m[5]) if label]
                    # print(lineage_labels)
        elif current_section == "(B)":
            if line.startswith("(B)"):
                runtime.logger.info(f"{len(lineage_labels)} Lineages identified: {lineage_labels}")
                runtime.logger.info(f"{n_partitions_expected} partitions expected")
                continue
            assert lineage_labels
            assert n_partitions_expected
            parts = line.strip().split()
            frequency = float(parts[1])
            num_subsets = int(parts[2])
            species_subsets_str = " ".join(parts[3:]).strip("()")
            species_subsets = []
            current_subset = []
            temp_lineage_label = ""
            for char in species_subsets_str:
                if char == ' ':
                    if current_subset:
                        species_subsets.append(current_subset)
                        current_subset = []
                else:
                    temp_lineage_label += char
                    if temp_lineage_label in lineage_labels:
                        current_subset.append(temp_lineage_label)
                        temp_lineage_label = ""
            if current_subset:
                species_subsets.append(current_subset)
            assert len(species_subsets) == num_subsets
            partition_d = {
                "frequency": frequency,
                "n_subsets": num_subsets,
                "subsets": species_subsets
            }
            runtime.logger.info(f"Partition {len(partition_info)+1} of {n_partitions_expected}: {num_subsets} clusters, probability = {frequency}")
            partition_info.append(partition_d)
        elif current_section == "post":
            pass
        else:
            runtime.logger.log_warning(f"Unhandled line: {line_idx}: '{line}'")
    if not partition_info:
        runtime.terminate_error("No species delimitation partitions parsed from source", exit_code=1)
    assert len(partition_info) == n_partitions_expected
    for ptn_idx, ptn_info in enumerate(partition_info):
        runtime.logger.info(
            f"Storing partition {ptn_idx+1} of {len(partition_info)}"
        )
        metadata_d = {
            "support": ptn_info["frequency"],
        }
        kwargs = {
            "metadata_d": metadata_d,
        }
        kwargs["subsets"] = ptn_info["subsets"]
        partition = partition_factory(**kwargs)
        partition._origin_size = len(partition_info)
        partition._origin_offset = ptn_idx
        yield partition

def parse_json_generic_lists(
    source_stream,
    partition_factory,
):
    runtime.logger.info("Parsing 'json-lists' format")
    source_data = source_stream.read()
    data_d = json.loads(source_data)
    runtime.logger.info(f"{len(data_d)} partitions in source")
    source_data = json.loads(source_data)
    for ptn_idx, ptn in enumerate(source_data):
        runtime.logger.info(
            f"Partition {ptn_idx+1:>5d} of {len(data_d)} ({len(ptn)} subsets)"
        )
        partition = partition_factory(
            subsets=ptn,
            metadata_d={},
        )
        partition._origin_size = len(source_data)
        partition._origin_offset = ptn_idx
        yield partition

def parse_spart_xml(
    source_stream,
    partition_factory,
):
    runtime.logger.info("Parsing 'spart-xml' format")
    source_data = source_stream.read()
    root = ET.fromstring(source_data)
    sparts = root.findall(".//spartition")
    runtime.logger.info(f"{len(sparts)} partitions in source")
    for ptn_idx, spartition_element in enumerate(sparts):
        subsets = []
        spart_subsets = spartition_element.findall(".//subset")
        runtime.logger.info(
            f"Partition {ptn_idx+1:>5d} of {len(sparts)} ({len(spart_subsets)} subsets)"
        )
        for subset_idx, subset_element in enumerate(spart_subsets):
            subset = []
            for individual_element in subset_element.findall(".//individual"):
                subset.append(individual_element.get("ref"))
            subsets.append(subset)
        metadata_d = {}
        for key in [
            "label",
            "spartitionScore",
        ]:
            if (d := spartition_element.attrib.get(key, None)):
                metadata_d[key]  = d
        partition = partition_factory(
            subsets=subsets,
            metadata_d=metadata_d,
        )
        partition._origin_size = len(sparts)
        partition._origin_offset = ptn_idx
        yield partition

class Parser:

    format_parser_map = {
        "piikun": parse_piikun_json,
        "delineate": parse_delineate,
        "bpp-a11": parse_bpp_a11,
        "json-lists": parse_json_generic_lists,
        "spart-xml": parse_spart_xml,
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
            partition_factory=self.partition_factory,
            source_stream=source,
        )
