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
    source_data = source_stream.read()
    data_d = json.loads(source_data)
    partition_ds = data_d["partitions"]
    for ptn_idx, (partition_key, partition_d) in enumerate(partition_ds.items()):
        subsets = partition_d["subsets"]
        metadata_d = partition_d["metadata"]
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
    n_lineages_expected = None
    n_lineages_expected_set_on_line = None
    n_partitions_expected = None
    for line_offset, line in enumerate(source_stream):
        line_idx = line_offset + 1
        line = line.strip()
        if not line:
            continue
        if line.startswith("COMPRESSED ALIGNMENTS") and not line.startswith("COMPRESSED ALIGNMENTS AFTER"):
            assert current_section == "pre", current_section
            current_section = "alignments"
        elif line.startswith("(A) List of best models"):
            assert current_section == "alignments"
            current_section = "(A)"
        # elif line.startswith("(B) "):
        elif (m := parsebpp.patterns["a11_section_b"].match(line)):
            assert current_section == "(A)"
            current_section = "(B)"
            n_partitions_expected = int(m[1])
        elif line.startswith("(C"):
            current_section = "post"
        if current_section == "pre":
            pass
        elif current_section == "alignments":
            if lineage_labels and n_lineages_expected and len(lineage_labels) == n_lineages_expected:
                continue
            if line.startswith("COMPRESSED ALIGNMENTS"):
                continue
            m = parsebpp.patterns["alignment_ntax_nchar"].match(line)
            if m:
                if n_lineages_expected:
                    runtime.RuntimeClient._logger.error(f"Unexpected alignment label and character description (already set to {n_lineages_expected} on {n_lineages_expected_set_on_line}): line {line_idx}: '{line}'")
                    assert not n_lineages_expected
                    continue
                n_lineages_expected = int(m[1])
                n_lineages_expected_set_on_line = line_idx
                continue
            if not n_lineages_expected:
                # runtime.RuntimeClient._logger.warn(f"Expected alignment label and character description: line {line_idx}: '{line}'")
                runtime.RuntimeClient._logger.error(f"Missing alignment label and character description: line {line_idx}: '{line}'")
                assert not n_lineages_expected
            m = parsebpp.patterns["alignment_sequence"].match(line)
            if not m:
                runtime.RuntimeClient._logger.error(f"Expected sequence data: line {line_idx}: '{line}'")
                assert m
            if len(lineage_labels) == n_lineages_expected:
                runtime.RuntimeClient._logger.error(f"Unexpected sequence definition ({n_lineages_expected} labels already parsed): line {line_idx}: '{line}'")
                assert m
            if m:
                lineage_labels.append(m[1])
                continue
            else:
                runtime.RuntimeClient._logger.error(f"Unable to parse sequence: line {line_idx}: '{line}'")
                assert m
        elif current_section == "(A)":
            continue
            # if line.startswith("(A)"):
            #     continue
            # elif not lineage_labels:
            #     m = parsebpp.patterns["a11_treemodel_entry"].match(line)
            #     if not m:
            #         runtime.RuntimeClient._logger.warn(f"Expecting species tree model data: line {line_idx}: '{line}'")
            #         # runtime.Runtime._logger.log_warning(f"Expecting species tree model data: line {line_idx}: '{line}'")
            #     else:
            #         lineage_labels = [label for label in parsebpp.patterns["strip_tree_tokens"].split(m[5]) if label]
            #         # print(lineage_labels)
        elif current_section == "(B)":
            if len(lineage_labels) != n_lineages_expected:
                runtime.RuntimeClient._logger.error(f"{n_lineages_expected} lineages expected but {len(lineage_labels)} lineages identified: {lineage_labels}")
            if line.startswith("(B)"):
                runtime.RuntimeClient._logger.info(f"{len(lineage_labels)} lineages identified: {lineage_labels}")
                # runtime.logger.info(f"{len(lineage_labels)} Lineages identified: {lineage_labels}")
                # runtime.logger.info(f"{n_partitions_expected} partitions expected")
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
            partition_d = {
                "frequency": frequency,
                "n_subsets": num_subsets,
                "subsets": species_subsets
            }
            # runtime.logger.info(f"Partition {len(partition_info)+1} of {n_partitions_expected}: {num_subsets} clusters, probability = {frequency}")
            partition_info.append(partition_d)
        elif current_section == "post":
            pass
        else:
            runtime.RuntimeClient._logger.warn(f"Unhandled line: {line_idx}: '{line}'")
    if not partition_info:
        runtime.terminate_error("No species delimitation partitions parsed from source", exit_code=1)
    assert len(partition_info) == n_partitions_expected
    for ptn_idx, ptn_info in enumerate(partition_info):
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
    source_data = source_stream.read()
    data_d = json.loads(source_data)
    source_data = json.loads(source_data)
    for ptn_idx, ptn in enumerate(source_data):
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
    source_data = source_stream.read()
    root = ET.fromstring(source_data)
    sparts = root.findall(".//spartition")
    for ptn_idx, spartition_element in enumerate(sparts):
        subsets = []
        spart_subsets = spartition_element.findall(".//subset")
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
