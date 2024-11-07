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
import re
import resource
import subprocess

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from matplotlib.colors import LogNorm, Normalize
from itertools import combinations, product

import yakherd
from piikun import partitionmodel
from piikun import utility
from piikun import parsebpp


class PartitionCoordinator(utility.RuntimeContext):
    def __init__(
        self,
        config_d=None,
        log_base=2,
        runtime_context=None,
    ):
        super().__init__(runtime_context=runtime_context)
        self.log_base = log_base
        self.configure(config_d=config_d)
        self.reset()

    def configure(self, config_d):
        if not config_d:
            return
        self.log_frequency = config_d.get("progress_report_frequency", 0.01)
        if self.log_frequency is None:
            self.log_frequency = 0.01
        self.bpp_control_file = config_d.get("bpp_control_file", None)

    def reset(self):
        self.partitions = []

    def new_partition(self, **kwargs):
        kwargs["log_base"] = self.log_base
        ptn = partitionmodel.Partition(**kwargs)
        self.partitions.append(ptn)
        return ptn

    def read_partitions(
        self,
        src_paths,
        source_format,
        limit_records=None,
    ):
        yfn = None
        if source_format == "bpp-a10":
            self.logger.log_critical("Unfortunately, score for this format is still under development.")
            sys.exit(1)
        elif source_format == "bpp-a11":
            yfn = self.parse_bpp_a11
        elif source_format == "json-list":
            yfn = self.parse_json
        elif source_format == "spart-xml":
            yfn = self.parse_spart_xml
        elif source_format == "delineate" or not source_format:
            yfn = self.parse_delineate
        else:
            raise ValueError(source_format)
        n_partitions_parsed = 0
        for src_idx, src_path in enumerate(src_paths):
            self.logger.log_info(
                f"Reading source {src_idx+1} of {len(src_paths)}: '{src_path}'"
            )
            src = open(src_path)
            for pidx, partition in enumerate(yfn(src)):
                n_partitions_parsed += 1
                if limit_records and n_partitions_parsed == limit_records:
                    break

    def parse_delineate(self, src):
        delineate_results = json.load(src)
        src_partitions = delineate_results["partitions"]
        self.logger.log_info(f"{len(src_partitions)} partitions in source")
        for spart_idx, src_partition in enumerate(src_partitions):
            self.logger.log_info(
                f"Storing partition {spart_idx+1} of {len(src_partitions)}"
            )
            metadata_d = {
                "constrained_probability": src_partition.get(
                    "constrained_probability", 0
                ),
                "unconstrained_probability": src_partition.get(
                    "unconstrained_probability", 0
                ),
                "score": src_partition.get("unconstrained_probability", 0),
            }
            kwargs = {
                "label": spart_idx + 1,
                "metadata_d": metadata_d,
            }
            partition_data = src_partition["species_leafsets"]
            if not isinstance(partition_data, dict):
                # legacy format!
                kwargs["subsets"] = partition_data
            else:
                kwargs["partition_d"] = partition_data
            partition = self.new_partition(**kwargs)
            yield partition

    def parse_bpp_a11(self, src):
        current_section = "pre"
        lineage_labels = []
        partition_info = []
        line = None
        line_idx = 0
        n_partitions_expected = None
        for line_idx, line in enumerate(src):
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
                        self.logger.log_warning(f"Expecting species tree model data: line {line_idx}: '{line}'")
                    else:
                        lineage_labels = [label for label in parsebpp.strip_tree_tokens.split(m[5]) if label]
                        # print(lineage_labels)
            elif current_section == "(B)":
                if line.startswith("(B)"):
                    self.logger.log_info(f"{len(lineage_labels)} Lineages identified: {lineage_labels}")
                    self.logger.log_info(f"{n_partitions_expected} partitions expected")
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
                self.logger.log_info(f"Partition {len(partition_info)+1} of {n_partitions_expected}: {num_subsets} clusters, probability = {frequency}")
                partition_info.append(partition_d)
            elif current_section == "post":
                pass
            else:
                self.logger.log_warning(f"Unhandled line: {line_idx}: '{line}'")
        if not partition_info:
            self.logger.log_error("No species delimitation partitions parsed from source")
            sys.exit(1)
        assert len(partition_info) == n_partitions_expected
        for ptn_idx, ptn_info in enumerate(partition_info):
            self.logger.log_info(
                f"Storing partition {ptn_idx+1} of {len(partition_info)}"
            )
            metadata_d = {
                "score": ptn_info["frequency"],
            }
            kwargs = {
                "label": ptn_idx + 1,
                "metadata_d": metadata_d,
            }
            kwargs["subsets"] = ptn_info["subsets"]
            partition = self.new_partition(**kwargs)
            yield partition

    def parse_json(self, src_data):
        src_data = json.loads(src_data)
        for ptn_idx, ptn in enumerate(src_data):
            partition = self.new_partition(
                label=ptn_idx + 1,
                subsets=ptn,
            )
            yield partition

    def parse_spart_xml(self, src_data):
        root = ET.fromstring(src_data)
        for spart_idx, spartition_element in enumerate(root.findall(".//spartition")):
            subsets = []
            for subset_idx, subset_element in enumerate(
                spartition_element.findall(".//subset")
            ):
                subset = []
                for individual_element in subset_element.findall(".//individual"):
                    subset.append(individual_element.get("ref"))
                subsets.append(subset)
            partition = self.new_partition(
                label=spartition_element.attrib.get("label", spart_idx + 1),
                subsets=subsets,
            )
            yield partition

    @property
    def partition_list(self):
        if not hasattr(self, "_partition_list") or self._partition_list is None:
            self._partition_list = list(self.partitions.values())
        return self._partition_list

    @property
    def partition_profile_store(self):
        if (
            not hasattr(self, "_partition_profile_store")
            or self._partition_profile_store is None
        ):
            self._partition_profile_store = self.runtime_context.ensure_store(
                key="partition-profile",
                name_parts=[
                    "profiles",
                ],
                separator="-",
                extension="tsv",
            )
        return self._partition_profile_store

    @property
    def partition_oneway_distances(self):
        if (
            not hasattr(self, "_partition_oneway_distances")
            or self._partition_oneway_distances is None
        ):
            self._partition_oneway_distances = self.runtime_context.ensure_store(
                key="partition-oneway-distances",
                name_parts=["d1"],
                separator="-",
                extension="tsv",
            )
        return self._partition_oneway_distances

    @property
    def partition_twoway_distances(self):
        if (
            not hasattr(self, "_partition_twoway_distance_store")
            or self._partition_score_distance_store is None
        ):
            self._partition_score_distance_store = self.runtime_context.ensure_store(
                key="partition-twoway-distances",
                name_parts=["distances"],
                separator="-",
                extension="tsv",
            )
        return self._partition_score_distance_store

    # @property
    # def partition_profiled_comparison_store(self):
    #     if (
    #         not hasattr(self, "_partition_profiled_comparison_store")
    #         or self._partition_profiled_comparison_store is None
    #     ):
    #         self._partition_profiled_comparison_store = self.runtime_context.ensure_store(
    #             key="partition-profiled_comparisons",
    #             name_parts=["partition", "profiled_comparisons"],
    #             separator="-",
    #             extension="tsv",
    #         )
    #     return self._partition_profiled_comparison_store

    def analyze_partitions(self, is_mirror=False):
        if is_mirror:
            n_expected_cmps = len(self.partitions) * len(self.partitions)
        else:
            n_expected_cmps = int(len(self.partitions) * len(self.partitions) / 2)
        progress_step = int(n_expected_cmps * self.log_frequency)
        if progress_step < 1:
            progress_step = 1
        n_comparisons = 0
        # comparisons = []
        seen_compares = set()
        for pidx1, ptn1 in enumerate(self.partitions):
            profile_d = {
                "partition_id": pidx1,
                "label": ptn1.label,
                "n_elements": ptn1.n_elements,
                "n_subsets": ptn1.n_subsets,
                "vi_entropy": ptn1.vi_entropy(),
            }
            if ptn1.metadata_d:
                profile_d.update(ptn1.metadata_d)
            self.partition_profile_store.write_d(profile_d)
            ptn1_metadata = {}
            for k, v in ptn1.metadata_d.items():
                ptn1_metadata[f"ptn1_{k}"] = v
            # print("Memory usage: {} MB".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
            for pidx2, ptn2 in enumerate(self.partitions):
                cmp_key = frozenset([ptn1, ptn2])
                if not is_mirror and cmp_key in seen_compares:
                    continue
                seen_compares.add(cmp_key)
                if n_comparisons == 0 or (n_comparisons % progress_step) == 0:
                    self.logger.log_info(
                        f"[ {int(n_comparisons * 100/n_expected_cmps): 4d} % ] Comparison {n_comparisons} of {n_expected_cmps}: Partition {ptn1.label} vs. partition {ptn2.label}"
                    )
                n_comparisons += 1
                comparison_d = {
                    "ptn1": ptn1.label,
                    "ptn2": ptn2.label,
                }
                comparison_d.update(ptn1_metadata)
                for k, v in ptn2.metadata_d.items():
                    comparison_d[f"ptn2_{k}"] = v
                comparison_d["vi_entropy_ptn1"] = ptn1.vi_entropy()
                comparison_d["vi_entropy_ptn2"] = ptn2.vi_entropy()
                for value_fieldname, value_fn in (
                    ("vi_mi", ptn1.vi_mutual_information),
                    ("vi_joint_entropy", ptn1.vi_joint_entropy),
                    ("vi_distance", ptn1.vi_distance),
                    ("vi_normalized_kraskov", ptn1.vi_normalized_kraskov),
                ):
                    comparison_d[value_fieldname] = value_fn(ptn2)
                self.partition_oneway_distances.write_d(comparison_d)
        self.partition_profile_store.close()
        self.partition_oneway_distances.close()
        utility.create_full_profile_distance_df(
            profiles_path=self.partition_profile_store.path,
            distances_path=self.partition_oneway_distances.path,
            merged_path=self.partition_twoway_distances.path,
            logger=self.logger,
        )
        self.partition_twoway_distances.close()

    # def analyze_partitions_x(
    #     self, is_assume_symmetry=True, is_include_self_comparisons=True
    # ):
    #     n_partitions = len(self.partitions)
    #     n_expected_cmps = (
    #         n_partitions * (n_partitions - 1)
    #         if is_assume_symmetry
    #         else math.comb(n_partitions, 2)
    #     )
    #     if is_include_self_comparisons:
    #         n_expected_cmps += n_partitions
    #     progress_step = max(1, int(n_expected_cmps * self.log_frequency))
    #     n_comparisons = 0

    #     self.logger.log_info(
    #         f"{n_partitions} partitions: {n_expected_cmps} expected distinct comparisons"
    #     )

    #     metadata_keys = (
    #         list(self.partitions[0].metadata_d.keys())
    #         if self.partitions[0].metadata_d
    #         else []
    #     )

    #     for pidx1, ptn1 in enumerate(self.partitions):
    #         vi_entropy1 = ptn1.vi_entropy()

    #         # Directly write to file
    #         # self.partition_profile_store.write_d(
    #         #     pidx1, ptn1.label, ptn1.n_elements, ptn1.n_subsets, vi_entropy1, *(ptn1.metadata_d.get(k, "NaN") for k in metadata_keys)
    #         # )

    #         print(
    #             "Memory usage: {} MB".format(
    #                 resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    #             )
    #         )

    #         for pidx2, ptn2 in enumerate(
    #             self.partitions
    #             if is_assume_symmetry
    #             else combinations(self.partitions, 2)
    #         ):
    #             if not is_include_self_comparisons and ptn1 == ptn2:
    #                 continue

    #             n_comparisons += 1

    #             if n_comparisons % progress_step == 0:
    #                 self.logger.log_info(
    #                     f"[ {int(n_comparisons * 100 / n_expected_cmps): 4d} % ] Comparison {n_comparisons} of {n_expected_cmps}: Partition {ptn1.label} vs. partition {ptn2.label}"
    #                 )

    #             vi_entropy2 = ptn2.vi_entropy()

    #             pairs = (
    #                 [(ptn1, ptn2), (ptn2, ptn1)]
    #                 if is_assume_symmetry and ptn1 != ptn2
    #                 else [(ptn1, ptn2)]
    #             )

    #             for ptn_a, ptn_b in pairs:
    #                 vi_mi = ptn_a.vi_mutual_information(ptn_b)
    #                 vi_joint_entropy = ptn_a.vi_joint_entropy(ptn_b)
    #                 vi_distance = ptn_a.vi_distance(ptn_b)
    #                 vi_normalized_kraskov = ptn_a.vi_normalized_kraskov(ptn_b)

    #                 # Directly write to file
    #                 # self.partition_oneway_distances.write_d(
    #                 #     ptn_a.label, ptn_b.label, ptn_a.n_elements, ptn_b.n_elements, ptn_a.n_subsets, ptn_b.n_subsets,
    #                 #     vi_entropy1, vi_entropy2, vi_mi, vi_joint_entropy, vi_distance, vi_normalized_kraskov,
    #                 #     *(ptn_a.metadata_d.get(k, "NaN") for k in metadata_keys), *(ptn_b.metadata_d.get(k, "NaN") for k in metadata_keys)
    #                 # )
    #     self.partition_profile_store.close()
    #     self.partition_oneway_distances.close()

    # def analyze_partitions_beta(
    #     self, is_assume_symmetry=True, is_include_self_comparisons=True
    # ):
    #     n_partitions = len(self.partitions)
    #     n_expected_cmps = (
    #         n_partitions * (n_partitions - 1)
    #         if is_assume_symmetry
    #         else math.comb(n_partitions, 2)
    #     )
    #     if is_include_self_comparisons:
    #         n_expected_cmps += n_partitions
    #     progress_step = max(1, int(n_expected_cmps * self.log_frequency))
    #     n_comparisons = 0
    #     self.logger.log_info(
    #         f"{n_partitions} partitions: {n_expected_cmps} expected distinct comparisons"
    #     )
    #     metadata_keys = (
    #         list(self.partitions[0].metadata_d.keys())
    #         if self.partitions[0].metadata_d
    #         else []
    #     )
    #     for pidx1, ptn1 in enumerate(self.partitions):
    #         vi_entropy1 = ptn1.vi_entropy()
    #         profile_d = {
    #             "partition_id": pidx1,
    #             "label": ptn1.label,
    #             "n_elements": ptn1.n_elements,
    #             "n_subsets": ptn1.n_subsets,
    #             "vi_entropy": vi_entropy1,
    #         }
    #         if ptn1.metadata_d:
    #             profile_d.update(ptn1.metadata_d)
    #         # ## writes the dictionary out as a tab-delimited string to a file handle
    #         self.partition_profile_store.write_d(profile_d)
    #         print(
    #             "Memory usage: {} MB".format(
    #                 resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    #             )
    #         )
    #         for pidx2, ptn2 in enumerate(
    #             self.partitions
    #             if is_assume_symmetry
    #             else combinations(self.partitions, 2)
    #         ):
    #             if not is_include_self_comparisons and ptn1 == ptn2:
    #                 continue
    #             n_comparisons += 1
    #             if n_comparisons % progress_step == 0:
    #                 self.logger.log_info(
    #                     f"[ {int(n_comparisons * 100 / n_expected_cmps): 4d} % ] Comparison {n_comparisons} of {n_expected_cmps}: Partition {ptn1.label} vs. partition {ptn2.label}"
    #                 )
    #             vi_entropy2 = ptn2.vi_entropy()
    #             # dists_d = {
    #             #     "vi_mi": ptn1.vi_mutual_information(ptn2),
    #             #     "vi_joint_entropy": ptn1.vi_joint_entropy(ptn2),
    #             #     "vi_distance": ptn1.vi_distance(ptn2),
    #             #     "vi_normalized_kraskov": ptn1.vi_normalized_kraskov(ptn2),
    #             # }

    #             pairs = (
    #                 [(ptn1, ptn2), (ptn2, ptn1)]
    #                 if is_assume_symmetry and ptn1 != ptn2
    #                 else [(ptn1, ptn2)]
    #             )
    #             for ptn_a, ptn_b in pairs:
    #                 pass
    #                 # comparison_d = {
    #                 #     "ptn1": ptn_a.label,
    #                 #     "ptn2": ptn_b.label,
    #                 #     "ptn1_n_elements": ptn_a.n_elements,
    #                 #     "ptn2_n_elements": ptn_b.n_elements,
    #                 #     "ptn1_n_subsets": ptn_a.n_subsets,
    #                 #     "ptn2_n_subsets": ptn_b.n_subsets,
    #                 #     "ptn1_vi_entropy": vi_entropy1,
    #                 #     "ptn2_vi_entropy": vi_entropy2,
    #                 # }
    #                 # for k in metadata_keys:
    #                 #     comparison_d[f"ptn1_{k}"] = ptn_a.metadata_d.get(k, "NaN")
    #                 #     comparison_d[f"ptn2_{k}"] = ptn_b.metadata_d.get(k, "NaN")
    #                 # comparison_d.update(dists_d)
    #                 # ## writes the dictionary out as a tab-delimited string to a file handle
    #                 # self.partition_oneway_distances.write_d(comparison_d)
    #     self.partition_profile_store.close()
    #     self.partition_oneway_distances.close()

    # def analyze_partitions_alpha(
    #     self,
    #     is_assume_symmetry=True,
    #     is_include_self_comparisons=True,
    # ):
    #     if is_assume_symmetry:
    #         n_expected_cmps = len(self.partitions) * (len(self.partitions) - 1)
    #     else:
    #         n_expected_cmps = math.comb(len(self.partitions), 2)
    #     if is_include_self_comparisons:
    #         n_expected_cmps += len(self.partitions)
    #     progress_step = int(n_expected_cmps * self.log_frequency)
    #     if progress_step < 1:
    #         progress_step = 1
    #     n_comparisons = 0
    #     # comparisons = []
    #     seen_compares = set()
    #     self.logger.log_info(
    #         f"{len(self.partitions)} partitions: {n_expected_cmps} expected distinct comparisons"
    #     )
    #     metadata_keys = None
    #     for pidx1, ptn1 in enumerate(self.partitions):
    #         profile_d = {
    #             "partition_id": pidx1,
    #             "label": ptn1.label,
    #             "n_elements": ptn1.n_elements,
    #             "n_subsets": ptn1.n_subsets,
    #             "vi_entropy": ptn1.vi_entropy(),
    #         }
    #         if ptn1.metadata_d:
    #             profile_d.update(ptn1.metadata_d)
    #             if not metadata_keys:
    #                 metadata_keys = list(ptn1.metadata_d.keys())
    #         self.partition_profile_store.write_d(profile_d)
    #         for pidx2, ptn2 in enumerate(self.partitions):
    #             if not is_include_self_comparisons and ptn1 == ptn2:
    #                 continue
    #             cmp_key = frozenset([ptn1, ptn2])
    #             if not is_assume_symmetry and cmp_key in seen_compares:
    #                 continue
    #             seen_compares.add(cmp_key)
    #             n_comparisons += 1
    #             if n_comparisons == 0 or (n_comparisons % progress_step) == 0:
    #                 self.logger.log_info(
    #                     f"[ {int(n_comparisons * 100/n_expected_cmps): 4d} % ] Comparison {n_comparisons} of {n_expected_cmps}: Partition {ptn1.label} vs. partition {ptn2.label}"
    #                 )

    #             dists_d = {}
    #             for value_fieldname, value_fn in (
    #                 ("vi_mi", ptn1.vi_mutual_information),
    #                 ("vi_joint_entropy", ptn1.vi_joint_entropy),
    #                 ("vi_distance", ptn1.vi_distance),
    #                 ("vi_normalized_kraskov", ptn1.vi_normalized_kraskov),
    #             ):
    #                 dists_d[value_fieldname] = value_fn(ptn2)

    #             if is_assume_symmetry and ptn1 != ptn2:
    #                 pairs = (
    #                     (ptn1, ptn2),
    #                     (ptn2, ptn1),
    #                 )
    #             else:
    #                 pairs = ((ptn1, ptn2),)
    #             for ptn_pair in pairs:
    #                 comparison_d = {
    #                     "ptn1": ptn_pair[0].label,
    #                     "ptn2": ptn_pair[1].label,
    #                 }
    #                 for k in [
    #                     "n_elements",
    #                     "n_subsets",
    #                 ]:
    #                     comparison_d[f"ptn1_{k}"] = getattr(ptn_pair[0], k, "NaN")
    #                     comparison_d[f"ptn2_{k}"] = getattr(ptn_pair[1], k, "NaN")
    #                 for k in metadata_keys:
    #                     comparison_d[f"ptn1_{k}"] = ptn_pair[0].metadata_d[k]
    #                     comparison_d[f"ptn2_{k}"] = ptn_pair[1].metadata_d[k]
    #                 comparison_d["ptn1_vi_entropy"] = ptn_pair[0].vi_entropy()
    #                 comparison_d["ptn2_vi_entropy"] = ptn_pair[1].vi_entropy()
    #                 comparison_d.update(dists_d)
    #                 self.partition_oneway_distances.write_d(comparison_d)
    #     self.partition_profile_store.close()
    #     self.partition_oneway_distances.close()


def main(args=None):
    parent_parser = argparse.ArgumentParser()
    parent_parser.set_defaults(func=lambda x: parent_parser.print_help())
    # subparsers = parent_parser.add_subparsers()

    # analyze_parser = subparsers.add_parser("analyze", help="Analyze a collection of partitions.")
    # parent_parser.set_defaults(func=execute_analysis)
    input_options = parent_parser.add_argument_group("Input Options")
    input_options.add_argument(
        "src_path",
        action="store",
        metavar="FILE",
        nargs="+",
        help="Path to data source file.",
    )
    input_options.add_argument(
        "-f",
        "--source-format",
        action="store",
        dest="source_format",
        default=None,
        choices=[
            "delineate",
            "bpp-a10",
            "bpp-a11",
            "json-list",
            "spart-xml",
        ],
        help="Format of species delimitation data: [default='delineate'].",
    )
    # input_options.add_argument(
    #     "--bpp-control-file",
    #     action="store",
    #     metavar="FILE",
    #     help="Path to BP&P control file (assumes format == 'bpp-a11').",
    # )
    input_options.add_argument(
        "--limit-partitions",
        action="store",
        default=None,
        type=int,
        help="Limit data to this number of partitions.",
    )
    output_options = parent_parser.add_argument_group("Output Options")
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

    # cluster_plot_options = parent_parser.add_argument_group("Cluster Plot Options")
    # cluster_plot_options.add_argument(
    #     "--cluster-rows",
    #     action=argparse.BooleanOptionalAction,
    #     dest="is_cluster_rows",
    #     default=False,
    #     help="Reorder / do not reorder partition rows to show clusters clearly",
    # )
    # cluster_plot_options.add_argument(
    #     "--cluster-cols",
    #     action=argparse.BooleanOptionalAction,
    #     dest="is_cluster_cols",
    #     default=False,
    #     help="Reorder / do not reorder partition colums to show clusters clearly",
    # )


    logger_configuration_parser = yakherd.LoggerConfigurationParser(name="piikun")
    logger_configuration_parser.attach(parent_parser)
    logger_configuration_parser.console_logging_parser_group.add_argument(
        "--progress-report-frequency",
        type=int,
        action="store",
        help="Frequency of progress reporting.",
    )

    plot_options = parent_parser.add_argument_group("Plot Options")
    # plot_options.add_argument(
    #         "--plot",
    #         action="store",
    #         help="Argument and options to pass to to plotter.")
    plot_options.add_argument(
        "--no-plot", action="store_true", default=False, help="Do not plot results."
    )
    plot_options.add_argument(
        "--plot",
        dest="plot_args",
        nargs=argparse.REMAINDER,
        # nargs="*",
    )

    args = parent_parser.parse_args(args)
    if args.output_title is None:
        args.output_title = str(pathlib.Path(args.src_path[0]).stem)
    config_d = dict(vars(args))
    logger = logger_configuration_parser.get_logger(args_d=config_d)
    source_format = config_d["source_format"]
    # bpp_control_file = config_d["bpp_control_file"]
    # if bpp_control_file:
    #     logger.log_info(f"Processing output generated under BPP control file: '{bpp_control_file}'")
    #     if not source_format or source_format.lower() not in ["bpp-a11", "bpp1"]:
    #         logger.log_error(
    #             f"BPP results analysis requires '--data-format=bpp-a10 or '--data-format=bpp-a11'",
    #             )
    #         sys.exit(1)
    logger.log_info(f"Data source format: '{source_format}'")
    if not source_format:
        source_format = "delineate"
    if args.output_title is None:
        args.output_title = pathlib.Path(config_d["src_path"]).stem
    runtime_context = utility.RuntimeContext(
        logger=logger,
        random_seed=None,
        output_directory=args.output_directory,
        output_title=args.output_title,
        output_configuration=config_d,
    )
    pc = PartitionCoordinator(
        config_d=config_d,
        runtime_context=runtime_context,
    )
    pc.read_partitions(
        src_paths=config_d["src_path"],
        source_format=source_format,
        limit_records=config_d["limit_partitions"],
    )
    df = pc.analyze_partitions()
    if not args.no_plot:
        cmd = ["piikun-plot-metrics"]
        cmd.extend(["-O", args.output_directory])
        cmd.extend(["-o", args.output_title])
        if args.plot_args:
            cmd.extend(args.plot_args)
        cmd.append(pc.partition_twoway_distances.path)
        cmd = [str(s) for s in cmd]
        cmd_str = ' '.join(cmd)
        logger.log_info([
            "Executing plot:",
            f" {cmd_str}"
        ])
        rv = subprocess.run(cmd)


if __name__ == "__main__":
    main()
