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

import sys
import json
import math
import functools
import collections
import pathlib
import collections
import pandas as pd
from piikun import parse
from piikun import utility

# https://stackoverflow.com/a/30134039
def iterate_partitions(collection):
    if len(collection) == 1:
        yield [collection]
        return
    first = collection[0]
    for smaller in iterate_partitions(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
        # put `first` in its own subset
        yield [[first]] + smaller


class PartitionSubset:
    def __init__(
        self,
        elements,
    ):
        self._elements = set(elements)

    def __hash__(self):
        return id(self)

    def __len__(self):
        return len(self._elements)

    @functools.cache
    def intersection(self, other):
        s = self._elements.intersection(other._elements)
        return s

    def __str__(self):
        return str(self._elements)

    def __repr__(self):
        return str(self._elements)


class Partition:

    _n_instances = 0

    def __init__(
        self,
        *,
        subsets=None,
        log_base=2,
        metadata_d=None,
    ):
        self.log_base = log_base
        self.log_fn = lambda x: math.log(x, self.log_base)
        self._index = Partition._n_instances
        Partition._n_instances += 1
        self._subsets = []
        # self._label_subset_map = {}
        self._elements = set()
        self.metadata_d = {}
        if metadata_d:
            self.metadata_d.update(metadata_d)
        # if partition_d is not None:
        #     self.parse_subsets(partition_d.values())
        if subsets is not None:
            self.parse_subsets(subsets)

    @property
    def n_elements(
        self,
    ):
        return len(self._elements)

    @property
    def n_subsets(
        self,
    ):
        return len(self._subsets)

    def __hash__(self):
        return id(self)

    def compose_definition_d(self, key=None):
        d = {}
        if key is not None:
            d["label"] = key
        d["subsets"] = [sorted(subset._elements) for subset in self._subsets]
        d["metadata"] = self.metadata_d
        return d

    def new_subset(self, elements):
        for element in elements:
            if element in self._elements:
                pass
            assert element not in self._elements
            self._elements.add(element)
        s = PartitionSubset(
            elements=elements,
        )
        # self._label_subset_map[label] = s
        self._subsets.append(s)
        return s

    # def parse_partition_d(self, partition_d):
    #     for label, v in partition_d.items():
    #         self.new_subset(label=label, elements=v)

    def parse_subsets(self, subsets):
        for label, v in enumerate(subsets):
            self.new_subset(elements=v)

    def entropy(self, method="vi"):
        if method == "vi":
            return self.vi_entropy()
        else:
            raise ValueError(f"Unrecognized methods: '{method}'")

    def distance(self, method="vi"):
        if method == "vi":
            return self.vi_distance()
        else:
            raise ValueError(f"Unrecognized methods: '{method}'")

    @functools.cache
    def vi_mutual_information(self, other):
        """
        Following: Meila 2007.
        """

        vi_mi = 0.0
        assert self._elements == other._elements
        for ptn1_idx, ptn1_subset in enumerate(self._subsets):
            for ptn2_idx, ptn2_subset in enumerate(other._subsets):
                intersection = ptn1_subset.intersection(ptn2_subset)
                vi_joint_prob = len(intersection) / self.n_elements
                if vi_joint_prob:
                    vi_h = vi_joint_prob * self.log_fn(
                        vi_joint_prob
                        / math.prod(
                            [
                                len(ptn1_subset) / self.n_elements,
                                len(ptn2_subset) / other.n_elements,
                            ],
                        )
                    )
                    vi_mi += vi_h
        return vi_mi

    @functools.cache
    def vi_joint_entropy(self, other):
        return (
            self.vi_entropy() + other.vi_entropy() - self.vi_mutual_information(other)
        )

    @functools.cache
    def vi_distance(self, other):
        vi_dist = (
            self.vi_entropy()
            + other.vi_entropy()
            - (2 * self.vi_mutual_information(other))
        )
        return vi_dist

    @functools.cache
    def vi_normalized_kraskov(self, other):
        """
        Following Kraskov et al. (2005) in Vinh et al. (2010); (Table 3)
        """
        if self.vi_joint_entropy(other):
            return 1.0 - (
                self.vi_mutual_information(other) / self.vi_joint_entropy(other)
            )
        else:
            return None

    # Univariate

    @functools.cache
    def vi_entropy(self):
        result = 0.0
        for subset in self._subsets:
            prob = len(subset) / self.n_elements
            result -= prob * self.log_fn(prob)
        return result

    # Requires Probability Distribution

    @functools.cache
    def vi_jensen_shannon_distance(self, other):
        """
        Returns the square root of the Jensen Shannon divergence(i.e., the Jensen-Shannon * distance*) using Meila's encoding
        """
        from scipy.spatial.distance import jensenshannon

        P = []
        Q = []
        for ptn1_idx, ptn1_subset in enumerate(self._subsets):
            for ptn2_idx, ptn2_subset in enumerate(other._subsets):
                P.append(len(ptn1_subset) / self.n_elements)
                Q.append(len(ptn2_subset) / other.n_elements)
        return jensenshannon(P, Q, base=self.log_base)

    @functools.cache
    def ahrens_match_ratio(self, other):
        """
        Calculate match ratio (Ahrens, 2006) between this partition and another
        partition.

        Match ratio = 2 * Nm / (Np + Nq) where:

        - Nm is number of exact matches between subsets
        - Np is number of subsets in this partition
        - Nq is number of subsets in other partition

        Args:
            other: Another Partition instance to compare against

        Returns:
            float: Match ratio between 0 and 1
        """
        # Count exact matches
        n_matches = 0
        for subset1 in self._subsets:
            for subset2 in other._subsets:
                if subset1._elements == subset2._elements:
                    n_matches += 1
                    break

        # Get total number of subsets in each partition
        n_subsets_p = len(self._subsets)
        n_subsets_q = len(other._subsets)

        if n_subsets_p == 0 and n_subsets_q == 0:
            return 1.0

        # Calculate ratio
        ratio = (2 * n_matches) / (n_subsets_p + n_subsets_q)

        return ratio


    # @functools.cache
    # def mirkin_metric(self, other):
    #     """
    #     The fraction of element pairs that are clustered differently between the two partitions.
    #     """
    #     assert self._elements == other._elements

    #     total_pairs = 0
    #     mismatched_pairs = 0

    #     # Get list of elements to ensure consistent ordering
    #     elements = list(self._elements)

    #     # Create maps from elements to their subset indices
    #     self_element_to_subset = {}
    #     other_element_to_subset = {}

    #     for idx, subset in enumerate(self._subsets):
    #         for element in subset._elements:
    #             self_element_to_subset[element] = idx

    #     for idx, subset in enumerate(other._subsets):
    #         for element in subset._elements:
    #             other_element_to_subset[element] = idx

    #     # Compare all pairs of elements
    #     for i in range(len(elements)):
    #         for j in range(i + 1, len(elements)):
    #             elem1, elem2 = elements[i], elements[j]
    #             total_pairs += 1

    #             # Check if elements are in same subset in both partitions
    #             same_subset_self = (self_element_to_subset[elem1] == self_element_to_subset[elem2])
    #             same_subset_other = (other_element_to_subset[elem1] == other_element_to_subset[elem2])

    #             if same_subset_self != same_subset_other:
    #                 mismatched_pairs += 1

    #     return mismatched_pairs / total_pairs if total_pairs > 0 else 0.0

    # @functools.cache
    # def misclassification_error_meila_heckerman(self, other):
    #     """
    #     Calculate the classification error between two partitions as defined by Meilă-Heckerman.

    #     H(Δ,Δ') = (1/n) ∑_{k,k'=match(k)} n_{kk'}

    #     where:
    #     - n is total number of elementk
    #     - n_{kk'} is the number of elements in subset k that are placed in non-matching subset k'
    #     - match(k) determines the best matching between subsets
    #     """
    #     assert self._elements == other._elements

    #     # Get total number of elements
    #     n = len(self._elements)

    #     # Create contingency matrix
    #     contingency = {}
    #     for i, subset1 in enumerate(self._subsets):
    #         for j, subset2 in enumerate(other._subsets):
    #             intersection = subset1._elements.intersection(subset2._elements)
    #             contingency[(i,j)] = len(intersection)

    #     # Find optimal matching between subsets to minimize misclassifications
    #     # Using greedy matching - for each subset in self, match to largest overlapping subset in other
    #     matches = {}
    #     used_other_subsets = set()

    #     for i in range(len(self._subsets)):
    #         best_overlap = -1
    #         best_j = None
    #         for j in range(len(other._subsets)):
    #             if j not in used_other_subsets:
    #                 overlap = contingency.get((i,j), 0)
    #                 if overlap > best_overlap:
    #                     best_overlap = overlap
    #                     best_j = j
    #         if best_j is not None:
    #             matches[i] = best_j
    #             used_other_subsets.add(best_j)

    #     # Calculate misclassification count
    #     misclassifications = 0
    #     for i in range(len(self._subsets)):
    #         matched_j = matches.get(i)
    #         if matched_j is not None:
    #             # Add elements not in matched subset
    #             misclassifications += sum(contingency.get((i,j), 0)
    #                                 for j in range(len(other._subsets))
    #                                 if j != matched_j)
    #         else:
    #             # If no match found, all elements are misclassified
    #             misclassifications += sum(contingency.get((i,j), 0)
    #                                     for j in range(len(other._subsets)))

    #     # Return normalized error
    #     return misclassifications / n if n > 0 else 0.0


class PartitionCollection:
    def __init__(self, log_base=2.0):
        self.log_base = log_base
        self._partitions = {}

    def __len__(self):
        return len(self._partitions)

    def compose_partition_label(self, ptn, idx=None):
        if idx is None:
            idx = len(self._partitions) + 1
        disambigution_idx = 1
        label = f"{idx}"
        key = label
        while key in self._partitions:
            disambigution_idx += 1
            key = f"{label}_{disambigution_idx:04d}"
        return key

    def new_partition(
        self,
        # label,
        subsets,
        metadata_d,
    ):
        ptn = Partition(
            # label=label,
            subsets=subsets,
            metadata_d=metadata_d,
            log_base=self.log_base,
        )
        key = self.compose_partition_label(ptn)
        self._partitions[key] = ptn
        return ptn

    def update_metadata(self, update_d):
        for ptn in self._partitions:
            ptn.metadata_d.update(update_d)

    def export_definition_d(
        self,
    ):
        exported = {
            "partitions": {
                key: ptn.compose_definition_d() for key, ptn in self._partitions.items()
            }
        }
        return exported

    def validate(
        self,
        runtime_context=None,
    ):
        """
        Ensure every partition is: jointly-comprehensive and mutually-exclusive
        with respect to the whole set.
        """
        all_elements = None
        for ptn_key, ptn in self._partitions.items():
            ptn_elements = set()
            for subset_idx, subset in enumerate(ptn._subsets):
                for element_idx, element in enumerate(subset._elements):
                    assert element not in ptn_elements
                    ptn_elements.add(element)
            if all_elements is None:
                all_elements = ptn_elements
                runtime_context and runtime_context.logger.info(
                    f"Validating partitioning of {len(ptn_elements)} elements: {sorted(ptn_elements)}"
                )
            else:
                assert all_elements == ptn_elements
        runtime_context and runtime_context.logger.info(
            f"All partitions are mutually-exclusive and jointly comprehensive with respect to {len(ptn_elements)} elements."
        )

    def read(
        self,
        source_path,
        source_format,
        limit_partitions=None,
        is_store_source_path=True,
        update_metadata=None,
        runtime_context=None,
    ):
        runtime_context and runtime_context.logger.info(f"Reading source: '{source_path}'")
        parser = parse.Parser(
            source_format=source_format,
        )
        parser.partition_factory = self.new_partition
        start_len = len(self)
        n_source_partitions = None
        for pidx, ptn in enumerate(parser.read_path(source_path)):
            runtime_context and runtime_context.logger.info(
                # f"Partition {pidx+1:>5d} of {len(src_partitions)} ({len(subsets)} subsets)"
                # f"Partition {pidx+1:>5d} of {ptn._origin_size}: {ptn.n_elements} lineages organized into {ptn.n_subsets} species"
                f"Partition {ptn._origin_offset+1} of {ptn._origin_size}: {ptn.n_elements} lineages organized into {ptn.n_subsets} species"
            )
            if not n_source_partitions:
                n_source_partitions = ptn._origin_size
            # ptn.metadata_d["source_path"] = str(pathlib.Path(src_path).absolute())
            if is_store_source_path:
                ptn.metadata_d["source"] = str(pathlib.Path(source_path).absolute())
            if update_metadata:
                ptn.metadata_d.update(update_metadata)
            if limit_partitions and (pidx >= limit_partitions - 1):
                runtime_context and runtime_context.logger.info(
                    f"Number of partitions read is at limit ({limit_partitions}): skipping remaining"
                )
                break
        end_len = len(self)
        runtime_context and runtime_context.logger.info(
            f"Reading completed: {end_len - start_len} of {n_source_partitions} partitions read from source ({len(self)} read in total)"
        )

    def summarize(
        self,
        runtime_context,
    ):

        ptn_summaries = {}
        summary_name_config_maps = {}

        metadata_key_filter_fn = lambda key: True
        summary_name_config_maps["metadata_keys"] = {
                "summary_fn": lambda ptn: [ (md_key, 1) for md_key in ptn.metadata_d
                                                if ( (not md_key.startswith("__piikun")) and metadata_key_filter_fn(md_key))
                                          ],
                "column_names": ["metadata", "count"],
        }

        summary_name_config_maps["species_comps"] = {
                "summary_fn": lambda ptn: [ ("; ".join(sorted(subset._elements)), ptn.metadata_d.get("score", 1.0)) for subset in ptn._subsets ],
                "column_names": ["species_composition", "weight"],
        }
        summary_name_config_maps["num_species"] = {
                "summary_fn": lambda ptn: [(ptn.n_subsets, ptn.metadata_d.get("score", 1.0))],
                "column_names": ["num_species", "weight"],
        }

        for summary_idx, (summary_name, summary_config) in enumerate(summary_name_config_maps.items()):
            ptn_summaries[summary_name] = collections.Counter()
            for ptn_key, ptn in self._partitions.items():
                ptn.metadata_d["__piikun_key"] = ptn_key
                if "summary_fn" in summary_config:
                    for (aspect_id, aspect_score) in summary_config["summary_fn"](ptn):
                        ptn_summaries[summary_name][aspect_id] += aspect_score

        for summary_idx, (summary_name, summary_results) in enumerate(ptn_summaries.items()):
            output_filepath = runtime_context.compose_output_path(subtitle=f"summary-{summary_name}", ext="tsv")
            summary_config = summary_name_config_maps.get(summary_name, {})
            runtime_context.logger.info(f"Summarizing {summary_config.get('description', summary_name)}: {output_filepath}")
            column_names = summary_config.get("column_names")
            if not column_names:
                column_names = [summary_name, "score"]
            df = utility.dataframe_from_counter(
                summary_results,
                column_names=column_names,
            )
            df.to_csv(output_filepath, sep="\t")
            runtime_context.logger.info(df)



