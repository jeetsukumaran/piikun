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
from piikun import parse

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
        # label=None,
        # partition_d=None,
        subsets=None,
        log_base=2,
        metadata_d=None,
    ):
        self.log_base = log_base
        self.log_fn = lambda x: math.log(x, self.log_base)
        # self.log_fn = lambda x: math.log(x)
        # self.label = label
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

    # @property
    # def label(self):
    #     if not hasattr(self, "_label") or self._label is None:
    #         self._label = str(self._index)
    #     return self._label

    # @label.setter
    # def label(self, value):
    #     self._label = value

    # @label.deleter
    # def label(self):
    #     del self._label

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

    # @property
    # def source_data_d(self):
    #     if (
    #         not hasattr(self, "_definition_d")
    #         or self._definition_d is None
    #     ):
    #         d = {}
    #         d["label"] = self.label
    #         d["subsets"] = list(self._subsets)
    #         self._definition_d = d
    #     return self._definition_d

    def compose_definition_d(self, key=None):
        d = {}
        if key is not None:
            d["label"] = key
        d["subsets"] = [sorted(subset._elements) for subset in self._subsets]
        d["metadata"] = self.metadata_d
        return d

    def new_subset(self, elements):
        for element in elements:
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

    def export_definition_d(
        self,
    ):
        exported = { "partitions": { key:ptn.compose_definition_d() for key, ptn in self._partitions.items() } }
        return exported

    def validate(
        self,
        rc=None,
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
                rc and rc.logger.info(f"Validating partitioning of {len(ptn_elements)} elements: {sorted(ptn_elements)}")
                all_elements = ptn_elements
            else:
                assert all_elements == ptn_elements

    def read(
        self,
        source_path,
        source_format,
        limit_partitions=None,
        rc=None,
    ):
        rc and rc.logger.info(f"Reading source: '{source_path}'")
        parser = parse.Parser(
            source_format=source_format,
        )
        parser.partition_factory = self.new_partition
        start_len = len(self)
        for pidx, ptn in enumerate(parser.read_path(source_path)):
            rc and rc.logger.info(
                # f"Partition {pidx+1:>5d} of {len(src_partitions)} ({len(subsets)} subsets)"
                # f"Partition {pidx+1:>5d} of {ptn._origin_size}: {ptn.n_elements} lineages organized into {ptn.n_subsets} species"
                f"Partition {ptn._origin_offset+1} of {ptn._origin_size}: {ptn.n_elements} lineages organized into {ptn.n_subsets} species"
            )
            # ptn.metadata_d["source_path"] = str(pathlib.Path(src_path).absolute())
            ptn.metadata_d["origin"] = {}
            ptn.metadata_d["origin"]["source_path"] = str(pathlib.Path(source_path).absolute())
            ptn.metadata_d["origin"]["source_size"] = ptn._origin_size
            ptn.metadata_d["origin"]["source_offset"] = ptn._origin_offset
            # ptn.metadata_d["origin"]["source_read"] = src_idx + 1
            # if rc.output_title and rc.output_title != "-":
            # if args.is_validate:
            #     partitions.validate(logger=rc.logger)
            # -1 as we need to anticipate limit being reached in the next loop
            if limit_partitions and (pidx >= limit_partitions - 1):
                rc and rc.logger.info(
                    f"Number of partitions read is at limit ({limit_partitions}): skipping remaining"
                )
                break
        end_len = len(self)
        rc and rc.logger.info(
            f"Reading completed: {end_len - start_len} of {ptn.metadata_d['origin']['source_size']} partitions read from source ({len(self)} read in total)"
        )




