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
import yakherd

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
        label=None,
        partition_d=None,
        subsets=None,
        log_base=2,
        metadata_d=None,
    ):
        self.log_base = log_base
        self.log_fn = lambda x: math.log(x, self.log_base)
        # self.log_fn = lambda x: math.log(x)
        self.label = label
        self._index = Partition._n_instances
        Partition._n_instances += 1
        self._subsets = []
        self._label_subset_map = {}
        self._elements = set()
        self.metadata_d = {}
        if metadata_d:
            self.metadata_d.update(metadata_d)
        if partition_d is not None:
            self.parse_partition_d(partition_d)
        elif subsets is not None:
            self.parse_subsets(subsets)

    @property
    def label(self):
        if not hasattr(self, "_label") or self._label is None:
            self._label = str(self._index)
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @label.deleter
    def label(self):
        del self._label

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

    def new_subset(self, label, elements):
        for element in elements:
            assert element not in self._elements
            self._elements.add(element)
        s = PartitionSubset(
            elements=elements,
        )
        self._label_subset_map[label] = s
        self._subsets.append(s)
        return s

    def parse_partition_d(self, partition_d):
        for label, v in partition_d.items():
            self.new_subset(label=label, elements=v)

    def parse_subsets(self, subsets):
        for label, v in enumerate(subsets):
            self.new_subset(label=label, elements=v)

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
        vi_mi = 0.0
        assert self._elements == other._elements
        for ptn1_idx, ptn1_subset in enumerate(self._subsets):
            for ptn2_idx, ptn2_subset in enumerate(other._subsets):
                # Meilă, Marina. 2007. Comparing clusterings—an information based distance. Journal of Multivariate Analysis. 98 (5): 873-895.
                # Section 3: Variance of Information
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
