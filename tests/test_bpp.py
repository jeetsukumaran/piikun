#! /usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import re

if __name__ == "__main__":
    import _pathmap
else:
    from . import _pathmap

from piikun import parsebpp
from piikun import partitionmodel


def test_bpp_a10():
    source_path = _pathmap.TESTS_DATA_DIR / "bpp-a10.01.txt"
    expected_results = {'partitions': {'1': {'subsets': [['S1.sub1', 'S2.sub1', 'S3.sub1', 'S4.sub1', 'S5.sub1', 'S6.sub1']], 'metadata': {'support': 0.0, 'prior_probability': 0.166667, 'posterior_probability': 0.0}}, '2': {'subsets': [['S1.sub1', 'S3.sub1', 'S4.sub1', 'S5.sub1', 'S6.sub1'], ['S2.sub1']], 'metadata': {'support': 0.217, 'prior_probability': 0.166667, 'posterior_probability': 0.217}}, '3': {'subsets': [['S1.sub1'], ['S3.sub1', 'S4.sub1', 'S5.sub1', 'S6.sub1'], ['S2.sub1']], 'metadata': {'support': 0.21, 'prior_probability': 0.166667, 'posterior_probability': 0.21}}, '4': {'subsets': [['S1.sub1'], ['S4.sub1'], ['S3.sub1', 'S5.sub1', 'S6.sub1'], ['S2.sub1']], 'metadata': {'support': 0.176, 'prior_probability': 0.166667, 'posterior_probability': 0.176}}, '5': {'subsets': [['S1.sub1'], ['S4.sub1'], ['S3.sub1'], ['S5.sub1', 'S6.sub1'], ['S2.sub1']], 'metadata': {'support': 0.187, 'prior_probability': 0.166667, 'posterior_probability': 0.187}}, '6': {'subsets': [['S1.sub1'], ['S4.sub1'], ['S3.sub1'], ['S6.sub1'], ['S5.sub1'], ['S2.sub1']], 'metadata': {'support': 0.21, 'prior_probability': 0.166667, 'posterior_probability': 0.21}}}}
    pc = partitionmodel.PartitionCollection()
    for ptn_idx, ptn in enumerate(
        parsebpp.parse_bpp_a10(
            source_stream=open(source_path),
            partition_factory=pc.new_partition,
        )
    ):
        pass
    observed_results = pc.export_definition_d()
    # print(observed_results)
    assert observed_results == expected_results


# def parse_bpp_a11_data(labels, src_data):
#     parsed_data = []
#     for line in src_data.strip().split("\n"):
#         freq, num_blocks, partition_str = re.search(r'(\d+\.\d+)\s+(\d+)\s+\((.+)\)', line).groups()
#         freq = float(freq)
#         num_blocks = int(num_blocks)
#         partition_blocks = []
#         for partition in partition_str.split(' '):
#             block = []
#             remaining_str = partition
#             while remaining_str:
#                 for label in sorted(labels, key=len, reverse=True):  # To match longer labels first
#                     if remaining_str.startswith(label):
#                         block.append(label)
#                         remaining_str = remaining_str[len(label):]
#                         break
#             partition_blocks.append(block)
#         assert num_blocks == len(partition_blocks)
#         parsed_data.append((freq, num_blocks, partition_blocks))
#     return parsed_data

# labels = ["Ge", "Ta", "Tr", "SA", "CA", "MX", "Al", "El",]
# src_data = """
#   99180   0.99180   8 (Ge Ta Tr Al El MX CA SA)
#     801   0.00801   1 (GeTaTrAlElMXCASA)
#       6   0.00006   6 (Ge Ta Tr Al El MXCASA)
#       4   0.00004   3 (GeTa Tr AlElMXCASA)
#       3   0.00003   5 (Ge Ta Tr AlMXCASA El)
#       2   0.00002   2 (GeTaTr AlElMXCASA)
#       2   0.00002   4 (Ge Ta Tr AlElMXCASA)
#       2   0.00002   7 (Ge Ta Tr Al El MX CASA)
# """
# parsed_data = parse_bpp_a11_data(labels, src_data)
# print(parsed_data)
