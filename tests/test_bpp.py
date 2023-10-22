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
    expected_results = {
        "partitions": {
            "1": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1", "S4.sub1", "S5.sub1", "S6.sub1"]
                ],
                "metadata": {
                    "support": 0.0,
                    "prior_probability": 0.166667,
                    "posterior_probability": 0.0,
                },
            },
            "2": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S4.sub1", "S5.sub1", "S6.sub1"],
                    ["S2.sub1"],
                ],
                "metadata": {
                    "support": 0.217,
                    "prior_probability": 0.166667,
                    "posterior_probability": 0.217,
                },
            },
            "3": {
                "subsets": [
                    ["S1.sub1"],
                    ["S3.sub1", "S4.sub1", "S5.sub1", "S6.sub1"],
                    ["S2.sub1"],
                ],
                "metadata": {
                    "support": 0.21,
                    "prior_probability": 0.166667,
                    "posterior_probability": 0.21,
                },
            },
            "4": {
                "subsets": [
                    ["S1.sub1"],
                    ["S4.sub1"],
                    ["S3.sub1", "S5.sub1", "S6.sub1"],
                    ["S2.sub1"],
                ],
                "metadata": {
                    "support": 0.176,
                    "prior_probability": 0.166667,
                    "posterior_probability": 0.176,
                },
            },
            "5": {
                "subsets": [
                    ["S1.sub1"],
                    ["S4.sub1"],
                    ["S3.sub1"],
                    ["S5.sub1", "S6.sub1"],
                    ["S2.sub1"],
                ],
                "metadata": {
                    "support": 0.187,
                    "prior_probability": 0.166667,
                    "posterior_probability": 0.187,
                },
            },
            "6": {
                "subsets": [
                    ["S1.sub1"],
                    ["S4.sub1"],
                    ["S3.sub1"],
                    ["S6.sub1"],
                    ["S5.sub1"],
                    ["S2.sub1"],
                ],
                "metadata": {
                    "support": 0.21,
                    "prior_probability": 0.166667,
                    "posterior_probability": 0.21,
                },
            },
        }
    }
    pc = partitionmodel.PartitionCollection()
    for ptn_idx, ptn in enumerate(
        parsebpp.parse_bpp_a10(
            source_stream=open(source_path),
            partition_factory=pc.new_partition,
        )
    ):
        pass
    observed_results = pc.export_definition_d()
    print(observed_results)
    assert observed_results == expected_results


def test_bpp_a11():
    source_path = _pathmap.TESTS_DATA_DIR / "bpp-a11.01.txt"
    expected_results = {
        "partitions": {
            "1": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1", "S4.sub1", "S5.sub1", "S6.sub1"]
                ],
                "metadata": {"count": 74, "frequency": 0.074, "support": 0.074},
            },
            "2": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S3.sub1", "S4.sub1", "S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 45, "frequency": 0.045, "support": 0.045},
            },
            "3": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1", "S4.sub1", "S6.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 35, "frequency": 0.035, "support": 0.035},
            },
            "4": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S4.sub1", "S5.sub1", "S6.sub1"],
                    ["S2.sub1"],
                ],
                "metadata": {"count": 35, "frequency": 0.035, "support": 0.035},
            },
            "5": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1", "S5.sub1", "S6.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 35, "frequency": 0.035, "support": 0.035},
            },
            "6": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S4.sub1", "S5.sub1", "S6.sub1"],
                    ["S3.sub1"],
                ],
                "metadata": {"count": 34, "frequency": 0.034, "support": 0.034},
            },
            "7": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1", "S4.sub1", "S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 31, "frequency": 0.031, "support": 0.031},
            },
            "8": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 19, "frequency": 0.019, "support": 0.019},
            },
            "9": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S3.sub1", "S4.sub1", "S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 16, "frequency": 0.016, "support": 0.016},
            },
            "10": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S4.sub1", "S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 13, "frequency": 0.013, "support": 0.013},
            },
            "11": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1", "S5.sub1"],
                    ["S4.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 12, "frequency": 0.012, "support": 0.012},
            },
            "12": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S3.sub1", "S4.sub1", "S6.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 11, "frequency": 0.011, "support": 0.011},
            },
            "13": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S5.sub1"],
                    ["S2.sub1", "S4.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 11, "frequency": 0.011, "support": 0.011},
            },
            "14": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S3.sub1", "S4.sub1"],
                    ["S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 11, "frequency": 0.011, "support": 0.011},
            },
            "15": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S3.sub1", "S6.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 10, "frequency": 0.01, "support": 0.01},
            },
            "16": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S5.sub1", "S6.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 10, "frequency": 0.01, "support": 0.01},
            },
            "17": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S4.sub1", "S6.sub1"],
                    ["S3.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 10, "frequency": 0.01, "support": 0.01},
            },
            "18": {
                "subsets": [
                    ["S1.sub1", "S4.sub1"],
                    ["S2.sub1", "S3.sub1", "S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 9, "frequency": 0.009, "support": 0.009},
            },
            "19": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S4.sub1", "S5.sub1"],
                    ["S2.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 9, "frequency": 0.009, "support": 0.009},
            },
            "20": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S3.sub1"],
                    ["S4.sub1", "S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 8, "frequency": 0.008, "support": 0.008},
            },
            "21": {
                "subsets": [
                    ["S1.sub1", "S5.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 8, "frequency": 0.008, "support": 0.008},
            },
            "22": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S3.sub1", "S5.sub1"],
                    ["S4.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 8, "frequency": 0.008, "support": 0.008},
            },
            "23": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S4.sub1", "S5.sub1"],
                    ["S3.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 8, "frequency": 0.008, "support": 0.008},
            },
            "24": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S5.sub1", "S6.sub1"],
                    ["S2.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 8, "frequency": 0.008, "support": 0.008},
            },
            "25": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S4.sub1", "S6.sub1"],
                    ["S3.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 8, "frequency": 0.008, "support": 0.008},
            },
            "26": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1", "S4.sub1"],
                    ["S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 8, "frequency": 0.008, "support": 0.008},
            },
            "27": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1", "S5.sub1"],
                    ["S4.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 8, "frequency": 0.008, "support": 0.008},
            },
            "28": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1", "S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 8, "frequency": 0.008, "support": 0.008},
            },
            "29": {
                "subsets": [
                    ["S1.sub1", "S6.sub1"],
                    ["S2.sub1", "S3.sub1", "S4.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 8, "frequency": 0.008, "support": 0.008},
            },
            "30": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S4.sub1", "S5.sub1", "S6.sub1"],
                    ["S3.sub1"],
                ],
                "metadata": {"count": 8, "frequency": 0.008, "support": 0.008},
            },
            "31": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S5.sub1"],
                    ["S4.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 7, "frequency": 0.007, "support": 0.007},
            },
            "32": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 7, "frequency": 0.007, "support": 0.007},
            },
            "33": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S5.sub1"],
                    ["S2.sub1"],
                    ["S4.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 7, "frequency": 0.007, "support": 0.007},
            },
            "34": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1", "S6.sub1"],
                    ["S4.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 7, "frequency": 0.007, "support": 0.007},
            },
            "35": {
                "subsets": [
                    ["S1.sub1", "S2.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 7, "frequency": 0.007, "support": 0.007},
            },
            "36": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1", "S6.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 7, "frequency": 0.007, "support": 0.007},
            },
            "37": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S4.sub1", "S5.sub1"],
                    ["S2.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 7, "frequency": 0.007, "support": 0.007},
            },
            "38": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S4.sub1"],
                    ["S3.sub1"],
                    ["S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 7, "frequency": 0.007, "support": 0.007},
            },
            "39": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S3.sub1", "S5.sub1", "S6.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 7, "frequency": 0.007, "support": 0.007},
            },
            "40": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S5.sub1", "S6.sub1"],
                    ["S2.sub1", "S4.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "41": {
                "subsets": [
                    ["S1.sub1", "S3.sub1"],
                    ["S2.sub1"],
                    ["S4.sub1", "S6.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "42": {
                "subsets": [
                    ["S1.sub1", "S4.sub1", "S6.sub1"],
                    ["S2.sub1", "S3.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "43": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1"],
                    ["S4.sub1", "S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "44": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "45": {
                "subsets": [
                    ["S1.sub1", "S4.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1"],
                    ["S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "46": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S5.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "47": {
                "subsets": [
                    ["S1.sub1", "S4.sub1", "S6.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "48": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S6.sub1"],
                    ["S3.sub1", "S4.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "49": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S5.sub1", "S6.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "50": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S4.sub1"],
                    ["S2.sub1", "S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "51": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S6.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "52": {
                "subsets": [
                    ["S1.sub1", "S4.sub1", "S5.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "53": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1", "S4.sub1"],
                    ["S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "54": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S5.sub1"],
                    ["S3.sub1", "S6.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "55": {
                "subsets": [
                    ["S1.sub1", "S2.sub1"],
                    ["S3.sub1", "S4.sub1", "S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "56": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S6.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 6, "frequency": 0.006, "support": 0.006},
            },
            "57": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S5.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 5, "frequency": 0.005, "support": 0.005},
            },
            "58": {
                "subsets": [
                    ["S1.sub1", "S6.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 5, "frequency": 0.005, "support": 0.005},
            },
            "59": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S5.sub1"],
                    ["S3.sub1", "S4.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 5, "frequency": 0.005, "support": 0.005},
            },
            "60": {
                "subsets": [
                    ["S1.sub1", "S5.sub1"],
                    ["S2.sub1", "S3.sub1", "S4.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 5, "frequency": 0.005, "support": 0.005},
            },
            "61": {
                "subsets": [
                    ["S1.sub1", "S5.sub1", "S6.sub1"],
                    ["S2.sub1", "S3.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 5, "frequency": 0.005, "support": 0.005},
            },
            "62": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1", "S6.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 5, "frequency": 0.005, "support": 0.005},
            },
            "63": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S5.sub1"],
                    ["S2.sub1"],
                    ["S4.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 5, "frequency": 0.005, "support": 0.005},
            },
            "64": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S5.sub1"],
                    ["S2.sub1", "S6.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "65": {
                "subsets": [
                    ["S1.sub1", "S3.sub1"],
                    ["S2.sub1", "S4.sub1"],
                    ["S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "66": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S5.sub1", "S6.sub1"],
                    ["S3.sub1", "S4.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "67": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S4.sub1"],
                    ["S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "68": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S4.sub1"],
                    ["S3.sub1", "S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "69": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S3.sub1", "S4.sub1"],
                    ["S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "70": {
                "subsets": [
                    ["S1.sub1", "S6.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S4.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "71": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S4.sub1", "S6.sub1"],
                    ["S2.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "72": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S4.sub1", "S6.sub1"],
                    ["S3.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "73": {
                "subsets": [
                    ["S1.sub1", "S3.sub1"],
                    ["S2.sub1", "S4.sub1", "S6.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "74": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S6.sub1"],
                    ["S2.sub1", "S4.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "75": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S4.sub1", "S5.sub1"],
                    ["S3.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "76": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S3.sub1"],
                    ["S4.sub1", "S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "77": {
                "subsets": [
                    ["S1.sub1", "S5.sub1", "S6.sub1"],
                    ["S2.sub1", "S3.sub1", "S4.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "78": {
                "subsets": [
                    ["S1.sub1", "S4.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "79": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S4.sub1", "S6.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "80": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S3.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 4, "frequency": 0.004, "support": 0.004},
            },
            "81": {
                "subsets": [
                    ["S1.sub1", "S3.sub1"],
                    ["S2.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "82": {
                "subsets": [
                    ["S1.sub1", "S4.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1"],
                    ["S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "83": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S4.sub1", "S5.sub1"],
                    ["S3.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "84": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S5.sub1"],
                    ["S3.sub1", "S4.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "85": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S4.sub1"],
                    ["S2.sub1"],
                    ["S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "86": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S5.sub1", "S6.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "87": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S4.sub1"],
                    ["S3.sub1", "S6.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "88": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S6.sub1"],
                    ["S3.sub1", "S4.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "89": {
                "subsets": [
                    ["S1.sub1", "S6.sub1"],
                    ["S2.sub1", "S4.sub1"],
                    ["S3.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "90": {
                "subsets": [
                    ["S1.sub1", "S5.sub1"],
                    ["S2.sub1", "S3.sub1"],
                    ["S4.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "91": {
                "subsets": [
                    ["S1.sub1", "S4.sub1", "S5.sub1", "S6.sub1"],
                    ["S2.sub1", "S3.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "92": {
                "subsets": [
                    ["S1.sub1", "S4.sub1", "S5.sub1"],
                    ["S2.sub1", "S3.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "93": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S6.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "94": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S6.sub1"],
                    ["S3.sub1", "S4.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "95": {
                "subsets": [
                    ["S1.sub1", "S6.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "96": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S4.sub1", "S6.sub1"],
                    ["S2.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "97": {
                "subsets": [
                    ["S1.sub1", "S4.sub1"],
                    ["S2.sub1", "S6.sub1"],
                    ["S3.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "98": {
                "subsets": [
                    ["S1.sub1", "S2.sub1"],
                    ["S3.sub1", "S5.sub1", "S6.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "99": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S6.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "100": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1", "S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "101": {
                "subsets": [
                    ["S1.sub1", "S4.sub1"],
                    ["S2.sub1", "S5.sub1"],
                    ["S3.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "102": {
                "subsets": [
                    ["S1.sub1", "S4.sub1", "S6.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "103": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S4.sub1", "S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "104": {
                "subsets": [
                    ["S1.sub1", "S3.sub1"],
                    ["S2.sub1", "S4.sub1", "S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "105": {
                "subsets": [
                    ["S1.sub1", "S6.sub1"],
                    ["S2.sub1", "S4.sub1", "S5.sub1"],
                    ["S3.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "106": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S3.sub1", "S6.sub1"],
                    ["S4.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "107": {
                "subsets": [
                    ["S1.sub1", "S3.sub1"],
                    ["S2.sub1", "S5.sub1"],
                    ["S4.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "108": {
                "subsets": [
                    ["S1.sub1", "S3.sub1"],
                    ["S2.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "109": {
                "subsets": [
                    ["S1.sub1", "S2.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "110": {
                "subsets": [
                    ["S1.sub1", "S2.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1", "S6.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "111": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S3.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "112": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S4.sub1", "S5.sub1"],
                    ["S3.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "113": {
                "subsets": [
                    ["S1.sub1", "S6.sub1"],
                    ["S2.sub1", "S5.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "114": {
                "subsets": [
                    ["S1.sub1", "S4.sub1"],
                    ["S2.sub1", "S5.sub1"],
                    ["S3.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "115": {
                "subsets": [
                    ["S1.sub1", "S4.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S6.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "116": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S6.sub1"],
                    ["S3.sub1", "S4.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "117": {
                "subsets": [
                    ["S1.sub1", "S4.sub1", "S5.sub1", "S6.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "118": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S6.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 3, "frequency": 0.003, "support": 0.003},
            },
            "119": {
                "subsets": [
                    ["S1.sub1", "S2.sub1"],
                    ["S3.sub1", "S4.sub1", "S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "120": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S5.sub1"],
                    ["S2.sub1", "S4.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "121": {
                "subsets": [
                    ["S1.sub1", "S5.sub1", "S6.sub1"],
                    ["S2.sub1", "S4.sub1"],
                    ["S3.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "122": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S6.sub1"],
                    ["S2.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "123": {
                "subsets": [
                    ["S1.sub1", "S5.sub1", "S6.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S4.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "124": {
                "subsets": [
                    ["S1.sub1", "S4.sub1", "S6.sub1"],
                    ["S2.sub1", "S5.sub1"],
                    ["S3.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "125": {
                "subsets": [
                    ["S1.sub1", "S4.sub1"],
                    ["S2.sub1", "S5.sub1", "S6.sub1"],
                    ["S3.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "126": {
                "subsets": [
                    ["S1.sub1", "S5.sub1"],
                    ["S2.sub1", "S3.sub1"],
                    ["S4.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "127": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S5.sub1"],
                    ["S3.sub1", "S6.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "128": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S6.sub1"],
                    ["S4.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "129": {
                "subsets": [
                    ["S1.sub1", "S6.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S4.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "130": {
                "subsets": [
                    ["S1.sub1", "S5.sub1", "S6.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "131": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S6.sub1"],
                    ["S3.sub1", "S5.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "132": {
                "subsets": [
                    ["S1.sub1", "S4.sub1"],
                    ["S2.sub1", "S3.sub1", "S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "133": {
                "subsets": [
                    ["S1.sub1", "S4.sub1"],
                    ["S2.sub1", "S3.sub1"],
                    ["S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "134": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S6.sub1"],
                    ["S2.sub1", "S4.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "135": {
                "subsets": [
                    ["S1.sub1", "S5.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S6.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "136": {
                "subsets": [
                    ["S1.sub1", "S4.sub1"],
                    ["S2.sub1", "S6.sub1"],
                    ["S3.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "137": {
                "subsets": [
                    ["S1.sub1", "S5.sub1"],
                    ["S2.sub1", "S4.sub1"],
                    ["S3.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "138": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S5.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "139": {
                "subsets": [
                    ["S1.sub1", "S6.sub1"],
                    ["S2.sub1", "S3.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "140": {
                "subsets": [
                    ["S1.sub1", "S6.sub1"],
                    ["S2.sub1", "S4.sub1"],
                    ["S3.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "141": {
                "subsets": [
                    ["S1.sub1", "S6.sub1"],
                    ["S2.sub1", "S3.sub1", "S5.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "142": {
                "subsets": [
                    ["S1.sub1", "S4.sub1"],
                    ["S2.sub1", "S3.sub1", "S6.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "143": {
                "subsets": [
                    ["S1.sub1", "S5.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S4.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "144": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S5.sub1", "S6.sub1"],
                    ["S3.sub1", "S4.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "145": {
                "subsets": [
                    ["S1.sub1", "S5.sub1"],
                    ["S2.sub1", "S4.sub1"],
                    ["S3.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "146": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1"],
                    ["S4.sub1", "S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "147": {
                "subsets": [
                    ["S1.sub1", "S6.sub1"],
                    ["S2.sub1", "S5.sub1"],
                    ["S3.sub1", "S4.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "148": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S6.sub1"],
                    ["S3.sub1", "S5.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "149": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S6.sub1"],
                    ["S2.sub1", "S5.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "150": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S4.sub1"],
                    ["S3.sub1"],
                    ["S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "151": {
                "subsets": [
                    ["S1.sub1", "S4.sub1", "S6.sub1"],
                    ["S2.sub1", "S3.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "152": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S4.sub1"],
                    ["S2.sub1"],
                    ["S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 2, "frequency": 0.002, "support": 0.002},
            },
            "153": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S3.sub1", "S5.sub1"],
                    ["S4.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "154": {
                "subsets": [
                    ["S1.sub1", "S4.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "155": {
                "subsets": [
                    ["S1.sub1", "S2.sub1"],
                    ["S3.sub1", "S5.sub1"],
                    ["S4.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "156": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S4.sub1"],
                    ["S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "157": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S4.sub1", "S6.sub1"],
                    ["S3.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "158": {
                "subsets": [
                    ["S1.sub1", "S2.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1", "S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "159": {
                "subsets": [
                    ["S1.sub1", "S6.sub1"],
                    ["S2.sub1", "S3.sub1"],
                    ["S4.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "160": {
                "subsets": [
                    ["S1.sub1", "S5.sub1"],
                    ["S2.sub1", "S4.sub1", "S6.sub1"],
                    ["S3.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "161": {
                "subsets": [
                    ["S1.sub1", "S5.sub1"],
                    ["S2.sub1", "S6.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "162": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S4.sub1"],
                    ["S3.sub1", "S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "163": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S5.sub1"],
                    ["S3.sub1", "S4.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "164": {
                "subsets": [
                    ["S1.sub1", "S2.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1", "S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "165": {
                "subsets": [
                    ["S1.sub1", "S4.sub1", "S5.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "166": {
                "subsets": [
                    ["S1.sub1", "S3.sub1", "S4.sub1"],
                    ["S2.sub1", "S6.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "167": {
                "subsets": [
                    ["S1.sub1", "S3.sub1"],
                    ["S2.sub1", "S4.sub1", "S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "168": {
                "subsets": [
                    ["S1.sub1", "S3.sub1"],
                    ["S2.sub1"],
                    ["S4.sub1", "S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "169": {
                "subsets": [
                    ["S1.sub1", "S3.sub1"],
                    ["S2.sub1"],
                    ["S4.sub1", "S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "170": {
                "subsets": [
                    ["S1.sub1", "S4.sub1", "S5.sub1"],
                    ["S2.sub1", "S6.sub1"],
                    ["S3.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "171": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S5.sub1"],
                    ["S3.sub1"],
                    ["S4.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "172": {
                "subsets": [
                    ["S1.sub1", "S5.sub1"],
                    ["S2.sub1", "S3.sub1", "S4.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "173": {
                "subsets": [
                    ["S1.sub1", "S5.sub1"],
                    ["S2.sub1", "S3.sub1", "S6.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "174": {
                "subsets": [
                    ["S1.sub1", "S3.sub1"],
                    ["S2.sub1", "S6.sub1"],
                    ["S4.sub1", "S5.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "175": {
                "subsets": [
                    ["S1.sub1", "S3.sub1"],
                    ["S2.sub1", "S5.sub1", "S6.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "176": {
                "subsets": [
                    ["S1.sub1", "S3.sub1"],
                    ["S2.sub1", "S6.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "177": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S4.sub1"],
                    ["S3.sub1"],
                    ["S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "178": {
                "subsets": [
                    ["S1.sub1", "S5.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S4.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "179": {
                "subsets": [
                    ["S1.sub1", "S6.sub1"],
                    ["S2.sub1"],
                    ["S3.sub1", "S5.sub1"],
                    ["S4.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "180": {
                "subsets": [
                    ["S1.sub1", "S6.sub1"],
                    ["S2.sub1", "S3.sub1", "S4.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "181": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S4.sub1"],
                    ["S3.sub1", "S5.sub1"],
                    ["S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "182": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1"],
                    ["S4.sub1"],
                    ["S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "183": {
                "subsets": [
                    ["S1.sub1", "S2.sub1", "S3.sub1"],
                    ["S4.sub1", "S6.sub1"],
                    ["S5.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
            "184": {
                "subsets": [
                    ["S1.sub1"],
                    ["S2.sub1", "S4.sub1"],
                    ["S3.sub1", "S5.sub1", "S6.sub1"],
                ],
                "metadata": {"count": 1, "frequency": 0.001, "support": 0.001},
            },
        }
    }

    pc = partitionmodel.PartitionCollection()
    for ptn_idx, ptn in enumerate(
        parsebpp.parse_bpp_a11(
            source_stream=open(source_path),
            partition_factory=pc.new_partition,
        )
    ):
        pass
    observed_results = pc.export_definition_d()
    print(observed_results)
    assert observed_results == expected_results


if __name__ == "__main__":
    test_bpp_a11()


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
