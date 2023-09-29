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

import math
import pytest

if __name__ == "__main__":
    import _pathmap
else:
    from . import _pathmap
from piikun import partitionmodel

def test_variance_of_information_spot_check1():
    ptns = [
        [
            ["ZNF28_281_359", "Q5H9V1_HUMAN_495-573", "ZNF28_HUMAN_498-576"],
            ["ZN578_HUMAN_146-224", "Q3MI94_HUMAN_147-225"],
            ["Q6ZP55_HUMAN_245-323"],
        ],
        [
            ["Q3MI94_HUMAN_147-225", "ZN578_HUMAN_146-224", "Q6ZP55_HUMAN_245-323"],
            ["ZNF28_281_359", "Q5H9V1_HUMAN_495-573"],
            ["ZNF28_HUMAN_498-576"],
        ],
    ]
    expected_values = [
        (math.e, [[0.0, 0.636514], [0.636514, 0.0]]),
    ]
    for log_base, result in expected_values:
        for pidx1, ptn1_data in enumerate(ptns):
            ptn1 = partitionmodel.Partition(
                subsets=ptn1_data,
                log_base=log_base,
            )
            for pidx2, ptn2_data in enumerate(ptns):
                ptn2 = partitionmodel.Partition(
                    subsets=ptn2_data,
                    log_base=log_base,
                )
                vi = ptn1.vi_distance(ptn2)
                assert pytest.approx(result[pidx1][pidx2], 1e-6) == vi
