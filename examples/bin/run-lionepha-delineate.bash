#! /bin/bash

export DATABANK_ROOT=../data/
set -e -o pipefail

## Step-by-step
piikun-compile \
    --source-format delineate \
    --output-title lionepha-delineate \
    ${DATABANK_ROOT}/delineate/maddison-2020/lionepha-p090-hky.mcct-mean-age.delimitation-results.json
piikun-evaluate lionepha-delineate__partitions.json
piikun-visualize lionepha-delineate__distances.json


## All steps together
piikun-analyze \
    --source-format delineate \
    --output-title lionepha-delineate \
    ${DATABANK_ROOT}/delineate/maddison-2020/lionepha-p090-hky.mcct-mean-age.delimitation-results.json



