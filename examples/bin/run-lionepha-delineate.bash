#! /bin/bash

export DATABANK_ROOT=../data/
set -e -o pipefail

piikun-compile \
    --source-format delineate \
    --output-title lionepha-delineate \
    ${DATABANK_ROOT}/delineate/maddison-2020/lionepha-p090-hky.mcct-mean-age.delimitation-results.json
piikun-evaluate lionepha-delineate__partitions.json
piikun-visualize lionepha-delineate__distances.json


