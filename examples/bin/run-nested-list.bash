#! /bin/bash

export DATABANK_ROOT=../data/
set -e -o pipefail

piikun-compile --format delineate ${DATABANK_ROOT}/delineate/maddison-2020/lionepha-p090-hky.mcct-mean-age.delimitation-results.json
piikun-evaluate ex1__partitions.json
piikun-visualize ex1__distances.json
