#! /bin/bash

set -e -o pipefail

export DATASOURCE=/path/to/downloaded/data/lionepha.run1.delimitation-results.full.json

piikun-analyze \
    -f delineate \
    --output-title lionepha-delineate-n59-n200 \
    --limit 200 \
    ${DATASOURCE}


