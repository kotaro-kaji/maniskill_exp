#!/usr/bin/env bash
set -euo pipefail

# Run the panel-observation-noised cardboard curriculum in one command:
#   stage1 fixed delta=0.06
#   -> stage2 randomized-v1 delta=0.06
#   -> stage3 randomized-v2 delta=0.06
#   -> delta=0.03
#   -> delta=0.02
#
# Usage:
#   bash scripts/train_noised_cardboard_randomized_3stage_to_delta020.sh

THREE_STAGE_RUN_PREFIX="${THREE_STAGE_RUN_PREFIX:-noised_cardboard_3stage}" \
DELTA020_RUN_PREFIX="${DELTA020_RUN_PREFIX:-noised_cardboard_delta060_to_delta020}" \
bash scripts/train_cardboard_randomized_3stage_to_delta020.sh "$@"
