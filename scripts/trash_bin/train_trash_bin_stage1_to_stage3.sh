#!/usr/bin/env bash
set -euo pipefail

# Run the trash-bin rolling curriculum:
#   stage1 -> stage2 via scripts/trash_bin/train_trash_bin_stage1_to_stage2.sh
#   stage3 via scripts/trash_bin/train_trash_bin_stage3.sh from the stage2 best checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED=1
RUN_TAG="$(date +%Y%m%d_%H%M%S)"

export SEED
export RUN_TAG
export STAGE1_RUN_NAME="trash_bin_stage1_rotation_only_25m_seed${SEED}_${RUN_TAG}"
export STAGE2_RUN_NAME="trash_bin_stage2_from_stage1_20m_seed${SEED}_${RUN_TAG}"
STAGE3_RUN_NAME="trash_bin_stage3_from_stage2_best_seed${SEED}_${RUN_TAG}"

echo
echo "### curriculum stage1 -> stage2"
bash scripts/trash_bin/train_trash_bin_stage1_to_stage2.sh

stage2_best_ckpt="runs/${STAGE2_RUN_NAME}/best_ckpt.pt"
if [[ ! -f "${stage2_best_ckpt}" ]]; then
  echo "stage2 best checkpoint does not exist: ${stage2_best_ckpt}" >&2
  exit 1
fi

echo
echo "### curriculum stage3 from stage2 best checkpoint"
echo "checkpoint: ${stage2_best_ckpt}"
echo "run: runs/${STAGE3_RUN_NAME}"

RUN_NAME="${STAGE3_RUN_NAME}" bash scripts/trash_bin/train_trash_bin_stage3.sh "${stage2_best_ckpt}"

echo
echo "### trash-bin rolling stage1 -> stage3 curriculum finished"
echo "stage1 run: runs/${STAGE1_RUN_NAME}"
echo "stage2 run: runs/${STAGE2_RUN_NAME}"
echo "stage2 best checkpoint: ${stage2_best_ckpt}"
echo "stage3 run: runs/${STAGE3_RUN_NAME}"
