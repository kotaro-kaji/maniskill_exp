#!/usr/bin/env bash
set -euo pipefail

# Run the complete trash-bin curriculum through the low-learning-rate polish stage.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SEED="${SEED:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-trash_bin_stage1_rotation_only_25m_seed${SEED}_${RUN_TAG}}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-trash_bin_stage2_from_stage1_20m_seed${SEED}_${RUN_TAG}}"
STAGE3_RUN_NAME="${STAGE3_RUN_NAME:-trash_bin_stage3_from_stage2_best_seed${SEED}_${RUN_TAG}}"
STAGE4_RUN_NAME="${STAGE4_RUN_NAME:-trash_bin_stage4_from_stage3_best_seed${SEED}_${RUN_TAG}}"

STAGE3_TOTAL_TIMESTEPS="${STAGE3_TOTAL_TIMESTEPS:-100_000_000}"
STAGE4_TOTAL_TIMESTEPS="${STAGE4_TOTAL_TIMESTEPS:-20_000_000}"
STAGE4_LEARNING_RATE="${STAGE4_LEARNING_RATE:-3e-5}"
STAGE4_NUM_EVAL_ENVS="${STAGE4_NUM_EVAL_ENVS:-32}"
STAGE4_EVAL_FREQ="${STAGE4_EVAL_FREQ:-2}"

export SEED
export RUN_TAG
export STAGE1_RUN_NAME
export STAGE2_RUN_NAME
export STAGE3_RUN_NAME
export STAGE4_RUN_NAME

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
RUN_NAME="${STAGE3_RUN_NAME}" \
TOTAL_TIMESTEPS="${STAGE3_TOTAL_TIMESTEPS}" \
  bash scripts/trash_bin/train_trash_bin_stage3.sh "${stage2_best_ckpt}"

stage3_best_ckpt="runs/${STAGE3_RUN_NAME}/best_ckpt.pt"
if [[ ! -f "${stage3_best_ckpt}" ]]; then
  echo "stage3 best checkpoint does not exist: ${stage3_best_ckpt}" >&2
  exit 1
fi

echo
echo "### curriculum stage4 from stage3 best checkpoint"
RUN_NAME="${STAGE4_RUN_NAME}" \
TOTAL_TIMESTEPS="${STAGE4_TOTAL_TIMESTEPS}" \
LEARNING_RATE="${STAGE4_LEARNING_RATE}" \
NUM_EVAL_ENVS="${STAGE4_NUM_EVAL_ENVS}" \
EVAL_FREQ="${STAGE4_EVAL_FREQ}" \
  bash scripts/trash_bin/train_trash_bin_stage4.sh "${stage3_best_ckpt}"

stage4_best_ckpt="runs/${STAGE4_RUN_NAME}/best_ckpt.pt"
if [[ ! -f "${stage4_best_ckpt}" ]]; then
  echo "stage4 best checkpoint does not exist: ${stage4_best_ckpt}" >&2
  exit 1
fi

echo
echo "### trash-bin rolling curriculum stage1 -> stage4 finished"
echo "stage1 run: runs/${STAGE1_RUN_NAME}"
echo "stage2 run: runs/${STAGE2_RUN_NAME}"
echo "stage3 run: runs/${STAGE3_RUN_NAME}"
echo "stage4 run: runs/${STAGE4_RUN_NAME}"
echo "stage4 best checkpoint: ${stage4_best_ckpt}"
