#!/usr/bin/env bash
set -euo pipefail

# Train stages 1-4, then compare the polished policy with the fixed baseline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f ./server_env.sh ]]; then
  # shellcheck disable=SC1091
  source ./server_env.sh
fi

SEED="${SEED:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-trash_bin_stage1_rotation_only_25m_seed${SEED}_${RUN_TAG}}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-trash_bin_stage2_from_stage1_20m_seed${SEED}_${RUN_TAG}}"
STAGE3_RUN_NAME="${STAGE3_RUN_NAME:-trash_bin_stage3_from_stage2_best_seed${SEED}_${RUN_TAG}}"
STAGE4_RUN_NAME="${STAGE4_RUN_NAME:-trash_bin_stage4_from_stage3_best_seed${SEED}_${RUN_TAG}}"

BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-good_ckpts/TrashBinRolling/best_ckpt.pt}"
SMOOTHNESS_NUM_ENVS="${SMOOTHNESS_NUM_ENVS:-64}"
SMOOTHNESS_SEED="${SMOOTHNESS_SEED:-1}"
SMOOTHNESS_POLICY_NOISE_SEED="${SMOOTHNESS_POLICY_NOISE_SEED:-10001}"
SMOOTHNESS_SIM_BACKEND="${SMOOTHNESS_SIM_BACKEND:-physx_cuda}"
COMPARISON_OUTPUT_DIR="${COMPARISON_OUTPUT_DIR:-outputs_to_user/trash_bin_action_smoothness/${STAGE4_RUN_NAME}}"

export SEED
export RUN_TAG
export STAGE1_RUN_NAME
export STAGE2_RUN_NAME
export STAGE3_RUN_NAME
export STAGE4_RUN_NAME

echo
echo "### train trash-bin curriculum through stage4"
bash scripts/trash_bin/train_trash_bin_stage1_to_stage4.sh

candidate_checkpoint="runs/${STAGE4_RUN_NAME}/best_ckpt.pt"
if [[ ! -f "${candidate_checkpoint}" ]]; then
  echo "stage4 best checkpoint does not exist: ${candidate_checkpoint}" >&2
  exit 1
fi
if [[ ! -f "${BASELINE_CHECKPOINT}" ]]; then
  echo "baseline checkpoint does not exist: ${BASELINE_CHECKPOINT}" >&2
  exit 1
fi

echo
echo "### compare baseline and stage4 action smoothness"
echo "baseline: ${BASELINE_CHECKPOINT}"
echo "candidate: ${candidate_checkpoint}"
echo "output: ${COMPARISON_OUTPUT_DIR}"

uv run python eval_scripts/compare_trash_bin_action_smoothness.py \
  --baseline-checkpoint "${BASELINE_CHECKPOINT}" \
  --candidate-checkpoint "${candidate_checkpoint}" \
  --output-dir "${COMPARISON_OUTPUT_DIR}" \
  --num-envs "${SMOOTHNESS_NUM_ENVS}" \
  --seed "${SMOOTHNESS_SEED}" \
  --policy-noise-seed "${SMOOTHNESS_POLICY_NOISE_SEED}" \
  --sim-backend "${SMOOTHNESS_SIM_BACKEND}"

echo
echo "### training and smoothness comparison finished"
echo "stage4 best checkpoint: ${candidate_checkpoint}"
echo "comparison report: ${COMPARISON_OUTPUT_DIR}/index.html"
