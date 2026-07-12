#!/usr/bin/env bash
set -euo pipefail

# Train trash-bin rolling stage1 with fixed XY bin placement and randomized bin
# rotation, then initialize stage2 with the normal randomized bin placement.

if [[ -f ./server_env.sh ]]; then
  # shellcheck disable=SC1091
  source ./server_env.sh
fi

SEED="${SEED:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-trash_bin_stage1_rotation_only_40m_seed${SEED}_${RUN_TAG}}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-trash_bin_stage2_from_stage1_75m_seed${SEED}_${RUN_TAG}}"

STAGE1_TOTAL_TIMESTEPS="${STAGE1_TOTAL_TIMESTEPS:-40_000_000}"
STAGE2_TOTAL_TIMESTEPS="${STAGE2_TOTAL_TIMESTEPS:-75_000_000}"
NUM_ENVS="${NUM_ENVS:-1024}"
NUM_STEPS="${NUM_STEPS:-120}"
NUM_EVAL_STEPS="${NUM_EVAL_STEPS:-120}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-8}"
NUM_EVAL_VIDEO_ENVS="${NUM_EVAL_VIDEO_ENVS:-8}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-32}"
EVAL_FREQ="${EVAL_FREQ:-5}"
GAMMA="${GAMMA:-0.99}"
ROBOT_INIT_NOISE_SCALE="${ROBOT_INIT_NOISE_SCALE:-1.0}"

common_args=(
  --control-mode pd_joint_delta_pos
  --robot-init-noise-scale "${ROBOT_INIT_NOISE_SCALE}"
  --num-envs "${NUM_ENVS}"
  --num-steps "${NUM_STEPS}"
  --num-eval-steps "${NUM_EVAL_STEPS}"
  --num-eval-envs "${NUM_EVAL_ENVS}"
  --num-eval-video-envs "${NUM_EVAL_VIDEO_ENVS}"
  --update-epochs "${UPDATE_EPOCHS}"
  --num-minibatches "${NUM_MINIBATCHES}"
  --eval-freq "${EVAL_FREQ}"
  --gamma "${GAMMA}"
)

echo
echo "### stage1: MyDualTrashBinRollingRotationOnly-v0"
echo "run: runs/${STAGE1_RUN_NAME}"
python ppo_dual_xarm7.py \
  --exp-name "${STAGE1_RUN_NAME}" \
  --env-id MyDualTrashBinRollingRotationOnly-v0 \
  --total-timesteps "${STAGE1_TOTAL_TIMESTEPS}" \
  "${common_args[@]}"

stage1_ckpt="runs/${STAGE1_RUN_NAME}/final_ckpt.pt"
if [[ ! -f "${stage1_ckpt}" ]]; then
  echo "stage1 final checkpoint does not exist: ${stage1_ckpt}" >&2
  exit 1
fi

echo
echo "### stage2: MyDualTrashBinRollingStage2-v0"
echo "checkpoint: ${stage1_ckpt}"
echo "run: runs/${STAGE2_RUN_NAME}"
python ppo_dual_xarm7.py \
  --exp-name "${STAGE2_RUN_NAME}" \
  --checkpoint "${stage1_ckpt}" \
  --env-id MyDualTrashBinRollingStage2-v0 \
  --total-timesteps "${STAGE2_TOTAL_TIMESTEPS}" \
  "${common_args[@]}"

echo
echo "### trash-bin rolling curriculum finished"
echo "stage1 checkpoint: ${stage1_ckpt}"
echo "stage2 run: runs/${STAGE2_RUN_NAME}"
