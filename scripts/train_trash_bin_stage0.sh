#!/usr/bin/env bash
set -euo pipefail

# Train MyDualTrashBinRollingStage0-v0 from scratch.

if [[ -f ./server_env.sh ]]; then
  # shellcheck disable=SC1091
  source ./server_env.sh
fi

SEED="${SEED:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-trash_bin_stage0_tcp_rotation_35deg_seed${SEED}_${RUN_TAG}}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20_000_000}"
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

echo
echo "### stage0: MyDualTrashBinRollingStage0-v0"
echo "run: runs/${RUN_NAME}"

python ppo_dual_xarm7.py \
  --exp-name "${RUN_NAME}" \
  --env-id MyDualTrashBinRollingStage0-v0 \
  --control-mode pd_joint_delta_pos \
  --robot-init-noise-scale "${ROBOT_INIT_NOISE_SCALE}" \
  --num-envs "${NUM_ENVS}" \
  --num-steps "${NUM_STEPS}" \
  --num-eval-steps "${NUM_EVAL_STEPS}" \
  --num-eval-envs "${NUM_EVAL_ENVS}" \
  --num-eval-video-envs "${NUM_EVAL_VIDEO_ENVS}" \
  --update-epochs "${UPDATE_EPOCHS}" \
  --num-minibatches "${NUM_MINIBATCHES}" \
  --total-timesteps "${TOTAL_TIMESTEPS}" \
  --eval-freq "${EVAL_FREQ}" \
  --gamma "${GAMMA}"
