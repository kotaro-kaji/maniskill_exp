#!/usr/bin/env bash
set -euo pipefail

# Train only MyDualTrashBinRollingStage3-v0 from a user-specified checkpoint.

if [[ -f ./server_env.sh ]]; then
  # shellcheck disable=SC1091
  source ./server_env.sh
fi

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <stage2_checkpoint_path>" >&2
  exit 1
fi

INPUT_CHECKPOINT="$1"
if [[ ! -f "${INPUT_CHECKPOINT}" ]]; then
  echo "input checkpoint does not exist: ${INPUT_CHECKPOINT}" >&2
  exit 1
fi

SEED="${SEED:-1}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-trash_bin_stage3_contact_penalty_seed${SEED}_${RUN_TAG}}"

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-75_000_000}"
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
echo "### stage3: MyDualTrashBinRollingStage3-v0"
echo "checkpoint: ${INPUT_CHECKPOINT}"
echo "run: runs/${RUN_NAME}"

python ppo_dual_xarm7.py \
  --exp-name "${RUN_NAME}" \
  --checkpoint "${INPUT_CHECKPOINT}" \
  --env-id MyDualTrashBinRollingStage3-v0 \
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
