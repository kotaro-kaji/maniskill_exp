#!/usr/bin/env bash
set -euo pipefail

# Post-curriculum small-delta fine-tuning:
#   input checkpoint: randomized-v2 delta=0.06 best checkpoint
#   stage 1: continue with my_xarm7_delta03
#   stage 2: continue from the stage-1 best checkpoint with my_xarm7_delta02
#
# Usage:
#   bash scripts/train_cardboard_delta060_to_delta020.sh <randomized_v2_delta060_best_ckpt>
#   PYTHON_BIN=.venv/bin/python bash scripts/train_cardboard_delta060_to_delta020.sh <ckpt>

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <randomized_v2_delta060_best_ckpt>" >&2
  exit 2
fi

INPUT_CKPT="$1"
SEED="${SEED:-1}"
RUN_PREFIX="${RUN_PREFIX:-cardboard_delta060_to_delta020}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_ID="${ENV_ID:-MySingleCardboardCabinetRandomized-v2}"
ROBOT_INIT_NOISE_SCALE="${ROBOT_INIT_NOISE_SCALE:-0.05}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
TOTAL_TIMESTEPS_DELTA03="${TOTAL_TIMESTEPS_DELTA03:-25000000}"
TOTAL_TIMESTEPS_DELTA02="${TOTAL_TIMESTEPS_DELTA02:-25000000}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-64}"
NUM_EVAL_VIDEO_ENVS="${NUM_EVAL_VIDEO_ENVS:-4}"

best_ckpt() {
  local run_dir="$1"
  local num_envs="$2"
  local batch_size=$((num_envs * 100))
  ${PYTHON_BIN} - "${run_dir}" "${batch_size}" <<'PY'
from pathlib import Path
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_dir = Path(sys.argv[1])
batch_size = int(sys.argv[2])
event_files = list(run_dir.glob("events.out.tfevents*"))
if not event_files:
    raise SystemExit(f"no tensorboard event file found in {run_dir}")

ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags().get("scalars", [])
preferred = [
    "eval/return",
    "eval/reward",
    "eval/r",
]
tag = next((t for t in preferred if t in tags), None)
if tag is None:
    raise SystemExit(f"no usable eval scalar found in {run_dir}; tags={tags}")

best = max(ea.Scalars(tag), key=lambda item: item.value)
iteration = int(best.step // batch_size) + 1
ckpt = run_dir / f"ckpt_{iteration}.pt"
if not ckpt.exists():
    ckpt = run_dir / "final_ckpt.pt"
if not ckpt.exists():
    raise SystemExit(f"best checkpoint does not exist for {run_dir}")
print(ckpt)
PY
}

run_delta_stage() {
  local stage_name="$1"
  local robot_uid="$2"
  local checkpoint="$3"
  local total_timesteps="$4"
  local num_envs="$5"
  local num_minibatches="$6"
  local exp_name="${RUN_PREFIX}_seed${SEED}_${stage_name}"

  local cmd=(
    ${PYTHON_BIN}
    ppo_dual_xarm7.py
    --exp-name "${exp_name}"
    --env-id "${ENV_ID}"
    --robot-uid "${robot_uid}"
    --control-mode pd_joint_delta_pos
    --robot-init-noise-scale "${ROBOT_INIT_NOISE_SCALE}"
    --checkpoint "${checkpoint}"
    --learning-rate "${LEARNING_RATE}"
    --num-envs "${num_envs}"
    --num-steps 100
    --num-eval-steps 100
    --num-eval-envs "${NUM_EVAL_ENVS}"
    --num-eval-video-envs "${NUM_EVAL_VIDEO_ENVS}"
    --update-epochs 8
    --num-minibatches "${num_minibatches}"
    --total-timesteps "${total_timesteps}"
    --eval-freq 5
    --gamma 0.99
    --seed "${SEED}"
  )

  echo
  echo "### ${stage_name}: ${robot_uid}"
  printf '%q ' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

run_delta_stage "delta03_lr1e4_only" \
  my_xarm7_delta03 "${INPUT_CKPT}" "${TOTAL_TIMESTEPS_DELTA03}" 1024 32
delta03_ckpt="$(best_ckpt "runs/${RUN_PREFIX}_seed${SEED}_delta03_lr1e4_only" 1024)"

run_delta_stage "delta02_lr1e4_only" \
  my_xarm7_delta02 "${delta03_ckpt}" "${TOTAL_TIMESTEPS_DELTA02}" 512 16
delta02_ckpt="$(best_ckpt "runs/${RUN_PREFIX}_seed${SEED}_delta02_lr1e4_only" 512)"

echo
echo "Delta 0.03 best checkpoint: ${delta03_ckpt}"
echo "Delta 0.02 best checkpoint: ${delta02_ckpt}"
