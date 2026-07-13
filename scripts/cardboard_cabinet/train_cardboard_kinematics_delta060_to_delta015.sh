#!/usr/bin/env bash
set -euo pipefail

# Reproducible kinematics-URDF small-delta fine-tuning chain:
#   input checkpoint: randomized-v2 delta=0.06 best checkpoint
#   stage 1: continue with my_xarm7_kinematics_delta03
#   stage 2: continue from the stage-1 best checkpoint with my_xarm7_kinematics_delta02
#   stage 3: continue from the stage-2 best checkpoint with my_xarm7_kinematics_delta015

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <randomized_v2_delta060_best_ckpt>" >&2
  exit 2
fi

INPUT_CKPT="$1"
SEED="${SEED:-1}"
RUN_PREFIX="${RUN_PREFIX:-cardboard_kinematics_delta060_to_delta015}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENV_ID="${ENV_ID:-MySingleCardboardCabinetRandomized-v2}"
ROBOT_INIT_NOISE_SCALE="${ROBOT_INIT_NOISE_SCALE:-0.05}"
LEARNING_RATE_DELTA03="${LEARNING_RATE_DELTA03:-1e-4}"
LEARNING_RATE_DELTA02="${LEARNING_RATE_DELTA02:-1e-4}"
LEARNING_RATE_DELTA015="${LEARNING_RATE_DELTA015:-3e-5}"
ANCHOR_COEF_DELTA015="${ANCHOR_COEF_DELTA015:-0.05}"
CLIP_COEF_DELTA015="${CLIP_COEF_DELTA015:-0.05}"
TARGET_KL_DELTA015="${TARGET_KL_DELTA015:-0.03}"
TOTAL_TIMESTEPS_DELTA03="${TOTAL_TIMESTEPS_DELTA03:-25000000}"
TOTAL_TIMESTEPS_DELTA02="${TOTAL_TIMESTEPS_DELTA02:-25000000}"
TOTAL_TIMESTEPS_DELTA015="${TOTAL_TIMESTEPS_DELTA015:-25000000}"
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

event_file = max(event_files, key=lambda path: (path.stat().st_mtime, path.name))
ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags().get("scalars", [])
preferred = [
    "eval/drawer_open_success",
    "eval/success_at_end",
    "eval/success",
    "eval/return",
    "eval/reward",
    "eval/r",
]
tag = next((t for t in preferred if t in tags), None)
if tag is None:
    raise SystemExit(f"no usable eval scalar found in {event_file}; tags={tags}")

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
  local learning_rate="$4"
  local total_timesteps="$5"
  local num_envs="$6"
  local num_minibatches="$7"
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
    --learning-rate "${learning_rate}"
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

run_delta015_stage() {
  local checkpoint="$1"
  local exp_name="${RUN_PREFIX}_seed${SEED}_delta015_lowlr_anchor005"

  local cmd=(
    ${PYTHON_BIN}
    ppo_dual_xarm7.py
    --exp-name "${exp_name}"
    --env-id "${ENV_ID}"
    --robot-uid my_xarm7_kinematics_delta015
    --control-mode pd_joint_delta_pos
    --robot-init-noise-scale "${ROBOT_INIT_NOISE_SCALE}"
    --checkpoint "${checkpoint}"
    --anchor-checkpoint "${checkpoint}"
    --anchor-coef "${ANCHOR_COEF_DELTA015}"
    --learning-rate "${LEARNING_RATE_DELTA015}"
    --clip-coef "${CLIP_COEF_DELTA015}"
    --target-kl "${TARGET_KL_DELTA015}"
    --num-envs 512
    --num-steps 100
    --num-eval-steps 100
    --num-eval-envs "${NUM_EVAL_ENVS}"
    --num-eval-video-envs "${NUM_EVAL_VIDEO_ENVS}"
    --update-epochs 8
    --num-minibatches 16
    --total-timesteps "${TOTAL_TIMESTEPS_DELTA015}"
    --eval-freq 5
    --gamma 0.99
    --seed "${SEED}"
  )

  echo
  echo "### delta015_lowlr_anchor005: my_xarm7_kinematics_delta015"
  printf '%q ' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

run_delta_stage "delta03_lr1e4_only" \
  my_xarm7_kinematics_delta03 "${INPUT_CKPT}" "${LEARNING_RATE_DELTA03}" \
  "${TOTAL_TIMESTEPS_DELTA03}" 1024 32
delta03_ckpt="$(best_ckpt "runs/${RUN_PREFIX}_seed${SEED}_delta03_lr1e4_only" 1024)"

run_delta_stage "delta02_lr1e4_only" \
  my_xarm7_kinematics_delta02 "${delta03_ckpt}" "${LEARNING_RATE_DELTA02}" \
  "${TOTAL_TIMESTEPS_DELTA02}" 512 16
delta02_ckpt="$(best_ckpt "runs/${RUN_PREFIX}_seed${SEED}_delta02_lr1e4_only" 512)"

run_delta015_stage "${delta02_ckpt}"
delta015_ckpt="$(best_ckpt "runs/${RUN_PREFIX}_seed${SEED}_delta015_lowlr_anchor005" 512)"

echo
echo "Delta 0.03 best checkpoint: ${delta03_ckpt}"
echo "Delta 0.02 best checkpoint: ${delta02_ckpt}"
echo "Delta 0.015 best checkpoint: ${delta015_ckpt}"
