#!/usr/bin/env bash
set -euo pipefail

# Reproducible three-stage kinematics-URDF delta=0.06 curriculum:
# Stage 1: train MyDualCardboardCabinet-v1 from scratch.
# Stage 2: continue conservatively on randomized-v1.
# Stage 3: continue conservatively on randomized-v2.

SEED="${SEED:-1}"
RUN_PREFIX="${RUN_PREFIX:-cardboard_kinematics_3stage}"
NUM_ENVS="${NUM_ENVS:-1024}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-64}"
NUM_EVAL_VIDEO_ENVS="${NUM_EVAL_VIDEO_ENVS:-4}"
STAGE1_TIMESTEPS="${STAGE1_TIMESTEPS:-25000000}"
STAGE2_TIMESTEPS="${STAGE2_TIMESTEPS:-15000000}"
STAGE3_TIMESTEPS="${STAGE3_TIMESTEPS:-15000000}"
CHECKPOINT_STAGE1="${CHECKPOINT_STAGE1:-}"
CHECKPOINT_STAGE2="${CHECKPOINT_STAGE2:-}"
PYTHON_BIN="${PYTHON_BIN:-uv run python}"
FINETUNE_LR="${FINETUNE_LR:-1e-4}"
FINETUNE_ANCHOR_COEF="${FINETUNE_ANCHOR_COEF:-0.1}"
FINETUNE_CLIP_COEF="${FINETUNE_CLIP_COEF:-0.1}"
FINETUNE_TARGET_KL="${FINETUNE_TARGET_KL:-0.05}"

best_ckpt() {
  local run_dir="$1"
  local batch_size=$((NUM_ENVS * 100))
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
    "eval/drawer_open_success",
    "eval/success_at_end",
    "eval/success",
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

run_stage() {
  local stage_name="$1"
  local env_id="$2"
  local robot_noise="$3"
  local total_timesteps="$4"
  local checkpoint="$5"
  local exp_name="${RUN_PREFIX}_seed${SEED}_${stage_name}"

  local cmd=(
    ${PYTHON_BIN}
    ppo_dual_xarm7.py
    --exp-name "${exp_name}"
    --env-id "${env_id}"
    --control-mode pd_joint_delta_pos
    --robot-uid my_xarm7_kinematics
    --robot-init-noise-scale "${robot_noise}"
    --num-envs "${NUM_ENVS}"
    --num-steps 100
    --num-eval-steps 100
    --num-eval-envs "${NUM_EVAL_ENVS}"
    --num-eval-video-envs "${NUM_EVAL_VIDEO_ENVS}"
    --update-epochs 8
    --num-minibatches 32
    --total-timesteps "${total_timesteps}"
    --eval-freq 5
    --gamma 0.99
    --seed "${SEED}"
  )
  if [[ -n "${checkpoint}" ]]; then
    cmd+=(--checkpoint "${checkpoint}")
  fi

  echo
  echo "### ${stage_name}"
  printf '%q ' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

run_finetune_stage() {
  local stage_name="$1"
  local env_id="$2"
  local robot_noise="$3"
  local total_timesteps="$4"
  local checkpoint="$5"
  local exp_name="${RUN_PREFIX}_seed${SEED}_${stage_name}"

  local cmd=(
    ${PYTHON_BIN}
    ppo_dual_xarm7.py
    --exp-name "${exp_name}"
    --env-id "${env_id}"
    --control-mode pd_joint_delta_pos
    --robot-uid my_xarm7_kinematics
    --robot-init-noise-scale "${robot_noise}"
    --checkpoint "${checkpoint}"
    --anchor-checkpoint "${checkpoint}"
    --anchor-coef "${FINETUNE_ANCHOR_COEF}"
    --learning-rate "${FINETUNE_LR}"
    --clip-coef "${FINETUNE_CLIP_COEF}"
    --target-kl "${FINETUNE_TARGET_KL}"
    --num-envs "${NUM_ENVS}"
    --num-steps 100
    --num-eval-steps 100
    --num-eval-envs "${NUM_EVAL_ENVS}"
    --num-eval-video-envs "${NUM_EVAL_VIDEO_ENVS}"
    --update-epochs 8
    --num-minibatches 32
    --total-timesteps "${total_timesteps}"
    --eval-freq 5
    --gamma 0.99
    --seed "${SEED}"
  )

  echo
  echo "### ${stage_name}"
  printf '%q ' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

stage1_ckpt="${CHECKPOINT_STAGE1}"
if [[ -z "${stage1_ckpt}" ]]; then
  run_stage "stage1_fixed_delta060" \
    MyDualCardboardCabinet-v1 0.0 "${STAGE1_TIMESTEPS}" ""
  stage1_ckpt="$(best_ckpt "runs/${RUN_PREFIX}_seed${SEED}_stage1_fixed_delta060")"
fi

stage2_ckpt="${CHECKPOINT_STAGE2}"
if [[ -z "${stage2_ckpt}" ]]; then
  run_finetune_stage "stage2_randomized_v1_delta060_lr1e4_anchor01" \
    MySingleCardboardCabinetRandomized-v1 0.05 "${STAGE2_TIMESTEPS}" "${stage1_ckpt}"
  stage2_ckpt="$(best_ckpt "runs/${RUN_PREFIX}_seed${SEED}_stage2_randomized_v1_delta060_lr1e4_anchor01")"
fi

run_finetune_stage "stage3_randomized_v2_delta060_lr1e4_anchor01" \
  MySingleCardboardCabinetRandomized-v2 0.05 "${STAGE3_TIMESTEPS}" "${stage2_ckpt}"
stage3_ckpt="$(best_ckpt "runs/${RUN_PREFIX}_seed${SEED}_stage3_randomized_v2_delta060_lr1e4_anchor01")"

echo
echo "Stage 1 best checkpoint: ${stage1_ckpt}"
echo "Stage 2 best checkpoint: ${stage2_ckpt}"
echo "Stage 3 best checkpoint: ${stage3_ckpt}"
