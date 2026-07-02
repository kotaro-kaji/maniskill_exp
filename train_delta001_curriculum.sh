#!/usr/bin/env bash
set -euo pipefail

# Reproducible curriculum for learning a cardboard-opening policy that still
# works with arm_delta_pos_limit=0.001.
#
# Usage examples:
#   bash train_delta001_curriculum.sh
#   ROBOT_FAMILY=kinematics SEED=2 bash train_delta001_curriculum.sh
#   PYTHON_BIN=.venv/bin/python bash train_delta001_curriculum.sh

ROBOT_FAMILY="${ROBOT_FAMILY:-left}"
SEED="${SEED:-1}"
NUM_ENVS="${NUM_ENVS:-1024}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-160}"
NUM_EVAL_VIDEO_ENVS="${NUM_EVAL_VIDEO_ENVS:-4}"
TOTAL_TIMESTEPS_STAGE1="${TOTAL_TIMESTEPS_STAGE1:-25000000}"
TOTAL_TIMESTEPS_STAGE2="${TOTAL_TIMESTEPS_STAGE2:-25000000}"
TOTAL_TIMESTEPS_STAGE3="${TOTAL_TIMESTEPS_STAGE3:-25000000}"
CHECKPOINT_STAGE0="${CHECKPOINT_STAGE0:-}"
PYTHON_BIN="${PYTHON_BIN:-uv run python}"

case "${ROBOT_FAMILY}" in
  left)
    ROBOT_DELTA03="my_xarm7_delta03"
    ROBOT_DELTA003="my_xarm7_delta003"
    ROBOT_DELTA001="my_xarm7_delta001"
    ;;
  kinematics)
    ROBOT_DELTA03="my_xarm7_kinematics_delta03"
    ROBOT_DELTA003="my_xarm7_kinematics_delta003"
    ROBOT_DELTA001="my_xarm7_kinematics_delta001"
    ;;
  *)
    echo "ROBOT_FAMILY must be left or kinematics, got: ${ROBOT_FAMILY}" >&2
    exit 2
    ;;
esac

COMMON_ARGS=(
  --env-id MyDualCardboardCabinetSmallDelta-v1
  --control-mode pd_joint_delta_pos
  --robot-init-noise-scale 0.0
  --num-envs "${NUM_ENVS}"
  --num-steps 200
  --num-eval-steps 200
  --num-eval-envs "${NUM_EVAL_ENVS}"
  --num-eval-video-envs "${NUM_EVAL_VIDEO_ENVS}"
  --update-epochs 8
  --num-minibatches 32
  --eval-freq 5
  --gamma 0.99
  --seed "${SEED}"
)

best_ckpt() {
  local run_dir="$1"
  local batch_size=$((NUM_ENVS * 200))
  ${PYTHON_BIN} - "${run_dir}" "${batch_size}" <<'PY'
from pathlib import Path
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run_dir = Path(sys.argv[1])
batch_size = int(sys.argv[2])
event_files = list(run_dir.glob("events.out.tfevents*"))
if not event_files:
    final_ckpt = run_dir / "final_ckpt.pt"
    print(final_ckpt)
    raise SystemExit

event_file = max(event_files, key=lambda path: (path.stat().st_mtime, path.name))
ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags().get("scalars", [])
preferred = [
    "eval/success",
    "eval/drawer_open_success_reached",
    "eval/drawer_open_success",
    "eval/return",
    "eval/reward",
    "eval/r",
]
tag = next((t for t in preferred if t in tags), None)
if tag is None:
    print(run_dir / "final_ckpt.pt")
    raise SystemExit

best = max(ea.Scalars(tag), key=lambda item: item.value)
iteration = int(best.step // batch_size) + 1
ckpt = run_dir / f"ckpt_{iteration}.pt"
print(ckpt if ckpt.exists() else run_dir / "final_ckpt.pt")
PY
}

run_stage() {
  local stage_name="$1"
  local robot_uid="$2"
  local timesteps="$3"
  local checkpoint="$4"
  local exp_name="delta001_curriculum_${ROBOT_FAMILY}_seed${SEED}_${stage_name}"

  local cmd=(
    ${PYTHON_BIN}
    ppo_dual_xarm7_low_freq.py
    "${COMMON_ARGS[@]}"
    --robot-uid "${robot_uid}"
    --total-timesteps "${timesteps}"
    --exp-name "${exp_name}"
  )
  if [[ -n "${checkpoint}" ]]; then
    cmd+=(--checkpoint "${checkpoint}")
  fi

  echo
  echo "### ${stage_name}: ${robot_uid}"
  printf '%q ' "${cmd[@]}"
  echo
  "${cmd[@]}"
}

stage1_checkpoint="${CHECKPOINT_STAGE0}"
run_stage "stage1_delta03" "${ROBOT_DELTA03}" "${TOTAL_TIMESTEPS_STAGE1}" "${stage1_checkpoint}"
stage1_best="$(best_ckpt "runs/delta001_curriculum_${ROBOT_FAMILY}_seed${SEED}_stage1_delta03")"

run_stage "stage2_delta003" "${ROBOT_DELTA003}" "${TOTAL_TIMESTEPS_STAGE2}" "${stage1_best}"
stage2_best="$(best_ckpt "runs/delta001_curriculum_${ROBOT_FAMILY}_seed${SEED}_stage2_delta003")"

run_stage "stage3_delta001" "${ROBOT_DELTA001}" "${TOTAL_TIMESTEPS_STAGE3}" "${stage2_best}"
stage3_best="$(best_ckpt "runs/delta001_curriculum_${ROBOT_FAMILY}_seed${SEED}_stage3_delta001")"

echo
echo "Best final-stage checkpoint: ${stage3_best}"
