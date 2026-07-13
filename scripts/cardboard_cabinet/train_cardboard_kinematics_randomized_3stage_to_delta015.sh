#!/usr/bin/env bash
set -euo pipefail

# Run the kinematics-URDF cardboard curriculum in one command:
#   1. delta=0.06 three-stage training: fixed -> randomized-v1 -> randomized-v2
#   2. continue from the randomized-v2 best checkpoint:
#      delta=0.03 -> delta=0.02 -> delta=0.015

SEED="${SEED:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
THREE_STAGE_RUN_PREFIX="${THREE_STAGE_RUN_PREFIX:-cardboard_kinematics_3stage}"
DELTA015_RUN_PREFIX="${DELTA015_RUN_PREFIX:-cardboard_kinematics_delta060_to_delta015}"
THREE_STAGE_NUM_ENVS="${THREE_STAGE_NUM_ENVS:-${NUM_ENVS:-1024}}"

best_ckpt() {
  local run_dir="$1"
  local batch_size="$2"
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

echo
echo "### kinematics delta=0.06 randomized three-stage curriculum"
RUN_PREFIX="${THREE_STAGE_RUN_PREFIX}" \
PYTHON_BIN="${PYTHON_BIN}" \
NUM_ENVS="${THREE_STAGE_NUM_ENVS}" \
SEED="${SEED}" \
bash scripts/train_cardboard_kinematics_randomized_3stage.sh

stage3_run_dir="runs/${THREE_STAGE_RUN_PREFIX}_seed${SEED}_stage3_randomized_v2_delta060_lr1e4_anchor01"
stage3_batch_size=$((THREE_STAGE_NUM_ENVS * 100))
stage3_ckpt="$(best_ckpt "${stage3_run_dir}" "${stage3_batch_size}")"

echo
echo "### kinematics delta=0.06 randomized-v2 best checkpoint"
echo "${stage3_ckpt}"

echo
echo "### continue kinematics delta=0.06 -> delta=0.03 -> delta=0.02 -> delta=0.015"
RUN_PREFIX="${DELTA015_RUN_PREFIX}" \
PYTHON_BIN="${PYTHON_BIN}" \
SEED="${SEED}" \
bash scripts/train_cardboard_kinematics_delta060_to_delta015.sh "${stage3_ckpt}"

echo
echo "### kinematics full delta=0.015 curriculum input checkpoint summary"
echo "Delta 0.06 randomized-v2 best checkpoint: ${stage3_ckpt}"
