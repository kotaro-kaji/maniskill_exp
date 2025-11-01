from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
from typing import Iterable, List, Sequence

import torch

from ppo_xarm7 import Agent


# Absolute joint limits for the 7 arm joints plus the gripper drive joint.
# Arm limits mirror the URDF bounds ManiSkill uses, and the drive joint range
# follows the URDF gripper specification (0.05–0.84 rad). Mimic joints inherit
# the drive joint's limit through the controller so the single drive DOF is
# sufficient here.
JOINT_POSITION_LOW = torch.tensor(
    [
        -6.283185307179586,  # joint1
        -2.059,  # joint2
        -6.283185307179586,  # joint3
        -0.19198,  # joint4
        -6.283185307179586,  # joint5
        -1.69297,  # joint6
        -6.283185307179586,  # joint7
        0.05,  # drive_joint (gripper)
    ],
    dtype=torch.float32,
)

JOINT_POSITION_HIGH = torch.tensor(
    [
        6.283185307179586,  # joint1
        2.0944,  # joint2
        6.283185307179586,  # joint3
        3.927,  # joint4
        6.283185307179586,  # joint5
        3.141592653589793,  # joint6
        6.283185307179586,  # joint7
        0.84,  # drive_joint (gripper)
    ],
    dtype=torch.float32,
)


class _DummySpace:
    def __init__(self, shape: Sequence[int]):
        self.shape = tuple(shape)


class _DummyEnv:
    def __init__(self, obs_shape: Sequence[int], action_shape: Sequence[int]):
        self.single_observation_space = _DummySpace(obs_shape)
        self.single_action_space = _DummySpace(action_shape)


def gripper_q_maniskill_to_robomanip(q_maniskill: torch.Tensor) -> torch.Tensor:
    return q_maniskill * (-1000.0) + 840.0


def gripper_q_robomanip_to_maniskill(q_robomanip: torch.Tensor) -> torch.Tensor:
    return (q_robomanip - 840.0) / (-1000.0)


# ---------------------------------------------------------------------------
# Configuration (edit as needed)
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "runs/MyJointHold-v0__ppo_xarm7__1__1758957634/ckpt_76.pt"
CSV_PATH = "rollout_log_for_input.csv"
USE_CUDA = True
DETERMINISTIC_POLICY = True
PRINT_JSON = False
CONVERT_GRIPPER = True
MAX_ROWS = None  # set to an int to process fewer rows

# Physical delta bounds (pd_joint_delta_pos controller for my_xarm7).
# Order: joint1, joint2, joint3, joint4, joint5, joint6, joint7, drive_joint
_DELTA_PHYSICAL_LOW = torch.tensor(
    [
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
        -0.1,
    ],
    dtype=torch.float32,
)
_DELTA_PHYSICAL_HIGH = torch.tensor(
    [
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ],
    dtype=torch.float32,
)


def _print_duration_stats(label: str, durations: Iterable[float]) -> None:
    durations = list(durations)
    if not durations:
        return

    mean_duration = statistics.fmean(durations)
    if len(durations) >= 2:
        variance = statistics.pvariance(durations)
        quartiles = statistics.quantiles(durations, n=4, method="inclusive")
    else:
        variance = 0.0
        quartiles = [durations[0]] * 3

    min_duration = min(durations)
    max_duration = max(durations)
    median_duration = statistics.median(durations)
    print(
        f"{label} "
        f"mean={mean_duration:.6f} "
        f"variance={variance:.6f} "
        f"min={min_duration:.6f} "
        f"max={max_duration:.6f} "
        f"median={median_duration:.6f} "
        f"q1={quartiles[0]:.6f} "
        f"q2={quartiles[1]:.6f} "
        f"q3={quartiles[2]:.6f}"
    )


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    # Read CSV rows and parse observations
    observations: List[List[float]] = []

    with open(os.path.abspath(CSV_PATH), "r") as f:
        reader = csv.DictReader(f)
        if "observation" not in reader.fieldnames:
            raise KeyError("CSV must contain an 'observation' column")
        for idx, row in enumerate(reader):
            obs = json.loads(row["observation"])
            if not isinstance(obs, list):
                raise ValueError("observation column must be JSON array")
            observations.append(obs)
            if MAX_ROWS is not None and len(observations) >= MAX_ROWS:
                break

    if not observations:
        raise ValueError("No observations parsed from CSV")

    obs_dim = len(observations[0])

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else CHECKPOINT_PATH
    # Load checkpoint to extract action dimension before constructing agent
    state_dict = torch.load(checkpoint_path, map_location=device)
    if "actor_logstd" not in state_dict:
        raise KeyError("Checkpoint missing 'actor_logstd' parameter")
    action_dim = int(state_dict["actor_logstd"].shape[-1])

    dummy_env = _DummyEnv((obs_dim,), (action_dim,))
    agent = Agent(dummy_env).to(device)
    agent.load_state_dict(state_dict)
    agent.eval()

    normalized_low = torch.full((action_dim,), -1.0, device=device)
    normalized_high = torch.full((action_dim,), 1.0, device=device)

    if action_dim != len(_DELTA_PHYSICAL_LOW):
        raise ValueError(
            f"Checkpoint action dim {action_dim} does not match hard-coded bounds {len(_DELTA_PHYSICAL_LOW)}"
        )
    physical_low = _DELTA_PHYSICAL_LOW.to(device)
    physical_high = _DELTA_PHYSICAL_HIGH.to(device)
    joint_position_low = JOINT_POSITION_LOW.to(device=device)
    joint_position_high = JOINT_POSITION_HIGH.to(device=device)

    inference_durations: List[float] = []
    pipeline_durations: List[float] = []
    for step_idx, obs_values in enumerate(observations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pipeline_start = time.perf_counter()

        obs_tensor = torch.tensor(obs_values, dtype=torch.float32, device=device).unsqueeze(0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        with torch.no_grad():
            action = agent.get_action(obs_tensor, deterministic=DETERMINISTIC_POLICY)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_durations.append(time.perf_counter() - start_time)
        clipped_action = torch.clamp(action, normalized_low, normalized_high)

        # Map normalized delta action back to physical delta range and recover absolute joint targets.
        scale = (clipped_action - normalized_low) / (normalized_high - normalized_low)
        denormalized_delta = physical_low + scale * (physical_high - physical_low)
        current_joint_pos = obs_tensor[..., : action_dim]
        direct_joint_command = current_joint_pos + denormalized_delta

        direct_joint_command = torch.max(
            torch.min(direct_joint_command, joint_position_high), joint_position_low
        )

        if CONVERT_GRIPPER and direct_joint_command.shape[-1] > 0:
            gripper_q = direct_joint_command[..., -1]
            direct_joint_command = direct_joint_command.clone()
            direct_joint_command[..., -1] = gripper_q_maniskill_to_robomanip(gripper_q)

        direct_joint_np = direct_joint_command.squeeze(0).cpu().numpy().tolist()

        if PRINT_JSON:
            payload = {
                "step": step_idx,
                "direct_joint_command": direct_joint_np,
            }
            print(json.dumps(payload))
        else:
            print(f"step={step_idx} direct_joint_command={direct_joint_np}")

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pipeline_durations.append(time.perf_counter() - pipeline_start)

    _print_duration_stats("inference_timing_seconds", inference_durations)
    _print_duration_stats("pipeline_timing_seconds", pipeline_durations)


if __name__ == "__main__":
    main()
