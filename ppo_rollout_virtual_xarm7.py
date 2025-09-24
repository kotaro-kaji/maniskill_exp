from __future__ import annotations

import csv
import json
import os
from typing import List, Sequence

import torch

from ppo_xarm7 import Agent


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
CHECKPOINT_PATH = "runs/MyJointHold-v0__ppo_xarm7__1__1758689027/ckpt_76.pt"
CSV_PATH = "rollout_log_for_input.csv"
USE_CUDA = True
DETERMINISTIC_POLICY = True
PRINT_JSON = False
CONVERT_GRIPPER = True
MAX_ROWS = None  # set to an int to process fewer rows

# Action-space bounds (pd_joint_pos controller for my_xarm7).
# Order: joint1, joint2, joint3, joint4, joint5, joint6, joint7, drive_joint
_ACTION_LOW = torch.tensor(
    [
        -6.2831853,
        -2.059,
        -6.2831853,
        -0.19198,
        -6.2831853,
        -1.69297,
        -6.2831853,
        0.05,
    ],
    dtype=torch.float32,
)
_ACTION_HIGH = torch.tensor(
    [
        6.2831853,
        2.0944,
        6.2831853,
        3.927,
        6.2831853,
        3.1415927,
        6.2831853,
        0.84,
    ],
    dtype=torch.float32,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    normalized_low = _ACTION_LOW.to(device)
    normalized_high = _ACTION_HIGH.to(device)
    physical_low = _ACTION_LOW.to(device)
    physical_high = _ACTION_HIGH.to(device)

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

    # Load checkpoint to extract action dimension before constructing agent
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    if "actor_logstd" not in state_dict:
        raise KeyError("Checkpoint missing 'actor_logstd' parameter")
    action_dim = int(state_dict["actor_logstd"].shape[-1])

    dummy_env = _DummyEnv((obs_dim,), (action_dim,))
    agent = Agent(dummy_env).to(device)
    agent.load_state_dict(state_dict)
    agent.eval()

    if action_dim != len(_ACTION_LOW):
        raise ValueError(
            f"Checkpoint action dim {action_dim} does not match hard-coded bounds {len(_ACTION_LOW)}"
        )

    for step_idx, obs_values in enumerate(observations):
        obs_tensor = torch.tensor(obs_values, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = agent.get_action(obs_tensor, deterministic=DETERMINISTIC_POLICY)
        clipped_action = torch.clamp(action, normalized_low, normalized_high)

        center = 0.5 * (physical_high + physical_low)
        half_range = 0.5 * (physical_high - physical_low)
        physical_action = center + half_range * clipped_action

        if CONVERT_GRIPPER and physical_action.shape[-1] > 0:
            gripper_q = physical_action[..., -1]
            physical_action = physical_action.clone()
            physical_action[..., -1] = gripper_q_maniskill_to_robomanip(gripper_q)

        action_np = action.squeeze(0).cpu().numpy().tolist()
        clipped_np = clipped_action.squeeze(0).cpu().numpy().tolist()
        physical_np = physical_action.squeeze(0).cpu().numpy().tolist()

        if PRINT_JSON:
            payload = {
                "step": step_idx,
                "action": action_np,
                "clipped_action": clipped_np,
                "physical_action": physical_np,
            }
            print(json.dumps(payload))
        else:
            print(
                f"step={step_idx} action={action_np} clipped_action={clipped_np} physical_action={physical_np}"
            )


if __name__ == "__main__":
    main()
