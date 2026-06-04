from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import tyro

from ppo_rollout_dual_xarm7 import RolloutArgs, run_rollout


@dataclass
class SingleRolloutArgs(RolloutArgs):
    env_id: str = "MySingleCardboardCabinetRandomized-v1"
    """Single-arm randomized cardboard cabinet env id."""
    robot_init_noise_scale: float = 0.0
    """Use the same deterministic robot initialization as cardboard training."""
    gripper_joint_indices: Optional[str] = "7"
    """Single-arm action index for the gripper drive joint."""


if __name__ == "__main__":
    run_rollout(tyro.cli(SingleRolloutArgs))
