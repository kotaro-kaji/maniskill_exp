"""xArm7 variant that wires the new rate-limited joint controller.

This agent preserves the joint-delta control interface used by ``my_xarm7``
while replacing the legacy ad-hoc filter with ManiSkill's
``RateLimitedJointPosController`` so that simulation enforces the same
velocity/acceleration caps exposed by the real xArm SDK (Mode 6).

Users should update the rate-limit arrays and PD gains with the values
measured on their robot to ensure the simulation faithfully mirrors the
hardware behaviour.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Sequence

import numpy as np

from mani_skill.agents.controllers import (
    PDJointPosControllerConfig,
    RateLimitedJointPosControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent

from .my_xarm7 import Xarm7

# ---------------------------------------------------------------------------
# Default xArm Mode 6 limits (approximate SDK defaults). These should be
# replaced with the actual values measured on the user's robot.
# ---------------------------------------------------------------------------
_DEFAULT_ARM_MAX_VEL = float(np.deg2rad(20.0))  # ≈0.349 rad/s
_DEFAULT_ARM_MAX_ACC = float(np.deg2rad(500.0))  # ≈8.73 rad/s^2


def _broadcast(values: Sequence[float] | float, dof: int) -> np.ndarray:
    """Utility to produce per-joint arrays from scalars or sequences."""

    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0:
        return np.full(dof, float(arr), dtype=np.float32)
    if arr.shape[0] != dof:
        return np.broadcast_to(arr, (dof,)).astype(np.float32)
    return arr.astype(np.float32)


@register_agent()
class Xarm7Official(Xarm7):
    """xArm7 agent that exposes a rate-limited joint delta controller."""

    uid = "my_xarm7_official"

    @property
    def _controller_configs(self):
        base_configs = deepcopy_dict(super()._controller_configs)

        arm_dof = len(self.arm_joint_names)

        # Rate-limit parameters for the arm. Replace with real hardware values.
        arm_max_vel = _broadcast(_DEFAULT_ARM_MAX_VEL, arm_dof)
        arm_max_acc = _broadcast(_DEFAULT_ARM_MAX_ACC, arm_dof)

        # PD gains fall back to the environment defaults; users should update
        # them with the values reported by their controller firmware.
        arm_force_limits = _broadcast(self.arm_force_limit, arm_dof)

        arm_rate_limited = RateLimitedJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=_broadcast(self.arm_stiffness, arm_dof),
            damping=_broadcast(self.arm_damping, arm_dof),
            force_limit=arm_force_limits,
            use_delta=True,
            use_target=True,
            max_velocity=arm_max_vel,
            max_acceleration=arm_max_acc,
            max_jerk=None,  # Populate once jerk data is available.
        )

        controller_configs = OrderedDict()
        controller_configs["rate_limited_pd_joint_delta_pos"] = dict(
            arm=arm_rate_limited,
            gripper=base_configs["pd_joint_delta_pos"]["gripper"],
            balance_passive_force=False,
        )

        hardware_p = np.array(
            [1200.0, 1400.0, 1200.0, 850.0, 500.0, 500.0, 300.0],
            dtype=np.float32,
        )
        hardware_d = np.array(
            [10.0, 10.0, 5.0, 5.0, 1.0, 1.0, 1.0],
            dtype=np.float32,
        )
        force_limits = np.array(
            [50.0, 50.0, 30.0, 30.0, 30.0, 20.0, 20.0],
            dtype=np.float32,
        )

        official_arm = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=hardware_p,
            damping=np.maximum(hardware_d, 0.05),
            force_limit=force_limits,
            use_delta=True,
        )
        official_gripper = copy.deepcopy(base_configs["pd_joint_delta_pos"]["gripper"])

        controller_configs["official_pd_joint_delta_pos"] = dict(
            arm=official_arm,
            gripper=official_gripper,
            balance_passive_force=False,
        )

        # Keep access to existing controllers for backwards compatibility.
        for name, cfg in base_configs.items():
            controller_configs[name] = cfg

        return controller_configs
