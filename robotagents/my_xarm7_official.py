"""xArm7 controller variant that mirrors the behaviour of the official SDK.

The new controller keeps the 7-DoF joint delta interface used by the
`my_xarm7` agent while emulating the firmware-level smoothing in
`set_servo_angle_j`: at every control step we integrate joint targets subject
to the SDK's velocity and acceleration caps before forwarding them to the
underlying ManiSkill PD drives.
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import torch

from mani_skill.agents.controllers import (
    PDJointPosController,
    PDJointPosControllerConfig,
    deepcopy_dict,
)
from mani_skill.agents.registration import register_agent
from mani_skill.utils.structs.types import Array

from .my_xarm7 import Xarm7


def _to_tensor_parameter(
    value: Union[float, Sequence[float], np.ndarray],
    dof: int,
    device,
) -> torch.Tensor:
    """Utility to broadcast scalar/sequence parameters to ``(1, dof)`` tensors."""

    if value is None:
        raise ValueError("Expected a numeric value, got None")
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = np.full(dof, arr, dtype=np.float32)
    elif arr.size != dof:
        arr = np.broadcast_to(arr, (dof,)).astype(np.float32)
    return torch.from_numpy(arr).to(device=device)


@dataclass
class XArmSDKJointDeltaControllerConfig(PDJointPosControllerConfig):
    """Configuration for the SDK-inspired joint delta controller."""

    max_joint_speed: Union[float, Sequence[float]] = math.pi  # rad/s cap (≈180°/s)
    max_joint_acc: Union[float, Sequence[float]] = 20.0  # rad/s^2 cap from SDK
    positional_gain: Union[float, Sequence[float]] = 6.0
    velocity_damping: Union[float, Sequence[float]] = 1.5
    controller_cls = None  # populated after class declaration


class XArmSDKJointDeltaController(PDJointPosController):
    """Delta-angle controller that mimics xArm's ``set_servo_angle_j`` pathing.

    Each control tick we respect the SDK's joint velocity/acceleration limits
    while moving the internal waypoint toward the commanded joint targets; the
    resulting pose is then tracked by ManiSkill's PD drives.
    """

    config: "XArmSDKJointDeltaControllerConfig"

    def __init__(
        self,
        config: XArmSDKJointDeltaControllerConfig,
        articulation,
        control_freq: int,
        sim_freq: int = None,
        scene=None,
    ):
        super().__init__(config, articulation, control_freq, sim_freq, scene)
        self._sim_dt = float(self.articulation.px.timestep)
        self._control_dt = self._sim_dt * self._sim_steps
        dof = len(self.joints)

        # Hardware joint limits (same for every env instance)
        qlimits = (
            self.articulation.get_qlimits()[0, self.active_joint_indices]
            .detach()
            .cpu()
            .numpy()
        )
        self._joint_lower = torch.from_numpy(qlimits[:, 0]).to(self.device)
        self._joint_upper = torch.from_numpy(qlimits[:, 1]).to(self.device)

        # Pre-broadcasted motion constraints
        self._max_speed = _to_tensor_parameter(
            self.config.max_joint_speed, dof, self.device
        )
        self._max_acc = _to_tensor_parameter(
            self.config.max_joint_acc, dof, self.device
        )

        # Action bounds (delta joint limits)
        lower = self.config.lower if self.config.lower is not None else -0.1
        upper = self.config.upper if self.config.upper is not None else 0.1
        self._delta_lower = _to_tensor_parameter(lower, dof, self.device)
        self._delta_upper = _to_tensor_parameter(upper, dof, self.device)

        # Ring buffers for the servo filter
        self._commanded_target = None
        self._servo_target = None
        self._servo_velocity = None
        self._pos_tolerance = 1e-4
        self._vel_tolerance = 5e-4

    def reset(self):
        super().reset()
        if self._commanded_target is None:
            self._commanded_target = self._target_qpos.clone()
            self._servo_target = self._target_qpos.clone()
            self._servo_velocity = torch.zeros_like(self._target_qpos)
        else:
            mask = getattr(self.scene, "_reset_mask", None)
            if mask is None:
                self._commanded_target.copy_(self._target_qpos)
                self._servo_target.copy_(self._target_qpos)
                self._servo_velocity.zero_()
            else:
                self._commanded_target[mask] = self._target_qpos[mask]
                self._servo_target[mask] = self._target_qpos[mask]
                self._servo_velocity[mask] = 0.0
        self.set_drive_targets(self._servo_target)

    def set_action(self, action: Array):
        action = self._preprocess_action(action).to(self.device)
        self._step = 0

        if self._commanded_target is None:
            self._commanded_target = self.qpos.clone()
            self._servo_target = self.qpos.clone()
            self._servo_velocity = torch.zeros_like(self.qpos)

        delta = torch.clamp(action, self._delta_lower, self._delta_upper)
        base = self._commanded_target if self.config.use_target else self.qpos
        commanded = torch.clamp(base + delta, self._joint_lower, self._joint_upper)
        self._commanded_target = commanded.clone()

    def _integrate_servo(self, dt: float):
        if self._commanded_target is None:
            return

        error = self._commanded_target - self._servo_target

        # Snap to target when both position error and residual velocity are negligible.
        arrived = (torch.abs(error) <= self._pos_tolerance) & (
            torch.abs(self._servo_velocity) <= self._vel_tolerance
        )
        if arrived.any():
            self._servo_target = torch.where(
                arrived, self._commanded_target, self._servo_target
            )
            self._servo_velocity = torch.where(
                arrived, torch.zeros_like(self._servo_velocity), self._servo_velocity
            )
            # Recompute error for the remaining joints.
            error = self._commanded_target - self._servo_target

        # Maximum admissible velocity to stop under constant deceleration.
        abs_error = torch.abs(error)
        max_stoppable_speed = torch.sqrt(
            torch.clamp(2.0 * self._max_acc * abs_error, min=0.0)
        )
        target_speed = torch.minimum(self._max_speed, max_stoppable_speed)
        desired_velocity = torch.sign(error) * target_speed

        vel_error = desired_velocity - self._servo_velocity
        max_vel_delta = self._max_acc * dt
        vel_delta = torch.clamp(vel_error, -max_vel_delta, max_vel_delta)
        self._servo_velocity = self._servo_velocity + vel_delta

        # Zero out tiny velocities to avoid numerical drift.
        self._servo_velocity = torch.where(
            torch.abs(self._servo_velocity) <= self._vel_tolerance,
            torch.zeros_like(self._servo_velocity),
            self._servo_velocity,
        )

        new_target = self._servo_target + self._servo_velocity * dt

        # Prevent overshoot beyond the commanded waypoint.
        overshoot_pos = (error > 0) & (new_target > self._commanded_target)
        overshoot_neg = (error < 0) & (new_target < self._commanded_target)
        overshoot = overshoot_pos | overshoot_neg
        if overshoot.any():
            new_target = torch.where(overshoot, self._commanded_target, new_target)
            self._servo_velocity = torch.where(
                overshoot, torch.zeros_like(self._servo_velocity), self._servo_velocity
            )

        self._servo_target = torch.clamp(new_target, self._joint_lower, self._joint_upper)

        # Final snap when we hit limits or targets.
        error = self._commanded_target - self._servo_target
        at_lower = self._servo_target <= (self._joint_lower + self._pos_tolerance)
        at_upper = self._servo_target >= (self._joint_upper - self._pos_tolerance)
        stuck = (torch.abs(error) <= self._pos_tolerance) | at_lower | at_upper
        if stuck.any():
            self._servo_target = torch.where(
                stuck, self._commanded_target, self._servo_target
            )
            self._servo_velocity = torch.where(
                stuck, torch.zeros_like(self._servo_velocity), self._servo_velocity
            )

    def before_simulation_step(self):
        if self._servo_target is not None:
            self._integrate_servo(self._sim_dt)
            self._target_qpos = self._servo_target.clone()
            self.set_drive_targets(self._servo_target)

    def get_state(self) -> dict:
        state = super().get_state()
        state.update(
            {
                "sdk_command_qpos": self._commanded_target,
                "sdk_servo_qpos": self._servo_target,
                "sdk_servo_qvel": self._servo_velocity,
            }
        )
        return state


# Link the config to the controller implementation
XArmSDKJointDeltaControllerConfig.controller_cls = XArmSDKJointDeltaController


@register_agent()
class Xarm7Official(Xarm7):
    """XArm7 variant with an SDK-faithful joint delta position controller."""

    uid = "my_xarm7_official"

    @property
    def _controller_configs(self):
        base_configs = deepcopy_dict(super()._controller_configs)

        # SDK-aligned per-joint gains taken from UFactory's ROS configs so the
        # simulated response mirrors the firmware defaults. Gains are rescaled
        # from the integer PID units used by the embedded drives to radian space.
        hardware_p = np.array(
            [1200.0, 1400.0, 1200.0, 850.0, 500.0, 500.0, 300.0],
            dtype=np.float32,
        )
        hardware_d = np.array(
            [10.0, 10.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float32
        )
        force_limits = np.array([50.0, 50.0, 30.0, 30.0, 30.0, 20.0, 20.0], dtype=np.float32)

        official_pd_arm = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=hardware_p,
            damping=np.maximum(hardware_d, 0.05),
            force_limit=force_limits,
            use_delta=True,
            normalize_action=False,
        )

        sdk_arm = XArmSDKJointDeltaControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=force_limits,
            use_delta=True,
            use_target=False,
            normalize_action=False,
            positional_gain=hardware_p / 150.0,
            velocity_damping=np.maximum(hardware_d / 5.0, 0.05),
            max_joint_speed=math.pi,
            max_joint_acc=20.0,
        )

        gripper_cfg_official = copy.deepcopy(base_configs["pd_joint_delta_pos"]["gripper"])
        gripper_cfg_sdk = copy.deepcopy(base_configs["pd_joint_delta_pos"]["gripper"])

        controller_configs = OrderedDict()
        controller_configs["official_pd_joint_delta_pos"] = dict(
            arm=official_pd_arm,
            gripper=gripper_cfg_official,
            balance_passive_force=False,
        )
        controller_configs["sdk_joint_delta_pos"] = dict(
            arm=sdk_arm,
            gripper=gripper_cfg_sdk,
            balance_passive_force=False,
        )

        # Preserve access to the legacy controllers for backwards compatibility.
        for name, cfg in base_configs.items():
            controller_configs[name] = cfg

        return controller_configs
