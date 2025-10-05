"""xArm7 controller variant that mirrors the behaviour of the official SDK.

The new controller keeps the 7-DoF joint delta interface used by the
`my_xarm7` agent while emulating the firmware-level smoothing in
`set_servo_angle_j`: we integrate per-joint PID loops with the same
velocity/acceleration clamps and goal tolerances that the real robot enforces.
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
    min_joint_speed: Union[float, Sequence[float]] = 1e-4
    max_joint_acc: Union[float, Sequence[float]] = 20.0  # rad/s^2 cap from SDK
    positional_gain: Union[float, Sequence[float]] = 6.0
    integral_gain: Union[float, Sequence[float]] = 0.0
    velocity_damping: Union[float, Sequence[float]] = 1.5
    integral_clamp: Union[float, Sequence[float]] = 0.25
    error_tolerance: float = math.radians(0.01)
    velocity_tolerance: float = math.radians(0.02)
    controller_cls = None  # populated after class declaration


class XArmSDKJointDeltaController(PDJointPosController):
    """Delta-angle controller that mimics xArm's ``set_servo_angle_j`` pathing.

    We run a per-joint PID filter with SDK-matched velocity/acceleration
    clamps, goal tolerances, and soft minimum velocities. The output of this
    filter is forwarded to the regular ManiSkill PD drives, so the arm still
    benefits from the platform's stiffness/damping configuration.
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

        # Pre-broadcasted controller parameters
        self._max_speed = _to_tensor_parameter(
            self.config.max_joint_speed, dof, self.device
        )
        self._min_speed = _to_tensor_parameter(
            self.config.min_joint_speed, dof, self.device
        )
        self._max_acc = _to_tensor_parameter(
            self.config.max_joint_acc, dof, self.device
        )
        self._pos_gain = _to_tensor_parameter(
            self.config.positional_gain, dof, self.device
        )
        self._int_gain = _to_tensor_parameter(
            self.config.integral_gain, dof, self.device
        )
        self._vel_damp = _to_tensor_parameter(
            self.config.velocity_damping, dof, self.device
        )
        self._integral_limit = _to_tensor_parameter(
            self.config.integral_clamp, dof, self.device
        )
        self._error_tol = torch.full(
            (dof,), float(self.config.error_tolerance), device=self.device
        )
        self._velocity_tol = torch.full(
            (dof,), float(self.config.velocity_tolerance), device=self.device
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
        self._integral_error = None

    def reset(self):
        super().reset()
        if self._commanded_target is None:
            self._commanded_target = self._target_qpos.clone()
            self._servo_target = self._target_qpos.clone()
            self._servo_velocity = torch.zeros_like(self._target_qpos)
            self._integral_error = torch.zeros_like(self._target_qpos)
        else:
            mask = getattr(self.scene, "_reset_mask", None)
            if mask is None:
                self._commanded_target.copy_(self._target_qpos)
                self._servo_target.copy_(self._target_qpos)
                self._servo_velocity.zero_()
                self._integral_error.zero_()
            else:
                self._commanded_target[mask] = self._target_qpos[mask]
                self._servo_target[mask] = self._target_qpos[mask]
                self._servo_velocity[mask] = 0.0
                self._integral_error[mask] = 0.0
        self.set_drive_targets(self._servo_target)

    def set_action(self, action: Array):
        action = self._preprocess_action(action).to(self.device)
        self._step = 0

        if self._commanded_target is None:
            self._commanded_target = self.qpos.clone()
            self._servo_target = self.qpos.clone()
            self._servo_velocity = torch.zeros_like(self.qpos)
            self._integral_error = torch.zeros_like(self.qpos)

        delta = torch.maximum(
            torch.minimum(action, self._delta_upper), self._delta_lower
        )
        base = self._commanded_target if self.config.use_target else self.qpos
        raw_target = base + delta

        clamped_target = torch.minimum(
            torch.maximum(raw_target, self._joint_lower), self._joint_upper
        )
        self._commanded_target = clamped_target.clone()
        self._target_qpos = clamped_target.clone()

    def before_simulation_step(self):
        if self._commanded_target is None:
            return

        self._step += 1
        dt = self._sim_dt

        error = self._commanded_target - self._servo_target
        self._integral_error = torch.clamp(
            self._integral_error + error * dt,
            -self._integral_limit,
            self._integral_limit,
        )

        # PID-like acceleration request (units: rad/s^2)
        desired_acc = (
            self._pos_gain * error
            + self._int_gain * self._integral_error
            - self._vel_damp * self._servo_velocity
        )
        desired_acc = torch.clamp(desired_acc, -self._max_acc, self._max_acc)

        self._servo_velocity = torch.clamp(
            self._servo_velocity + desired_acc * dt,
            -self._max_speed,
            self._max_speed,
        )

        abs_vel = self._servo_velocity.abs()
        close_to_goal = (error.abs() <= self._error_tol) & (
            abs_vel <= self._velocity_tol
        )
        below_min = (abs_vel < self._min_speed) & (~close_to_goal)
        # Stick to the minimum servo velocity when travelling, stop once both
        # error and speed are within SDK tolerances.
        self._servo_velocity = torch.where(
            close_to_goal,
            torch.zeros_like(self._servo_velocity),
            torch.where(
                below_min,
                torch.sign(self._servo_velocity) * self._min_speed,
                self._servo_velocity,
            ),
        )

        self._servo_target = self._servo_target + self._servo_velocity * dt

        overshoot_pos = (self._servo_velocity > 0) & (
            self._servo_target >= self._commanded_target
        )
        overshoot_neg = (self._servo_velocity < 0) & (
            self._servo_target <= self._commanded_target
        )
        overshoot_mask = overshoot_pos | overshoot_neg
        if overshoot_mask.any():
            self._servo_target = torch.where(
                overshoot_mask, self._commanded_target, self._servo_target
            )
            self._servo_velocity = torch.where(
                overshoot_mask,
                torch.zeros_like(self._servo_velocity),
                self._servo_velocity,
            )
            self._integral_error = torch.where(
                overshoot_mask,
                torch.zeros_like(self._integral_error),
                self._integral_error,
            )

        self._servo_target = torch.minimum(
            torch.maximum(self._servo_target, self._joint_lower), self._joint_upper
        )

        contact_limit = (self._servo_target <= self._joint_lower + 1e-6) | (
            self._servo_target >= self._joint_upper - 1e-6
        )
        if contact_limit.any():
            self._servo_velocity = torch.where(
                contact_limit,
                torch.zeros_like(self._servo_velocity),
                self._servo_velocity,
            )
            self._integral_error = torch.where(
                contact_limit,
                torch.zeros_like(self._integral_error),
                self._integral_error,
            )

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

        # SDK-aligned per-joint gains taken from UFactory's ROS configs
        # (scaled to radian units so that the simulated motion profile matches
        # the embedded servo behaviour).
        hardware_p = np.array(
            [1200.0, 1400.0, 1200.0, 850.0, 500.0, 500.0, 300.0],
            dtype=np.float32,
        )
        hardware_d = np.array(
            [10.0, 10.0, 5.0, 5.0, 1.0, 1.0, 1.0], dtype=np.float32
        )
        hardware_i = np.array(
            [5.0, 5.0, 5.0, 3.0, 3.0, 1.0, 0.05], dtype=np.float32
        )

        sdk_arm = XArmSDKJointDeltaControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
            use_target=True,
            normalize_action=False,
            positional_gain=hardware_p / 150.0,
            velocity_damping=np.maximum(hardware_d / 5.0, 0.05),
            integral_gain=hardware_i / 100.0,
            integral_clamp=np.clip(hardware_p / 6000.0, 0.02, 0.3),
            max_joint_speed=math.pi,
            max_joint_acc=20.0,
            min_joint_speed=1e-4,
            error_tolerance=math.radians(0.01),
            velocity_tolerance=math.radians(0.02),
        )

        gripper_cfg = copy.deepcopy(base_configs["pd_joint_delta_pos"]["gripper"])

        controller_configs = OrderedDict()
        controller_configs["sdk_joint_delta_pos"] = dict(
            arm=sdk_arm,
            gripper=gripper_cfg,
            balance_passive_force=False,
        )

        # Preserve access to the legacy controllers for backwards compatibility.
        for name, cfg in base_configs.items():
            controller_configs[name] = cfg

        return controller_configs
