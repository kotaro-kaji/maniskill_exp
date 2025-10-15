from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch

from mani_skill.agents.controllers import (
    PDJointPosController,
    PDJointPosControllerConfig,
)
from mani_skill.utils import common
from mani_skill.utils.structs.types import Array


def _broadcast_limit(
    value: Optional[Union[float, Sequence[float], np.ndarray]],
    dof: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        arr = np.full(dof, float(arr), dtype=np.float32)
    else:
        arr = np.broadcast_to(arr, (dof,)).astype(np.float32)
    return common.to_tensor(arr, device=device).unsqueeze(0)


def _apply_rate_limits(
    desired_position: torch.Tensor,
    prev_position: torch.Tensor,
    prev_velocity: Optional[torch.Tensor],
    dt: float,
    max_velocity: Optional[torch.Tensor],
    max_acceleration: Optional[torch.Tensor],
    prev_acceleration: Optional[torch.Tensor] = None,
    max_jerk: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if dt <= 0:
        raise ValueError("Control timestep must be positive.")

    device = desired_position.device
    dtype = desired_position.dtype

    if prev_velocity is None:
        prev_velocity = torch.zeros_like(desired_position, device=device, dtype=dtype)
    else:
        prev_velocity = prev_velocity.to(device=device, dtype=dtype)

    if prev_acceleration is None:
        prev_acceleration = torch.zeros_like(desired_position, device=device, dtype=dtype)
    else:
        prev_acceleration = prev_acceleration.to(device=device, dtype=dtype)

    desired_velocity = (desired_position - prev_position) / dt
    if max_velocity is not None:
        desired_velocity = torch.clamp(desired_velocity, -max_velocity, max_velocity)

    if max_acceleration is not None:
        max_delta_vel = max_acceleration * dt
        desired_velocity = torch.clamp(
            desired_velocity,
            prev_velocity - max_delta_vel,
            prev_velocity + max_delta_vel,
        )

    new_acceleration = (desired_velocity - prev_velocity) / dt

    if max_jerk is not None:
        max_delta_acc = max_jerk * dt
        new_acceleration = torch.clamp(
            new_acceleration,
            prev_acceleration - max_delta_acc,
            prev_acceleration + max_delta_acc,
        )
        if max_acceleration is not None:
            new_acceleration = torch.clamp(
                new_acceleration, -max_acceleration, max_acceleration
            )
        desired_velocity = prev_velocity + new_acceleration * dt
        if max_velocity is not None:
            desired_velocity = torch.clamp(desired_velocity, -max_velocity, max_velocity)
        new_acceleration = (desired_velocity - prev_velocity) / dt

    new_position = prev_position + desired_velocity * dt
    return new_position, desired_velocity, new_acceleration


@dataclass
class RateLimitedJointPosControllerConfig(PDJointPosControllerConfig):
    max_velocity: Optional[Union[float, Sequence[float]]] = None
    max_acceleration: Optional[Union[float, Sequence[float]]] = None
    max_jerk: Optional[Union[float, Sequence[float]]] = None
    controller_cls = None  # populated after class definition


class RateLimitedJointPosController(PDJointPosController):
    config: "RateLimitedJointPosControllerConfig"

    def __init__(
        self,
        config: RateLimitedJointPosControllerConfig,
        articulation,
        control_freq: int,
        sim_freq: int = None,
        scene=None,
    ):
        super().__init__(config, articulation, control_freq, sim_freq, scene)
        self._sim_dt = float(self.articulation.px.timestep)
        self._control_dt = self._sim_dt * self._sim_steps if self._sim_steps > 0 else self._sim_dt
        self._dof = len(self.joints)

        limits = self._get_joint_limits()
        self._joint_lower = common.to_tensor(limits[:, 0], device=self.device).unsqueeze(0)
        self._joint_upper = common.to_tensor(limits[:, 1], device=self.device).unsqueeze(0)

        self._max_velocity = _broadcast_limit(self.config.max_velocity, self._dof, self.device)
        self._max_acceleration = _broadcast_limit(
            self.config.max_acceleration, self._dof, self.device
        )
        self._max_jerk = _broadcast_limit(self.config.max_jerk, self._dof, self.device)

        self._desired_qpos: Optional[torch.Tensor] = None
        self._rate_velocity: Optional[torch.Tensor] = None
        self._rate_acceleration: Optional[torch.Tensor] = None

    def reset(self):
        super().reset()
        if self._desired_qpos is None:
            self._desired_qpos = self._target_qpos.clone()
            self._rate_velocity = torch.zeros_like(self._target_qpos)
            self._rate_acceleration = torch.zeros_like(self._target_qpos)
        else:
            mask = getattr(self.scene, "_reset_mask", None)
            if mask is None:
                self._desired_qpos.copy_(self._target_qpos)
                self._rate_velocity.zero_()
                self._rate_acceleration.zero_()
            else:
                self._desired_qpos[mask] = self._target_qpos[mask]
                self._rate_velocity[mask] = 0.0
                self._rate_acceleration[mask] = 0.0
        self.set_drive_targets(self._target_qpos)

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        self._step = 0

        if self._desired_qpos is None:
            self._desired_qpos = self.qpos.clone()
            self._rate_velocity = torch.zeros_like(self._desired_qpos)
            self._rate_acceleration = torch.zeros_like(self._desired_qpos)

        if self.config.use_delta:
            base = self._desired_qpos if self.config.use_target else self.qpos
            desired = base + action
        else:
            desired = torch.broadcast_to(action, self.qpos.shape).clone()

        desired = torch.clamp(desired, self._joint_lower, self._joint_upper)
        self._desired_qpos = desired
        self._rate_limit_step(self._control_dt)

    def before_simulation_step(self):
        self._step += 1
        self._rate_limit_step(self._sim_dt)

    def _rate_limit_step(self, dt: float):
        if (
            self._desired_qpos is None
            or self._target_qpos is None
            or self._rate_velocity is None
        ):
            return

        desired = torch.clamp(self._desired_qpos, self._joint_lower, self._joint_upper)
        prev_acc = self._rate_acceleration if self._max_jerk is not None else None
        new_target, new_velocity, new_acc = _apply_rate_limits(
            desired,
            self._target_qpos,
            self._rate_velocity,
            dt,
            self._max_velocity,
            self._max_acceleration,
            prev_acceleration=prev_acc,
            max_jerk=self._max_jerk,
        )

        new_target = torch.clamp(new_target, self._joint_lower, self._joint_upper)
        self._target_qpos = new_target
        self._rate_velocity = new_velocity
        self._rate_acceleration = new_acc

        self.set_drive_targets(self._target_qpos)

    def get_state(self) -> dict:
        state = super().get_state()
        state.update(
            {
                "desired_qpos": self._desired_qpos,
                "rate_target_qpos": self._target_qpos,
                "rate_velocity": self._rate_velocity,
                "rate_acceleration": self._rate_acceleration,
            }
        )
        return state

    def set_state(self, state: dict):
        if "target_qpos" in state:
            super().set_state(state)
        self._desired_qpos = state.get("desired_qpos")
        self._target_qpos = state.get("rate_target_qpos", self._target_qpos)
        self._rate_velocity = state.get("rate_velocity")
        self._rate_acceleration = state.get("rate_acceleration")

        if self._desired_qpos is None:
            self._desired_qpos = self.qpos.clone()
        if self._target_qpos is None:
            self._target_qpos = self.qpos.clone()
        if self._rate_velocity is None:
            self._rate_velocity = torch.zeros_like(self._target_qpos)
        if self._rate_acceleration is None:
            self._rate_acceleration = torch.zeros_like(self._target_qpos)

        self.set_drive_targets(self._target_qpos)


RateLimitedJointPosControllerConfig.controller_cls = RateLimitedJointPosController
