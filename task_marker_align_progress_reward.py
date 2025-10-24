"""Experimental variant of the marker-alignment task with progress-based rewards.

This script keeps the original environment intact while providing a subclass
that awards a small shaped reward when the TCP moves closer to the marker and
a large bonus upon success.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from mani_skill.utils.registration import register_env

from task_marker_align_official import (
    MARKER_NORMAL_OFFSET,
    MyEEAlignMarkerEnv,
)


PROGRESS_REWARD = 0.1
SUCCESS_BONUS = 5.0
PROGRESS_EPS = 2e-4  # 0.2 mm
STILLNESS_TOLERANCE = 0.02  # rad, average per joint
STILLNESS_REWARD = 0.02  # tiny bonus for staying near the start pose


@register_env("MyEEAlignMarkerProgress-v0", max_episode_steps=100)
class MyEEAlignMarkerProgressEnv(MyEEAlignMarkerEnv):
    """Marker alignment environment with shaped progress and success rewards."""

    def __init__(
        self,
        *args,
        robot_uids: str = "my_xarm7_official",
        marker_pose: Optional[Any] = None,
        **kwargs,
    ):
        self._prev_distance: Optional[torch.Tensor] = None
        self._initial_qpos: Optional[torch.Tensor] = None
        super().__init__(*args, robot_uids=robot_uids, marker_pose=marker_pose, **kwargs)

    def _clear(self):
        super()._clear()
        self._prev_distance = None
        self._initial_qpos = None

    def _ensure_prev_distance(self):
        if self._prev_distance is None or len(self._prev_distance) != self.num_envs:
            self._prev_distance = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
        elif self._prev_distance.device != self.device:
            self._prev_distance = self._prev_distance.to(
                device=self.device, dtype=torch.float32
            )

    def _current_distance(self) -> torch.Tensor:
        marker_pose, _, _, normal = self._marker_frame_axes()
        target_point = marker_pose.p + normal * MARKER_NORMAL_OFFSET
        tcp_position = self._get_tcp_pose_on_device().p
        return torch.linalg.norm(tcp_position - target_point, dim=1)

    def _update_prev_distance(self, env_idx: torch.Tensor):
        self._ensure_prev_distance()
        if env_idx.ndim == 0:
            env_idx = env_idx.unsqueeze(0)
        env_idx = env_idx.to(device=self.device, dtype=torch.long)
        if env_idx.numel() == 0:
            return
        current = self._current_distance()
        if current.device != self.device:
            current = current.to(self.device)
        selected = current.index_select(0, env_idx)
        self._prev_distance.index_copy_(0, env_idx, selected)

    def _get_current_qpos(self) -> torch.Tensor:
        qpos = self.agent.robot.get_qpos()
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        if qpos.device != self.device:
            qpos = qpos.to(self.device)
        return qpos

    def _ensure_initial_qpos_buffer(self, qpos_template: torch.Tensor):
        if qpos_template.ndim == 1:
            qpos_template = qpos_template.unsqueeze(0)
        expected_shape = (self.num_envs, qpos_template.shape[-1])
        need_new = (
            self._initial_qpos is None
            or self._initial_qpos.shape[0] != expected_shape[0]
            or self._initial_qpos.shape[1] != expected_shape[1]
        )
        if need_new:
            new_buffer = torch.zeros(
                expected_shape, dtype=qpos_template.dtype, device=qpos_template.device
            )
            if self._initial_qpos is not None:
                rows = min(self._initial_qpos.shape[0], expected_shape[0])
                cols = min(self._initial_qpos.shape[1], expected_shape[1])
                new_buffer[:rows, :cols] = self._initial_qpos[:rows, :cols].to(
                    device=qpos_template.device, dtype=qpos_template.dtype
                )
            self._initial_qpos = new_buffer
        elif (
            self._initial_qpos.device != qpos_template.device
            or self._initial_qpos.dtype != qpos_template.dtype
        ):
            self._initial_qpos = self._initial_qpos.to(
                device=qpos_template.device, dtype=qpos_template.dtype
            )

    def _cache_initial_qpos(self, env_idx: torch.Tensor):
        if env_idx.ndim == 0:
            env_idx = env_idx.unsqueeze(0)
        if env_idx.numel() == 0:
            return
        qpos = self._get_current_qpos()
        self._ensure_initial_qpos_buffer(qpos)
        env_idx = env_idx.to(device=qpos.device, dtype=torch.long)
        selected = qpos.index_select(0, env_idx).detach()
        self._initial_qpos.index_copy_(0, env_idx, selected)

    def _initialize_success_counters(self, env_idx: torch.Tensor):
        super()._initialize_success_counters(env_idx)
        self._ensure_prev_distance()
        if env_idx.numel() > 0:
            inf_values = torch.full(
                (env_idx.numel(),), float("inf"), device=self.device
            )
            env_idx = env_idx.to(device=self.device, dtype=torch.long)
            self._prev_distance.index_copy_(0, env_idx, inf_values)

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict):
        super()._initialize_episode(env_idx, options)
        self._update_prev_distance(env_idx)
        self._cache_initial_qpos(env_idx)

    def _respawn_marker(self, env_idx: torch.Tensor):
        super()._respawn_marker(env_idx)
        self._update_prev_distance(env_idx)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        self._ensure_prev_distance()
        current_distance = self._current_distance()
        prev_distance = self._prev_distance
        if prev_distance.device != current_distance.device:
            prev_distance = prev_distance.to(current_distance.device)
            self._prev_distance = prev_distance

        improved = (prev_distance - current_distance) > PROGRESS_EPS
        reward = improved.float() * PROGRESS_REWARD

        success = info.get("success")
        if success is not None:
            if not isinstance(success, torch.Tensor):
                success_tensor = torch.as_tensor(success, device=self.device)
            else:
                success_tensor = success.to(self.device)
            if success_tensor.ndim == 0:
                success_tensor = success_tensor.unsqueeze(0)
            success_tensor = success_tensor.to(current_distance.device, dtype=torch.float32)
            reward = reward + success_tensor * SUCCESS_BONUS

        current_qpos = self._get_current_qpos()
        self._ensure_initial_qpos_buffer(current_qpos)
        initial_qpos = self._initial_qpos
        if initial_qpos is None:
            initial_qpos = current_qpos.detach().clone()
            self._initial_qpos = initial_qpos
        elif initial_qpos.device != current_qpos.device:
            initial_qpos = initial_qpos.to(current_qpos.device)
            self._initial_qpos = initial_qpos
        avg_joint_delta = torch.mean(torch.abs(current_qpos - initial_qpos), dim=1)
        stillness_factor = torch.clamp(
            1.0 - (avg_joint_delta / STILLNESS_TOLERANCE), min=0.0, max=1.0
        )
        reward = reward + stillness_factor * STILLNESS_REWARD

        self._prev_distance = current_distance.detach().clone()
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = SUCCESS_BONUS + PROGRESS_REWARD
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
