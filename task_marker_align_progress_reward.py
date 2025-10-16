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
        super().__init__(*args, robot_uids=robot_uids, marker_pose=marker_pose, **kwargs)

    def _clear(self):
        super()._clear()
        self._prev_distance = None

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

        self._prev_distance = current_distance.detach().clone()
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = SUCCESS_BONUS + PROGRESS_REWARD
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
