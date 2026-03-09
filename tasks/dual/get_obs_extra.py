from typing import Any, Dict

import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from mani_skill.utils.structs.pose import Pose


def _build_obs_dict(
    env, info: Dict[str, Any], include_tcp: bool, include_pushpoints: bool
) -> Dict[str, torch.Tensor]:
    obs: Dict[str, torch.Tensor] = {}
    if not isinstance(env.agent, MultiAgent):
        return obs
    obs_mode = str(getattr(env, "obs_mode", "")).lower()
    include_box_pose = "rgb" not in obs_mode

    def _ensure_batch(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0) if t.ndim == 1 else t

    def _rotation_to_theta_deg(rotation: torch.Tensor) -> torch.Tensor:
        yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
        theta = torch.rad2deg(yaw)
        theta = torch.remainder(theta, 360.0)
        theta = torch.remainder(-theta, 360.0)
        return theta

    env._ensure_pushpoint_buffers()
    box_pose = Pose.create(env.box.pose, device=env.device)
    box_position = _ensure_batch(box_pose.p)
    box_rotation = quaternion_to_matrix(box_pose.q)
    box_rotation = _ensure_batch(box_rotation)
    assert box_position.ndim == 2
    assert box_rotation.ndim == 3
    box_position_jittered, box_rotation_jittered = env._apply_box_obs_offset(
        box_position, box_rotation
    )
    box_position_jittered, box_rotation_jittered = env._apply_box_obs_step_jitter(
        box_position_jittered, box_rotation_jittered
    )

    if include_box_pose:
        box_quat = matrix_to_quaternion(box_rotation_jittered)
        box_pose_offset = Pose.create_from_pq(
            p=box_position_jittered, q=box_quat, device=env.device
        )
        obs["box_pose_6d_from_bimanual_center"] = env._pose_to_6d(
            box_pose_offset, center_frame=True
        )

    if include_tcp:
        left_tcp_pose = Pose.create(env.agent.agents[0].tcp_pose, device=env.device)
        right_tcp_pose = Pose.create(env.agent.agents[1].tcp_pose, device=env.device)
        obs["left_tcp_pose_6d_from_bimanual_center"] = env._pose_to_6d(
            left_tcp_pose, center_frame=True
        )
        obs["right_tcp_pose_6d_from_bimanual_center"] = env._pose_to_6d(
            right_tcp_pose, center_frame=True
        )


    if include_pushpoints:
        box_center = box_position_jittered
        rotation_current = box_rotation_jittered
        theta = _rotation_to_theta_deg(rotation_current)
        (
            target_pushpoint_left,
            target_pushpoint_right,
            _,
            _,
            _,
        ) = env._compute_pushpoints_from_pose(box_center, rotation_current, theta)
        center_vec = torch.tensor(
            env.bimanual_center_pose.p,
            dtype=target_pushpoint_left.dtype,
            device=target_pushpoint_left.device,
        )
        obs["pushpoint_left_from_bimanual_center"] = (
            target_pushpoint_left - center_vec
        )
        obs["pushpoint_right_from_bimanual_center"] = (
            target_pushpoint_right - center_vec
        )

    return obs


def get_obs_extra_full(env, info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Return the full observation dictionary (original behavior)."""
    return _build_obs_dict(env, info, include_tcp=True, include_pushpoints=True)


def get_obs_extra_ablation(env, info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Return an ablated observation dictionary (no TCP or pushpoint terms)."""
    return _build_obs_dict(env, info, include_tcp=False, include_pushpoints=False)
