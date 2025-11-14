from typing import Any, Dict

import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.structs.pose import Pose


def _build_obs_dict(
    env, info: Dict[str, Any], include_tcp: bool, include_pushpoints: bool
) -> Dict[str, torch.Tensor]:
    obs: Dict[str, torch.Tensor] = {}
    if not isinstance(env.agent, MultiAgent):
        return obs

    def _ensure_batch(t: torch.Tensor) -> torch.Tensor:
        return t.unsqueeze(0) if t.ndim == 1 else t

    if include_tcp:
        left_tcp_pose = Pose.create(env.agent.agents[1].tcp.pose, device=env.device)
        right_tcp_pose = Pose.create(env.agent.agents[0].tcp.pose, device=env.device)
        obs["left_tcp_pose_6d_from_bimanual_center"] = env._pose_to_6d(
            left_tcp_pose, center_frame=True
        )
        obs["right_tcp_pose_6d_from_bimanual_center"] = env._pose_to_6d(
            right_tcp_pose, center_frame=True
        )

    env._ensure_pushpoint_buffers()
    box_pose = Pose.create(env.box.pose, device=env.device)
    obs["box_pose_6d_from_bimanual_center"] = env._pose_to_6d(
        box_pose, center_frame=True
    )

    if include_pushpoints:
        box_center = _ensure_batch(box_pose.p)
        rotation_current = quaternion_to_matrix(box_pose.q)
        world_pushpoint_right = box_center + torch.einsum(
            "bij,bj->bi", rotation_current, env.pushpoint_local_right
        )
        world_pushpoint_left = box_center + torch.einsum(
            "bij,bj->bi", rotation_current, env.pushpoint_local_left
        )
        center_vec = torch.tensor(
            env.bimanual_center_pose.p,
            dtype=world_pushpoint_left.dtype,
            device=world_pushpoint_left.device,
        )
        obs["pushpoint_left_from_bimanual_center"] = (
            world_pushpoint_left - center_vec
        )
        obs["pushpoint_right_from_bimanual_center"] = (
            world_pushpoint_right - center_vec
        )

    return obs


def get_obs_extra_full(env, info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Return the full observation dictionary (original behavior)."""
    return _build_obs_dict(env, info, include_tcp=True, include_pushpoints=True)


def get_obs_extra_ablation(env, info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Return an ablated observation dictionary (no TCP or pushpoint terms)."""
    return _build_obs_dict(env, info, include_tcp=False, include_pushpoints=False)
