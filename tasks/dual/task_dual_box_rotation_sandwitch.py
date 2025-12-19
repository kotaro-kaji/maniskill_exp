import torch
from typing import Tuple

from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.agents.multi_agent import MultiAgent

from .task_dual_box_rotation import MyDualBoxRotationEnv


@register_env("MyDualBoxRotationSandwitch-v0", max_episode_steps=200)
class MyDualBoxRotationSandwitchEnv(MyDualBoxRotationEnv):
    """Variant that sandwiches the box along the Y faces and drops TCP lead reward."""

    def _update_initial_pushpoints(
        self, env_idx: torch.Tensor, positions: torch.Tensor, orientations: torch.Tensor
    ):
        """Place pushpoints on the +Y and -Y face centers for a sandwich push."""
        batch_size = positions.shape[0]
        if batch_size == 0:
            return
        self._ensure_pushpoint_buffers()
        rotations = quaternion_to_matrix(orientations)
        hy = float(self.BOX_HALF_SIZE[1])
        # place targets slightly outside the Y faces so debug markers are visible
        offset_scale = 1.05

        offset_right_local = torch.tensor(
            [0.0, offset_scale * hy, 0.0], device=self.device, dtype=torch.float32
        )
        offset_left_local = torch.tensor(
            [0.0, -offset_scale * hy, 0.0], device=self.device, dtype=torch.float32
        )

        # Swap assignment so that initial "right" pushpoint is on -Y (inward from the right arm side).
        pushpoint_by_right = positions + torch.einsum(
            "bij,j->bi", rotations, offset_right_local
        )
        pushpoint_by_left = positions + torch.einsum(
            "bij,j->bi", rotations, offset_left_local
        )

        env_idx_long = env_idx.long()
        self.initial_forward_side_center[env_idx_long] = positions
        # Assign swapped so initial visual matches desired handedness.
        self.initial_pushpoint_by_right[env_idx_long] = pushpoint_by_left
        self.initial_pushpoint_by_left[env_idx_long] = pushpoint_by_right
        self.initial_box_center[env_idx_long] = positions

        rotations_T = rotations.transpose(1, 2)
        local_right = torch.einsum(
            "bij,bj->bi", rotations_T, pushpoint_by_left - positions
        )
        local_left = torch.einsum(
            "bij,bj->bi", rotations_T, pushpoint_by_right - positions
        )
        self.pushpoint_local_right[env_idx_long] = local_right
        self.pushpoint_local_left[env_idx_long] = local_left

        if getattr(self, "_enable_pushpoint_debug", False):
            if self.pushpoint_left_site is not None:
                self.pushpoint_left_site.set_pose(
                    Pose.create_from_pq(p=self.initial_pushpoint_by_left)
                )
            if self.pushpoint_right_site is not None:
                self.pushpoint_right_site.set_pose(
                    Pose.create_from_pq(p=self.initial_pushpoint_by_right)
                )

    def _compute_pushpoints_from_pose(
        self,
        current_box_center: torch.Tensor,
        rotation_current: torch.Tensor,
        theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute pushpoints without inversion: always +Y for right, -Y for left."""
        world_pushpoint_right = current_box_center + torch.einsum(
            "bij,bj->bi", rotation_current, self.pushpoint_local_right
        )
        world_pushpoint_left = current_box_center + torch.einsum(
            "bij,bj->bi", rotation_current, self.pushpoint_local_left
        )
        inversion_conditions = torch.zeros_like(theta, dtype=torch.bool)
        return (
            world_pushpoint_left.clone(),
            world_pushpoint_right.clone(),
            inversion_conditions,
            world_pushpoint_left,
            world_pushpoint_right,
        )

    def compute_normalized_dense_reward(self, obs, action, info):
        """Same reward as base, but removes tcp_lead term for sandwich pushing."""
        if not isinstance(self.agent, MultiAgent):
            return torch.zeros(self.num_envs, device=self.device)

        context = self._gather_reward_context(info)
        self._log_observation_to_info(info)

        pushpoint_score, push_info, pushpoint_stage_complete = self._pushpoint_tracking_reward(
            context.left_tcp_pos,
            context.right_tcp_pos,
            context.target_pushpoint_left,
            context.target_pushpoint_right,
        )
        translation_penalty_score, translation_info = self._box_translation_penalty(
            context.current_box_center
        )
        rotation_score, rotation_info = self._box_yaw_rotation(context.box_theta_deg)
        intrusion_penalty_score, intrusion_info = self._box_intrusion_penalty(
            context.current_box_center,
            context.left_tcp_pos,
            context.right_tcp_pos,
            context.box_rotation_matrix,
        )

        contact_penalty_score = self._compute_contact_penalty(info)

        reward_pushpoint = pushpoint_score

        positive_rotation = rotation_score > 0.0
        reward_rotation = torch.where(
            positive_rotation, rotation_score, torch.zeros_like(rotation_score)
        )
        penalty_inverse_rotation = torch.where(
            positive_rotation, torch.zeros_like(rotation_score), rotation_score
        )

        penalty_translation = -translation_penalty_score
        penalty_contact = -contact_penalty_score

        reward = reward_pushpoint

        stage_2_reward = 1.0 + reward_rotation
        mask = pushpoint_stage_complete.bool()
        reward[mask] = stage_2_reward[mask]

        reward = reward + penalty_contact + penalty_translation + penalty_inverse_rotation

        info["reward_pushpoint"] = reward_pushpoint.detach().cpu()
        info["reward_rotation"] = reward_rotation.detach().cpu()
        info["penalty_translation"] = penalty_translation.detach().cpu()
        info["penalty_contact"] = penalty_contact.detach().cpu()
        info["penalty_inverse_rotation"] = penalty_inverse_rotation.detach().cpu()
        info["rewards_t"] = reward.detach().cpu()

        info["pushpoint_left_distance"] = push_info["distance_left"].detach().cpu()
        info["pushpoint_left_reached"] = push_info["reached_left"].detach().cpu()
        info["pushpoint_right_distance"] = push_info["distance_right"].detach().cpu()
        info["pushpoint_right_reached"] = push_info["reached_right"].detach().cpu()
        info["pushpoint_stage_complete"] = push_info["pushpoint_stage_complete"].detach().cpu()
        info["pushpoint_left_world"] = context.target_pushpoint_left.detach().cpu()
        info["pushpoint_right_world"] = context.target_pushpoint_right.detach().cpu()
        info["box_translation_distance"] = translation_info["translation_distance"].detach().cpu()
        info["box_translation_penalty"] = translation_info["translation_penalty"].detach().cpu()
        info["box_rotation_delta"] = rotation_info["yaw_delta_deg"].detach().cpu()
        info["box_rotation_reward"] = rotation_info["yaw_rotation_reward"].detach().cpu()
        info["box_intrusion_penalty"] = intrusion_info["intrusion_penalty"].detach().cpu()
        info["box_intrusion_left"] = intrusion_info["intrusion_left"].detach().cpu()
        info["box_intrusion_right"] = intrusion_info["intrusion_right"].detach().cpu()
        info["box_rotation_progress"] = rotation_info["yaw_rotation_progress_deg"].detach().cpu()
        info["box_rotation_target_delta"] = rotation_info[
            "yaw_rotation_target_delta_deg"
        ].detach().cpu()
        info["box_intrusion_depth_left"] = intrusion_info["intrusion_depth_left"].detach().cpu()
        info["box_intrusion_depth_right"] = intrusion_info["intrusion_depth_right"].detach().cpu()

        try:
            left_qpos = self.agent.agents[0].robot.get_qpos()
            right_qpos = self.agent.agents[1].robot.get_qpos()
            left_idx = self._agent_obs_joint_indices.get(f"{self.agent.agents[0].uid}-0")
            right_idx = self._agent_obs_joint_indices.get(f"{self.agent.agents[1].uid}-1")
            if left_idx is not None:
                left_qpos = self._index_select_joint_tensor(left_qpos, left_idx)
            if right_idx is not None:
                right_qpos = self._index_select_joint_tensor(right_qpos, right_idx)
            info["mesured_q"] = torch.cat([left_qpos, right_qpos], dim=-1).detach().cpu()
        except Exception:
            pass

        if reward.ndim == 0:
            reward = reward.unsqueeze(0)
        return reward.to(self.device)
