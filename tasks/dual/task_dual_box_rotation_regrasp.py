"""
Regrasp-oriented variant of the dual box rotation task.
Extends the base environment to:
- Use a tightened normal/inversion range: normal when 0 <= yaw < 145 or yaw >= 315.
- Track previous normal state (step 0 uses the previous state), inversion flips, and signed positive flip count.
- Switch the yaw goal dynamically: 145 deg when normal, otherwise 315 deg.
Reward logic is inherited from the base environment.
"""

from typing import Any, Dict, Tuple

import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.utils.registration import register_env

from .task_dual_box_rotation import MyDualBoxRotationEnv


@register_env("MyDualBoxRotationRegrasp-v0", max_episode_steps=200)
class MyDualBoxRotationRegraspEnv(MyDualBoxRotationEnv):
    """Specialized dual box rotation env with inversion tracking and dynamic goals."""

    def _ensure_rotation_buffers(self) -> bool:
        num_envs = getattr(self, "num_envs", 1)
        needs_init = not hasattr(self, "_prev_box_theta")
        if not needs_init:
            needs_init = self._prev_box_theta.shape[0] != num_envs
        if not needs_init:
            return False
        zeros = torch.zeros((num_envs,), device=self.device, dtype=torch.float32)
        self._prev_box_theta = zeros.clone()
        self._positive_yaw_progress = zeros.clone()
        self._prev_normal_conditions = torch.zeros(
            (num_envs,), device=self.device, dtype=torch.bool
        )
        self._rotation_step_count = torch.zeros(
            (num_envs,), device=self.device, dtype=torch.long
        )
        self._inversion_count = torch.zeros(
            (num_envs,), device=self.device, dtype=torch.float32
        )
        self._positive_flip_count = torch.zeros(
            (num_envs,), device=self.device, dtype=torch.float32
        )
        self._prev_pushpoint_complete = torch.zeros(
            (num_envs,), device=self.device, dtype=torch.bool
        )
        return True

    def _reset_rotation_buffers(
        self, env_idx: torch.Tensor, theta_deg: torch.Tensor
    ):
        self._ensure_rotation_buffers()
        if env_idx.numel() == 0:
            return
        env_idx_long = env_idx.long()
        # Align previous theta with the actual box yaw at reset to prevent a large first-step delta.
        current_theta = self._get_box_theta_deg().to(
            device=self.device, dtype=torch.float32
        )
        if current_theta.ndim == 0:
            current_theta = current_theta.unsqueeze(0)
        self._prev_box_theta[env_idx_long] = current_theta[env_idx_long]
        self._positive_yaw_progress[env_idx_long] = 0.0
        # Keep normal/inversion tracking consistent with the normalized yaw we store.
        raw_normal = self._compute_raw_normal_conditions(current_theta)
        self._prev_normal_conditions[env_idx_long] = raw_normal[env_idx_long]
        self._rotation_step_count[env_idx_long] = 0
        self._inversion_count[env_idx_long] = 0.0
        self._positive_flip_count[env_idx_long] = 0.0
        self._prev_pushpoint_complete[env_idx_long] = False

    def _compute_raw_normal_conditions(self, theta: torch.Tensor) -> torch.Tensor:
        theta = theta.to(device=self.device, dtype=torch.float32)
        return ((theta >= 0.0) & (theta < 125.0)) | (theta >= 305.0)

    def _get_normal_conditions(
        self, theta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_rotation_buffers()
        raw_normal = self._compute_raw_normal_conditions(theta)
        step_zero_mask = self._rotation_step_count == 0
        prev_normal = self._prev_normal_conditions
        normal_conditions = torch.where(step_zero_mask, prev_normal, raw_normal)
        return normal_conditions, raw_normal

    def _compute_pushpoints_from_pose(
        self,
        current_box_center: torch.Tensor,
        rotation_current: torch.Tensor,
        theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        world_pushpoint_right = current_box_center + torch.einsum(
            "bij,bj->bi", rotation_current, self.pushpoint_local_right
        )
        world_pushpoint_left = current_box_center + torch.einsum(
            "bij,bj->bi", rotation_current, self.pushpoint_local_left
        )
        normal_conditions, _ = self._get_normal_conditions(theta)
        inversion_conditions = ~normal_conditions
        inversion_conditions_mask = inversion_conditions.unsqueeze(-1)
        target_pushpoint_by_right = torch.where(
            inversion_conditions_mask, world_pushpoint_left, world_pushpoint_right
        )
        target_pushpoint_by_left = torch.where(
            inversion_conditions_mask, world_pushpoint_right, world_pushpoint_left
        )
        return (
            target_pushpoint_by_left.clone(),
            target_pushpoint_by_right.clone(),
            inversion_conditions,
            world_pushpoint_left,
            world_pushpoint_right,
        )

    def _box_yaw_rotation(
        self, theta_deg: torch.Tensor, pushpoint_complete: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        buffers_reset = self._ensure_rotation_buffers()
        theta_deg = theta_deg.to(device=self.device, dtype=torch.float32)
        pushpoint_complete = pushpoint_complete.to(device=self.device)
        if buffers_reset:
            self._prev_box_theta = theta_deg.clone()
            delta = torch.zeros_like(theta_deg)
        else:
            prev_theta = self._prev_box_theta
            delta = torch.remainder(theta_deg - prev_theta + 180.0, 360.0) - 180.0
        signed_delta = delta
        self._prev_box_theta = theta_deg
        self._positive_yaw_progress = torch.clamp(
            self._positive_yaw_progress + signed_delta, min=0.0
        )

        normal_conditions, raw_normal_conditions = self._get_normal_conditions(theta_deg)
        prev_normal_conditions = self._prev_normal_conditions
        flipped = normal_conditions != prev_normal_conditions
        delta_sign = torch.sign(signed_delta)
        zeros_like_delta = torch.zeros_like(signed_delta)
        prev_pp_complete = self._prev_pushpoint_complete
        flipped_and_positive = flipped & (delta_sign > 0) & prev_pp_complete
        flipped_and_negative = flipped & (delta_sign < 0)
        pos_increment = torch.where(
            flipped_and_positive,
            torch.ones_like(signed_delta),
            zeros_like_delta,
        )
        neg_decrement = torch.where(
            flipped_and_negative,
            torch.ones_like(signed_delta),
            zeros_like_delta,
        )
        inversion_increment = flipped.float()
        self._positive_flip_count = torch.clamp(
            self._positive_flip_count + pos_increment - neg_decrement, min=0.0
        )
        self._inversion_count = self._inversion_count + inversion_increment
        self._prev_normal_conditions = normal_conditions
        self._rotation_step_count = self._rotation_step_count + 1
        self._prev_pushpoint_complete = pushpoint_complete.bool()

        target_delta_value = max(self.MAX_ROTATION_PACE * self.control_timestep, 1e-6)
        target_delta = torch.tensor(
            target_delta_value, device=self.device, dtype=torch.float32
        )
        step_score = signed_delta / target_delta
        step_score = torch.clamp(step_score, min=-1.0, max=1.0)
        delta_score = 2.0 * step_score  # range [-2, 2]

        goal_theta = torch.where(
            normal_conditions,
            torch.tensor(125.0, device=self.device, dtype=torch.float32),
            torch.tensor(305.0, device=self.device, dtype=torch.float32),
        )
        angle_diff_rad = torch.deg2rad(goal_theta - theta_deg)
        alignment_reward = 1.0 + torch.cos(angle_diff_rad)  # max = 2.0, min = 0.0
        info = {
            "yaw_delta_deg": signed_delta,
            "yaw_rotation_reward": alignment_reward,
            "yaw_step_score": delta_score,
            "yaw_rotation_progress_deg": self._positive_yaw_progress,
            "yaw_rotation_target_delta_deg": signed_delta.new_full(
                signed_delta.shape, target_delta_value
            ),
            "normal_conditions": normal_conditions.detach().cpu(),
            "previous_normal_conditions": prev_normal_conditions.detach().cpu(),
            "raw_normal_conditions": raw_normal_conditions.detach().cpu(),
            "normal_conditions_flipped": flipped.detach().cpu(),
            "inversion_count": self._inversion_count.detach().cpu(),
            "positive_flip_count": self._positive_flip_count.detach().cpu(),
            "yaw_goal_deg": goal_theta.detach().cpu(),
        }
        return delta_score, info

    def compute_normalized_dense_reward(self, obs, action, info):
        """
        Reward override: copied from base for easy customization, but now uses the
        subclass yaw/goal/inversion tracking logic.
        """
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
        box_min_x_penalty_score, box_min_x_info = self._box_min_x_penalty(
            context.current_box_center
        )
        rotation_step_score, rotation_info = self._box_yaw_rotation(context.box_theta_deg, pushpoint_stage_complete)
        tcp_lead_score, tcp_lead_info = self._tcp_leading_reward(
            context.current_box_center,
            context.box_rotation_matrix,
            context.left_tcp_pos,
            context.right_tcp_pos,
            context.target_pushpoint_left,
            context.target_pushpoint_right,
        )
        tcp_height_penalty_score, tcp_height_info = self._tcp_height_penalty(
            context.current_box_center,
            context.left_tcp_pos,
            context.right_tcp_pos,
        )

        contact_penalty_score = self._compute_contact_penalty(info)  # non-negative penalty

        reward_pushpoint = pushpoint_score  # min = 0.0, max = 1.0

        mask = pushpoint_stage_complete.bool()
        rotation_alignment_reward = rotation_info["yaw_rotation_reward"]
        reward_rotation = torch.where(
            mask, rotation_alignment_reward, torch.zeros_like(rotation_alignment_reward)
        )  # min = -3.0, max = 3.0 (only after pushpoint stage)

        positive_rotation = rotation_step_score > 0.0
        penalty_inverse_rotation = torch.where(
            positive_rotation, torch.zeros_like(rotation_step_score), rotation_step_score
        )  # min = -2.0, max = 0.0
        penalty_forward_rotation_stage1 = torch.where(
            mask, torch.zeros_like(rotation_step_score), -torch.clamp(rotation_step_score, min=0.0)
        )  # min = -2.0, max = 0.0 (only before pushpoints reached)
        reward_tcp_lead = tcp_lead_score  # min = -TCP_LEAD_MAX, max = TCP_LEAD_MAX

        penalty_translation = -translation_penalty_score  # min = -0.1 (scale=0.25), max = 0.0
        penalty_contact = -contact_penalty_score  # min = -1.0, max = 0.0
        penalty_tcp_height = -tcp_height_penalty_score  # min = -TCP_HEIGHT_PENALTY, max = 0.0
        penalty_box_min_x = -box_min_x_penalty_score  # min = -1.0, max = 0.0

        # Stage 1: move tcp to pushpoint
        reward = reward_pushpoint + reward_tcp_lead  # min = -1.0, max = 1 + TCP_LEAD_MAX

        # Stage 2: rotate box to pushpoint
        stage_2_reward = reward_pushpoint + self.TCP_LEAD_MAX + 1e-2 + reward_rotation
        reward[mask] = stage_2_reward[mask]  # approx range: (1 + TCP_LEAD_MAX - 0.0, 1 + TCP_LEAD_MAX + 2.0)

        #_positive_inversion_count
        reward = reward + self._positive_flip_count*(1.0+2.0+self.TCP_LEAD_MAX)
        
        # Add a constant penalty that is independent of the stage.
        reward = (
            reward
            + penalty_contact
            + penalty_translation
            + penalty_inverse_rotation
            + penalty_forward_rotation_stage1
            + penalty_tcp_height
            + penalty_box_min_x
        )

        info["reward_pushpoint"] = reward_pushpoint.detach().cpu()
        info["reward_rotation"] = reward_rotation.detach().cpu()
        info["penalty_translation"] = penalty_translation.detach().cpu()
        info["penalty_contact"] = penalty_contact.detach().cpu()
        info["penalty_inverse_rotation"] = penalty_inverse_rotation.detach().cpu()
        info["penalty_forward_rotation_stage1"] = penalty_forward_rotation_stage1.detach().cpu()
        info["penalty_tcp_height"] = penalty_tcp_height.detach().cpu()
        info["penalty_box_min_x"] = penalty_box_min_x.detach().cpu()
        info["box_min_x_penalty_norm"] = box_min_x_info["box_min_x_penalty"].detach().cpu()
        info["box_min_x_reference"] = box_min_x_info["box_min_x_reference"].detach().cpu()
        info["box_min_x_span"] = box_min_x_info["box_min_x_span"].detach().cpu()
        info["reward_tcp_lead"] = reward_tcp_lead.detach().cpu()
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
        info["box_rotation_step_score"] = rotation_info["yaw_step_score"].detach().cpu()
        info["box_rotation_progress"] = rotation_info["yaw_rotation_progress_deg"].detach().cpu()
        info["box_rotation_target_delta"] = rotation_info["yaw_rotation_target_delta_deg"].detach().cpu()
        info["positive_flip_count"] = rotation_info["positive_flip_count"].detach().cpu()
        info["tcp_lead_reward"] = tcp_lead_info["tcp_lead_reward"].detach().cpu()
        info["tcp_lead_right_component"] = tcp_lead_info["tcp_lead_right_component"].detach().cpu()
        info["tcp_lead_left_component"] = tcp_lead_info["tcp_lead_left_component"].detach().cpu()
        info["tcp_height_violation_left"] = tcp_height_info["tcp_height_violation_left"].detach().cpu()
        info["tcp_height_violation_right"] = tcp_height_info["tcp_height_violation_right"].detach().cpu()

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
