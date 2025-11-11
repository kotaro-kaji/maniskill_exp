from typing import Any, Dict, Tuple
from dataclasses import dataclass

import torch
import sapien

from mani_skill.agents.multi_agent import MultiAgent

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.structs.pose import Pose
from robotagents.xarm_ball_ee import Xarm7BallEE


from scenebuilders.dual_xarm7_table_scene_builder import (
    DualXarm7TableSceneBuilder,
    PRIMARY_ARM_Y_OFFSET,
    SECONDARY_ARM_Y_OFFSET,
)
from scenebuilders.xarm7_table_scene_builder import (
    PEDESTAL_HEIGHT,
    ROBOT_BASE_X_OFFSET,
)


def smoothstep(x: torch.Tensor) -> torch.Tensor:
    """
    Smootherstep function (0 < x < 1)
    f(x) = 6x^5 - 15x^4 + 10x^3
    Properties:
      f(0)=0, f(1)=1
      f'(0)=f'(1)=0
      f''(x) changes sign at x=0.5
      smoother near endpoints than smoothstep
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    x = torch.clamp(x, 0.0, 1.0)
    return 6 * x**5 - 15 * x**4 + 10 * x**3


@register_env("MyDualBoxRotation-v0", max_episode_steps=200)
class MyDualBoxRotationEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("xarm7_ball_ee", "xarm7_ball_ee")]
    agent: MultiAgent[Tuple[Xarm7BallEE, Xarm7BallEE]]

    BOX_HALF_SIZE = (0.06, 0.09, 0.06)
    BOX_DENSITY = 200.0
    BOX_X_OFFSET_FROM_BASE = 0.43
    BOX_Y_JITTER = 0.05
    BOX_X_JITTER = 0.03
    DISTANCE_SCALE = 4.0
    PUSHPOINT_DISTANCE_SCALE = 5.0
    PUSHPOINT_DISTANCE_THRESHOLD = 0.05
    BOX_CENTER_MAX_OFFSET = 0.15
    BOX_CENTER_PENALTY_WEIGHT = 0.5
    MAX_ROTATION_PACE = 540.0 / 10.0  # degrees per second for full reward
    BOX_INTRUSION_MARGIN = 0.025
    BOX_INTRUSION_SCALE = 2.0
    TCP_LEAD_SATURATION = 0.03

    def __init__(self, *args, robot_uids=("xarm7_ball_ee", "xarm7_ball_ee"), robot_init_qpos_noise=0.02,**kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise #この引数は現在は未使用です。
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: Dict[str, Any]):
        super()._load_agent(options, [sapien.Pose(p=[0,-1,0]), sapien.Pose(p=[0,1,0])])
        #super()._load_agent(options, sapien.Pose[p=])

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.2, 1.0, 1.0], [0.0, 0.0, 0.4])
        return CameraConfig(
            "render_camera", pose=pose, width=800, height=800, fov=1.0, near=0.01, far=5.0
        )

    def _load_scene(self, options: Dict[str, Any]):
        self.table_scene = DualXarm7TableSceneBuilder(env=self)
        self.table_scene.build()
        self.bimanual_center_pose = self._compute_bimanual_center_pose()
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=self.BOX_HALF_SIZE,
            density=self.BOX_DENSITY,
        )
        self._enable_pushpoint_debug = False
        if True:
            self._enable_pushpoint_debug = True
        box_material = sapien.render.RenderMaterial()
        box_material.set_base_color([1.0, 1.0, 1.0, 1.0])
        builder.add_box_visual(
            half_size=self.BOX_HALF_SIZE,
            material=box_material,
        )
        # Initial pose will be overwritten during episode init; place safely above table for now.
        builder.initial_pose = sapien.Pose(
            p=self._bimanual_center_point_to_world(
                [
                    self.BOX_X_OFFSET_FROM_BASE,
                    0.0,
                    self.BOX_HALF_SIZE[2] * 2,
                ]
            )
        )
        self.box = builder.build(name="box_between_arms")
        if self._enable_pushpoint_debug:
            pushpoint_radius = 0.02
            self.pushpoint_left_site = actors.build_sphere(
                self.scene,
                radius=pushpoint_radius,
                color=(1.0, 0.0, 0.0, 1.0),
                name="pushpoint_left_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(
                    p=self._bimanual_center_point_to_world(
                        [0.0, 0.0, self.BOX_HALF_SIZE[2]]
                    )
                ),
            )
            self.pushpoint_right_site = actors.build_sphere(
                self.scene,
                radius=pushpoint_radius,
                color=(0.3, 0.3, 0.9, 0.8),
                name="pushpoint_right_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(
                    p=self._bimanual_center_point_to_world(
                        [0.0, 0.0, self.BOX_HALF_SIZE[2]]
                    )
                ),
            )
        else:
            self.pushpoint_left_site = None
            self.pushpoint_right_site = None

    def _ensure_pushpoint_buffers(self):
        num_envs = getattr(self, "num_envs", 1)
        needs_init = not hasattr(self, "initial_forward_side_center")
        if not needs_init:
            needs_init = self.initial_forward_side_center.shape[0] != num_envs
        if not needs_init:
            return
        zeros = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)
        self.initial_forward_side_center = zeros.clone()
        self.initial_pushpoint_by_right = zeros.clone()
        self.initial_pushpoint_by_left = zeros.clone()
        self.initial_box_center = zeros.clone()
        self.pushpoint_local_right = zeros.clone()
        self.pushpoint_local_left = zeros.clone()

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
        return True

    def _reset_rotation_buffers(
        self, env_idx: torch.Tensor, theta_deg: torch.Tensor
    ):
        self._ensure_rotation_buffers()
        if env_idx.numel() == 0:
            return
        env_idx_long = env_idx.long()
        theta_deg = theta_deg.to(device=self.device, dtype=torch.float32)
        self._prev_box_theta[env_idx_long] = theta_deg
        self._positive_yaw_progress[env_idx_long] = 0.0

    def _theta_to_quaternion(self, theta_deg: torch.Tensor) -> torch.Tensor:
        """Convert planar angle in degrees (around z) to quaternion."""
        if not isinstance(theta_deg, torch.Tensor):
            theta_deg = torch.tensor(theta_deg, device=self.device, dtype=torch.float32)
        else:
            theta_deg = theta_deg.to(device=self.device, dtype=torch.float32)
        theta_rad = theta_deg * (torch.pi / 180.0)
        half = 0.5 * theta_rad
        zeros = torch.zeros_like(theta_rad)
        w = torch.cos(half)
        z = torch.sin(half)
        return torch.stack((w, zeros, zeros, z), dim=-1)

    def _get_box_theta_deg(self) -> torch.Tensor:
        """Return box yaw in degrees within [0, 360)."""
        box_pose = Pose.create(self.box.pose, device=self.device)
        rotation = quaternion_to_matrix(box_pose.q)
        yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
        theta = torch.rad2deg(yaw)
        theta = torch.remainder(theta, 360.0)
        theta = torch.remainder(-theta, 360.0)
        return theta

    def _update_initial_pushpoints(
        self, env_idx: torch.Tensor, positions: torch.Tensor, orientations: torch.Tensor
    ):
        batch_size = positions.shape[0]
        if batch_size == 0:
            return
        rotations = quaternion_to_matrix(orientations)
        hx = float(self.BOX_HALF_SIZE[0])
        hy = float(self.BOX_HALF_SIZE[1])

        long_side_offsets_local = torch.tensor(
            [[hx, 0.0, 0.0], [-hx, 0.0, 0.0]],
            device=self.device,
            dtype=torch.float32,
        )
        long_side_offsets_local = long_side_offsets_local.transpose(0, 1)  # (3, 2)
        long_side_offsets_world = torch.matmul(
            rotations, long_side_offsets_local.unsqueeze(0).expand(batch_size, -1, -1)
        )  # (batch, 3, 2)
        long_side_offsets_world = long_side_offsets_world.transpose(1, 2)  # (batch, 2, 3)
        long_side_centers_world = positions.unsqueeze(1) + long_side_offsets_world
        mask = long_side_centers_world[:, 0, 0] >= long_side_centers_world[:, 1, 0]
        initial_forward_side_center = torch.where(
            mask.unsqueeze(-1),
            long_side_centers_world[:, 0, :],
            long_side_centers_world[:, 1, :],
        )

        push_offset_local = torch.tensor(
            [0.0, -0.8 * hy, 0.0],
            device=self.device,
            dtype=torch.float32,
        )
        push_offset_world = torch.einsum("bij,j->bi", rotations, push_offset_local)
        initial_pushpoint_by_right = initial_forward_side_center + push_offset_world
        initial_pushpoint_by_left = (
            2.0 * positions - initial_pushpoint_by_right
        )
        env_idx_long = env_idx.long()
        self.initial_forward_side_center[env_idx_long] = initial_forward_side_center
        self.initial_pushpoint_by_right[env_idx_long] = initial_pushpoint_by_right
        self.initial_pushpoint_by_left[env_idx_long] = initial_pushpoint_by_left
        self.initial_box_center[env_idx_long] = positions
        rotations_T = rotations.transpose(1, 2)
        local_right = torch.einsum(
            "bij,bj->bi", rotations_T, initial_pushpoint_by_right - positions
        )
        local_left = torch.einsum(
            "bij,bj->bi", rotations_T, initial_pushpoint_by_left - positions
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

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict[str, Any]):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            batch_size = len(env_idx)
            if batch_size == 0:
                return

            self._ensure_pushpoint_buffers()
            x_center = (
                self.BOX_X_OFFSET_FROM_BASE
                + (torch.rand(batch_size, device=self.device) - 0.5)
                * 2
                * self.BOX_X_JITTER
            )
            y_center = (
                (torch.rand(batch_size, device=self.device) - 0.5)
                * 2
                * self.BOX_Y_JITTER
            )
            z_center = torch.full(
                (batch_size,),
                self.BOX_HALF_SIZE[2],
                device=self.device,
                dtype=torch.float32,
            )
            positions = torch.stack((x_center, y_center, z_center), dim=-1)
            positions = self._bimanual_center_tensor_to_world(positions)
            theta = torch.zeros(
                batch_size, device=self.device, dtype=torch.float32
            )
            orientations = self._theta_to_quaternion(theta)
            self.box.set_pose(Pose.create_from_pq(positions, orientations))
            self._update_initial_pushpoints(env_idx, positions, orientations)
            self._reset_rotation_buffers(env_idx, theta)

    def _pose_to_6d(self, pose: Pose, *, center_frame: bool = False) -> torch.Tensor:
        matrix = pose.to_transformation_matrix()[..., :3, :3]
        position = pose.p
        if position.ndim == 1:
            position = position.unsqueeze(0)
            matrix = matrix.unsqueeze(0)
        if center_frame:
            center = torch.tensor(
                self.bimanual_center_pose.p,
                dtype=position.dtype,
                device=position.device,
            )
            position = position - center
        return torch.cat([position, matrix[..., :, 0], matrix[..., :, 1]], dim=-1)

    @dataclass
    class RewardContext:
        current_box_center: torch.Tensor
        target_pushpoint_left: torch.Tensor
        target_pushpoint_right: torch.Tensor
        left_tcp_pos: torch.Tensor
        right_tcp_pos: torch.Tensor
        box_theta_deg: torch.Tensor
        box_rotation_matrix: torch.Tensor

    def _gather_reward_context(
        self, info: Dict[str, Any]
    ) -> "MyDualBoxRotationEnv.RewardContext":
        theta = self._get_box_theta_deg()
        if theta.ndim == 0:
            theta = theta.unsqueeze(0)
        theta = theta.to(device=self.device, dtype=torch.float32)
        self._ensure_pushpoint_buffers()
        box_pose = Pose.create(self.box.pose, device=self.device)
        current_box_center = box_pose.p
        if current_box_center.ndim == 1:
            current_box_center = current_box_center.unsqueeze(0)
        rotation_current = quaternion_to_matrix(box_pose.q)
        world_pushpoint_right = current_box_center + torch.einsum(
            "bij,bj->bi", rotation_current, self.pushpoint_local_right
        )
        world_pushpoint_left = current_box_center + torch.einsum(
            "bij,bj->bi", rotation_current, self.pushpoint_local_left
        )
        normal_conditions = (theta >= 0.0) & (theta < 179.0) | (theta >= 359.0)
        inversion_conditions = ~normal_conditions 
        inversion_conditions_mask = inversion_conditions.unsqueeze(-1)
        current_pushpoint_by_right = torch.where(
            inversion_conditions_mask, world_pushpoint_left,  world_pushpoint_right
        )

        current_pushpoint_by_left = torch.where(
            inversion_conditions_mask, world_pushpoint_right, world_pushpoint_left
        )

        left_tcp_pos = Pose.create(
            self.agent.agents[1].tcp.pose, device=self.device
        ).p
        left_tcp_pos = (
            left_tcp_pos.squeeze(0)
            if left_tcp_pos.ndim == 2 and left_tcp_pos.shape[0] == 1
            else left_tcp_pos
        )
        right_tcp_pos = Pose.create(
            self.agent.agents[0].tcp.pose, device=self.device
        ).p
        right_tcp_pos = (
            right_tcp_pos.squeeze(0)
            if right_tcp_pos.ndim == 2 and right_tcp_pos.shape[0] == 1
            else right_tcp_pos
        )

        target_pushpoint_left = current_pushpoint_by_left.clone()
        target_pushpoint_right = current_pushpoint_by_right.clone()

        info["box_theta_deg"] = theta.detach().cpu()
        info["pushpoint_inversion_conditions"] = inversion_conditions.detach().cpu()
        info["initial_forward_side_center"] = (
            self.initial_forward_side_center.detach().cpu()
        )
        info["initial_pushpoint_by_right"] = (
            self.initial_pushpoint_by_right.detach().cpu()
        )
        info["initial_pushpoint_by_left"] = (
            self.initial_pushpoint_by_left.detach().cpu()
        )
        info["current_pushpoint_by_right"] = target_pushpoint_right.detach().cpu()
        info["current_pushpoint_by_left"] = target_pushpoint_left.detach().cpu()

        if getattr(self, "_enable_pushpoint_debug", False):
            if self.pushpoint_left_site is not None:
                self.pushpoint_left_site.set_pose(
                    Pose.create_from_pq(p=target_pushpoint_left)
                )
            if self.pushpoint_right_site is not None:
                self.pushpoint_right_site.set_pose(
                    Pose.create_from_pq(p=target_pushpoint_right)
                )

        return self.RewardContext(
            current_box_center=current_box_center,
            target_pushpoint_left=target_pushpoint_left,
            target_pushpoint_right=target_pushpoint_right,
            left_tcp_pos=left_tcp_pos,
            right_tcp_pos=right_tcp_pos,
            box_theta_deg=theta,
            box_rotation_matrix=rotation_current,
        )

    def _get_obs_extra(self, info: Dict[str, Any]):
        obs = {}
        if not isinstance(self.agent, MultiAgent):
            return obs

        def _ensure_batch(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0) if t.ndim == 1 else t

        left_tcp_pose = Pose.create(self.agent.agents[1].tcp.pose, device=self.device)
        right_tcp_pose = Pose.create(self.agent.agents[0].tcp.pose, device=self.device)
        obs["left_tcp_pose_6d_from_bimanual_center"] = self._pose_to_6d(
            left_tcp_pose, center_frame=True
        )
        obs["right_tcp_pose_6d_from_bimanual_center"] = self._pose_to_6d(
            right_tcp_pose, center_frame=True
        )

        self._ensure_pushpoint_buffers()
        box_pose = Pose.create(self.box.pose, device=self.device)
        obs["box_pose_6d_from_bimanual_center"] = self._pose_to_6d(
            box_pose, center_frame=True
        )

        box_center = box_pose.p
        box_center = _ensure_batch(box_center)
        rotation_current = quaternion_to_matrix(box_pose.q)
        world_pushpoint_right = box_center + torch.einsum(
            "bij,bj->bi", rotation_current, self.pushpoint_local_right
        )
        world_pushpoint_left = box_center + torch.einsum(
            "bij,bj->bi", rotation_current, self.pushpoint_local_left
        )
        center_vec = torch.tensor(
            self.bimanual_center_pose.p,
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

    def _compute_bimanual_center_pose(self) -> sapien.Pose:
        base_right = torch.tensor(
            [ROBOT_BASE_X_OFFSET, PRIMARY_ARM_Y_OFFSET, PEDESTAL_HEIGHT],
            dtype=torch.float32,
        )
        base_left = torch.tensor(
            [ROBOT_BASE_X_OFFSET, SECONDARY_ARM_Y_OFFSET, PEDESTAL_HEIGHT],
            dtype=torch.float32,
        )
        center = 0.5 * (base_right + base_left)
        return sapien.Pose(p=center.tolist())

    def _bimanual_center_point_to_world(self, point):
        center = self.bimanual_center_pose.p
        return [
            center[0] + point[0],
            center[1] + point[1],
            center[2] + point[2],
        ]

    def _bimanual_center_tensor_to_world(self, tensor: torch.Tensor) -> torch.Tensor:
        center = torch.tensor(
            self.bimanual_center_pose.p, dtype=tensor.dtype, device=tensor.device
        )
        if tensor.ndim == 1:
            return tensor + center
        if tensor.ndim == 2:
            return tensor + center.unsqueeze(0)
        raise ValueError("Expected tensor to have rank 1 or 2.")

    def _pushpoint_tracking_reward(
        self,
        left_tcp_pos: torch.Tensor,
        right_tcp_pos: torch.Tensor,
        target_pushpoint_left: torch.Tensor,
        target_pushpoint_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        left_tcp_pos = left_tcp_pos.to(device=self.device, dtype=torch.float32)
        right_tcp_pos = right_tcp_pos.to(device=self.device, dtype=torch.float32)
        target_pushpoint_left = target_pushpoint_left.to(
            device=self.device, dtype=torch.float32
        )
        target_pushpoint_right = target_pushpoint_right.to(
            device=self.device, dtype=torch.float32
        )
        if left_tcp_pos.ndim == 1:
            left_tcp_pos = left_tcp_pos.unsqueeze(0)
        if right_tcp_pos.ndim == 1:
            right_tcp_pos = right_tcp_pos.unsqueeze(0)
        distance_left = torch.linalg.norm(
            left_tcp_pos - target_pushpoint_left, dim=-1
        )
        distance_right = torch.linalg.norm(
            right_tcp_pos - target_pushpoint_right, dim=-1
        )
        base_reward_left = 1 - torch.tanh(
            self.PUSHPOINT_DISTANCE_SCALE * distance_left
        )
        reward_left = torch.where(
            distance_left < self.PUSHPOINT_DISTANCE_THRESHOLD,
            torch.ones_like(base_reward_left),
            base_reward_left,
        )
        base_reward_right = 1 - torch.tanh(
            self.PUSHPOINT_DISTANCE_SCALE * distance_right
        )
        reward_right = torch.where(
            distance_right < self.PUSHPOINT_DISTANCE_THRESHOLD,
            torch.ones_like(base_reward_right),
            base_reward_right,
        )
        reward = 0.5 * (reward_left + reward_right)
        reached_left = distance_left < self.PUSHPOINT_DISTANCE_THRESHOLD
        reached_right = distance_right < self.PUSHPOINT_DISTANCE_THRESHOLD
        pushpoint_stage_complete = (reached_left & reached_right).float()
        info = {
            "distance_left": distance_left,
            "distance_right": distance_right,
            "reached_left": reached_left,
            "reached_right": reached_right,
            "pushpoint_stage_complete": pushpoint_stage_complete,
        }
        return reward, info, pushpoint_stage_complete

    def _box_translation_penalty(
        self, current_box_center: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        translation_vector = current_box_center - self.initial_box_center
        translation_distance = torch.linalg.norm(translation_vector, dim=-1)
        normalized_distance = torch.clamp(
            translation_distance / self.BOX_CENTER_MAX_OFFSET, 0.0, 1.0
        )
        normalized_penalty = smoothstep(normalized_distance)
        translation_penalty = self.BOX_CENTER_PENALTY_WEIGHT * normalized_penalty
        info = {
            "translation_distance": translation_distance,
            "normalized_translation_distance": normalized_distance,
            "normalized_translation_penalty": normalized_penalty,
            "translation_penalty": translation_penalty,
        }
        return translation_penalty, info

    def _box_yaw_rotation(
        self, theta_deg: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        buffers_reset = self._ensure_rotation_buffers()
        theta_deg = theta_deg.to(device=self.device, dtype=torch.float32)
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
        target_delta_value = max(self.MAX_ROTATION_PACE * self.control_timestep, 1e-6)
        target_delta = torch.tensor(
            target_delta_value, device=self.device, dtype=torch.float32
        )
        step_reward = signed_delta / target_delta
        rotation_reward = torch.clamp(step_reward, min=-1.0, max=1.0)
        info = {
            "yaw_delta_deg": signed_delta,
            "yaw_rotation_reward": rotation_reward,
            "yaw_rotation_progress_deg": self._positive_yaw_progress,
            "yaw_rotation_target_delta_deg": signed_delta.new_full(
                signed_delta.shape, target_delta_value
            ),
        }
        return rotation_reward, info

    def _box_intrusion_penalty(
        self,
        current_box_center: torch.Tensor,
        left_tcp_pos: torch.Tensor,
        right_tcp_pos: torch.Tensor,
        rotation_matrix: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        half_sizes = torch.tensor(
            self.BOX_HALF_SIZE, device=self.device, dtype=torch.float32
        )
        inflated_half = half_sizes + self.BOX_INTRUSION_MARGIN

        def _intrusion(tcp_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            delta_world = tcp_pos - current_box_center
            delta_local = torch.einsum(
                "bij,bj->bi", rotation_matrix.transpose(1, 2), delta_world
            )
            delta_abs = torch.abs(delta_local)
            inside = torch.all(delta_abs <= inflated_half, dim=-1)
            penetration = torch.clamp(inflated_half - delta_abs, min=0.0)
            depth = torch.linalg.norm(penetration, dim=-1)
            return inside, depth

        inside_left, depth_left = _intrusion(left_tcp_pos)
        inside_right, depth_right = _intrusion(right_tcp_pos)
        penalty = self.BOX_INTRUSION_SCALE * (depth_left + depth_right)
        info = {
            "intrusion_left": inside_left,
            "intrusion_right": inside_right,
            "intrusion_depth_left": depth_left,
            "intrusion_depth_right": depth_right,
            "intrusion_penalty": penalty,
        }
        return penalty, info

    def _tcp_leading_reward(
        self,
        current_box_center: torch.Tensor,
        rotation_matrix: torch.Tensor,
        left_tcp_pos: torch.Tensor,
        right_tcp_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Reward when TCP passes the pushpoint along +/-X on the box surface."""
        self._ensure_pushpoint_buffers()

        def _ensure_batch(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0) if t.ndim == 1 else t

        current_box_center = _ensure_batch(current_box_center)
        left_tcp_pos = _ensure_batch(left_tcp_pos)
        right_tcp_pos = _ensure_batch(right_tcp_pos)
        if rotation_matrix.ndim == 2:
            rotation_matrix = rotation_matrix.unsqueeze(0)

        rot_T = rotation_matrix.transpose(1, 2)
        left_local = torch.einsum(
            "bij,bj->bi", rot_T, left_tcp_pos - current_box_center
        )
        right_local = torch.einsum(
            "bij,bj->bi", rot_T, right_tcp_pos - current_box_center
        )
        push_left_local = self.pushpoint_local_left.to(
            device=self.device, dtype=torch.float32
        )
        push_right_local = self.pushpoint_local_right.to(
            device=self.device, dtype=torch.float32
        )

        def _lead(push_local: torch.Tensor, tcp_local: torch.Tensor) -> torch.Tensor:
            push_x = push_local[..., 0]
            tcp_x = tcp_local[..., 0]
            direction = torch.sign(push_x)
            progress = direction * (tcp_x - push_x)
            progress = progress * direction.abs()
            progress = torch.clamp(progress, max=self.TCP_LEAD_SATURATION)
            scale = self.TCP_LEAD_SATURATION + 1e-8
            return torch.tanh((progress / scale)*0.35) #最高でも0.35点に

        lead_right = _lead(push_right_local, right_local)
        lead_left = _lead(push_left_local, left_local)
        reward = 0.5 * (lead_right + lead_left)
        info = {
            "tcp_lead_reward": reward,
            "tcp_lead_right_component": lead_right,
            "tcp_lead_left_component": lead_left,
        }
        return reward, info



    def compute_normalized_dense_reward(self, obs, action, info):

        if not isinstance(self.agent, MultiAgent):
            return torch.zeros(self.num_envs, device=self.device)

        context = self._gather_reward_context(info)


        pushpoint_reward, push_info, pushpoint_stage = self._pushpoint_tracking_reward(
            context.left_tcp_pos,
            context.right_tcp_pos,
            context.target_pushpoint_left,
            context.target_pushpoint_right,
        )
        translation_penalty, translation_info = self._box_translation_penalty(
            context.current_box_center
        )
        rotation_reward, rotation_info = self._box_yaw_rotation(context.box_theta_deg)
        intrusion_penalty, intrusion_info = self._box_intrusion_penalty(
            context.current_box_center,
            context.left_tcp_pos,
            context.right_tcp_pos,
            context.box_rotation_matrix,
        )
        tcp_lead_reward, tcp_lead_info = self._tcp_leading_reward(
            context.current_box_center,
            context.box_rotation_matrix,
            context.left_tcp_pos,
            context.right_tcp_pos,
        )

        reward_pushpoint = pushpoint_reward

        # 負の回転は常に抑制、正の回転はpushpoint_stageのときだけ（envごとに判定）
        positive_rotation = rotation_reward > 0.0
        reward_rotation = torch.where(
            positive_rotation, rotation_reward * pushpoint_stage, rotation_reward
        )
        reward_translation = -translation_penalty
        reward_intrusion = -intrusion_penalty * (1.0 - pushpoint_stage)
        reward_tcp_lead = tcp_lead_reward * (1.0 - pushpoint_stage)

        #腕を振り回すというだけの報酬も用意してみました。
        #reward = (torch.linalg.norm(self.agent.agents[0].tcp.get_linear_velocity(), dim = -1) + torch.linalg.norm(self.agent.agents[1].tcp.get_linear_velocity(), dim = -1))/20.0 
        
        #このreward_rotationとrotation_rewardは異なる。後者のほうが生の報酬です。
        reward = rotation_reward + reward_pushpoint
        #reward = reward_pushpoint + reward_rotation + reward_translation + reward_tcp_lead
        # reward_intrusion is tracked in info but currently excluded from the final sum.

        info["reward_pushpoint"] = reward_pushpoint.detach().cpu()
        info["reward_rotation"] = reward_rotation.detach().cpu()
        info["reward_translation"] = reward_translation.detach().cpu()
        info["reward_intrusion"] = reward_intrusion.detach().cpu()
        info["reward_tcp_lead"] = reward_tcp_lead.detach().cpu()
        info["rewards_t"] = reward.detach().cpu()


        info["pushpoint_left_distance"] = push_info["distance_left"].detach().cpu()
        info["pushpoint_left_reached"] = push_info["reached_left"].detach().cpu()
        info["pushpoint_right_distance"] = push_info["distance_right"].detach().cpu()
        info["pushpoint_right_reached"] = push_info["reached_right"].detach().cpu()
        info["pushpoint_stage_complete"] = push_info[
            "pushpoint_stage_complete"
        ].detach().cpu()
        info["pushpoint_left_world"] = context.target_pushpoint_left.detach().cpu()
        info["pushpoint_right_world"] = context.target_pushpoint_right.detach().cpu()
        info["box_translation_distance"] = translation_info[
            "translation_distance"
        ].detach().cpu()
        info["box_translation_penalty"] = translation_info[
            "translation_penalty"
        ].detach().cpu()
        info["box_rotation_delta"] = rotation_info["yaw_delta_deg"].detach().cpu()
        info["box_rotation_reward"] = rotation_info[
            "yaw_rotation_reward"
        ].detach().cpu()
        info["box_intrusion_penalty"] = intrusion_info["intrusion_penalty"].detach().cpu()
        info["box_intrusion_left"] = intrusion_info["intrusion_left"].detach().cpu()
        info["box_intrusion_right"] = intrusion_info["intrusion_right"].detach().cpu()
        info["box_rotation_progress"] = rotation_info[
            "yaw_rotation_progress_deg"
        ].detach().cpu()
        info["box_rotation_target_delta"] = rotation_info[
            "yaw_rotation_target_delta_deg"
        ].detach().cpu()
        info["tcp_lead_reward"] = tcp_lead_info["tcp_lead_reward"].detach().cpu()
        info["tcp_lead_right_component"] = tcp_lead_info[
            "tcp_lead_right_component"
        ].detach().cpu()
        info["tcp_lead_left_component"] = tcp_lead_info[
            "tcp_lead_left_component"
        ].detach().cpu()
        info["box_intrusion_depth_left"] = intrusion_info[
            "intrusion_depth_left"
        ].detach().cpu()
        info["box_intrusion_depth_right"] = intrusion_info[
            "intrusion_depth_right"
        ].detach().cpu()
        if reward.ndim == 0:
            reward = reward.unsqueeze(0)
        return reward.to(self.device)
