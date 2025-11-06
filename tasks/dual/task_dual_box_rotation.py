from typing import Any, Dict, Tuple

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


from scenebuilders.dual_xarm7_table_scene_builder import DualXarm7TableSceneBuilder
from scenebuilders.xarm7_table_scene_builder import ROBOT_BASE_X_OFFSET


@register_env("MyDualBoxRotation-v0", max_episode_steps=200)
class MyDualBoxRotationEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("xarm7_ball_ee", "xarm7_ball_ee")]
    agent: MultiAgent[Tuple[Xarm7BallEE, Xarm7BallEE]]

    BOX_HALF_SIZE = (0.06, 0.09, 0.06)
    BOX_DENSITY = 200.0
    BOX_X_OFFSET_FROM_BASE = 0.43
    BOX_Y_JITTER = 0.05
    BOX_X_JITTER = 0.03
    LEFT_TARGET_POS = (0.5, 0.3, 0.5)
    RIGHT_TARGET_POS = (0.5, -0.3, 0.5)
    DISTANCE_SCALE = 4.0
    PUSHPOINT_DISTANCE_SCALE = 5.0
    PUSHPOINT_DISTANCE_THRESHOLD = 0.05

    def __init__(self, *args, robot_uids=("xarm7_ball_ee", "xarm7_ball_ee"), robot_init_qpos_noise=0.02,**kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
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
            [
                ROBOT_BASE_X_OFFSET + self.BOX_X_OFFSET_FROM_BASE,
                0.0,
                self.BOX_HALF_SIZE[2] * 2,
            ]
        )
        self.box = builder.build(name="box_between_arms")
        if self._enable_pushpoint_debug:
            pushpoint_radius = 0.012
            self.pushpoint_left_site = actors.build_sphere(
                self.scene,
                radius=pushpoint_radius,
                color=(0.9, 0.3, 0.3, 0.8),
                name="pushpoint_left_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(p=[0.0, 0.0, self.BOX_HALF_SIZE[2]]),
            )
            self.pushpoint_right_site = actors.build_sphere(
                self.scene,
                radius=pushpoint_radius,
                color=(0.3, 0.3, 0.9, 0.8),
                name="pushpoint_right_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(p=[0.0, 0.0, self.BOX_HALF_SIZE[2]]),
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
        return theta

    def _update_initial_pushpoints(
        self, env_idx: torch.Tensor, positions: torch.Tensor, orientations: torch.Tensor
    ):
        batch_size = positions.shape[0]
        if batch_size == 0:
            return
        rotations = quaternion_to_matrix(orientations)
        half_sizes = torch.tensor(
            self.BOX_HALF_SIZE, device=self.device, dtype=torch.float32
        )
        hy = half_sizes[1]
        side_offsets_local = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
            device=self.device,
            dtype=torch.float32,
        ) * hy
        side_offsets_local = side_offsets_local.transpose(0, 1)  # (3, 2)
        side_offsets_world = torch.matmul(
            rotations, side_offsets_local.unsqueeze(0).expand(batch_size, -1, -1)
        )  # (batch, 3, 2)
        side_offsets_world = side_offsets_world.transpose(1, 2)  # (batch, 2, 3)
        side_centers_world = positions.unsqueeze(1) + side_offsets_world
        mask = side_centers_world[:, 0, 0] >= side_centers_world[:, 1, 0]
        initial_forward_side_center = torch.where(
            mask.unsqueeze(-1), side_centers_world[:, 0, :], side_centers_world[:, 1, :]
        )

        push_offset_local = torch.tensor(
            [0.0, -0.8, 0.0],
            device=self.device,
            dtype=torch.float32,
        ) * hy
        push_offset_world = torch.einsum("bij,j->bi", rotations, push_offset_local)
        initial_pushpoint_by_right = initial_forward_side_center + push_offset_world
        initial_pushpoint_by_left = (
            2.0 * positions - initial_pushpoint_by_right
        )
        env_idx_long = env_idx.long()
        self.initial_forward_side_center[env_idx_long] = initial_forward_side_center
        self.initial_pushpoint_by_right[env_idx_long] = initial_pushpoint_by_right
        self.initial_pushpoint_by_left[env_idx_long] = initial_pushpoint_by_left
        if getattr(self, "_enable_pushpoint_debug", False):
            left_vis = self.initial_pushpoint_by_left.clone()
            right_vis = self.initial_pushpoint_by_right.clone()
            left_vis[..., 2] = self.BOX_HALF_SIZE[2]
            right_vis[..., 2] = self.BOX_HALF_SIZE[2]
            if self.pushpoint_left_site is not None:
                self.pushpoint_left_site.set_pose(Pose.create_from_pq(p=left_vis))
            if self.pushpoint_right_site is not None:
                self.pushpoint_right_site.set_pose(Pose.create_from_pq(p=right_vis))

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict[str, Any]):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            batch_size = len(env_idx)
            if batch_size == 0:
                return

            self._ensure_pushpoint_buffers()
            x_center = (
                ROBOT_BASE_X_OFFSET
                + self.BOX_X_OFFSET_FROM_BASE
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
            theta = torch.zeros(
                batch_size, device=self.device, dtype=torch.float32
            )
            orientations = self._theta_to_quaternion(theta)
            self.box.set_pose(Pose.create_from_pq(positions, orientations))
            self._update_initial_pushpoints(env_idx, positions, orientations)

    def compute_normalized_dense_reward(self, obs, action, info):

        if not isinstance(self.agent, MultiAgent):
            return torch.zeros(self.num_envs, device=self.device)

        theta = self._get_box_theta_deg()
        if theta.ndim == 0:
            theta = theta.unsqueeze(0)
        self._ensure_pushpoint_buffers()
        use_initial_mask = (theta >= 0.0) & (theta < 180.0)
        current_pushpoint_by_right = torch.where(
            use_initial_mask.unsqueeze(-1),
            self.initial_pushpoint_by_right,
            self.initial_pushpoint_by_left,
        )
        current_pushpoint_by_left = torch.where(
            use_initial_mask.unsqueeze(-1),
            self.initial_pushpoint_by_left,
            self.initial_pushpoint_by_right,
        )

        left_tcp_pos = Pose.create(
            self.agent.agents[1].tcp.pose, device=self.device
        ).p
        left_tcp_pos = left_tcp_pos.squeeze(0) if left_tcp_pos.ndim == 2 and left_tcp_pos.shape[0] == 1 else left_tcp_pos

        target_pushpoint_left = current_pushpoint_by_left.clone()
        target_pushpoint_left[..., 2] = self.BOX_HALF_SIZE[2]
        target_pushpoint_right = current_pushpoint_by_right.clone()
        target_pushpoint_right[..., 2] = self.BOX_HALF_SIZE[2]

        left_tcp_pos = left_tcp_pos.to(device=self.device, dtype=torch.float32)
        target_pushpoint_left = target_pushpoint_left.to(
            device=self.device, dtype=torch.float32
        )
        target_pushpoint_right = target_pushpoint_right.to(
            device=self.device, dtype=torch.float32
        )

        if left_tcp_pos.ndim == 1:
            left_tcp_pos = left_tcp_pos.unsqueeze(0)

        distance_left = torch.linalg.norm(
            left_tcp_pos - target_pushpoint_left, dim=-1
        )
        base_reward = 1 - torch.tanh(self.PUSHPOINT_DISTANCE_SCALE * distance_left)
        reward = torch.where(
            distance_left < self.PUSHPOINT_DISTANCE_THRESHOLD,
            torch.ones_like(base_reward),
            base_reward,
        )
        reached = distance_left < self.PUSHPOINT_DISTANCE_THRESHOLD

        info["box_theta_deg"] = theta.detach().cpu()
        info["pushpoint_use_initial_mask"] = use_initial_mask.detach().cpu()
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
        info["pushpoint_left_distance"] = distance_left.detach().cpu()
        info["pushpoint_left_reached"] = reached.detach().cpu()
        if getattr(self, "_enable_pushpoint_debug", False):
            if self.pushpoint_left_site is not None:
                self.pushpoint_left_site.set_pose(
                    Pose.create_from_pq(p=target_pushpoint_left)
                )
            if self.pushpoint_right_site is not None:
                self.pushpoint_right_site.set_pose(
                    Pose.create_from_pq(p=target_pushpoint_right)
                )

        if reward.ndim == 0:
            reward = reward.unsqueeze(0)
        return reward.to(self.device)
