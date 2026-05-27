from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from robotagents.my_xarm7 import Xarm7
from scenebuilders.cardboard_cabinet_builder import (
    DEFAULT_CARDBOARD_CABINET_SPEC,
    build_cardboard_cabinet_actor,
    build_cardboard_inner_box_actor,
    cardboard_cabinet_quaternion,
    make_cardboard_inner_box_spec,
)
from scenebuilders.dual_xarm7_table_scene_builder import (
    DualXarm7TableSceneBuilder,
    LEFT_ARM_Y_OFFSET,
    RIGHT_ARM_Y_OFFSET,
)
from scenebuilders.xarm7_table_scene_builder import (
    PEDESTAL_HEIGHT,
    ROBOT_BASE_X_OFFSET,
)


@register_env("MyDualCardboardCabinet-v1", max_episode_steps=200)
class MyDualCardboardCabinetEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("my_xarm7", "my_xarm7")]
    agent: MultiAgent[Tuple[Xarm7, Xarm7]]

    OUTER_CARDBOARD_BASE_SPEC = DEFAULT_CARDBOARD_CABINET_SPEC
    CABINET_SPEC = replace(OUTER_CARDBOARD_BASE_SPEC, color_hex="#4C78A8")
    INNER_BOX_SPEC = replace(
        make_cardboard_inner_box_spec(OUTER_CARDBOARD_BASE_SPEC),
        wall_thickness=0.002,
        density=6000.0,
        static_friction=0.2,
        dynamic_friction=0.1,
    )
    INNER_BOX_WORLD_Y_OFFSET = 0.0035
    BOX_X_OFFSET_FROM_BASE = 0.30
    INSERT_TARGET_RADIUS = 0.008
    INSERT_TARGET_COLOR = (0.5, 1.0, 0.0, 1.0)
    INSERT_TARGET_SIDE_LOCAL = (0.0, 0.0, 0.00925)
    TCP_TO_TARGET_REWARD_SCALE = 5.0
    GRIPPER_OPENING_TARGET_QPOS = 0.44
    GRIPPER_OPENING_REWARD_WEIGHT = 0.05
    GRIPPER_OPENING_REWARD_SCALE = 8.0

    def __init__(
        self,
        *args,
        robot_uids=("my_xarm7", "my_xarm7"),
        robot_init_qpos_noise=0.02,
        robot_init_noise_scale: float = 1.0,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.robot_init_noise_scale = robot_init_noise_scale
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: Dict[str, Any]):
        super()._load_agent(
            options,
            [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])],
        )

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.2, 1.0, 0.9], [0.0, 0.0, 0.22])
        return CameraConfig(
            "render_camera",
            pose=pose,
            width=960,
            height=960,
            fov=1.0,
            near=0.01,
            far=5.0,
        )

    def _load_scene(self, options: Dict[str, Any]):
        self.table_scene = DualXarm7TableSceneBuilder(env=self)
        self.table_scene.build()
        self.bimanual_center_pose = self._compute_bimanual_center_pose()

        self.cardboard_cabinet = build_cardboard_cabinet_actor(
            self.scene,
            initial_pose=self._initial_cabinet_pose(),
            spec=self.CABINET_SPEC,
        )
        self.cardboard_inner_box = build_cardboard_inner_box_actor(
            self.scene,
            initial_pose=self._initial_inner_box_pose(),
            spec=self.INNER_BOX_SPEC,
        )
        self.insert_target_site = actors.build_sphere(
            self.scene,
            radius=self.INSERT_TARGET_RADIUS,
            color=self.INSERT_TARGET_COLOR,
            name="insert_target_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict[str, Any]):
        with torch.device(self.device):
            noise_scale = self.robot_init_noise_scale
            if options is not None:
                noise_scale = float(options.get("robot_init_noise_scale", noise_scale))
            if hasattr(self.table_scene, "set_noise_scale"):
                self.table_scene.set_noise_scale(noise_scale)
            self.table_scene.initialize(env_idx)
            if len(env_idx) == 0:
                return

            positions = self._initial_box_world_position().unsqueeze(0).repeat(
                len(env_idx), 1
            )
            orientations = cardboard_cabinet_quaternion(
                self.CABINET_SPEC,
                device=self.device,
            ).unsqueeze(0).repeat(len(env_idx), 1)
            self.cardboard_cabinet.set_pose(Pose.create_from_pq(positions, orientations))
            self.cardboard_inner_box.set_pose(
                Pose.create_from_pq(
                    self._initial_inner_box_world_position().unsqueeze(0).repeat(
                        len(env_idx), 1
                    ),
                    orientations,
                )
            )
            self._sync_target_sites(env_idx)

    def _compute_bimanual_center_pose(self) -> sapien.Pose:
        base_left = torch.tensor(
            [ROBOT_BASE_X_OFFSET, LEFT_ARM_Y_OFFSET, PEDESTAL_HEIGHT],
            dtype=torch.float32,
        )
        base_right = torch.tensor(
            [ROBOT_BASE_X_OFFSET, RIGHT_ARM_Y_OFFSET, PEDESTAL_HEIGHT],
            dtype=torch.float32,
        )
        center = 0.5 * (base_right + base_left)
        return sapien.Pose(p=center.tolist())

    def _initial_box_world_position(self) -> torch.Tensor:
        local = torch.tensor(
            [
                self.BOX_X_OFFSET_FROM_BASE,
                0.0,
                self.CABINET_SPEC.outer_height_z / 2 - PEDESTAL_HEIGHT,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        center = torch.tensor(
            self.bimanual_center_pose.p, dtype=torch.float32, device=self.device
        )
        return local + center

    def _initial_cabinet_pose(self) -> sapien.Pose:
        return sapien.Pose(
            p=self._initial_box_world_position().detach().cpu().tolist(),
            q=cardboard_cabinet_quaternion(self.CABINET_SPEC).tolist(),
        )

    def _initial_inner_box_pose(self) -> sapien.Pose:
        return sapien.Pose(
            p=self._initial_inner_box_world_position().detach().cpu().tolist(),
            q=cardboard_cabinet_quaternion(self.CABINET_SPEC).tolist(),
        )

    def _initial_inner_box_world_position(self) -> torch.Tensor:
        position = self._initial_box_world_position().clone()
        position[1] += self.INNER_BOX_WORLD_Y_OFFSET
        return position

    def _insert_side_origin_box_local(self) -> torch.Tensor:
        spec = self.INNER_BOX_SPEC
        return torch.tensor(
            [
                -spec.outer_length_x / 2 + spec.wall_thickness / 2,
                0.0,
                0.0,
            ],
            dtype=torch.float32,
            device=self.device,
        )

    def _insert_target_side_local(self) -> torch.Tensor:
        return torch.tensor(
            self.INSERT_TARGET_SIDE_LOCAL,
            dtype=torch.float32,
            device=self.device,
        )

    def _box_insert_target_local(self) -> torch.Tensor:
        return self._insert_side_origin_box_local() + self._insert_target_side_local()

    def _box_local_point_world(self, local_point: torch.Tensor) -> torch.Tensor:
        pose = self.cardboard_inner_box.pose
        matrix = pose.to_transformation_matrix()[..., :3, :3]
        position = pose.p
        if position.ndim == 1:
            position = position.unsqueeze(0)
            matrix = matrix.unsqueeze(0)
        local_point = local_point.unsqueeze(0).repeat(position.shape[0], 1)
        return torch.matmul(local_point.unsqueeze(1), matrix.transpose(-1, -2)).squeeze(
            1
        ) + position

    def _current_insert_target_world(self) -> torch.Tensor:
        return self._box_local_point_world(self._box_insert_target_local())

    def _tcp_position(self) -> torch.Tensor:
        position = self.agent.agents[0].tcp.pose.p
        if position.ndim == 1:
            position = position.unsqueeze(0)
        return position

    def _tcp_to_target_distance(self) -> torch.Tensor:
        return torch.linalg.norm(
            self._tcp_position() - self._current_insert_target_world(),
            dim=-1,
        )

    def _tcp_to_target_reward(self) -> torch.Tensor:
        return 1 - torch.tanh(
            self.TCP_TO_TARGET_REWARD_SCALE * self._tcp_to_target_distance()
        )  # min = 0.0, max = 1.0

    def _gripper_drive_qpos(self) -> torch.Tensor:
        qpos = self.agent.agents[0].robot.get_qpos()
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        return qpos[:, 7]

    def _gripper_opening_reward(self) -> torch.Tensor:
        gripper_error = torch.abs(
            self._gripper_drive_qpos() - self.GRIPPER_OPENING_TARGET_QPOS
        )
        return 1 - torch.tanh(
            self.GRIPPER_OPENING_REWARD_SCALE * gripper_error
        )  # min = 0.0, max = 1.0

    def _sync_target_sites(self, env_idx: Optional[torch.Tensor] = None):
        target_pos = self._current_insert_target_world()
        if env_idx is not None:
            target_pos = target_pos[env_idx]
        self.insert_target_site.set_pose(Pose.create_from_pq(p=target_pos))

    def evaluate(self):
        self._sync_target_sites()
        return {
            "tcp_to_target_distance": self._tcp_to_target_distance(),
        }

    def _get_obs_extra(self, info: Dict[str, Any]):
        pose = self.cardboard_inner_box.pose
        matrix = pose.to_transformation_matrix()[..., :3, :3]
        position = pose.p
        if position.ndim == 1:
            position = position.unsqueeze(0)
            matrix = matrix.unsqueeze(0)
        rotation_6d = matrix[..., :, :2].reshape(position.shape[0], 6)
        return {
            "inner_box_position": position,
            "inner_box_rotation_6d": rotation_6d,
        }

    def compute_normalized_dense_reward(self, obs, action, info):
        reward = (
            self._tcp_to_target_reward()
            + self.GRIPPER_OPENING_REWARD_WEIGHT * self._gripper_opening_reward()
        )
        return reward / (1 + self.GRIPPER_OPENING_REWARD_WEIGHT)
