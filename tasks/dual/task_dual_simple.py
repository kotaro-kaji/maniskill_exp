import os
from typing import Any, Dict, Tuple

import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from robotagents.xarm_ball_ee import Xarm7BallEE


from scenebuilders.dual_xarm7_table_scene_builder import (
    DualXarm7TableSceneBuilder,
    LEFT_ARM_Y_OFFSET,
    RIGHT_ARM_Y_OFFSET,
)
from scenebuilders.xarm7_table_scene_builder import (
    PEDESTAL_HEIGHT,
    ROBOT_BASE_X_OFFSET,
)


@register_env("MyDualSimple-v0", max_episode_steps=200)
class MyDualSimpleEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("xarm7_ball_ee", "xarm7_ball_ee")]
    agent: MultiAgent[Tuple[Xarm7BallEE, Xarm7BallEE]]

    BOX_HALF_SIZE = (0.18, 0.12, 0.12)
    BOX_DENSITY = 200.0
    BOX_X_OFFSET_FROM_BASE = 0.28
    BOX_Y_JITTER = 0.05
    BOX_X_JITTER = 0.03
    LEFT_TARGET_POS_FROM_BIMANUAL_CENTER = (0.5, 0.3, 0.35)
    RIGHT_TARGET_POS_FROM_BIMANUAL_CENTER = (0.5, -0.3, 0.35)
    DISTANCE_SCALE = 4.0
    TARGET_RADIUS = 0.02
    LEFT_TARGET_COLOR = (0.1, 0.8, 0.2, 1.0) #緑色
    RIGHT_TARGET_COLOR = (0.2, 0.4, 1.0, 1.0) #青色

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
        self.bimanual_center_pose = self._compute_bimanual_center_pose()
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            half_size=self.BOX_HALF_SIZE,
            density=self.BOX_DENSITY,
        )
        visual_file = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "assets",
                "cardboard_box",
                "textured.obj",
            )
        )
        builder.add_visual_from_file(
            filename=visual_file,
            scale=[0.12, 0.12, 0.12],
            pose=sapien.Pose(),
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
        self.left_target_site = actors.build_sphere(
            self.scene,
            radius=self.TARGET_RADIUS,
            color=self.LEFT_TARGET_COLOR,
            name="left_target_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(
                p=self._bimanual_center_point_to_world(
                    self.LEFT_TARGET_POS_FROM_BIMANUAL_CENTER
                )
            ),
        )
        self.right_target_site = actors.build_sphere(
            self.scene,
            radius=self.TARGET_RADIUS,
            color=self.RIGHT_TARGET_COLOR,
            name="right_target_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(
                p=self._bimanual_center_point_to_world(
                    self.RIGHT_TARGET_POS_FROM_BIMANUAL_CENTER
                )
            ),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict[str, Any]):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            batch_size = len(env_idx)
            if batch_size == 0:
                return

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
            orientations = torch.zeros(
                (batch_size, 4), device=self.device, dtype=torch.float32
            )
            orientations[..., 0] = 1.0
            self.box.set_pose(Pose.create_from_pq(positions, orientations))
            left_override = None if options is None else options.get("left_target_pos")
            right_override = None if options is None else options.get("right_target_pos")
            left_target = (
                torch.tensor(
                    self.LEFT_TARGET_POS_FROM_BIMANUAL_CENTER,
                    dtype=torch.float32,
                    device=self.device,
                )
                if left_override is None
                else torch.as_tensor(
                    left_override,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            right_target = (
                torch.tensor(
                    self.RIGHT_TARGET_POS_FROM_BIMANUAL_CENTER,
                    dtype=torch.float32,
                    device=self.device,
                )
                if right_override is None
                else torch.as_tensor(
                    right_override,
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            if left_target.ndim == 1:
                left_target = left_target.unsqueeze(0).repeat(batch_size, 1)
            elif left_target.shape[0] != batch_size:
                left_target = left_target[0:1].repeat(batch_size, 1)
            if right_target.ndim == 1:
                right_target = right_target.unsqueeze(0).repeat(batch_size, 1)
            elif right_target.shape[0] != batch_size:
                right_target = right_target[0:1].repeat(batch_size, 1)
            left_target = self._bimanual_center_tensor_to_world(left_target)
            right_target = self._bimanual_center_tensor_to_world(right_target)
            self.left_target_site.set_pose(Pose.create_from_pq(p=left_target))
            self.right_target_site.set_pose(Pose.create_from_pq(p=right_target))

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

    def compute_normalized_dense_reward(self, obs, action, info):
        if not isinstance(self.agent, MultiAgent):
            return torch.zeros(self.num_envs, device=self.device)

        # Agent index 0 now sits at y > 0 (left-hand side), index 1 at y < 0 (right-hand side).
        left_tcp_pos = self.agent.agents[0].tcp.pose.p
        right_tcp_pos = self.agent.agents[1].tcp.pose.p

        left_target = self.left_target_site.pose.p
        right_target = self.right_target_site.pose.p

        left_dist = torch.linalg.norm(left_tcp_pos - left_target, dim=-1)
        right_dist = torch.linalg.norm(right_tcp_pos - right_target, dim=-1)

        left_reward = 1 - torch.tanh(self.DISTANCE_SCALE * left_dist)
        right_reward = 1 - torch.tanh(self.DISTANCE_SCALE * right_dist)

        reward = 0.5 * (left_reward + right_reward)
        if reward.ndim == 0:
            reward = reward.unsqueeze(0)
        return reward
