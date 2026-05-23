from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.utils import get_active_joint_indices
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


@register_env("MyDualCardboardCabinet-v0", max_episode_steps=200)
class MyDualCardboardCabinetEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("my_xarm7", "my_xarm7")]
    agent: MultiAgent[Tuple[Xarm7, Xarm7]]

    OUTER_CARDBOARD_BASE_SPEC = DEFAULT_CARDBOARD_CABINET_SPEC
    CABINET_SPEC = replace(OUTER_CARDBOARD_BASE_SPEC, color_hex="#4C78A8")
    INNER_BOX_SPEC = replace(
        make_cardboard_inner_box_spec(OUTER_CARDBOARD_BASE_SPEC),
        wall_thickness=OUTER_CARDBOARD_BASE_SPEC.wall_thickness / 2,
    )
    INNER_BOX_WORLD_Y_OFFSET = 0.0035
    BOX_X_OFFSET_FROM_BASE = 0.30
    INSERT_TARGET_RADIUS = 0.004
    INSERT_TARGET_COLOR = (0.5, 1.0, 0.0, 1.0)
    INSERT_TARGET_SIDE_LOCAL = (0.0105, 0.0, 0.03925)
    FINGER_INSERT_MARKER_LOCAL = (0.0, -0.01790, 0.05340)
    INSERT_REWARD_DISTANCE_SCALE = 20.0
    INSERT_PRECISION_REWARD_WEIGHT = 0.0
    INSERT_PRECISION_REWARD_DISTANCE_SCALE = 80.0
    INSERT_AXIS_REWARD_WEIGHT = 0.0
    INSERT_AXIS_REWARD_DISTANCE_SCALE = 80.0
    INSERT_YZ_REWARD_WEIGHT = 0.0
    INSERT_YZ_REWARD_DISTANCE_SCALE = 60.0
    INSERT_SUCCESS_DISTANCE = 0.005
    BOX_POSITION_SHIFT_TOLERANCE = 0.002
    BOX_SUCCESS_MAX_SHIFT = 0.005
    BOX_POSITION_PENALTY_SCALE = 250.0
    BOX_POSITION_PENALTY_MAX = 0.995

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
        self._agent_obs_joint_indices: Dict[str, torch.Tensor] = dict()
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: Dict[str, Any]):
        super()._load_agent(
            options,
            [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])],
        )
        self._configure_observed_joint_indices()

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

    def _clear(self):
        super()._clear()
        self._agent_obs_joint_indices = dict()

    def _configure_observed_joint_indices(self):
        self._agent_obs_joint_indices = dict()
        if not isinstance(self.agent, MultiAgent):
            return
        for idx, sub_agent in enumerate(self.agent.agents):
            key = f"{sub_agent.uid}-{idx}"
            indices = self._build_agent_joint_indices(sub_agent)
            if indices is None:
                if hasattr(sub_agent, "_obs_joint_indices"):
                    delattr(sub_agent, "_obs_joint_indices")
                continue
            self._agent_obs_joint_indices[key] = indices
            sub_agent._obs_joint_indices = indices

    def _build_agent_joint_indices(self, sub_agent) -> Optional[torch.Tensor]:
        observed_joint_names: List[str] = []
        arm_joint_names = getattr(sub_agent, "arm_joint_names", None)
        if arm_joint_names is not None:
            observed_joint_names.extend(list(arm_joint_names))
        gripper_joint_names = getattr(sub_agent, "gripper_joint_names", None)
        if gripper_joint_names:
            drive_joint = gripper_joint_names[0]
            if drive_joint not in observed_joint_names:
                observed_joint_names.append(drive_joint)
        if not observed_joint_names:
            return None
        return get_active_joint_indices(sub_agent.robot, observed_joint_names).long()

    def _get_obs_agent(self):
        obs = super()._get_obs_agent()
        if not isinstance(obs, dict):
            return obs
        index_map = getattr(self, "_agent_obs_joint_indices", None)
        if not index_map:
            return obs
        filtered = {}
        for key, value in obs.items():
            indices = index_map.get(key)
            if indices is None:
                filtered[key] = value
                continue
            filtered[key] = self._filter_agent_obs_entry(value, indices)
        return filtered

    def _filter_agent_obs_entry(self, obs_entry: Any, indices: torch.Tensor) -> Any:
        if not isinstance(obs_entry, dict):
            return obs_entry
        filtered_entry = dict(obs_entry)
        if "qpos" in obs_entry and obs_entry["qpos"] is not None:
            filtered_entry["qpos"] = self._index_select_joint_tensor(
                obs_entry["qpos"], indices
            )
        if "qvel" in filtered_entry:
            filtered_entry.pop("qvel", None)
        return filtered_entry

    def _index_select_joint_tensor(
        self, tensor: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        if tensor is None:
            return tensor
        idx = indices.to(device=tensor.device, dtype=torch.long)
        dim = tensor.dim() - 1
        return tensor.index_select(dim, idx)

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

    def _current_insert_target_world(self) -> torch.Tensor:
        pose = self.cardboard_inner_box.pose
        matrix = pose.to_transformation_matrix()[..., :3, :3]
        position = pose.p
        if position.ndim == 1:
            position = position.unsqueeze(0)
            matrix = matrix.unsqueeze(0)
        local_point = self._box_insert_target_local().unsqueeze(0).repeat(
            position.shape[0], 1
        )
        return torch.matmul(local_point.unsqueeze(1), matrix.transpose(-1, -2)).squeeze(
            1
        ) + position

    def _finger_insert_marker_world(self) -> torch.Tensor:
        pose = self.agent.agents[0].finger1_link.pose
        matrix = pose.to_transformation_matrix()[..., :3, :3]
        position = pose.p
        if position.ndim == 1:
            position = position.unsqueeze(0)
            matrix = matrix.unsqueeze(0)
        local_point = torch.tensor(
            self.FINGER_INSERT_MARKER_LOCAL,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0).repeat(position.shape[0], 1)
        return torch.matmul(local_point.unsqueeze(1), matrix.transpose(-1, -2)).squeeze(
            1
        ) + position

    def _inner_box_position_shift(self) -> torch.Tensor:
        position = self.cardboard_inner_box.pose.p
        if position.ndim == 1:
            position = position.unsqueeze(0)
        initial_position = self._initial_inner_box_world_position().unsqueeze(0)
        return torch.linalg.norm(position - initial_position, dim=-1)

    def _inner_box_position_penalty(self) -> torch.Tensor:
        shift = self._inner_box_position_shift()
        penalized_shift = torch.clamp(
            shift - self.BOX_POSITION_SHIFT_TOLERANCE,
            min=0.0,
        )
        return self.BOX_POSITION_PENALTY_MAX * torch.tanh(
            self.BOX_POSITION_PENALTY_SCALE * penalized_shift
        )

    def _sync_target_sites(self, env_idx: Optional[torch.Tensor] = None):
        target_pos = self._current_insert_target_world()
        if env_idx is not None:
            target_pos = target_pos[env_idx]
        self.insert_target_site.set_pose(
            Pose.create_from_pq(p=target_pos)
        )

    def evaluate(self):
        marker_pos = self._finger_insert_marker_world()
        target_pos = self._current_insert_target_world()
        distance = torch.linalg.norm(marker_pos - target_pos, dim=-1)
        box_position_shift = self._inner_box_position_shift()
        self._sync_target_sites()
        return {
            "success": (distance < self.INSERT_SUCCESS_DISTANCE)
            & (box_position_shift < self.BOX_SUCCESS_MAX_SHIFT),
            "insert_marker_distance": distance,
            "box_position_shift": box_position_shift,
            "box_position_penalty": self._inner_box_position_penalty(),
        }

    def _get_obs_extra(self, info: Dict[str, Any]):
        marker_pos = self._finger_insert_marker_world()
        target_pos = self._current_insert_target_world()
        center = torch.tensor(
            self.bimanual_center_pose.p,
            dtype=torch.float32,
            device=self.device,
        )
        return {
            "finger_insert_marker_from_bimanual_center": marker_pos - center,
            "insert_target_from_bimanual_center": target_pos - center,
            "insert_target_delta": target_pos - marker_pos,
            "inner_box_position_shift": self._inner_box_position_shift().unsqueeze(-1),
        }

    def compute_normalized_dense_reward(self, obs, action, info):
        marker_pos = self._finger_insert_marker_world()
        target_pos = self._current_insert_target_world()
        delta = target_pos - marker_pos
        distance = torch.linalg.norm(delta, dim=-1)
        reward = 1 - torch.tanh(self.INSERT_REWARD_DISTANCE_SCALE * distance)
        if self.INSERT_PRECISION_REWARD_WEIGHT > 0:
            precision_reward = 1 - torch.tanh(
                self.INSERT_PRECISION_REWARD_DISTANCE_SCALE * distance
            )
            reward = (
                reward
                + self.INSERT_PRECISION_REWARD_WEIGHT * precision_reward
            ) / (1 + self.INSERT_PRECISION_REWARD_WEIGHT)
        if self.INSERT_AXIS_REWARD_WEIGHT > 0:
            axis_reward = 1 - torch.tanh(
                self.INSERT_AXIS_REWARD_DISTANCE_SCALE * torch.abs(delta)
            )
            axis_reward = axis_reward.mean(dim=-1)
            reward = (
                reward
                + self.INSERT_AXIS_REWARD_WEIGHT * axis_reward
            ) / (1 + self.INSERT_AXIS_REWARD_WEIGHT)
        if self.INSERT_YZ_REWARD_WEIGHT > 0:
            yz_distance = torch.linalg.norm(delta[:, 1:], dim=-1)
            yz_reward = 1 - torch.tanh(
                self.INSERT_YZ_REWARD_DISTANCE_SCALE * yz_distance
            )
            reward = (
                reward
                + self.INSERT_YZ_REWARD_WEIGHT * yz_reward
            ) / (1 + self.INSERT_YZ_REWARD_WEIGHT)
        reward[distance < self.INSERT_SUCCESS_DISTANCE] = 1.0
        return reward * (1 - self._inner_box_position_penalty())


@register_env("MyDualCardboardCabinetPrecision-v0", max_episode_steps=200)
class MyDualCardboardCabinetPrecisionEnv(MyDualCardboardCabinetEnv):
    INSERT_PRECISION_REWARD_WEIGHT = 1.0


@register_env("MyDualCardboardCabinetAxis-v0", max_episode_steps=200)
class MyDualCardboardCabinetAxisEnv(MyDualCardboardCabinetEnv):
    INSERT_AXIS_REWARD_WEIGHT = 1.0


@register_env("MyDualCardboardCabinetYzShaped-v0", max_episode_steps=200)
class MyDualCardboardCabinetYzShapedEnv(MyDualCardboardCabinetEnv):
    INSERT_YZ_REWARD_WEIGHT = 0.5


@register_env("MyDualCardboardCabinetSlotCenter-v0", max_episode_steps=200)
class MyDualCardboardCabinetSlotCenterEnv(MyDualCardboardCabinetYzShapedEnv):
    INSERT_TARGET_SIDE_LOCAL = (0.0105, 0.0, 0.04825)


@register_env("MyDualCardboardCabinetTallSlot-v0", max_episode_steps=200)
class MyDualCardboardCabinetTallSlotEnv(MyDualCardboardCabinetYzShapedEnv):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        notch_height_z=0.04,
    )
    INSERT_TARGET_SIDE_LOCAL = (0.0105, 0.0, 0.03725)


@register_env("MyDualCardboardCabinetNoNotchCollision-v0", max_episode_steps=200)
class MyDualCardboardCabinetNoNotchCollisionEnv(MyDualCardboardCabinetYzShapedEnv):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        notch_side_collision=False,
    )
