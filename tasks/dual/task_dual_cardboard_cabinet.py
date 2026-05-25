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
        wall_thickness=0.002,
    )
    INNER_BOX_WORLD_Y_OFFSET = 0.0035
    BOX_X_OFFSET_FROM_BASE = 0.30
    INSERT_TARGET_RADIUS = 0.004
    INSERT_TARGET_COLOR = (0.5, 1.0, 0.0, 1.0)
    INSERT_TARGET_SIDE_LOCAL = (0.0105, 0.0, 0.04025)
    INSERT_WAYPOINT_RADIUS = 0.0035
    INSERT_WAYPOINT_COLORS = (
        (1.0, 0.55, 0.0, 1.0),
    )
    INSERT_WAYPOINT_SIDE_LOCALS = (
        (-0.0295, 0.0, 0.04025),
    )
    EEF_FRAME_AXIS_LENGTH = 0.035
    EEF_FRAME_MARKER_RADIUS = 0.003
    FINGER_INSERT_MARKER_LOCAL = (0.0, -0.01790, 0.05340)
    INSERT_REWARD_DISTANCE_SCALE = 20.0
    INSERT_PRECISION_REWARD_WEIGHT = 0.0
    INSERT_PRECISION_REWARD_DISTANCE_SCALE = 80.0
    INSERT_AXIS_REWARD_WEIGHT = 0.0
    INSERT_AXIS_REWARD_DISTANCE_SCALE = 80.0
    INSERT_YZ_REWARD_WEIGHT = 0.0
    INSERT_YZ_REWARD_DISTANCE_SCALE = 60.0
    INSERT_Z_REWARD_WEIGHT = 0.0
    INSERT_Z_REWARD_DISTANCE_SCALE = 120.0
    INSERT_APPROACH_REWARD_WEIGHT = 0.0
    INSERT_APPROACH_DISTANCE_SCALE = 40.0
    INSERT_APPROACH_GATE_DISTANCE = 0.04
    INSERT_APPROACH_LOCAL_OFFSET = (-0.04, 0.0, 0.0)
    INSERT_SUCCESS_DISTANCE = 0.005
    BOX_POSITION_SHIFT_TOLERANCE = 0.003
    BOX_SUCCESS_MAX_SHIFT = 0.007
    BOX_POSITION_PENALTY_SCALE = 180.0
    BOX_POSITION_PENALTY_MAX = 0.995
    USE_EPISODE_MAX_BOX_SHIFT = False
    BOX_FAIL_MAX_SHIFT = None
    USE_HOLD_SUCCESS = False
    HOLD_SUCCESS_STEPS = 5
    HOLD_REWARD_WEIGHT = 0.0
    INSERT_AXIS_ALIGN_REWARD_WEIGHT = 0.0
    GRIPPER_OPENING_TARGET_QPOS = 0.44
    GRIPPER_OPENING_REWARD_WEIGHT = 0.0
    GRIPPER_OPENING_REWARD_SCALE = 8.0
    EEF_X_WORLD_REWARD_WEIGHT = 0.0
    EEF_X_WORLD_REWARD_SCALE = 8.0
    EEF_X_WORLD_MIN_DOT = 1.0

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
        self.insert_waypoint_sites = []
        for idx, color in enumerate(self.INSERT_WAYPOINT_COLORS):
            site = actors.build_sphere(
                self.scene,
                radius=self.INSERT_WAYPOINT_RADIUS,
                color=color,
                name=f"insert_waypoint_site_{idx}",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(),
            )
            self.insert_waypoint_sites.append(site)
        self.eef_frame_sites = {
            "origin": actors.build_sphere(
                self.scene,
                radius=self.EEF_FRAME_MARKER_RADIUS,
                color=(1.0, 1.0, 1.0, 1.0),
                name="eef_frame_origin_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(),
            ),
            "x": actors.build_sphere(
                self.scene,
                radius=self.EEF_FRAME_MARKER_RADIUS,
                color=(1.0, 0.0, 0.0, 1.0),
                name="eef_frame_x_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(),
            ),
            "y": actors.build_sphere(
                self.scene,
                radius=self.EEF_FRAME_MARKER_RADIUS,
                color=(0.0, 1.0, 0.0, 1.0),
                name="eef_frame_y_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(),
            ),
            "z": actors.build_sphere(
                self.scene,
                radius=self.EEF_FRAME_MARKER_RADIUS,
                color=(0.0, 0.25, 1.0, 1.0),
                name="eef_frame_z_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(),
            ),
        }

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
            self._reset_episode_box_shift(env_idx)
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

    def _insert_waypoint_side_locals(self) -> torch.Tensor:
        return torch.tensor(
            self.INSERT_WAYPOINT_SIDE_LOCALS,
            dtype=torch.float32,
            device=self.device,
        )

    def _box_insert_waypoint_locals(self) -> torch.Tensor:
        return (
            self._insert_side_origin_box_local().unsqueeze(0)
            + self._insert_waypoint_side_locals()
        )

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

    def _initial_box_local_point_world(self, local_point: torch.Tensor) -> torch.Tensor:
        current_position = self.cardboard_inner_box.pose.p
        batch_size = 1 if current_position.ndim == 1 else current_position.shape[0]
        position = self._initial_inner_box_world_position().unsqueeze(0).repeat(
            batch_size, 1
        )
        orientation = cardboard_cabinet_quaternion(
            self.CABINET_SPEC,
            device=self.device,
        ).unsqueeze(0).repeat(batch_size, 1)
        pose = Pose.create_from_pq(position, orientation)
        matrix = pose.to_transformation_matrix()[..., :3, :3]
        local_point = local_point.unsqueeze(0).repeat(batch_size, 1)
        return torch.matmul(local_point.unsqueeze(1), matrix.transpose(-1, -2)).squeeze(
            1
        ) + position

    def _current_insert_target_world(self) -> torch.Tensor:
        return self._box_local_point_world(self._box_insert_target_local())

    def _current_insert_approach_world(self) -> torch.Tensor:
        offset = torch.tensor(
            self.INSERT_APPROACH_LOCAL_OFFSET,
            dtype=torch.float32,
            device=self.device,
        )
        return self._box_local_point_world(self._box_insert_target_local() + offset)

    def _current_insert_waypoint_worlds(self) -> List[torch.Tensor]:
        return [
            self._box_local_point_world(local_point)
            for local_point in self._box_insert_waypoint_locals()
        ]

    def _current_reward_target_world(self) -> torch.Tensor:
        return self._current_insert_target_world()

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

    def _finger_insert_axes_world(self) -> torch.Tensor:
        pose = self.agent.agents[0].finger1_link.pose
        matrix = pose.to_transformation_matrix()[..., :3, :3]
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        return matrix

    def _eef_frame_points_world(self) -> Dict[str, torch.Tensor]:
        pose = self.agent.agents[0].tcp.pose
        matrix = pose.to_transformation_matrix()[..., :3, :3]
        position = pose.p
        if position.ndim == 1:
            position = position.unsqueeze(0)
            matrix = matrix.unsqueeze(0)
        axis_length = self.EEF_FRAME_AXIS_LENGTH
        return {
            "origin": position,
            "x": position + axis_length * matrix[..., :, 0],
            "y": position + axis_length * matrix[..., :, 1],
            "z": position + axis_length * matrix[..., :, 2],
        }

    def _eef_x_axis_world(self) -> torch.Tensor:
        pose = self.agent.agents[0].tcp.pose
        matrix = pose.to_transformation_matrix()[..., :3, :3]
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        return matrix[..., :, 0]

    def _eef_x_world_error(self) -> torch.Tensor:
        x_axis = self._eef_x_axis_world()
        target = torch.tensor(
            [1.0, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        return torch.linalg.norm(x_axis - target, dim=-1)

    def _gripper_drive_qpos(self) -> torch.Tensor:
        qpos = self.agent.agents[0].robot.get_qpos()
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        return qpos[:, 7]

    def _apply_gripper_and_eef_rewards(
        self,
        reward: torch.Tensor,
        info: Dict[str, Any],
    ) -> torch.Tensor:
        if self.GRIPPER_OPENING_REWARD_WEIGHT > 0:
            gripper_error = torch.abs(
                info["gripper_drive_qpos"] - self.GRIPPER_OPENING_TARGET_QPOS
            )
            gripper_reward = 1 - torch.tanh(
                self.GRIPPER_OPENING_REWARD_SCALE * gripper_error
            )
            reward = (
                reward
                + self.GRIPPER_OPENING_REWARD_WEIGHT * gripper_reward
            ) / (1 + self.GRIPPER_OPENING_REWARD_WEIGHT)
        if self.EEF_X_WORLD_REWARD_WEIGHT > 0:
            eef_x_dot = info["eef_x_axis_world"][:, 0]
            eef_error = torch.clamp(self.EEF_X_WORLD_MIN_DOT - eef_x_dot, min=0.0)
            eef_reward = 1 - torch.tanh(
                self.EEF_X_WORLD_REWARD_SCALE * eef_error
            )
            eef_gate = (
                (1 - self.EEF_X_WORLD_REWARD_WEIGHT)
                + self.EEF_X_WORLD_REWARD_WEIGHT * eef_reward
            )
            reward = reward * eef_gate
        return reward

    def _target_insert_direction_world(self) -> torch.Tensor:
        offset = torch.tensor(
            [0.001, 0.0, 0.0],
            dtype=torch.float32,
            device=self.device,
        )
        current_target = self._initial_box_local_point_world(
            self._box_insert_target_local()
        )
        next_target = self._initial_box_local_point_world(
            self._box_insert_target_local() + offset
        )
        direction = next_target - current_target
        return torch.nn.functional.normalize(direction, dim=-1)

    def _finger_axis_alignment(self) -> torch.Tensor:
        direction = self._target_insert_direction_world()
        axes = self._finger_insert_axes_world()
        return torch.matmul(axes.transpose(-1, -2), direction.unsqueeze(-1)).squeeze(-1)

    def _inner_box_position_shift(self) -> torch.Tensor:
        position = self.cardboard_inner_box.pose.p
        if position.ndim == 1:
            position = position.unsqueeze(0)
        initial_position = self._initial_inner_box_world_position().unsqueeze(0)
        return torch.linalg.norm(position - initial_position, dim=-1)

    def _reset_episode_box_shift(self, env_idx: torch.Tensor):
        if not hasattr(self, "_episode_max_inner_box_shift"):
            self._episode_max_inner_box_shift = torch.zeros(
                self.num_envs,
                dtype=torch.float32,
                device=self.device,
            )
        self._episode_max_inner_box_shift[env_idx] = 0.0
        if not hasattr(self, "_valid_close_streak"):
            self._valid_close_streak = torch.zeros(
                self.num_envs,
                dtype=torch.long,
                device=self.device,
            )
        self._valid_close_streak[env_idx] = 0

    def _update_valid_close_streak(
        self,
        distance: torch.Tensor,
        success_box_shift: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self, "_valid_close_streak"):
            self._valid_close_streak = torch.zeros(
                self.num_envs,
                dtype=torch.long,
                device=self.device,
            )
        valid_close = (distance < self.INSERT_SUCCESS_DISTANCE) & (
            success_box_shift < self.BOX_SUCCESS_MAX_SHIFT
        )
        self._valid_close_streak = torch.where(
            valid_close,
            self._valid_close_streak + 1,
            torch.zeros_like(self._valid_close_streak),
        )
        return self._valid_close_streak

    def _inner_box_episode_max_shift(self) -> torch.Tensor:
        current_shift = self._inner_box_position_shift()
        if not hasattr(self, "_episode_max_inner_box_shift"):
            self._episode_max_inner_box_shift = torch.zeros_like(current_shift)
        self._episode_max_inner_box_shift = torch.maximum(
            self._episode_max_inner_box_shift,
            current_shift,
        )
        return self._episode_max_inner_box_shift

    def _box_position_shift_for_reward(self) -> torch.Tensor:
        if self.USE_EPISODE_MAX_BOX_SHIFT:
            return self._inner_box_episode_max_shift()
        return self._inner_box_position_shift()

    def _box_position_shift_for_success(self) -> torch.Tensor:
        if self.USE_EPISODE_MAX_BOX_SHIFT:
            return self._inner_box_episode_max_shift()
        return self._inner_box_position_shift()

    def _inner_box_position_penalty_from_shift(self, shift: torch.Tensor) -> torch.Tensor:
        penalized_shift = torch.clamp(
            shift - self.BOX_POSITION_SHIFT_TOLERANCE,
            min=0.0,
        )
        return self.BOX_POSITION_PENALTY_MAX * torch.tanh(
            self.BOX_POSITION_PENALTY_SCALE * penalized_shift
        )

    def _inner_box_position_penalty(self) -> torch.Tensor:
        return self._inner_box_position_penalty_from_shift(
            self._box_position_shift_for_reward()
        )

    def _sync_target_sites(self, env_idx: Optional[torch.Tensor] = None):
        target_pos = self._current_insert_target_world()
        if env_idx is not None:
            target_pos = target_pos[env_idx]
        self.insert_target_site.set_pose(
            Pose.create_from_pq(p=target_pos)
        )
        for site, waypoint_pos in zip(
            self.insert_waypoint_sites,
            self._current_insert_waypoint_worlds(),
        ):
            if env_idx is not None:
                waypoint_pos = waypoint_pos[env_idx]
            site.set_pose(Pose.create_from_pq(p=waypoint_pos))
        eef_frame_points = self._eef_frame_points_world()
        for axis_name, site in self.eef_frame_sites.items():
            position = eef_frame_points[axis_name]
            if env_idx is not None:
                position = position[env_idx]
            site.set_pose(Pose.create_from_pq(p=position))

    def evaluate(self):
        marker_pos = self._finger_insert_marker_world()
        target_pos = self._current_reward_target_world()
        distance = torch.linalg.norm(marker_pos - target_pos, dim=-1)
        box_position_shift = self._inner_box_position_shift()
        box_episode_max_shift = self._inner_box_episode_max_shift()
        success_box_shift = self._box_position_shift_for_success()
        valid_close_streak = self._update_valid_close_streak(
            distance,
            success_box_shift,
        )
        close_success = (distance < self.INSERT_SUCCESS_DISTANCE) & (
            success_box_shift < self.BOX_SUCCESS_MAX_SHIFT
        )
        if self.USE_HOLD_SUCCESS:
            success = valid_close_streak >= self.HOLD_SUCCESS_STEPS
        else:
            success = close_success
        fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.BOX_FAIL_MAX_SHIFT is not None:
            fail = box_episode_max_shift > self.BOX_FAIL_MAX_SHIFT
        self._sync_target_sites()
        return {
            "success": success,
            "fail": fail,
            "insert_marker_distance": distance,
            "box_position_shift": box_position_shift,
            "box_episode_max_shift": box_episode_max_shift,
            "finger_axis_alignment": self._finger_axis_alignment(),
            "eef_x_axis_world": self._eef_x_axis_world(),
            "eef_x_world_error": self._eef_x_world_error(),
            "gripper_drive_qpos": self._gripper_drive_qpos(),
            "valid_close_streak": valid_close_streak,
            "box_position_penalty": self._inner_box_position_penalty(),
        }

    def _get_obs_extra(self, info: Dict[str, Any]):
        marker_pos = self._finger_insert_marker_world()
        target_pos = self._current_reward_target_world()
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
        target_pos = self._current_reward_target_world()
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
        if self.INSERT_Z_REWARD_WEIGHT > 0:
            z_reward = 1 - torch.tanh(
                self.INSERT_Z_REWARD_DISTANCE_SCALE * torch.abs(delta[:, 2])
            )
            reward = (
                reward
                + self.INSERT_Z_REWARD_WEIGHT * z_reward
            ) / (1 + self.INSERT_Z_REWARD_WEIGHT)
        if self.INSERT_APPROACH_REWARD_WEIGHT > 0:
            approach_delta = self._current_insert_approach_world() - marker_pos
            approach_distance = torch.linalg.norm(approach_delta, dim=-1)
            approach_reward = 1 - torch.tanh(
                self.INSERT_APPROACH_DISTANCE_SCALE * approach_distance
            )
            approach_gate = torch.clamp(
                1 - approach_distance / self.INSERT_APPROACH_GATE_DISTANCE,
                min=0.0,
                max=1.0,
            )
            staged_reward = (
                (1 - approach_gate) * approach_reward
                + approach_gate * reward
            )
            reward = (
                reward
                + self.INSERT_APPROACH_REWARD_WEIGHT * staged_reward
            ) / (1 + self.INSERT_APPROACH_REWARD_WEIGHT)
        if self.HOLD_REWARD_WEIGHT > 0:
            hold_progress = torch.clamp(
                info["valid_close_streak"].to(torch.float32)
                / float(self.HOLD_SUCCESS_STEPS),
                min=0.0,
                max=1.0,
            )
            reward = (
                reward
                + self.HOLD_REWARD_WEIGHT * hold_progress
            ) / (1 + self.HOLD_REWARD_WEIGHT)
        if self.INSERT_AXIS_ALIGN_REWARD_WEIGHT > 0:
            axis_alignment = info["finger_axis_alignment"][:, 1]
            align_reward = torch.clamp((axis_alignment + 1.0) / 2.0, min=0.0, max=1.0)
            reward = (
                reward
                + self.INSERT_AXIS_ALIGN_REWARD_WEIGHT * align_reward
            ) / (1 + self.INSERT_AXIS_ALIGN_REWARD_WEIGHT)
        reward = self._apply_gripper_and_eef_rewards(reward, info)
        if self.USE_HOLD_SUCCESS:
            reward[info["success"]] = 1.0
        else:
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


@register_env("MyDualCardboardCabinetBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetBoxStrictEnv(MyDualCardboardCabinetEnv):
    BOX_POSITION_SHIFT_TOLERANCE = 0.0005
    BOX_POSITION_PENALTY_SCALE = 1200.0
    BOX_POSITION_PENALTY_MAX = 0.999


@register_env("MyDualCardboardCabinetAxisBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetAxisBoxStrictEnv(MyDualCardboardCabinetAxisEnv):
    BOX_POSITION_SHIFT_TOLERANCE = 0.0005
    BOX_POSITION_PENALTY_SCALE = 1200.0
    BOX_POSITION_PENALTY_MAX = 0.999


@register_env("MyDualCardboardCabinetYzBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetYzBoxStrictEnv(MyDualCardboardCabinetYzShapedEnv):
    BOX_POSITION_SHIFT_TOLERANCE = 0.0005
    BOX_POSITION_PENALTY_SCALE = 1200.0
    BOX_POSITION_PENALTY_MAX = 0.999


@register_env("MyDualCardboardCabinetWaypoint0BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetWaypoint0BoxStrictEnv(MyDualCardboardCabinetYzBoxStrictEnv):
    INSERT_REWARD_DISTANCE_SCALE = 40.0
    INSERT_YZ_REWARD_DISTANCE_SCALE = 80.0

    def _current_reward_target_world(self) -> torch.Tensor:
        return self._current_insert_waypoint_worlds()[0]


@register_env("MyDualCardboardCabinetWaypoint1BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetWaypoint1BoxStrictEnv(MyDualCardboardCabinetWaypoint0BoxStrictEnv):
    def _current_reward_target_world(self) -> torch.Tensor:
        return self._current_insert_target_world()


@register_env("MyDualCardboardCabinetWaypoint1FinalBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetWaypoint1FinalBoxStrictEnv(MyDualCardboardCabinetYzBoxStrictEnv):
    WAYPOINT1_GATE_DISTANCE = 0.012
    WAYPOINT1_REWARD_DISTANCE_SCALE = 80.0
    FINAL_REWARD_DISTANCE_SCALE = 120.0

    def compute_normalized_dense_reward(self, obs, action, info):
        marker_pos = self._finger_insert_marker_world()
        waypoint1_pos = self._current_insert_waypoint_worlds()[0]
        target_pos = self._current_insert_target_world()

        waypoint1_distance = torch.linalg.norm(waypoint1_pos - marker_pos, dim=-1)
        target_distance = torch.linalg.norm(target_pos - marker_pos, dim=-1)

        waypoint1_reward = 1 - torch.tanh(
            self.WAYPOINT1_REWARD_DISTANCE_SCALE * waypoint1_distance
        )
        target_reward = 1 - torch.tanh(
            self.FINAL_REWARD_DISTANCE_SCALE * target_distance
        )
        waypoint1_gate = torch.clamp(
            1 - waypoint1_distance / self.WAYPOINT1_GATE_DISTANCE,
            min=0.0,
            max=1.0,
        )
        reward = (waypoint1_reward + waypoint1_gate * target_reward) / (
            1 + waypoint1_gate
        )
        reward[target_distance < self.INSERT_SUCCESS_DISTANCE] = 1.0
        return reward * (1 - self._inner_box_position_penalty())


@register_env("MyDualCardboardCabinetMidpointBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetMidpointBoxStrictEnv(MyDualCardboardCabinetWaypoint0BoxStrictEnv):
    MIDPOINT_SIDE_LOCAL = (0.0105, 0.0, 0.04675)

    def _current_reward_target_world(self) -> torch.Tensor:
        midpoint = self._insert_side_origin_box_local() + torch.tensor(
            self.MIDPOINT_SIDE_LOCAL,
            dtype=torch.float32,
            device=self.device,
        )
        return self._box_local_point_world(midpoint)


@register_env("MyDualCardboardCabinetMidpointDescentBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetMidpointDescentBoxStrictEnv(
    MyDualCardboardCabinetMidpointBoxStrictEnv
):
    XY_REWARD_DISTANCE_SCALE = 150.0
    Z_REWARD_DISTANCE_SCALE = 200.0

    def compute_normalized_dense_reward(self, obs, action, info):
        marker_pos = self._finger_insert_marker_world()
        target_pos = self._current_reward_target_world()
        delta = target_pos - marker_pos

        xy_distance = torch.linalg.norm(delta[:, :2], dim=-1)
        z_distance = torch.abs(delta[:, 2])
        distance = torch.linalg.norm(delta, dim=-1)

        xy_reward = 1 - torch.tanh(self.XY_REWARD_DISTANCE_SCALE * xy_distance)
        z_reward = 1 - torch.tanh(self.Z_REWARD_DISTANCE_SCALE * z_distance)
        point_reward = 1 - torch.tanh(self.INSERT_REWARD_DISTANCE_SCALE * distance)

        reward = (xy_reward + z_reward + point_reward) / 3
        reward[distance < self.INSERT_SUCCESS_DISTANCE] = 1.0
        return reward * (1 - self._inner_box_position_penalty())


@register_env("MyDualCardboardCabinetLowerApproachBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerApproachBoxStrictEnv(
    MyDualCardboardCabinetWaypoint0BoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (-0.0295, 0.0, 0.03925)

    def _current_reward_target_world(self) -> torch.Tensor:
        lower_approach = self._insert_side_origin_box_local() + torch.tensor(
            self.LOWER_APPROACH_SIDE_LOCAL,
            dtype=torch.float32,
            device=self.device,
        )
        return self._box_local_point_world(lower_approach)


@register_env("MyDualCardboardCabinetLowerMidBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerMidBoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (-0.0095, 0.0, 0.03925)


@register_env("MyDualCardboardCabinetLowerNearBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerNearBoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.0005, 0.0, 0.03925)


@register_env("MyDualCardboardCabinetLowerNearSettlePenalty-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerNearSettlePenaltyEnv(
    MyDualCardboardCabinetLowerNearBoxStrictEnv
):
    BOX_POSITION_SHIFT_TOLERANCE = 0.002
    BOX_POSITION_PENALTY_SCALE = 900.0


@register_env("MyDualCardboardCabinetLowerAlmostFinal0BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal0BoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.003, 0.0, 0.03925)


@register_env("MyDualCardboardCabinetLowerNearPlus0BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerNearPlus0BoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.0015, 0.0, 0.03925)


@register_env("MyDualCardboardCabinetLowerNearPlus1BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerNearPlus1BoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.002, 0.0, 0.03925)


@register_env("MyDualCardboardCabinetLowerNearPlus2BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerNearPlus2BoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.0025, 0.0, 0.03925)


@register_env("MyDualCardboardCabinetLowerNearPlus0FixedTargetBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerNearPlus0FixedTargetBoxStrictEnv(
    MyDualCardboardCabinetLowerNearPlus0BoxStrictEnv
):
    def _current_reward_target_world(self) -> torch.Tensor:
        local_point = self._insert_side_origin_box_local() + torch.tensor(
            self.LOWER_APPROACH_SIDE_LOCAL,
            dtype=torch.float32,
            device=self.device,
        )
        return self._initial_box_local_point_world(local_point)


@register_env("MyDualCardboardCabinetLowerAlmostFinal0FixedTargetBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal0FixedTargetBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal0BoxStrictEnv
):
    def _current_reward_target_world(self) -> torch.Tensor:
        local_point = self._insert_side_origin_box_local() + torch.tensor(
            self.LOWER_APPROACH_SIDE_LOCAL,
            dtype=torch.float32,
            device=self.device,
        )
        return self._initial_box_local_point_world(local_point)


@register_env("MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.0055, 0.0, 0.03925)

    def _current_reward_target_world(self) -> torch.Tensor:
        local_point = self._insert_side_origin_box_local() + torch.tensor(
            self.LOWER_APPROACH_SIDE_LOCAL,
            dtype=torch.float32,
            device=self.device,
        )
        return self._initial_box_local_point_world(local_point)


@register_env("MyDualCardboardCabinetLowerAlmostFinal1FixedLanePosBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal1FixedLanePosBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.0055, 0.0015, 0.03925)


@register_env("MyDualCardboardCabinetLowerAlmostFinal1FixedLaneNegBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal1FixedLaneNegBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.0055, -0.0015, 0.03925)


@register_env("MyDualCardboardCabinetLowerAlmostFinal1FixedHighBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal1FixedHighBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.0055, 0.0, 0.04125)


@register_env("MyDualCardboardCabinetLowerAlmostFinal1FixedLowBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal1FixedLowBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.0055, 0.0, 0.03725)


@register_env("MyDualCardboardCabinetLowerAlmostFinal1FixedHeavyBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal1FixedHeavyBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=1000.0,
    )


@register_env("MyDualCardboardCabinetLowerAlmostFinal1FixedDensity500BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal1FixedDensity500BoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=500.0,
    )


@register_env("MyDualCardboardCabinetLowerAlmostFinal1FixedDensity750BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal1FixedDensity750BoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=750.0,
    )


@register_env("MyDualCardboardCabinetLowerAlmostFinal1FixedDensity1500BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal1FixedDensity1500BoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=1500.0,
    )


@register_env("MyDualCardboardCabinetLowerAlmostFinal1FixedDensity5000LowFrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal1FixedDensity5000LowFrictionBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=5000.0,
        static_friction=0.2,
        dynamic_friction=0.1,
    )


@register_env("MyDualCardboardCabinetLowerAlmostFinal1FixedVeryHeavyBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal1FixedVeryHeavyBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=2500.0,
    )


@register_env("MyDualCardboardCabinetLowerAlmostFinal03FixedTargetBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal03FixedTargetBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.004, 0.0, 0.03925)


@register_env("MyDualCardboardCabinetLowerAlmostFinal04FixedTargetBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal04FixedTargetBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.0045, 0.0, 0.03925)


@register_env("MyDualCardboardCabinetLowerAlmostFinal05FixedTargetBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal05FixedTargetBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.005, 0.0, 0.03925)


@register_env("MyDualCardboardCabinetLowerAlmostFinal05FixedCumulativeBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal05FixedCumulativeBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal05FixedTargetBoxStrictEnv
):
    USE_EPISODE_MAX_BOX_SHIFT = True
    BOX_POSITION_SHIFT_TOLERANCE = 0.0005
    BOX_POSITION_PENALTY_SCALE = 1200.0
    BOX_POSITION_PENALTY_MAX = 0.999


@register_env("MyDualCardboardCabinetLowerAlmostFinal05FixedFailBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal05FixedFailBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal05FixedCumulativeBoxStrictEnv
):
    BOX_FAIL_MAX_SHIFT = 0.005


@register_env("MyDualCardboardCabinetLowerAlmostFinal05FixedHoldBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal05FixedHoldBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal05FixedCumulativeBoxStrictEnv
):
    USE_HOLD_SUCCESS = True
    HOLD_SUCCESS_STEPS = 5
    HOLD_REWARD_WEIGHT = 1.0


@register_env("MyDualCardboardCabinetLowerAlmostFinal05FixedAlignBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal05FixedAlignBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal05FixedCumulativeBoxStrictEnv
):
    INSERT_AXIS_ALIGN_REWARD_WEIGHT = 0.5


@register_env("MyDualCardboardCabinetLowerAlmostFinal1FixedNoPushBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal1FixedNoPushBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv
):
    BOX_POSITION_SHIFT_TOLERANCE = 0.0
    BOX_POSITION_PENALTY_SCALE = 4000.0
    BOX_POSITION_PENALTY_MAX = 0.9999


@register_env("MyDualCardboardCabinetLowerAlmostFinal1FixedCumulativeBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal1FixedCumulativeBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal1FixedTargetBoxStrictEnv
):
    USE_EPISODE_MAX_BOX_SHIFT = True
    BOX_POSITION_SHIFT_TOLERANCE = 0.0005
    BOX_POSITION_PENALTY_SCALE = 1200.0
    BOX_POSITION_PENALTY_MAX = 0.999


@register_env("MyDualCardboardCabinetLowerAlmostFinal2FixedTargetBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal2FixedTargetBoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.008, 0.0, 0.03925)

    def _current_reward_target_world(self) -> torch.Tensor:
        local_point = self._insert_side_origin_box_local() + torch.tensor(
            self.LOWER_APPROACH_SIDE_LOCAL,
            dtype=torch.float32,
            device=self.device,
        )
        return self._initial_box_local_point_world(local_point)


@register_env("MyDualCardboardCabinetLowerAlmostFinal2FixedDensity1500BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal2FixedDensity1500BoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal2FixedTargetBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=1500.0,
    )


@register_env("MyDualCardboardCabinetFixedTargetBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetFixedTargetBoxStrictEnv(
    MyDualCardboardCabinetYzBoxStrictEnv
):
    def _current_reward_target_world(self) -> torch.Tensor:
        return self._initial_box_local_point_world(self._box_insert_target_local())


@register_env("MyDualCardboardCabinetFixedTargetDensity1500BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetFixedTargetDensity1500BoxStrictEnv(
    MyDualCardboardCabinetFixedTargetBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=1500.0,
    )


@register_env("MyDualCardboardCabinetLowerAlmostFinal2FixedDensity1500FrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal2FixedDensity1500FrictionBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal2FixedTargetBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=1500.0,
        static_friction=2.0,
        dynamic_friction=1.5,
    )


@register_env("MyDualCardboardCabinetLowerAlmostFinal2FixedDensity1500LowFrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal2FixedDensity1500LowFrictionBoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal2FixedTargetBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=1500.0,
        static_friction=0.2,
        dynamic_friction=0.1,
    )


@register_env("MyDualCardboardCabinetFixedTargetDensity1500FrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetFixedTargetDensity1500FrictionBoxStrictEnv(
    MyDualCardboardCabinetFixedTargetBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=1500.0,
        static_friction=2.0,
        dynamic_friction=1.5,
    )


@register_env("MyDualCardboardCabinetFixedTargetDensity1500LowFrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetFixedTargetDensity1500LowFrictionBoxStrictEnv(
    MyDualCardboardCabinetFixedTargetBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=1500.0,
        static_friction=0.2,
        dynamic_friction=0.1,
    )


@register_env("MyDualCardboardCabinetLowerFixedStagedFinalBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedStagedFinalBoxStrictEnv(
    MyDualCardboardCabinetFixedTargetBoxStrictEnv
):
    STAGE_SIDE_LOCALS = (
        (0.0055, 0.0, 0.03925),
        (0.0080, 0.0, 0.03925),
        (0.0105, 0.0, 0.03925),
    )
    STAGE_REWARD_DISTANCE_SCALE = 120.0
    STAGE_GATE_DISTANCE = 0.007

    def _stage_targets_world(self) -> List[torch.Tensor]:
        targets = []
        for side_local in self.STAGE_SIDE_LOCALS:
            local_point = self._insert_side_origin_box_local() + torch.tensor(
                side_local,
                dtype=torch.float32,
                device=self.device,
            )
            targets.append(self._initial_box_local_point_world(local_point))
        return targets

    def compute_normalized_dense_reward(self, obs, action, info):
        marker_pos = self._finger_insert_marker_world()
        stage_rewards = []
        stage_distances = []
        for target_pos in self._stage_targets_world():
            distance = torch.linalg.norm(target_pos - marker_pos, dim=-1)
            stage_distances.append(distance)
            stage_rewards.append(
                1 - torch.tanh(self.STAGE_REWARD_DISTANCE_SCALE * distance)
            )

        reward = stage_rewards[0] / len(stage_rewards)
        for stage_idx in range(1, len(stage_rewards)):
            passed_previous = stage_distances[stage_idx - 1] < self.STAGE_GATE_DISTANCE
            stage_reward = (stage_idx + stage_rewards[stage_idx]) / len(stage_rewards)
            reward = torch.where(passed_previous, stage_reward, reward)

        final_distance = stage_distances[-1]
        reward[final_distance < self.INSERT_SUCCESS_DISTANCE] = 1.0
        reward = self._apply_gripper_and_eef_rewards(reward, info)
        return reward * (1 - self._inner_box_position_penalty())


@register_env("MyDualCardboardCabinetLowerFixedStagedFinalDensity1500LowFrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedStagedFinalDensity1500LowFrictionBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedStagedFinalBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=1500.0,
        static_friction=0.2,
        dynamic_friction=0.1,
    )


@register_env("MyDualCardboardCabinetLowerFixedStagedFinalDensity1500LowFrictionFailBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedStagedFinalDensity1500LowFrictionFailBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedStagedFinalDensity1500LowFrictionBoxStrictEnv
):
    USE_EPISODE_MAX_BOX_SHIFT = True
    BOX_FAIL_MAX_SHIFT = 0.005


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity1500LowFrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity1500LowFrictionBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedStagedFinalDensity1500LowFrictionBoxStrictEnv
):
    STAGE_SIDE_LOCALS = (
        (0.0055, 0.0, 0.04025),
        (0.0105, 0.0, 0.04025),
    )
    STAGE_GATE_DISTANCE = 0.006
    STAGE_REWARD_DISTANCE_SCALE = 140.0


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity3000LowFrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity3000LowFrictionBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity1500LowFrictionBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=3000.0,
        static_friction=0.2,
        dynamic_friction=0.1,
    )


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity4000LowFrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity4000LowFrictionBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity1500LowFrictionBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=4000.0,
        static_friction=0.2,
        dynamic_friction=0.1,
    )


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity4500LowFrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity4500LowFrictionBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity1500LowFrictionBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=4500.0,
        static_friction=0.2,
        dynamic_friction=0.1,
    )


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000LowFrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000LowFrictionBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity1500LowFrictionBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=5000.0,
        static_friction=0.2,
        dynamic_friction=0.1,
    )


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000LowFrictionSettlePenalty-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000LowFrictionSettlePenaltyEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000LowFrictionBoxStrictEnv
):
    BOX_POSITION_SHIFT_TOLERANCE = 0.002
    BOX_POSITION_PENALTY_SCALE = 900.0
    BOX_POSITION_PENALTY_MAX = 0.999


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000VeryLowFrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000VeryLowFrictionBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity1500LowFrictionBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=5000.0,
        static_friction=0.1,
        dynamic_friction=0.05,
    )


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000MediumFrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000MediumFrictionBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity1500LowFrictionBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=5000.0,
        static_friction=0.35,
        dynamic_friction=0.2,
    )


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000LowFrictionStrictBonus-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000LowFrictionStrictBonusEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000LowFrictionBoxStrictEnv
):
    def compute_normalized_dense_reward(self, obs, action, info):
        marker_pos = self._finger_insert_marker_world()
        stage_rewards = []
        stage_distances = []
        for target_pos in self._stage_targets_world():
            distance = torch.linalg.norm(target_pos - marker_pos, dim=-1)
            stage_distances.append(distance)
            stage_rewards.append(
                1 - torch.tanh(self.STAGE_REWARD_DISTANCE_SCALE * distance)
            )

        reward = stage_rewards[0] / len(stage_rewards)
        for stage_idx in range(1, len(stage_rewards)):
            passed_previous = stage_distances[stage_idx - 1] < self.STAGE_GATE_DISTANCE
            stage_reward = (stage_idx + stage_rewards[stage_idx]) / len(stage_rewards)
            reward = torch.where(passed_previous, stage_reward, reward)

        reward = reward * (1 - self._inner_box_position_penalty())
        reward[info["success"]] = 1.0
        return reward


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000LowFrictionActionHint-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000LowFrictionActionHintEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000LowFrictionStrictBonusEnv
):
    ACTION_HINT_WEIGHT = 0.05

    def compute_normalized_dense_reward(self, obs, action, info):
        reward = super().compute_normalized_dense_reward(obs, action, info)
        if isinstance(action, dict):
            action = torch.cat([action["my_xarm7-0"], action["my_xarm7-1"]], dim=-1)
        marker_pos = self._finger_insert_marker_world()
        first_target = self._stage_targets_world()[0]
        first_distance = torch.linalg.norm(first_target - marker_pos, dim=-1)
        near_contact = first_distance < 0.012
        box_is_stable = self._box_position_shift_for_reward() < self.BOX_SUCCESS_MAX_SHIFT
        action_hint = (
            torch.clamp(action[:, 1], min=0.0, max=1.0)
            + torch.clamp(action[:, 4], min=0.0, max=1.0)
        ) / 2
        hint_gate = (near_contact & box_is_stable).to(torch.float32)
        reward = reward + self.ACTION_HINT_WEIGHT * hint_gate * action_hint
        return torch.clamp(reward, max=1.0)


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000LowFrictionYFinal-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000LowFrictionYFinalEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5000LowFrictionBoxStrictEnv
):
    INSERT_Y_DISTANCE_SCALE = 160.0
    INSERT_XZ_DISTANCE_SCALE = 200.0
    INSERT_Y_REWARD_WEIGHT = 4.0

    def compute_normalized_dense_reward(self, obs, action, info):
        marker_pos = self._finger_insert_marker_world()
        first_target, final_target = self._stage_targets_world()

        first_distance = torch.linalg.norm(first_target - marker_pos, dim=-1)
        first_reward = 1 - torch.tanh(
            self.STAGE_REWARD_DISTANCE_SCALE * first_distance
        )

        final_delta = final_target - marker_pos
        y_distance = torch.abs(final_delta[:, 1])
        xz_distance = torch.linalg.norm(final_delta[:, [0, 2]], dim=-1)
        final_distance = torch.linalg.norm(final_delta, dim=-1)

        y_reward = 1 - torch.tanh(self.INSERT_Y_DISTANCE_SCALE * y_distance)
        xz_reward = 1 - torch.tanh(self.INSERT_XZ_DISTANCE_SCALE * xz_distance)
        point_reward = 1 - torch.tanh(
            self.STAGE_REWARD_DISTANCE_SCALE * final_distance
        )
        final_reward = (
            self.INSERT_Y_REWARD_WEIGHT * y_reward
            + xz_reward
            + point_reward
        ) / (self.INSERT_Y_REWARD_WEIGHT + 2)

        reward = first_reward / 2
        passed_first = first_distance < self.STAGE_GATE_DISTANCE
        reward = torch.where(passed_first, (1 + final_reward) / 2, reward)
        reward[final_distance < self.INSERT_SUCCESS_DISTANCE] = 1.0
        return reward * (1 - self._inner_box_position_penalty())


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5500LowFrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity5500LowFrictionBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity1500LowFrictionBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=5500.0,
        static_friction=0.2,
        dynamic_friction=0.1,
    )


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity1500LowFrictionBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=6000.0,
        static_friction=0.2,
        dynamic_friction=0.1,
    )


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionPoseGripBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionPoseGripBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionBoxStrictEnv
):
    GRIPPER_OPENING_REWARD_WEIGHT = 0.5
    EEF_X_WORLD_REWARD_WEIGHT = 1.0
    EEF_X_WORLD_MIN_DOT = 0.85


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionGripBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionGripBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionBoxStrictEnv
):
    GRIPPER_OPENING_REWARD_WEIGHT = 0.5


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripOpenBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripOpenBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionBoxStrictEnv
):
    INSERT_Y_DISTANCE_SCALE = 160.0
    INSERT_XZ_DISTANCE_SCALE = 200.0
    INSERT_Y_REWARD_WEIGHT = 4.0
    GRIPPER_MIN_QPOS = 0.40
    GRIPPER_MIN_REWARD_WEIGHT = 0.25
    GRIPPER_MIN_REWARD_SCALE = 8.0

    def compute_normalized_dense_reward(self, obs, action, info):
        marker_pos = self._finger_insert_marker_world()
        first_target, final_target = self._stage_targets_world()

        first_distance = torch.linalg.norm(first_target - marker_pos, dim=-1)
        first_reward = 1 - torch.tanh(
            self.STAGE_REWARD_DISTANCE_SCALE * first_distance
        )

        final_delta = final_target - marker_pos
        y_distance = torch.abs(final_delta[:, 1])
        xz_distance = torch.linalg.norm(final_delta[:, [0, 2]], dim=-1)
        final_distance = torch.linalg.norm(final_delta, dim=-1)

        y_reward = 1 - torch.tanh(self.INSERT_Y_DISTANCE_SCALE * y_distance)
        xz_reward = 1 - torch.tanh(self.INSERT_XZ_DISTANCE_SCALE * xz_distance)
        point_reward = 1 - torch.tanh(
            self.STAGE_REWARD_DISTANCE_SCALE * final_distance
        )
        final_reward = (
            self.INSERT_Y_REWARD_WEIGHT * y_reward
            + xz_reward
            + point_reward
        ) / (self.INSERT_Y_REWARD_WEIGHT + 2)

        reward = first_reward / 2
        passed_first = first_distance < self.STAGE_GATE_DISTANCE
        reward = torch.where(passed_first, (1 + final_reward) / 2, reward)

        gripper_shortfall = torch.clamp(
            self.GRIPPER_MIN_QPOS - info["gripper_drive_qpos"],
            min=0.0,
        )
        gripper_reward = 1 - torch.tanh(
            self.GRIPPER_MIN_REWARD_SCALE * gripper_shortfall
        )
        shaped_reward = (
            reward + self.GRIPPER_MIN_REWARD_WEIGHT * gripper_reward
        ) / (1 + self.GRIPPER_MIN_REWARD_WEIGHT)
        reward = shaped_reward

        reward[final_distance < self.INSERT_SUCCESS_DISTANCE] = 1.0
        return reward * (1 - self._inner_box_position_penalty())


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripActionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripActionBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripOpenBoxStrictEnv
):
    pass


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripAction005BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripAction005BoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripActionBoxStrictEnv
):
    pass


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripAction020BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripAction020BoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripActionBoxStrictEnv
):
    pass


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripAction020EefBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripAction020EefBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripAction020BoxStrictEnv
):
    EEF_X_WORLD_REWARD_WEIGHT = 0.10
    EEF_X_WORLD_REWARD_SCALE = 30.0

    def compute_normalized_dense_reward(self, obs, action, info):
        reward = super().compute_normalized_dense_reward(obs, action, info)
        eef_x_dot = info["eef_x_axis_world"][:, 0]
        eef_angle_error = torch.acos(torch.clamp(eef_x_dot, min=-1.0, max=1.0))
        eef_reward = 1 - torch.tanh(
            self.EEF_X_WORLD_REWARD_SCALE * eef_angle_error
        )
        return (
            reward + self.EEF_X_WORLD_REWARD_WEIGHT * eef_reward
        ) / (1 + self.EEF_X_WORLD_REWARD_WEIGHT)


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity7000LowFrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity7000LowFrictionBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity1500LowFrictionBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=7000.0,
        static_friction=0.2,
        dynamic_friction=0.1,
    )


@register_env("MyDualCardboardCabinetLowerFixedTwoPointFinalDensity10000LowFrictionBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerFixedTwoPointFinalDensity10000LowFrictionBoxStrictEnv(
    MyDualCardboardCabinetLowerFixedTwoPointFinalDensity1500LowFrictionBoxStrictEnv
):
    INNER_BOX_SPEC = replace(
        MyDualCardboardCabinetEnv.INNER_BOX_SPEC,
        density=10000.0,
        static_friction=0.2,
        dynamic_friction=0.1,
    )


@register_env("MyDualCardboardCabinetFixedTargetAxisBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetFixedTargetAxisBoxStrictEnv(
    MyDualCardboardCabinetFixedTargetBoxStrictEnv
):
    XY_REWARD_DISTANCE_SCALE = 160.0
    Z_REWARD_DISTANCE_SCALE = 200.0
    POINT_REWARD_DISTANCE_SCALE = 80.0
    Z_REWARD_WEIGHT = 2.0

    def compute_normalized_dense_reward(self, obs, action, info):
        marker_pos = self._finger_insert_marker_world()
        target_pos = self._current_reward_target_world()
        delta = target_pos - marker_pos

        xy_distance = torch.linalg.norm(delta[:, :2], dim=-1)
        z_distance = torch.abs(delta[:, 2])
        point_distance = torch.linalg.norm(delta, dim=-1)

        xy_reward = 1 - torch.tanh(self.XY_REWARD_DISTANCE_SCALE * xy_distance)
        z_reward = 1 - torch.tanh(self.Z_REWARD_DISTANCE_SCALE * z_distance)
        point_reward = 1 - torch.tanh(
            self.POINT_REWARD_DISTANCE_SCALE * point_distance
        )

        reward = (
            xy_reward + self.Z_REWARD_WEIGHT * z_reward + point_reward
        ) / (self.Z_REWARD_WEIGHT + 2)
        reward[point_distance < self.INSERT_SUCCESS_DISTANCE] = 1.0
        return reward * (1 - self._inner_box_position_penalty())


@register_env("MyDualCardboardCabinetAlmostFinal0XNeg1BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetAlmostFinal0XNeg1BoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal0BoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.003, -0.0015, 0.03925)


@register_env("MyDualCardboardCabinetAlmostFinal0XNeg2BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetAlmostFinal0XNeg2BoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal0BoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.003, -0.003, 0.03925)


@register_env("MyDualCardboardCabinetAlmostFinal0XNeg3BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetAlmostFinal0XNeg3BoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal0BoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.003, -0.0045, 0.03925)


@register_env("MyDualCardboardCabinetAlmostFinal0Z1BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetAlmostFinal0Z1BoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal0BoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.003, 0.0, 0.04175)


@register_env("MyDualCardboardCabinetAlmostFinal0Z2BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetAlmostFinal0Z2BoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal0BoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.003, 0.0, 0.04425)


@register_env("MyDualCardboardCabinetAlmostFinal0Z3BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetAlmostFinal0Z3BoxStrictEnv(
    MyDualCardboardCabinetLowerAlmostFinal0BoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.003, 0.0, 0.04675)


@register_env("MyDualCardboardCabinetLowerAlmostFinal0SettlePenalty-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal0SettlePenaltyEnv(
    MyDualCardboardCabinetLowerAlmostFinal0BoxStrictEnv
):
    BOX_POSITION_SHIFT_TOLERANCE = 0.002
    BOX_POSITION_PENALTY_SCALE = 900.0


@register_env("MyDualCardboardCabinetLowerAlmostFinal1BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal1BoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.0055, 0.0, 0.03925)


@register_env("MyDualCardboardCabinetLowerAlmostFinal1SettlePenalty-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal1SettlePenaltyEnv(
    MyDualCardboardCabinetLowerAlmostFinal1BoxStrictEnv
):
    BOX_POSITION_SHIFT_TOLERANCE = 0.002
    BOX_POSITION_PENALTY_SCALE = 900.0


@register_env("MyDualCardboardCabinetLowerAlmostFinal2BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal2BoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.008, 0.0, 0.03925)


@register_env("MyDualCardboardCabinetLowerAlmostFinal2SettlePenalty-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerAlmostFinal2SettlePenaltyEnv(
    MyDualCardboardCabinetLowerAlmostFinal2BoxStrictEnv
):
    BOX_POSITION_SHIFT_TOLERANCE = 0.002
    BOX_POSITION_PENALTY_SCALE = 900.0


@register_env("MyDualCardboardCabinetLowerNearHalfBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerNearHalfBoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (-0.002, 0.0, 0.03925)


@register_env("MyDualCardboardCabinetLowerMidNearBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerMidNearBoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (-0.0045, 0.0, 0.03925)


@register_env("MyDualCardboardCabinetLowerMidNearHalfBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerMidNearHalfBoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (-0.007, 0.0, 0.03925)


@register_env("MyDualCardboardCabinetLowerMidNearThreeQuarterBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerMidNearThreeQuarterBoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (-0.00575, 0.0, 0.03925)


@register_env("MyDualCardboardCabinetMidNearRaisedBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetMidNearRaisedBoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (-0.0045, 0.0, 0.04675)


@register_env("MyDualCardboardCabinetMidNearSoftBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetMidNearSoftBoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (-0.0045, -0.0025, 0.03925)


@register_env("MyDualCardboardCabinetLowerNearLaneBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerNearLaneBoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.0005, 0.003, 0.03925)


@register_env("MyDualCardboardCabinetFinalLaneBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetFinalLaneBoxStrictEnv(MyDualCardboardCabinetYzBoxStrictEnv):
    INSERT_TARGET_SIDE_LOCAL = (0.0105, 0.003, 0.03925)


@register_env("MyDualCardboardCabinetNearRaisedLaneBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetNearRaisedLaneBoxStrictEnv(
    MyDualCardboardCabinetLowerApproachBoxStrictEnv
):
    LOWER_APPROACH_SIDE_LOCAL = (0.0005, 0.003, 0.04675)


@register_env("MyDualCardboardCabinetInsertionYBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetInsertionYBoxStrictEnv(MyDualCardboardCabinetYzBoxStrictEnv):
    INSERT_Y_DISTANCE_SCALE = 80.0
    INSERT_XZ_DISTANCE_SCALE = 200.0
    INSERT_Y_REWARD_WEIGHT = 4.0

    def compute_normalized_dense_reward(self, obs, action, info):
        marker_pos = self._finger_insert_marker_world()
        target_pos = self._current_insert_target_world()
        delta = target_pos - marker_pos

        y_distance = torch.abs(delta[:, 1])
        xz_distance = torch.linalg.norm(delta[:, [0, 2]], dim=-1)
        point_distance = torch.linalg.norm(delta, dim=-1)

        y_reward = 1 - torch.tanh(self.INSERT_Y_DISTANCE_SCALE * y_distance)
        xz_reward = 1 - torch.tanh(self.INSERT_XZ_DISTANCE_SCALE * xz_distance)
        point_reward = 1 - torch.tanh(
            self.INSERT_REWARD_DISTANCE_SCALE * point_distance
        )

        reward = (
            self.INSERT_Y_REWARD_WEIGHT * y_reward
            + xz_reward
            + point_reward
        ) / (self.INSERT_Y_REWARD_WEIGHT + 2)
        reward[point_distance < self.INSERT_SUCCESS_DISTANCE] = 1.0
        return reward * (1 - self._inner_box_position_penalty())


@register_env("MyDualCardboardCabinetMidNearYBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetMidNearYBoxStrictEnv(
    MyDualCardboardCabinetLowerMidNearBoxStrictEnv
):
    INSERT_Y_DISTANCE_SCALE = 160.0
    INSERT_XZ_DISTANCE_SCALE = 200.0
    INSERT_Y_REWARD_WEIGHT = 4.0

    def compute_normalized_dense_reward(self, obs, action, info):
        marker_pos = self._finger_insert_marker_world()
        target_pos = self._current_reward_target_world()
        delta = target_pos - marker_pos

        y_distance = torch.abs(delta[:, 1])
        xz_distance = torch.linalg.norm(delta[:, [0, 2]], dim=-1)
        point_distance = torch.linalg.norm(delta, dim=-1)

        y_reward = 1 - torch.tanh(self.INSERT_Y_DISTANCE_SCALE * y_distance)
        xz_reward = 1 - torch.tanh(self.INSERT_XZ_DISTANCE_SCALE * xz_distance)
        point_reward = 1 - torch.tanh(
            self.INSERT_REWARD_DISTANCE_SCALE * point_distance
        )

        reward = (
            self.INSERT_Y_REWARD_WEIGHT * y_reward
            + xz_reward
            + point_reward
        ) / (self.INSERT_Y_REWARD_WEIGHT + 2)
        reward[point_distance < self.INSERT_SUCCESS_DISTANCE] = 1.0
        return reward * (1 - self._inner_box_position_penalty())


@register_env("MyDualCardboardCabinetLowerNearHalfYBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerNearHalfYBoxStrictEnv(
    MyDualCardboardCabinetLowerNearHalfBoxStrictEnv
):
    INSERT_Y_DISTANCE_SCALE = 160.0
    INSERT_XZ_DISTANCE_SCALE = 200.0
    INSERT_Y_REWARD_WEIGHT = 4.0

    def compute_normalized_dense_reward(self, obs, action, info):
        marker_pos = self._finger_insert_marker_world()
        target_pos = self._current_reward_target_world()
        delta = target_pos - marker_pos

        y_distance = torch.abs(delta[:, 1])
        xz_distance = torch.linalg.norm(delta[:, [0, 2]], dim=-1)
        point_distance = torch.linalg.norm(delta, dim=-1)

        y_reward = 1 - torch.tanh(self.INSERT_Y_DISTANCE_SCALE * y_distance)
        xz_reward = 1 - torch.tanh(self.INSERT_XZ_DISTANCE_SCALE * xz_distance)
        point_reward = 1 - torch.tanh(
            self.INSERT_REWARD_DISTANCE_SCALE * point_distance
        )

        reward = (
            self.INSERT_Y_REWARD_WEIGHT * y_reward
            + xz_reward
            + point_reward
        ) / (self.INSERT_Y_REWARD_WEIGHT + 2)
        reward[point_distance < self.INSERT_SUCCESS_DISTANCE] = 1.0
        return reward * (1 - self._inner_box_position_penalty())


@register_env("MyDualCardboardCabinetLowerNearAction5BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerNearAction5BoxStrictEnv(
    MyDualCardboardCabinetLowerNearBoxStrictEnv
):
    ACTION5_REWARD_WEIGHT = 0.15

    def compute_normalized_dense_reward(self, obs, action, info):
        reward = super().compute_normalized_dense_reward(obs, action, info)
        if isinstance(action, dict):
            action = torch.cat([action["my_xarm7-0"], action["my_xarm7-1"]], dim=-1)
        action5_reward = torch.clamp((action[:, 5] + 1.0) / 2.0, min=0.0, max=1.0)
        return (
            reward + self.ACTION5_REWARD_WEIGHT * action5_reward
        ) / (1 + self.ACTION5_REWARD_WEIGHT)


@register_env("MyDualCardboardCabinetLowerStagedNearBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerStagedNearBoxStrictEnv(MyDualCardboardCabinetYzBoxStrictEnv):
    STAGE_SIDE_LOCALS = (
        (-0.0295, 0.0, 0.03925),
        (-0.0095, 0.0, 0.03925),
        (-0.0045, 0.0, 0.03925),
        (0.0005, 0.0, 0.03925),
    )
    STAGE_REWARD_DISTANCE_SCALE = 80.0
    STAGE_GATE_DISTANCE = 0.007
    STAGE_REWARD_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

    def _stage_targets_world(self) -> List[torch.Tensor]:
        targets = []
        for side_local in self.STAGE_SIDE_LOCALS:
            local_point = self._insert_side_origin_box_local() + torch.tensor(
                side_local,
                dtype=torch.float32,
                device=self.device,
            )
            targets.append(self._box_local_point_world(local_point))
        return targets

    def _current_reward_target_world(self) -> torch.Tensor:
        return self._stage_targets_world()[-1]

    def compute_normalized_dense_reward(self, obs, action, info):
        marker_pos = self._finger_insert_marker_world()
        gate = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        reward = torch.zeros_like(gate)
        weight = torch.zeros_like(gate)
        final_distance = None

        for target_pos, stage_weight in zip(
            self._stage_targets_world(),
            self.STAGE_REWARD_WEIGHTS,
        ):
            distance = torch.linalg.norm(target_pos - marker_pos, dim=-1)
            stage_reward = 1 - torch.tanh(
                self.STAGE_REWARD_DISTANCE_SCALE * distance
            )
            weighted_gate = gate * stage_weight
            reward = reward + weighted_gate * stage_reward
            weight = weight + weighted_gate
            gate = gate * torch.clamp(
                1 - distance / self.STAGE_GATE_DISTANCE,
                min=0.0,
                max=1.0,
            )
            final_distance = distance

        reward = reward / torch.clamp(weight, min=1e-6)
        reward[final_distance < self.INSERT_SUCCESS_DISTANCE] = 1.0
        return reward * (1 - self._inner_box_position_penalty())


@register_env("MyDualCardboardCabinetLowerStagedAlmostFinal0BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerStagedAlmostFinal0BoxStrictEnv(
    MyDualCardboardCabinetLowerStagedNearBoxStrictEnv
):
    STAGE_SIDE_LOCALS = (
        (-0.0295, 0.0, 0.03925),
        (-0.0095, 0.0, 0.03925),
        (-0.0045, 0.0, 0.03925),
        (0.0005, 0.0, 0.03925),
        (0.003, 0.0, 0.03925),
    )
    STAGE_REWARD_WEIGHTS = (0.1, 0.2, 0.4, 1.0, 2.0)


@register_env("MyDualCardboardCabinetLowerStagedNearWeightedBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerStagedNearWeightedBoxStrictEnv(
    MyDualCardboardCabinetLowerStagedNearBoxStrictEnv
):
    STAGE_REWARD_WEIGHTS = (0.1, 0.2, 0.5, 2.0)


@register_env("MyDualCardboardCabinetLowerHardStagedNearBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerHardStagedNearBoxStrictEnv(
    MyDualCardboardCabinetLowerStagedNearBoxStrictEnv
):
    STAGE_GATE_DISTANCE = 0.007
    STAGE_REWARD_DISTANCE_SCALE = 120.0

    def compute_normalized_dense_reward(self, obs, action, info):
        marker_pos = self._finger_insert_marker_world()
        stage_rewards = []
        stage_distances = []
        for target_pos in self._stage_targets_world():
            distance = torch.linalg.norm(target_pos - marker_pos, dim=-1)
            stage_distances.append(distance)
            stage_rewards.append(
                1 - torch.tanh(self.STAGE_REWARD_DISTANCE_SCALE * distance)
            )

        reward = stage_rewards[0] / len(stage_rewards)
        for stage_idx in range(1, len(stage_rewards)):
            passed_previous = stage_distances[stage_idx - 1] < self.STAGE_GATE_DISTANCE
            stage_reward = (stage_idx + stage_rewards[stage_idx]) / len(stage_rewards)
            reward = torch.where(passed_previous, stage_reward, reward)

        final_distance = stage_distances[-1]
        reward[final_distance < self.INSERT_SUCCESS_DISTANCE] = 1.0
        return reward * (1 - self._inner_box_position_penalty())


@register_env("MyDualCardboardCabinetLowerHardStagedNearPlus0BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerHardStagedNearPlus0BoxStrictEnv(
    MyDualCardboardCabinetLowerHardStagedNearBoxStrictEnv
):
    STAGE_SIDE_LOCALS = (
        (-0.0295, 0.0, 0.03925),
        (-0.0095, 0.0, 0.03925),
        (-0.0045, 0.0, 0.03925),
        (0.0005, 0.0, 0.03925),
        (0.0015, 0.0, 0.03925),
    )


@register_env("MyDualCardboardCabinetLowerHardStagedAlmostFinal0BoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetLowerHardStagedAlmostFinal0BoxStrictEnv(
    MyDualCardboardCabinetLowerHardStagedNearBoxStrictEnv
):
    STAGE_SIDE_LOCALS = (
        (-0.0295, 0.0, 0.03925),
        (-0.0095, 0.0, 0.03925),
        (-0.0045, 0.0, 0.03925),
        (0.0005, 0.0, 0.03925),
        (0.0015, 0.0, 0.03925),
        (0.003, 0.0, 0.03925),
    )


@register_env("MyDualCardboardCabinetZBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetZBoxStrictEnv(MyDualCardboardCabinetYzBoxStrictEnv):
    INSERT_Z_REWARD_WEIGHT = 1.0


@register_env("MyDualCardboardCabinetApproachBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetApproachBoxStrictEnv(MyDualCardboardCabinetYzBoxStrictEnv):
    INSERT_APPROACH_REWARD_WEIGHT = 1.0


@register_env("MyDualCardboardCabinetApproachZBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetApproachZBoxStrictEnv(MyDualCardboardCabinetApproachBoxStrictEnv):
    INSERT_Z_REWARD_WEIGHT = 0.5


@register_env("MyDualCardboardCabinetOpeningTargetBoxStrict-v0", max_episode_steps=200)
class MyDualCardboardCabinetOpeningTargetBoxStrictEnv(MyDualCardboardCabinetApproachZBoxStrictEnv):
    INSERT_TARGET_SIDE_LOCAL = (0.0105, 0.0, 0.05425)
