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
    FIRST_WAYPOINT_GATE_DISTANCE = 0.005
    FIRST_WAYPOINT_REWARD_DISTANCE_SCALE = 5.0
    FIRST_WAYPOINT_FINE_DISTANCE_SCALE = 40.0
    FINAL_REWARD_DISTANCE_SCALE = 40.0
    FINAL_Y_REWARD_DISTANCE_SCALE = 40.0
    FINAL_Y_REWARD_WEIGHT = 0.25
    INSERT_SUCCESS_DISTANCE = 0.003
    BOX_POSITION_SHIFT_TOLERANCE = 0.003
    BOX_SUCCESS_MAX_SHIFT = 0.007
    BOX_POSITION_PENALTY_SCALE = 180.0
    BOX_POSITION_PENALTY_MAX = 0.995
    GRIPPER_OPENING_TARGET_QPOS = 0.44
    GRIPPER_OPENING_REWARD_WEIGHT = 0.05
    GRIPPER_OPENING_REWARD_SCALE = 8.0
    EEF_X_WORLD_REWARD_WEIGHT = 0.20
    EEF_X_WORLD_REWARD_SCALE = 20.0

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

    def _current_insert_waypoint_worlds(self) -> List[torch.Tensor]:
        return [
            self._box_local_point_world(local_point)
            for local_point in self._box_insert_waypoint_locals()
        ]

    def _stage_targets_world(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._current_insert_waypoint_worlds()[0],
            self._current_insert_target_world(),
        )

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

    def _eef_x_roll_world(self) -> torch.Tensor:
        pose = self.agent.agents[0].tcp.pose
        matrix = pose.to_transformation_matrix()[..., :3, :3]
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        y_axis = matrix[..., :, 1]
        return torch.atan2(y_axis[..., 2], y_axis[..., 1])

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
        if not hasattr(self, "_first_waypoint_reached"):
            self._first_waypoint_reached = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        self._first_waypoint_reached[env_idx] = False

    def _update_first_waypoint_reached(
        self,
        first_distance: torch.Tensor,
    ) -> torch.Tensor:
        if not hasattr(self, "_first_waypoint_reached"):
            self._first_waypoint_reached = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        self._first_waypoint_reached = self._first_waypoint_reached | (
            first_distance < self.FIRST_WAYPOINT_GATE_DISTANCE
        )
        return self._first_waypoint_reached

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
        return self._inner_box_position_shift()

    def _box_position_shift_for_success(self) -> torch.Tensor:
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
        first_target, target_pos = self._stage_targets_world()
        first_distance = torch.linalg.norm(marker_pos - first_target, dim=-1)
        distance = torch.linalg.norm(marker_pos - target_pos, dim=-1)
        first_waypoint_reached = self._update_first_waypoint_reached(first_distance)
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
        success = close_success
        fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._sync_target_sites()
        return {
            "success": success,
            "fail": fail,
            "insert_marker_distance": distance,
            "box_position_shift": box_position_shift,
            "box_episode_max_shift": box_episode_max_shift,
            "finger_axis_alignment": self._finger_axis_alignment(),
            "eef_x_axis_world": self._eef_x_axis_world(),
            "eef_x_roll_world": self._eef_x_roll_world(),
            "eef_x_world_error": self._eef_x_world_error(),
            "gripper_drive_qpos": self._gripper_drive_qpos(),
            "valid_close_streak": valid_close_streak,
            "box_position_penalty": self._inner_box_position_penalty(),
            "first_waypoint_distance": first_distance,
            "first_waypoint_reached": first_waypoint_reached,
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
        marker_pos = self._finger_insert_marker_world()
        first_target, final_target = self._stage_targets_world()

        # Stage 1: move the insert marker to the first waypoint.
        first_distance = torch.linalg.norm(first_target - marker_pos, dim=-1)
        first_reached = self._update_first_waypoint_reached(first_distance)
        first_coarse_reward = 1 - torch.tanh(
            self.FIRST_WAYPOINT_REWARD_DISTANCE_SCALE * first_distance
        )  # min = 0.0, max = 1.0
        first_fine_reward = 1 - torch.tanh(
            self.FIRST_WAYPOINT_FINE_DISTANCE_SCALE * first_distance
        )  # min = 0.0, max = 1.0
        reward_first_waypoint = (
            0.7 * first_coarse_reward + 0.3 * first_fine_reward
        )  # min = 0.0, max = 1.0

        # Stage 2: after the first waypoint has ever been reached, move to final.
        final_delta = final_target - marker_pos
        final_distance = torch.linalg.norm(final_delta, dim=-1)
        final_point_reward = 1 - torch.tanh(
            self.FINAL_REWARD_DISTANCE_SCALE * final_distance
        )  # min = 0.0, max = 1.0
        final_y_reward = 1 - torch.tanh(
            self.FINAL_Y_REWARD_DISTANCE_SCALE * torch.abs(final_delta[:, 1])
        )  # min = 0.0, max = 1.0
        reward_final_target = (
            final_point_reward + self.FINAL_Y_REWARD_WEIGHT * final_y_reward
        ) / (1 + self.FINAL_Y_REWARD_WEIGHT)  # min = 0.0, max = 1.0

        stage_reward = torch.where(
            first_reached,
            0.5 + 0.5 * reward_final_target,
            0.5 * reward_first_waypoint,
        )  # min = 0.0, max = 1.0

        # Keep the left gripper near the desired opening through the episode.
        gripper_error = torch.abs(
            info["gripper_drive_qpos"] - self.GRIPPER_OPENING_TARGET_QPOS
        )
        reward_gripper_opening = 1 - torch.tanh(
            self.GRIPPER_OPENING_REWARD_SCALE * gripper_error
        )  # min = 0.0, max = 1.0
        reward = (
            stage_reward
            + self.GRIPPER_OPENING_REWARD_WEIGHT * reward_gripper_opening
        ) / (1 + self.GRIPPER_OPENING_REWARD_WEIGHT)  # min = 0.0, max = 1.0

        # After the first waypoint, align the EEF local x-axis with world +x.
        if self.EEF_X_WORLD_REWARD_WEIGHT > 0:
            eef_x_dot = info["eef_x_axis_world"][:, 0]
            eef_angle_error = torch.acos(
                torch.clamp(eef_x_dot, min=-1.0, max=1.0)
            )
            reward_eef_x_world = 1 - torch.tanh(
                self.EEF_X_WORLD_REWARD_SCALE * eef_angle_error
            )  # min = 0.0, max = 1.0
            reward_with_pose = (
                reward + self.EEF_X_WORLD_REWARD_WEIGHT * reward_eef_x_world
            ) / (1 + self.EEF_X_WORLD_REWARD_WEIGHT)  # min = 0.0, max = 1.0
            reward = torch.where(first_reached, reward_with_pose, reward)

        # Final target success overrides shaping before the box-shift multiplier.
        reward[final_distance < self.INSERT_SUCCESS_DISTANCE] = 1.0

        box_position_penalty = self._inner_box_position_penalty()
        reward = reward * (1 - box_position_penalty)
        return reward
