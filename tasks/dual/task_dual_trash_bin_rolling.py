import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.utils import get_active_joint_indices
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from robotagents.xarm_ball_ee_kinematics import (
    Xarm7BallEEKinematicsLeft,
    Xarm7BallEEKinematicsRight,
)
from robotagents.xarm_ball_ee_wo_force_sensor import Xarm7BallEEWoForceSensor
from scenebuilders.dual_xarm7_table_scene_builder import (
    DualXarm7TableSceneBuilder,
    LEFT_ARM_Y_OFFSET,
    RIGHT_ARM_Y_OFFSET,
)
from scenebuilders.trash_bin_mesh import ensure_trash_bin_frustum_mesh
from scenebuilders.xarm7_table_scene_builder import (
    PEDESTAL_HEIGHT,
    ROBOT_BASE_X_OFFSET,
)

TRASH_BIN_ROLLING_MAX_EPISODE_STEPS = 120

@register_env("MyDualTrashBinRolling-v0", max_episode_steps=TRASH_BIN_ROLLING_MAX_EPISODE_STEPS)
class MyDualTrashBinRollingEnv(BaseEnv):
    MAX_EPISODE_STEPS = TRASH_BIN_ROLLING_MAX_EPISODE_STEPS
    SUPPORTED_ROBOTS = [
        ("xarm7_ball_ee_kinematics_left", "xarm7_ball_ee_kinematics_right"),
        ("xarm7_ball_ee_wo_force_sensor", "xarm7_ball_ee_wo_force_sensor"),
    ]
    agent: MultiAgent[
        Tuple[
            Xarm7BallEEKinematicsLeft | Xarm7BallEEWoForceSensor,
            Xarm7BallEEKinematicsRight | Xarm7BallEEWoForceSensor,
        ]
    ]

    BIN_HEIGHT = 0.2727
    BIN_BOTTOM_RADIUS = 0.184 / 2.0
    BIN_TOP_RADIUS = 0.218 / 2.0
    BIN_RIM_RADIUS = 0.224 / 2.0
    BIN_RIM_THICKNESS = 0.003
    BIN_DENSITY = 240.0
    BIN_X_OFFSET_FROM_BASE = 0.31425
    BIN_INITIAL_Z = 0.113
    BIN_X_RANDOMIZATION_BACKWARD = 0.05
    BIN_X_RANDOMIZATION_FORWARD = 0.10
    BIN_Y_RANDOMIZATION = 0.10
    RANDOMIZE_BIN_TRANSLATION = True
    BIN_LOCAL_Z_RANDOMIZATION_DEG = 45.0
    BIN_LOCAL_X_RANDOMIZATION_DEG = 15.0
    FACE_MARK_THICKNESS = 0.003
    FRAME_AXIS_LENGTH = 0.06
    FRAME_AXIS_RADIUS = 0.003
    TCP_REACH_TARGET_DISTANCE = 0.12 + 0.02
    TCP_REACH_DISTANCE_SCALE = 10.0
    TCP_REACH_REWARD_MAX = 0.5
    FINE_ORIENTATION_REWARD_MAX = 1.0
    FINE_ORIENTATION_DISTANCE_SCALE = 50.0
    SUCCESS_LOCAL_Y_WORLD_Y = 0.995

    def __init__(
        self,
        *args,
        robot_uids=("xarm7_ball_ee_kinematics_left", "xarm7_ball_ee_kinematics_right"),
        robot_init_noise_scale: float = 1.0,
        initial_bin_xy: Optional[tuple[float, float]] = None,
        **kwargs,
    ):
        self.robot_init_noise_scale = robot_init_noise_scale
        self.initial_bin_xy = initial_bin_xy
        self._agent_obs_joint_indices: Dict[str, torch.Tensor] = dict()
        self._episode_bin_initial_xy: Optional[torch.Tensor] = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: Dict[str, Any]):
        super()._load_agent(options, [sapien.Pose(p=[0, 1, 0]), sapien.Pose(p=[0, -1, 0])])
        self._configure_observed_joint_indices()

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.1, 1.0, 0.8], [0.0, 0.0, 0.18])
        return CameraConfig(
            "render_camera", pose=pose, width=800, height=800, fov=1.0, near=0.01, far=5.0
        )

    def _load_scene(self, options: Dict[str, Any]):
        self.table_scene = DualXarm7TableSceneBuilder(env=self)
        self.table_scene.build()
        self.bimanual_center_pose = self._compute_bimanual_center_pose()

        mesh_path = ensure_trash_bin_frustum_mesh(
            height=self.BIN_HEIGHT,
            bottom_radius=self.BIN_BOTTOM_RADIUS,
            top_radius=self.BIN_TOP_RADIUS,
        )
        material = sapien.render.RenderMaterial(
            base_color=[0.85, 0.85, 0.88, 1.0],
            roughness=0.45,
            metallic=0.0,
        )
        rim_material = sapien.render.RenderMaterial(
            base_color=[0.70, 0.70, 0.74, 1.0],
            roughness=0.45,
            metallic=0.0,
        )
        physical_material = sapien.physx.PhysxMaterial(
            static_friction=2.0,
            dynamic_friction=1.8,
            restitution=0.05,
        )

        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(
            filename=str(mesh_path),
            material=physical_material,
            density=self.BIN_DENSITY,
        )
        builder.add_visual_from_file(filename=str(mesh_path), material=material)
        rim_pose = sapien.Pose(
            p=[0.0, 0.0, -self.BIN_HEIGHT / 2.0 - self.BIN_RIM_THICKNESS / 2.0],
            q=[math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0],
        )
        builder.add_cylinder_collision(
            pose=rim_pose,
            radius=self.BIN_RIM_RADIUS,
            half_length=self.BIN_RIM_THICKNESS / 2.0,
            material=physical_material,
            density=self.BIN_DENSITY,
        )
        builder.add_cylinder_visual(
            pose=rim_pose,
            radius=self.BIN_RIM_RADIUS,
            half_length=self.BIN_RIM_THICKNESS / 2.0,
            material=rim_material,
        )
        self._add_small_face_mark_visual(builder)
        builder.initial_pose = self._initial_bin_pose()
        self.trash_bin = builder.build(name="trash_bin")
        self._build_trash_bin_frame_sites()

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict[str, Any]):
        with torch.device(self.device):
            noise_scale = self.robot_init_noise_scale
            if options is not None:
                noise_scale = float(options.get("robot_init_noise_scale", noise_scale))
            if hasattr(self.table_scene, "set_noise_scale"):
                self.table_scene.set_noise_scale(noise_scale)
            self.table_scene.initialize(env_idx)

            batch_size = len(env_idx)
            if batch_size == 0:
                return

            initial_pose = self._initial_bin_pose()
            positions = torch.as_tensor(
                np.array([initial_pose.p], dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)
            if self.initial_bin_xy is None:
                orientations = self._randomized_bin_orientations(batch_size)
                if self.RANDOMIZE_BIN_TRANSLATION:
                    positions[:, 0] += self._sample_bin_x_offset(batch_size)
                    positions[:, 1] += self._sample_bin_y_offset(batch_size)
            else:
                assert len(self.initial_bin_xy) == 2, self.initial_bin_xy
                center = torch.tensor(
                    self.bimanual_center_pose.p,
                    dtype=positions.dtype,
                    device=positions.device,
                )
                positions[:, 0] = center[0] + float(self.initial_bin_xy[0])
                positions[:, 1] = center[1] + float(self.initial_bin_xy[1])
                orientations = self._fixed_initial_bin_orientations(batch_size)
            positions[:, 2] = self._initial_bin_z()
            self._reset_episode_bin_initial_xy(env_idx, positions)
            self.trash_bin.set_pose(Pose.create_from_pq(positions, orientations))
            self._update_trash_bin_frame_sites()

    def _after_control_step(self):
        self._update_trash_bin_frame_sites()

    def _initial_bin_pose(self) -> sapien.Pose:
        p = self._bimanual_center_point_to_world(
            [self.BIN_X_OFFSET_FROM_BASE, 0.0, self._initial_bin_z() - PEDESTAL_HEIGHT]
        )
        return sapien.Pose(p=p, q=[math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])

    def _initial_bin_z(self) -> float:
        return self.BIN_INITIAL_Z

    def _sample_bin_x_offset(self, batch_size: int) -> torch.Tensor:
        random_range = self.BIN_X_RANDOMIZATION_BACKWARD + self.BIN_X_RANDOMIZATION_FORWARD
        return (
            torch.rand(batch_size, dtype=torch.float32, device=self.device) * random_range
            - self.BIN_X_RANDOMIZATION_BACKWARD
        )

    def _sample_bin_y_offset(self, batch_size: int) -> torch.Tensor:
        return (
            torch.rand(batch_size, dtype=torch.float32, device=self.device) * 2.0 - 1.0
        ) * self.BIN_Y_RANDOMIZATION

    def _ensure_episode_bin_initial_xy(self):
        if (
            self._episode_bin_initial_xy is not None
            and self._episode_bin_initial_xy.shape == (self.num_envs, 2)
        ):
            return

        initial_xy = torch.tensor(
            self._initial_bin_pose().p[:2],
            dtype=torch.float32,
            device=self.device,
        )
        self._episode_bin_initial_xy = initial_xy.repeat(self.num_envs, 1)

    def _reset_episode_bin_initial_xy(self, env_idx: torch.Tensor, positions: torch.Tensor):
        self._ensure_episode_bin_initial_xy()
        self._episode_bin_initial_xy[env_idx.long()] = positions[:, :2]

    def _randomized_bin_orientations(self, batch_size: int) -> torch.Tensor:
        base_q = self._fixed_initial_bin_orientations(batch_size)
        local_z = self._sample_angle(batch_size, self.BIN_LOCAL_Z_RANDOMIZATION_DEG)
        local_x = self._sample_angle(batch_size, self.BIN_LOCAL_X_RANDOMIZATION_DEG)
        local_z_q = self._axis_angle_quat(torch.tensor([0.0, 0.0, 1.0], device=self.device), local_z)
        local_x_q = self._axis_angle_quat(torch.tensor([1.0, 0.0, 0.0], device=self.device), local_x)
        return self._quat_mul(self._quat_mul(base_q, local_z_q), local_x_q)

    def _fixed_initial_bin_orientations(self, batch_size: int) -> torch.Tensor:
        return torch.tensor(
            [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0],
            dtype=torch.float32,
            device=self.device,
        ).repeat(batch_size, 1)

    def _sample_angle(self, batch_size: int, half_range_deg: float) -> torch.Tensor:
        half_range = math.radians(half_range_deg)
        return (torch.rand(batch_size, dtype=torch.float32, device=self.device) * 2.0 - 1.0) * half_range

    def _axis_angle_quat(self, axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        half_angle = angle / 2.0
        quat = torch.zeros((angle.shape[0], 4), dtype=torch.float32, device=self.device)
        quat[:, 0] = torch.cos(half_angle)
        quat[:, 1:] = axis * torch.sin(half_angle).unsqueeze(1)
        return quat

    def _quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(dim=1)
        w2, x2, y2, z2 = q2.unbind(dim=1)
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=1,
        )

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

    def _add_small_face_mark_visual(self, builder):
        material = sapien.render.RenderMaterial(
            base_color=[0.0, 0.0, 0.0, 1.0],
            roughness=0.45,
            metallic=0.0,
        )
        z = self.BIN_HEIGHT / 2.0 + self.FACE_MARK_THICKNESS / 2.0
        half_thickness = self.FACE_MARK_THICKNESS / 2.0
        head_angle = math.radians(45.0)
        builder.add_box_visual(
            pose=sapien.Pose(p=[0.005, 0.0, z]),
            half_size=[0.050, 0.006, half_thickness],
            material=material,
        )
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[-0.040, 0.014, z],
                q=[math.cos(head_angle / 2.0), 0.0, 0.0, math.sin(head_angle / 2.0)],
            ),
            half_size=[0.028, 0.006, half_thickness],
            material=material,
        )
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[-0.040, -0.014, z],
                q=[math.cos(-head_angle / 2.0), 0.0, 0.0, math.sin(-head_angle / 2.0)],
            ),
            half_size=[0.028, 0.006, half_thickness],
            material=material,
        )

    def _build_trash_bin_frame_sites(self):
        axis_specs = self._frame_axis_specs(self.FRAME_AXIS_LENGTH, self.FRAME_AXIS_RADIUS)
        self.trash_bin_frame_sites = self._build_frame_site_set(
            "trash_bin_local_frame", axis_specs
        )

    def _frame_axis_specs(self, length: float, radius: float):
        return {
            "x": (
                [length / 2.0, radius, radius],
                [1.0, 0.05, 0.05, 1.0],
            ),
            "y": (
                [radius, length / 2.0, radius],
                [0.05, 0.85, 0.05, 1.0],
            ),
            "z": (
                [radius, radius, length / 2.0],
                [0.05, 0.20, 1.0, 1.0],
            ),
        }

    def _build_frame_site_set(self, name_prefix: str, axis_specs):
        frame_sites = {}
        for axis_name, (half_size, color) in axis_specs.items():
            builder = self.scene.create_actor_builder()
            builder.add_box_visual(
                half_size=half_size,
                material=sapien.render.RenderMaterial(base_color=color),
            )
            builder.initial_pose = sapien.Pose()
            frame_sites[axis_name] = builder.build_kinematic(
                name=f"{name_prefix}_{axis_name}"
            )
        return frame_sites

    def _update_trash_bin_frame_sites(self):
        if not hasattr(self, "trash_bin_frame_sites"):
            return
        axis_offset = self.FRAME_AXIS_LENGTH / 2.0
        origin = self._trash_bin_obs_frame_local_origin()
        local_poses = {
            "x": sapien.Pose(p=[origin[0] + axis_offset, origin[1], origin[2]]),
            "y": sapien.Pose(p=[origin[0], origin[1] + axis_offset, origin[2]]),
            "z": sapien.Pose(p=[origin[0], origin[1], origin[2] + axis_offset]),
        }
        trash_bin_pose = Pose.create(self.trash_bin.pose)
        for axis_name, local_pose in local_poses.items():
            self.trash_bin_frame_sites[axis_name].set_pose(
                trash_bin_pose * Pose.create(local_pose, device=self.device)
            )

    def _trash_bin_obs_frame_local_origin(self):
        return [0.0, 0.0, self.BIN_HEIGHT / 2.0]

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
            else:
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
        if "qvel" in obs_entry and obs_entry["qvel"] is not None:
            filtered_entry["qvel"] = self._index_select_joint_tensor(
                obs_entry["qvel"], indices
            )
        return filtered_entry

    def _index_select_joint_tensor(
        self, tensor: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        idx = indices.to(device=tensor.device, dtype=torch.long)
        return tensor.index_select(tensor.dim() - 1, idx)

    def _get_obs_extra(self, info: Dict[str, Any]):
        bin_pose = Pose.create(self.trash_bin.pose, device=self.device)
        bin_quat = bin_pose.q
        bin_quat = bin_quat / torch.linalg.norm(bin_quat, dim=1, keepdim=True).clamp_min(1e-6)
        bin_rotation = quaternion_to_matrix(bin_quat)
        frame_origin = torch.tensor(
            self._trash_bin_obs_frame_local_origin(),
            dtype=bin_rotation.dtype,
            device=self.device,
        )
        bin_position = bin_pose.p + torch.bmm(
            bin_rotation,
            frame_origin.reshape(1, 3, 1).repeat(bin_quat.shape[0], 1, 1),
        ).squeeze(-1)
        bin_rotation_6d = torch.cat(
            [bin_rotation[..., :, 0], bin_rotation[..., :, 1]],
            dim=-1,
        )
        center = torch.tensor(
            self.bimanual_center_pose.p,
            dtype=bin_position.dtype,
            device=self.device,
        )
        return {
            "trash_bin_position": bin_position - center,
            "trash_bin_rotation_6d": bin_rotation_6d,
        }

    def _bin_axis_scalars(self) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.trash_bin.pose.q
        q = q / torch.linalg.norm(q, dim=1, keepdim=True).clamp_min(1e-6)
        w, x, y, z = q.unbind(dim=1)

        # IMPORTANT: RecordEpisode tiles 8-env videos column-major with 2 rows:
        # left-top env0, left-bottom env1, second-column-top env2, second-column-bottom env3, ...
        # Keep this order in mind when comparing these per-env scalars with recorded videos.
        local_z_world_x = 2.0 * (x * z + w * y)
        local_y_world_y = 1.0 - 2.0 * (x * x + z * z)
        return local_z_world_x, local_y_world_y

    def _tcp_ball_trash_bin_force_norms(self) -> Tuple[torch.Tensor, torch.Tensor]:
        force_norms = []
        for sub_agent in self.agent.agents:
            ball_link = sub_agent.robot.links_map.get("link_tcp_ball")
            assert ball_link is not None, sub_agent.robot.links_map.keys()
            force = self.scene.get_pairwise_contact_forces(ball_link, self.trash_bin).to(
                self.device
            )
            force_norms.append(torch.linalg.norm(force, dim=-1))
        assert len(force_norms) == 2, len(force_norms)
        return force_norms[0], force_norms[1]

    def evaluate(self):
        local_z_world_x, local_y_world_y = self._bin_axis_scalars()
        left_force_norm, right_force_norm = self._tcp_ball_trash_bin_force_norms()
        final_step = self.elapsed_steps >= self.MAX_EPISODE_STEPS
        success = final_step & (
            local_y_world_y >= self.SUCCESS_LOCAL_Y_WORLD_Y
        )
        return {
            "success": success,
            "trash_bin_local_z_world_x": local_z_world_x,
            "trash_bin_local_y_world_y": local_y_world_y,
            "contact/left_tcp_ball_trash_bin_force_norm": left_force_norm,
            "contact/right_tcp_ball_trash_bin_force_norm": right_force_norm,
            "contact/max_tcp_ball_trash_bin_force_norm": torch.maximum(
                left_force_norm,
                right_force_norm,
            ),
        }

    def _farthest_tcp_to_bin_origin_dist(self) -> torch.Tensor:
        left_tcp_pos = Pose.create(self.agent.agents[0].tcp_pose, device=self.device).p
        right_tcp_pos = Pose.create(self.agent.agents[1].tcp_pose, device=self.device).p
        bin_origin = self.trash_bin.pose.p
        assert left_tcp_pos.shape == right_tcp_pos.shape == bin_origin.shape

        left_dist = torch.linalg.norm(left_tcp_pos - bin_origin, dim=1)
        right_dist = torch.linalg.norm(right_tcp_pos - bin_origin, dim=1)
        return torch.maximum(left_dist, right_dist)

    def _tcp_reaching_reward(self) -> torch.Tensor:
        farthest_dist = self._farthest_tcp_to_bin_origin_dist()
        outside_target = torch.clamp(
            farthest_dist - self.TCP_REACH_TARGET_DISTANCE,
            min=0.0,
        )
        return self.TCP_REACH_REWARD_MAX * (
            1.0 - torch.tanh(self.TCP_REACH_DISTANCE_SCALE * outside_target)
        )

    def _fine_local_y_world_y_reward(self) -> torch.Tensor:
        _, local_y_world_y = self._bin_axis_scalars()
        orientation_error = torch.clamp(
            1.0 - local_y_world_y,
            min=0.0,
        )
        return self.FINE_ORIENTATION_REWARD_MAX * (
            1.0
            - torch.tanh(
                self.FINE_ORIENTATION_DISTANCE_SCALE * orientation_error
            )
        )

    def compute_dense_reward(self, obs, action, info):
        _, local_y_world_y = self._bin_axis_scalars()
        orientation_reward = local_y_world_y + 1.0
        return (
            orientation_reward
            + self._fine_local_y_world_y_reward()
            + self._tcp_reaching_reward()
        )

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)


@register_env("MyDualTrashBinRollingRotationOnly-v0", max_episode_steps=TRASH_BIN_ROLLING_MAX_EPISODE_STEPS)
class MyDualTrashBinRollingRotationOnlyEnv(MyDualTrashBinRollingEnv):
    RANDOMIZE_BIN_TRANSLATION = False


@register_env("MyDualTrashBinRollingStage2-v0", max_episode_steps=TRASH_BIN_ROLLING_MAX_EPISODE_STEPS)
class MyDualTrashBinRollingStage2Env(MyDualTrashBinRollingEnv):
    pass
