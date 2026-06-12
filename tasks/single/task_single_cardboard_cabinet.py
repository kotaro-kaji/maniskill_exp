from dataclasses import replace
import math
from typing import Any, Dict, Optional

import numpy as np
import sapien
import torch

from mani_skill.agents.utils import get_active_joint_indices
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from robotagents.my_xarm7 import Xarm7
from robotagents.my_xarm7_small_delta import Xarm7SmallDelta
from scenebuilders.cardboard_cabinet_builder import (
    DEFAULT_CARDBOARD_CABINET_SPEC,
    build_cardboard_cabinet_actor,
    build_cardboard_inner_box_actor,
    cardboard_cabinet_panel_specs,
    cardboard_cabinet_quaternion,
    cardboard_inner_box_panel_specs,
    INNER_MARKER_PANEL_INDEX,
    make_cardboard_inner_box_spec,
    OUTER_MARKER_PANEL_INDEX,
)
from scenebuilders.dual_xarm7_table_scene_builder import (
    DualXarm7TableSceneBuilder,
    LEFT_ARM_Y_OFFSET,
)
from scenebuilders.xarm7_table_scene_builder import (
    PEDESTAL_HEIGHT,
    ROBOT_BASE_X_OFFSET,
)
from scenebuilders.xarm7_initial_randomization_scene_builder import (
    Xarm7InitialRandomizationSceneBuilder,
)


@register_env("MyDualCardboardCabinet-v1", max_episode_steps=100)
class MyDualCardboardCabinetEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["my_xarm7"]
    agent: Xarm7

    OUTER_CARDBOARD_BASE_SPEC = DEFAULT_CARDBOARD_CABINET_SPEC
    CABINET_SPEC = replace(
        OUTER_CARDBOARD_BASE_SPEC,
        color_hex="#4C78A8",
        density=2500.0,
    )
    INNER_BOX_SPEC = replace(
        make_cardboard_inner_box_spec(OUTER_CARDBOARD_BASE_SPEC),
        wall_thickness=0.002,
        density=OUTER_CARDBOARD_BASE_SPEC.density,
        static_friction=0.2,
        dynamic_friction=0.1,
    )
    INNER_BOX_WORLD_Y_OFFSET = 0.0035
    BOX_X_OFFSET_FROM_BASE = 0.30
    INSERT_TARGET_RADIUS = 0.008
    INSERT_TARGET_COLOR = (0.5, 1.0, 0.0, 1.0)
    INSERT_TARGET_SIDE_LOCAL = (0.0, 0.0, 0.00925)
    TCP_TO_TARGET_REWARD_SCALE = 5.0
    FINE_TCP_TO_TARGET_REWARD_SCALE = 30.0
    OPEN_STARTED_DISTANCE = 0.005
    INNER_BOX_OPEN_GOAL_DISTANCE = 0.12
    INNER_BOX_OPEN_REWARD_TARGET_DISTANCE = 0.125
    OUTER_BOX_STABLE_DISTANCE = 0.02
    OUTER_BOX_SHIFT_PENALTY_SCALE = 0.05 / math.atanh(0.95)
    GRIPPER_OPENING_TARGET_QPOS = 0.44
    GRIPPER_CLOSING_TARGET_QPOS = 0.62
    GRIPPER_CLOSE_DISTANCE = 0.04
    GRIPPER_OPENING_REWARD_WEIGHT = 0.05
    GRIPPER_OPENING_REWARD_SCALE = 8.0
    RETURN_TARGET_TCP_POSITION = torch.tensor(
        [-0.2891441583633423, 0.32909828424453735, 0.35221153497695923],
        dtype=torch.float32,
    )
    RETURN_TARGET_TCP_ROTATION = torch.tensor(
        [
            [1.0, 4.522452854871517e-06, 1.1265297871432267e-05],
            [4.52255198979401e-06, -1.0, -8.828912541503087e-06],
            [1.1265257853665389e-05, 8.828963473206386e-06, -1.0],
        ],
        dtype=torch.float32,
    )
    RETURN_TCP_POSITION_REWARD_MAX_DISTANCE = 0.60
    RETURN_TARGET_QPOS = torch.tensor(
        [
            -0.00001,
            -0.5236051678657532,
            0.00,
            0.7853981852531433,
            -0.00001,
            1.30899178981781,
            -0.000001,
        ],
        dtype=torch.float32,
    )

    def __init__(
        self,
        *args,
        robot_uids="my_xarm7",
        robot_init_qpos_noise=0.02,
        robot_init_noise_scale: float = 1.0,
        initial_box_xy: Optional[tuple[float, float]] = None,
        collect_rmb_data: bool = False,
        **kwargs,
    ):
        assert robot_uids in self.SUPPORTED_ROBOTS, robot_uids
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.robot_init_noise_scale = robot_init_noise_scale
        self.initial_box_xy = initial_box_xy
        self.collect_rmb_data = bool(collect_rmb_data)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: Dict[str, Any]):
        super()._load_agent(
            options,
            sapien.Pose(p=[ROBOT_BASE_X_OFFSET, LEFT_ARM_Y_OFFSET, PEDESTAL_HEIGHT]),
        )
        self._configure_observed_joint_indices()

    def _configure_observed_joint_indices(self):
        arm_joint_names = list(self.agent.arm_joint_names)
        gripper_drive_joint = self.agent.gripper_joint_names[0]
        self._obs_qpos_joint_indices = get_active_joint_indices(
            self.agent.robot,
            arm_joint_names + [gripper_drive_joint],
        ).long()
        self._obs_qvel_joint_indices = get_active_joint_indices(
            self.agent.robot,
            arm_joint_names,
        ).long()

    @property
    def _default_sensor_configs(self):
        workspace_center_pose = self._compute_workspace_center_pose()
        workspace_center_T_front_camera = np.array(
            [
                [-0.72152435, 0.30910393, -0.61956224, 0.78648193],
                [0.69229153, 0.33708384, -0.63804931, 0.72505581],
                [0.01162087, -0.88928581, -0.45720423, 0.53946321],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        rotation_np = workspace_center_T_front_camera[:3, :3]
        y_c_rotation_rad = math.radians(0.0)
        y_c_rotation = np.array(
            [
                [math.cos(y_c_rotation_rad), 0.0, math.sin(y_c_rotation_rad)],
                [0.0, 1.0, 0.0],
                [-math.sin(y_c_rotation_rad), 0.0, math.cos(y_c_rotation_rad)],
            ],
            dtype=np.float32,
        )
        rotation_np = rotation_np @ y_c_rotation
        # 実機カメラ座標とManiSkill上のカメラ座標の定義差を吸収する変換。
        # これは「カメラ座標系だけ」を入れ替える解析的な固定回転で、
        # workspace_center座標系には手を入れない。
        # 軸対応: x_new = -y_old, y_new = -z_old, z_new = x_old
        camera_axis_map = np.array(
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        rotation_np = rotation_np @ camera_axis_map
        rotation = torch.tensor(rotation_np, dtype=torch.float32)
        translation = torch.tensor(
            workspace_center_T_front_camera[:3, 3], dtype=torch.float32
        )
        front_camera_pose_in_workspace_center = Pose.create_from_pq(
            p=translation,
            q=matrix_to_quaternion(rotation),
        )
        pose = Pose.create(workspace_center_pose) * front_camera_pose_in_workspace_center
        scale = 1.0 if self.collect_rmb_data else 0.4
        # Real front camera calibration for the cardboard cabinet task.
        base_intrinsic = np.array(
            [
                [606.135498046875, 0.0, 330.1974182128906],
                [0.0, 605.4591674804688, 244.23666381835938],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        intrinsic = base_intrinsic.copy()
        intrinsic[0, 0] *= scale
        intrinsic[1, 1] *= scale
        intrinsic[0, 2] *= scale
        intrinsic[1, 2] *= scale
        return [
            CameraConfig(
                "top",
                pose=pose,
                width=int(640 * scale),
                height=int(480 * scale),
                intrinsic=intrinsic,
                fov=None,
                near=0.01,
                far=100,
            )
        ]

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
        self.workspace_center_pose = self._compute_workspace_center_pose()

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
        self.inner_marker_panel_frame_site = self._build_marker_frame_site(
            "inner_marker_panel_frame_site"
        )
        self.outer_marker_panel_frame_site = self._build_marker_frame_site(
            "outer_marker_panel_frame_site"
        )

    def _build_marker_frame_site(self, name: str):
        builder = self.scene.create_actor_builder()
        axis_length = 0.035
        axis_half_width = 0.0015
        axis_specs = [
            ([axis_length / 2, 0.0, 0.0], [axis_length / 2, axis_half_width, axis_half_width], (1.0, 0.0, 0.0, 1.0)),
            ([0.0, axis_length / 2, 0.0], [axis_half_width, axis_length / 2, axis_half_width], (0.0, 1.0, 0.0, 1.0)),
            ([0.0, 0.0, axis_length / 2], [axis_half_width, axis_half_width, axis_length / 2], (0.0, 0.2, 1.0, 1.0)),
        ]
        for local_position, half_size, color in axis_specs:
            builder.add_box_visual(
                pose=sapien.Pose(p=local_position),
                half_size=half_size,
                material=sapien.render.RenderMaterial(base_color=color),
            )
        builder.initial_pose = sapien.Pose()
        return builder.build_kinematic(name=name)

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
            if not hasattr(self, "drawer_open_success_reached"):
                self.drawer_open_success_reached = torch.zeros(
                    self.num_envs,
                    dtype=torch.bool,
                    device=self.device,
                )
            self.drawer_open_success_reached[env_idx] = False

            positions = self._sample_initial_box_world_positions(env_idx)
            self._set_initial_box_reference_positions(env_idx, positions)
            orientations = self._sample_initial_box_orientations(env_idx)
            self.cardboard_cabinet.set_pose(Pose.create_from_pq(positions, orientations))
            self.cardboard_inner_box.set_pose(
                Pose.create_from_pq(
                    self._initial_inner_box_world_positions(positions),
                    orientations,
                )
            )
            self._sync_target_sites(env_idx)
            self._sync_marker_frame_sites(env_idx)

    def _compute_workspace_center_pose(self) -> sapien.Pose:
        return sapien.Pose(p=[ROBOT_BASE_X_OFFSET, 0.0, PEDESTAL_HEIGHT])

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
            self.workspace_center_pose.p, dtype=torch.float32, device=self.device
        )
        return local + center

    def _sample_initial_box_world_positions(
        self, env_idx: torch.Tensor
    ) -> torch.Tensor:
        position = self._initial_box_world_position()
        if self.initial_box_xy is not None:
            assert len(self.initial_box_xy) == 2, self.initial_box_xy
            position = position.clone()
            center = torch.tensor(
                self.workspace_center_pose.p,
                dtype=position.dtype,
                device=position.device,
            )
            position[0] = center[0] + float(self.initial_box_xy[0])
            position[1] = center[1] + float(self.initial_box_xy[1])
        return position.unsqueeze(0).repeat(len(env_idx), 1)

    def _sample_initial_box_orientations(self, env_idx: torch.Tensor) -> torch.Tensor:
        return (
            cardboard_cabinet_quaternion(self.CABINET_SPEC, device=self.device)
            .unsqueeze(0)
            .repeat(len(env_idx), 1)
        )

    def _set_initial_box_reference_positions(
        self, env_idx: torch.Tensor, positions: torch.Tensor
    ):
        if not hasattr(self, "_initial_box_reference_positions"):
            self._initial_box_reference_positions = (
                self._initial_box_world_position()
                .unsqueeze(0)
                .repeat(self.num_envs, 1)
            )
        self._initial_box_reference_positions[env_idx.long()] = positions

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
        return self._initial_inner_box_world_positions(
            self._initial_box_world_position()
        )

    def _initial_inner_box_world_positions(
        self, box_positions: torch.Tensor
    ) -> torch.Tensor:
        positions = box_positions.clone()
        positions[..., 1] += self.INNER_BOX_WORLD_Y_OFFSET
        return positions

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

    def _actor_local_point_world(self, actor, local_point: torch.Tensor) -> torch.Tensor:
        pose = actor.pose
        matrix = pose.to_transformation_matrix()[..., :3, :3]
        position = pose.p
        if position.ndim == 1:
            position = position.unsqueeze(0)
            matrix = matrix.unsqueeze(0)
        local_point = local_point.unsqueeze(0).repeat(position.shape[0], 1)
        return torch.matmul(local_point.unsqueeze(1), matrix.transpose(-1, -2)).squeeze(
            1
        ) + position

    def _actor_rotation_6d(self, actor) -> torch.Tensor:
        matrix = actor.pose.to_transformation_matrix()[..., :3, :3]
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        return torch.cat([matrix[..., :, 0], matrix[..., :, 1]], dim=-1)

    def _box_local_point_world(self, local_point: torch.Tensor) -> torch.Tensor:
        return self._actor_local_point_world(self.cardboard_inner_box, local_point)

    def _inner_marker_panel_local_position(self) -> torch.Tensor:
        panel_pose, _ = cardboard_inner_box_panel_specs(self.INNER_BOX_SPEC)[
            INNER_MARKER_PANEL_INDEX
        ]
        return torch.tensor(panel_pose.p, dtype=torch.float32, device=self.device)

    def _outer_marker_panel_local_position(self) -> torch.Tensor:
        panel_pose, _ = cardboard_cabinet_panel_specs(self.CABINET_SPEC)[
            OUTER_MARKER_PANEL_INDEX
        ]
        return torch.tensor(panel_pose.p, dtype=torch.float32, device=self.device)

    def _marker_panel_world_pose(self, actor, local_position: torch.Tensor):
        position = self._actor_local_point_world(actor, local_position)
        quat = actor.pose.q
        if quat.ndim == 1:
            quat = quat.unsqueeze(0)
        return Pose.create_from_pq(p=position, q=quat)

    def _current_insert_target_world(self) -> torch.Tensor:
        return self._box_local_point_world(self._box_insert_target_local())

    def _tcp_position(self) -> torch.Tensor:
        position = self.agent.tcp.pose.p
        if position.ndim == 1:
            position = position.unsqueeze(0)
        return position

    def _tcp_rotation_matrix(self) -> torch.Tensor:
        matrix = self.agent.tcp.pose.to_transformation_matrix()[..., :3, :3]
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        return matrix

    def _tcp_to_target_distance(self) -> torch.Tensor:
        return torch.linalg.norm(
            self._tcp_position() - self._current_insert_target_world(),
            dim=-1,
        )

    def _tcp_to_target_reward(self) -> torch.Tensor:
        coarse_reward = 1 - torch.tanh(
            self.TCP_TO_TARGET_REWARD_SCALE * self._tcp_to_target_distance()
        )
        fine_reward = 1 - torch.tanh(
            self.FINE_TCP_TO_TARGET_REWARD_SCALE * self._tcp_to_target_distance()
        )
        return 0.5 * (coarse_reward + fine_reward)  # min = 0.0, max = 1.0

    def _inner_box_center_distance(self) -> torch.Tensor:
        return torch.linalg.norm(
            self.cardboard_inner_box.pose.p - self.cardboard_cabinet.pose.p,
            dim=-1,
        )

    def _inner_box_open_amount(self) -> torch.Tensor:
        closed_distance = abs(float(self.INNER_BOX_WORLD_Y_OFFSET))
        return torch.clamp(
            self._inner_box_center_distance() - closed_distance,
            min=0.0,
        )

    def _inner_box_open_fraction(self) -> torch.Tensor:
        return torch.clamp(
            self._inner_box_open_amount() / self.INNER_BOX_OPEN_GOAL_DISTANCE,
            min=0.0,
            max=1.0,
        )

    def _inner_box_open_target_reward(self) -> torch.Tensor:
        open_amount = self._inner_box_open_amount()
        target_distance = self.INNER_BOX_OPEN_REWARD_TARGET_DISTANCE
        below_goal = torch.clamp(
            open_amount / target_distance,
            min=0.0,
            max=1.0,
        )
        overshoot_fraction = torch.clamp(
            (open_amount - target_distance) / target_distance,
            min=0.0,
            max=1.0,
        )
        above_goal = 0.5 * (1.0 + torch.cos(math.pi * overshoot_fraction))
        return torch.where(
            open_amount <= target_distance,
            below_goal,
            above_goal,
        )

    def _inner_box_open_enough(self) -> torch.Tensor:
        return self._inner_box_open_amount() >= self.INNER_BOX_OPEN_GOAL_DISTANCE

    def _outer_box_shift(self) -> torch.Tensor:
        current_pos = self.cardboard_cabinet.pose.p
        if current_pos.ndim == 1:
            current_pos = current_pos.unsqueeze(0)
        if hasattr(self, "_initial_box_reference_positions"):
            initial_pos = self._initial_box_reference_positions[: current_pos.shape[0]]
        else:
            initial_pos = self._initial_box_world_position().reshape(1, 3)
        return torch.linalg.norm(current_pos - initial_pos, dim=-1)

    def _outer_box_shift_penalty(self) -> torch.Tensor:
        return torch.tanh(self._outer_box_shift() / self.OUTER_BOX_SHIFT_PENALTY_SCALE)

    def _outer_box_stable_enough(self) -> torch.Tensor:
        return self._outer_box_shift() <= self.OUTER_BOX_STABLE_DISTANCE

    def _gripper_drive_qpos(self) -> torch.Tensor:
        qpos = self.agent.robot.get_qpos()
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        return qpos[:, 7]

    def _gripper_target_qpos(self) -> torch.Tensor:
        close_gripper = (self._tcp_to_target_distance() <= self.GRIPPER_CLOSE_DISTANCE) | (
            self._inner_box_open_amount() >= self.OPEN_STARTED_DISTANCE
        )
        return torch.where(
            close_gripper,
            torch.full_like(self._gripper_drive_qpos(), self.GRIPPER_CLOSING_TARGET_QPOS),
            torch.full_like(self._gripper_drive_qpos(), self.GRIPPER_OPENING_TARGET_QPOS),
        )

    def _gripper_opening_reward(self) -> torch.Tensor:
        gripper_error = torch.abs(
            self._gripper_drive_qpos() - self._gripper_target_qpos()
        )
        return 1 - torch.tanh(
            self.GRIPPER_OPENING_REWARD_SCALE * gripper_error
        )  # min = 0.0, max = 1.0

    def _return_to_target_qpos_reward(self) -> torch.Tensor:
        return self._return_to_target_arm_qpos_reward()

    def _return_target_tcp_position(self) -> torch.Tensor:
        return self.RETURN_TARGET_TCP_POSITION.to(
            device=self.device, dtype=torch.float32
        ).reshape(1, 3)

    def _return_target_tcp_rotation(self) -> torch.Tensor:
        return self.RETURN_TARGET_TCP_ROTATION.to(
            device=self.device, dtype=torch.float32
        ).reshape(1, 3, 3)

    def _return_to_target_tcp_position_error(self) -> torch.Tensor:
        return torch.linalg.norm(
            self._tcp_position() - self._return_target_tcp_position(),
            dim=-1,
        )

    def _return_to_target_tcp_euler_abs_error(self) -> torch.Tensor:
        current = self._tcp_rotation_matrix()
        target = self._return_target_tcp_rotation()
        relative = torch.matmul(target.transpose(-1, -2), current)
        pitch_arg = torch.clamp(-relative[:, 2, 0], min=-1.0, max=1.0)
        roll = torch.atan2(relative[:, 2, 1], relative[:, 2, 2])
        pitch = torch.asin(pitch_arg)
        yaw = torch.atan2(relative[:, 1, 0], relative[:, 0, 0])
        return torch.abs(torch.stack([roll, pitch, yaw], dim=-1))

    def _return_to_target_tcp_pose_reward(self) -> torch.Tensor:
        return torch.clamp(
            1.0
            - self._return_to_target_tcp_position_error()
            / self.RETURN_TCP_POSITION_REWARD_MAX_DISTANCE,
            min=0.0,
            max=1.0,
        )

    def _return_to_target_arm_qpos_reward(self):
        qpos = self.agent.robot.get_qpos()
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        target = self.RETURN_TARGET_QPOS.to(device=self.device, dtype=torch.float32)
        assert target.numel() == 7, target.numel()
        assert qpos.shape[-1] >= 7, qpos.shape[-1]
        joint_scores = 1.0 - torch.tanh(torch.abs(qpos[:, :7] - target))
        return joint_scores.mean(dim=-1)

    def _sync_target_sites(self, env_idx: Optional[torch.Tensor] = None):
        target_pos = self._current_insert_target_world()
        if env_idx is not None:
            target_pos = target_pos[env_idx]
        self.insert_target_site.set_pose(Pose.create_from_pq(p=target_pos))

    def _sync_marker_frame_sites(self, env_idx: Optional[torch.Tensor] = None):
        inner_pose = self._marker_panel_world_pose(
            self.cardboard_inner_box,
            self._inner_marker_panel_local_position(),
        )
        outer_pose = self._marker_panel_world_pose(
            self.cardboard_cabinet,
            self._outer_marker_panel_local_position(),
        )
        if env_idx is not None:
            inner_pose = Pose.create_from_pq(
                p=inner_pose.p[env_idx],
                q=inner_pose.q[env_idx],
            )
            outer_pose = Pose.create_from_pq(
                p=outer_pose.p[env_idx],
                q=outer_pose.q[env_idx],
            )
        self.inner_marker_panel_frame_site.set_pose(inner_pose)
        self.outer_marker_panel_frame_site.set_pose(outer_pose)

    def evaluate(self):
        self._sync_target_sites()
        self._sync_marker_frame_sites()
        open_enough = self._inner_box_open_enough()
        outer_box_stable_enough = self._outer_box_stable_enough()
        return_arm_reward = self._return_to_target_arm_qpos_reward()
        return_to_target_qpos_reward = self._return_to_target_qpos_reward()
        return_to_target_tcp_pose_reward = self._return_to_target_tcp_pose_reward()
        return_tcp_euler_abs_error = self._return_to_target_tcp_euler_abs_error()
        drawer_open_success = open_enough & outer_box_stable_enough
        if not hasattr(self, "drawer_open_success_reached"):
            self.drawer_open_success_reached = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        self.drawer_open_success_reached = drawer_open_success
        return {
            "tcp_to_target_distance": self._tcp_to_target_distance(),
            "inner_box_open_amount": self._inner_box_open_amount(),
            "inner_box_open_fraction": self._inner_box_open_fraction(),
            "inner_box_open_target_reward": self._inner_box_open_target_reward(),
            "open_enough": open_enough,
            "outer_box_shift": self._outer_box_shift(),
            "outer_box_shift_penalty": self._outer_box_shift_penalty(),
            "outer_box_stable_enough": outer_box_stable_enough,
            "gripper_drive_qpos": self._gripper_drive_qpos(),
            "gripper_target_qpos": self._gripper_target_qpos(),
            "return_to_target_qpos_reward": return_to_target_qpos_reward,
            "return_to_target_tcp_pose_reward": return_to_target_tcp_pose_reward,
            "return_tcp_position_error": self._return_to_target_tcp_position_error(),
            "return_tcp_roll_abs_error": return_tcp_euler_abs_error[:, 0],
            "return_tcp_pitch_abs_error": return_tcp_euler_abs_error[:, 1],
            "return_tcp_yaw_abs_error": return_tcp_euler_abs_error[:, 2],
            "return_arm_qpos_reward": return_arm_reward,
            "drawer_open_success": drawer_open_success,
            "drawer_open_success_reached": self.drawer_open_success_reached,
        }

    def _get_obs_extra(self, info: Dict[str, Any]):
        inner_marker_panel_position = self._actor_local_point_world(
            self.cardboard_inner_box,
            self._inner_marker_panel_local_position(),
        )
        outer_marker_panel_position = self._actor_local_point_world(
            self.cardboard_cabinet,
            self._outer_marker_panel_local_position(),
        )
        center = torch.tensor(
            self.workspace_center_pose.p,
            dtype=inner_marker_panel_position.dtype,
            device=inner_marker_panel_position.device,
        )
        return {
            "inner_marker_panel_position": inner_marker_panel_position - center,
            "inner_marker_panel_rotation_6d": self._actor_rotation_6d(
                self.cardboard_inner_box
            ),
            "outer_marker_panel_position": outer_marker_panel_position - center,
            "outer_marker_panel_rotation_6d": self._actor_rotation_6d(
                self.cardboard_cabinet
            ),
        }

    def _get_obs_agent(self):
        obs = super()._get_obs_agent()
        qpos_indices = self._obs_qpos_joint_indices.to(
            device=obs["qpos"].device,
            dtype=torch.long,
        )
        qvel_indices = self._obs_qvel_joint_indices.to(
            device=obs["qvel"].device,
            dtype=torch.long,
        )
        dim = obs["qpos"].dim() - 1
        obs["qpos"] = obs["qpos"].index_select(dim, qpos_indices)
        obs["qvel"] = obs["qvel"].index_select(dim, qvel_indices)
        return obs

    def compute_normalized_dense_reward(self, obs, action, info):
        reaching_reward = self._tcp_to_target_reward()
        open_amount = self._inner_box_open_amount()
        outer_box_stability = 1.0 - self._outer_box_shift_penalty()
        stable_open_target_reward = (
            self._inner_box_open_target_reward() * outer_box_stability
        )
        open_reward = 2.0 * stable_open_target_reward

        open_started = open_amount >= self.OPEN_STARTED_DISTANCE
        reaching_reward = torch.where(
            open_started,
            torch.full_like(reaching_reward, 2.0),
            reaching_reward,
        )  # min = 0.0, max = 2.0
        open_reward = torch.where(
            info["open_enough"],
            3.0 * stable_open_target_reward,
            open_reward,
        )  # min = 0.0, max = 3.0

        gripper_reward = (
            self.GRIPPER_OPENING_REWARD_WEIGHT * self._gripper_opening_reward()
        )
        reward = reaching_reward + open_reward + gripper_reward
        stage_return_mask = info["drawer_open_success_reached"]
        stage_return_reward = (
            4.0 * stable_open_target_reward
            + 4.0 * self._return_to_target_tcp_pose_reward()
        )
        reward = torch.where(
            stage_return_mask,
            stage_return_reward,
            outer_box_stability * reward,
        )
        return reward / (8.0 + self.GRIPPER_OPENING_REWARD_WEIGHT)


@register_env("MyDualCardboardCabinetSmallDelta-v1", max_episode_steps=200)
class MyDualCardboardCabinetSmallDeltaEnv(MyDualCardboardCabinetEnv):
    SUPPORTED_ROBOTS = [
        "my_xarm7_delta001",
        "my_xarm7_delta003",
        "my_xarm7_delta03",
    ]
    agent: Xarm7SmallDelta

    def __init__(self, *args, robot_uids="my_xarm7_delta003", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)


@register_env("MySingleCardboardCabinetRandomized-v1", max_episode_steps=100)
class MySingleCardboardCabinetRandomizedEnv(MyDualCardboardCabinetEnv):
    BOX_X_OFFSET_FROM_BASE = 0.3373
    BOX_POSITION_NOISE_LOW = (-0.01, -0.02, 0.0)
    BOX_POSITION_NOISE_HIGH = (0.01, 0.02, 0.0)
    BOX_YAW_NOISE_DEG = 5.0

    def __init__(self, *args, robot_init_noise_scale: float = 0.25, **kwargs):
        super().__init__(
            *args,
            robot_init_noise_scale=robot_init_noise_scale,
            **kwargs,
        )

    def _load_scene(self, options: Dict[str, Any]):
        super()._load_scene(options)
        self.table_scene.initial_qpos = (
            Xarm7InitialRandomizationSceneBuilder
            ._RESET_STATE_OF_ROBOMANIPBASELINES
            .clone()
        )

    def _sample_initial_box_world_positions(
        self, env_idx: torch.Tensor
    ) -> torch.Tensor:
        positions = super()._sample_initial_box_world_positions(env_idx)
        if self.initial_box_xy is not None:
            return positions
        noise_low = torch.tensor(
            self.BOX_POSITION_NOISE_LOW,
            dtype=torch.float32,
            device=self.device,
        )
        noise_high = torch.tensor(
            self.BOX_POSITION_NOISE_HIGH,
            dtype=torch.float32,
            device=self.device,
        )
        noise = noise_low + torch.rand_like(positions) * (noise_high - noise_low)
        return positions + noise

    def _sample_initial_box_orientations(self, env_idx: torch.Tensor) -> torch.Tensor:
        if self.initial_box_xy is not None:
            return super()._sample_initial_box_orientations(env_idx)
        base_yaw = math.radians(self.CABINET_SPEC.yaw_deg)
        yaw_noise = (
            2.0 * torch.rand((len(env_idx),), dtype=torch.float32, device=self.device)
            - 1.0
        ) * math.radians(self.BOX_YAW_NOISE_DEG)
        half_yaw = 0.5 * (base_yaw + yaw_noise)
        orientations = torch.zeros(
            (len(env_idx), 4), dtype=torch.float32, device=self.device
        )
        orientations[:, 0] = torch.cos(half_yaw)
        orientations[:, 3] = torch.sin(half_yaw)
        return orientations


@register_env("MySingleCardboardCabinetRandomized-v2", max_episode_steps=100)
class MySingleCardboardCabinetRandomizedWideYEnv(MySingleCardboardCabinetRandomizedEnv):
    BOX_POSITION_NOISE_LOW = (-0.03, -0.06, 0.0)
    BOX_POSITION_NOISE_HIGH = (0.03, 0.06, 0.0)


@register_env("MyDualCardboardCabinetNoGripperReward-v1", max_episode_steps=100)
class MyDualCardboardCabinetNoGripperRewardEnv(MyDualCardboardCabinetEnv):
    GRIPPER_OPENING_REWARD_WEIGHT = 0.0


@register_env("MyDualCardboardCabinetLegacyRotation6D-v1", max_episode_steps=100)
class MyDualCardboardCabinetLegacyRotation6DEnv(MyDualCardboardCabinetEnv):
    def _actor_rotation_6d(self, actor) -> torch.Tensor:
        matrix = actor.pose.to_transformation_matrix()[..., :3, :3]
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
        return matrix[..., :, :2].reshape(matrix.shape[0], 6)


@register_env("MyDualCardboardCabinet66de62b-v1", max_episode_steps=100)
class MyDualCardboardCabinet66de62bEnv(MyDualCardboardCabinetLegacyRotation6DEnv):
    RETURN_TARGET_QPOS_SUCCESS_SCORE = 0.80
    RETURN_ARM_REWARD_WEIGHT = 0.75
    RETURN_GRIPPER_REWARD_WEIGHT = 0.25
    RETURN_TARGET_QPOS = torch.tensor(
        [
            -0.00001,
            -0.5236051678657532,
            0.00,
            0.7853981852531433,
            -0.00001,
            1.30899178981781,
            -0.000001,
            0.7310,
            0.7310,
            0.7310,
            0.7310,
            0.7310,
            0.7310,
        ],
        dtype=torch.float32,
    )

    def _return_to_target_qpos_reward(self) -> torch.Tensor:
        arm_score, gripper_score = self._return_to_target_qpos_component_rewards()
        return (
            self.RETURN_ARM_REWARD_WEIGHT * arm_score
            + self.RETURN_GRIPPER_REWARD_WEIGHT * gripper_score
        )

    def _return_to_target_qpos_component_rewards(self):
        qpos = self.agent.robot.get_qpos()
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        target = self.RETURN_TARGET_QPOS.to(device=self.device, dtype=torch.float32)
        assert qpos.shape[-1] == target.numel(), (qpos.shape[-1], target.numel())
        joint_scores = 1.0 - torch.tanh(torch.abs(qpos - target))
        return joint_scores[:, :7].mean(dim=-1), joint_scores[:, 7:].mean(dim=-1)

    def evaluate(self):
        self._sync_target_sites()
        self._sync_marker_frame_sites()
        open_enough = self._inner_box_open_enough()
        outer_box_stable_enough = self._outer_box_stable_enough()
        return_arm_reward, return_gripper_reward = (
            self._return_to_target_qpos_component_rewards()
        )
        return_to_target_qpos_reward = self._return_to_target_qpos_reward()
        return_pose_enough = (
            return_to_target_qpos_reward >= self.RETURN_TARGET_QPOS_SUCCESS_SCORE
        )
        drawer_open_success = open_enough & outer_box_stable_enough
        if not hasattr(self, "drawer_open_success_reached"):
            self.drawer_open_success_reached = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        self.drawer_open_success_reached = (
            self.drawer_open_success_reached | drawer_open_success
        )
        return {
            "tcp_to_target_distance": self._tcp_to_target_distance(),
            "inner_box_open_amount": self._inner_box_open_amount(),
            "inner_box_open_fraction": self._inner_box_open_fraction(),
            "open_enough": open_enough,
            "outer_box_shift": self._outer_box_shift(),
            "outer_box_shift_penalty": self._outer_box_shift_penalty(),
            "outer_box_stable_enough": outer_box_stable_enough,
            "gripper_drive_qpos": self._gripper_drive_qpos(),
            "gripper_target_qpos": self._gripper_target_qpos(),
            "return_to_target_qpos_reward": return_to_target_qpos_reward,
            "return_arm_qpos_reward": return_arm_reward,
            "return_gripper_qpos_reward": return_gripper_reward,
            "return_pose_enough": return_pose_enough,
            "drawer_open_success": drawer_open_success,
            "drawer_open_success_reached": self.drawer_open_success_reached,
            "success": (
                self.drawer_open_success_reached
                & outer_box_stable_enough
                & return_pose_enough
            ),
        }

    def _get_obs_extra(self, info: Dict[str, Any]):
        inner_marker_panel_position = self._actor_local_point_world(
            self.cardboard_inner_box,
            self._inner_marker_panel_local_position(),
        )
        outer_marker_panel_position = self._actor_local_point_world(
            self.cardboard_cabinet,
            self._outer_marker_panel_local_position(),
        )
        return {
            "inner_marker_panel_position": inner_marker_panel_position,
            "inner_marker_panel_rotation_6d": self._actor_rotation_6d(
                self.cardboard_inner_box
            ),
            "outer_marker_panel_position": outer_marker_panel_position,
            "outer_marker_panel_rotation_6d": self._actor_rotation_6d(
                self.cardboard_cabinet
            ),
        }

    def compute_normalized_dense_reward(self, obs, action, info):
        reaching_reward = self._tcp_to_target_reward()
        open_amount = self._inner_box_open_amount()
        outer_box_stability = 1.0 - self._outer_box_shift_penalty()
        stable_open_fraction = self._inner_box_open_fraction() * outer_box_stability
        open_reward = 2.0 * stable_open_fraction

        open_started = open_amount >= self.OPEN_STARTED_DISTANCE
        reaching_reward = torch.where(
            open_started,
            torch.full_like(reaching_reward, 2.0),
            reaching_reward,
        )
        open_reward = torch.where(
            info["open_enough"],
            torch.full_like(open_reward, 3.0),
            open_reward,
        )

        gripper_reward = (
            self.GRIPPER_OPENING_REWARD_WEIGHT * self._gripper_opening_reward()
        )
        reward = reaching_reward + open_reward + gripper_reward
        stage_return_mask = info["drawer_open_success_reached"]
        stage_return_reward = 4.0 + 4.0 * self._return_to_target_qpos_reward()
        reward = torch.where(stage_return_mask, stage_return_reward, reward)
        reward = outer_box_stability * reward
        return reward / (8.0 + self.GRIPPER_OPENING_REWARD_WEIGHT)
