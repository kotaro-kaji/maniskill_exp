import sapien
import sapien.render
import torch
import torch.nn.functional as F

from typing import Any, Dict, Optional

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.utils import get_active_joint_indices
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from robotagents.my_xarm7_official import Xarm7Official
import robotagents.my_xarm7_mjcf
from scenebuilders.xarm7_joint_hold_scene_builder import (
    Xarm7JointHoldSceneBuilder,
)


MARKER_NORMAL_OFFSET = 0.2
ALIGNMENT_TOLERANCE = 0.02
POSITION_REWARD_LENGTH_SCALE = 0.05

MAX_SUCCESSES_PER_EPISODE = 3

VELOCITY_PENALTY_THRESHOLD = 0.15
VELOCITY_PENALTY_SCALE = 0.05
VELOCITY_PENALTY_EXP_MAX = 10.0

DEFAULT_MARKER_POSITION = torch.tensor([-0.15, 0.0, 0.0], dtype=torch.float32)
DEFAULT_MARKER_ORIENTATION = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
DEFAULT_MARKER_POSE = torch.cat((DEFAULT_MARKER_POSITION, DEFAULT_MARKER_ORIENTATION))

MARKER_BASE_HALF_SIZE = (0.04, 0.04, 0.002)
MARKER_GREEN_HALF_SIZE = (MARKER_BASE_HALF_SIZE[0], 0.003, 0.001)
MARKER_GREEN_Y_OFFSET = MARKER_BASE_HALF_SIZE[1] - MARKER_GREEN_HALF_SIZE[1]
MARKER_GREEN_Z_OFFSET = MARKER_BASE_HALF_SIZE[2] + MARKER_GREEN_HALF_SIZE[2]


@register_env("MyEEAlignMarker-v0", max_episode_steps=100)
class MyEEAlignMarkerEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["my_xarm7_official", "my_xarm7_over", "my_xarm7_mjcf"]
    agent: Xarm7Official

    def __init__(
        self,
        *args,
        robot_uids: str = "my_xarm7_official",
        marker_pose: Optional[Any] = None,
        **kwargs,
    ):
        if marker_pose is None:
            self._configured_marker_pose_base: Optional[Pose] = None
        else:
            self._configured_marker_pose_base = Pose.create(marker_pose)
        self._success_counts: Optional[torch.Tensor] = None
        self._obs_joint_indices: Optional[torch.Tensor] = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        base_pose = sapien.Pose(p=[0.0, 0.0, 0.0])
        super()._load_agent(options, base_pose)
        self._configure_observed_joint_indices()

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        self.table_scene: Xarm7JointHoldSceneBuilder = Xarm7JointHoldSceneBuilder(env=self)
        self.table_scene.build()

        builder = self.scene.create_actor_builder()
        builder.add_box_visual(
            half_size=list(MARKER_BASE_HALF_SIZE),
            material=sapien.render.RenderMaterial(base_color=[0.0, 0.0, 0.0, 1.0]),
        )
        green_material = sapien.render.RenderMaterial(base_color=[0.05, 0.9, 0.05, 1.0])
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[0.0, MARKER_GREEN_Y_OFFSET, MARKER_GREEN_Z_OFFSET]
            ),
            half_size=list(MARKER_GREEN_HALF_SIZE),
            material=green_material,
        )
        builder.add_box_visual(
            pose=sapien.Pose(
                p=[0.0, -MARKER_GREEN_Y_OFFSET, MARKER_GREEN_Z_OFFSET]
            ),
            half_size=list(MARKER_GREEN_HALF_SIZE),
            material=green_material,
        )
        builder.initial_pose = sapien.Pose()
        self.marker = builder.build_kinematic(name="alignment_marker")

    def _configure_observed_joint_indices(self):
        observed_joint_names = []
        arm_joint_names = getattr(self.agent, "arm_joint_names", None)
        if arm_joint_names is not None:
            observed_joint_names.extend(list(arm_joint_names))
        gripper_joint_names = getattr(self.agent, "gripper_joint_names", None)
        if gripper_joint_names:
            drive_joint = gripper_joint_names[0]
            if drive_joint not in observed_joint_names:
                observed_joint_names.append(drive_joint)
        if not observed_joint_names:
            self._obs_joint_indices = None
            if hasattr(self.agent, "_obs_joint_indices"):
                delattr(self.agent, "_obs_joint_indices")
            return
        indices = get_active_joint_indices(self.agent.robot, observed_joint_names).long()
        self._obs_joint_indices = indices
        self.agent._obs_joint_indices = indices

    def _clear(self):
        self._close_viewer()
        self.agent = None
        self._sensors = dict()
        self._human_render_cameras = dict()
        self.scene = None
        self._hidden_objects = []
        self._success_counts = None
        self._obs_joint_indices = None
        try:
            import gc as _gc

            _gc.collect()
        except Exception:
            pass

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            self._initialize_success_counters(env_idx)
            marker_pose_base = self._resolve_marker_pose(env_idx, options)
            if (
                self._configured_marker_pose_base is None
                and (options is None or options.get("marker_pose") is None)
            ):
                marker_pose_base = self._randomize_marker_pose_in_base(
                    marker_pose_base, env_idx
                )
            marker_pose_world = self._convert_pose_base_to_world(
                marker_pose_base, env_idx
            )
            self.marker.set_pose(marker_pose_world)

    def _get_robot_base_pose_on_device(
        self, env_idx: Optional[torch.Tensor] = None
    ) -> Pose:
        base_pose = self.agent.robot.pose
        if env_idx is not None:
            index = env_idx
            if not isinstance(index, torch.Tensor):
                index = torch.as_tensor(index, dtype=torch.long, device=base_pose.device)
            else:
                index = index.to(device=base_pose.device, dtype=torch.long)
            base_pose = Pose.create(base_pose.raw_pose.index_select(0, index))
        if base_pose.device != self.device:
            base_pose = base_pose.to(self.device)
        return base_pose

    def _convert_pose_world_to_base(
        self, pose_world: Pose, env_idx: Optional[torch.Tensor] = None
    ) -> Pose:
        base_pose = self._get_robot_base_pose_on_device(env_idx)
        pose_world = Pose.create(pose_world, device=self.device)
        if len(pose_world) == 1 and len(base_pose) > 1:
            pose_world = Pose.create(
                pose_world.raw_pose.repeat(len(base_pose), 1), device=self.device
            )
        elif len(pose_world) != len(base_pose):
            pose_world = Pose.create(
                pose_world.raw_pose[: len(base_pose)], device=self.device
            )
        return base_pose.inv() * pose_world

    def _convert_pose_base_to_world(
        self, pose_base: Pose, env_idx: Optional[torch.Tensor] = None
    ) -> Pose:
        base_pose = self._get_robot_base_pose_on_device(env_idx)
        pose_base = Pose.create(pose_base, device=self.device)
        if len(pose_base) == 1 and len(base_pose) > 1:
            pose_base = Pose.create(
                pose_base.raw_pose.repeat(len(base_pose), 1), device=self.device
            )
        elif len(pose_base) != len(base_pose):
            pose_base = Pose.create(
                pose_base.raw_pose[: len(base_pose)], device=self.device
            )
        return base_pose * pose_base

    def _default_marker_pose_in_base(self, env_idx: torch.Tensor) -> Pose:
        batch_size = len(env_idx)
        pose_world = Pose.create(DEFAULT_MARKER_POSE, device=self.device)
        if len(pose_world) == 1 and batch_size > 1:
            pose_world = Pose.create(
                pose_world.raw_pose.repeat(batch_size, 1), device=self.device
            )
        elif len(pose_world) > batch_size:
            pose_world = Pose.create(
                pose_world.raw_pose[:batch_size], device=self.device
            )
        return self._convert_pose_world_to_base(pose_world, env_idx)

    def _resolve_marker_pose(
        self, env_idx: torch.Tensor, options: Optional[dict]
    ) -> Pose:
        batch_size = len(env_idx)
        if options is not None and options.get("marker_pose") is not None:
            base_value = options["marker_pose"]
        elif self._configured_marker_pose_base is not None:
            base_value = self._configured_marker_pose_base
        else:
            base_value = self._default_marker_pose_in_base(env_idx)

        pose = Pose.create(base_value, device=self.device)
        if len(pose) == 1 and batch_size > 1:
            pose = Pose.create(pose.raw_pose.repeat(batch_size, 1), device=self.device)
        elif len(pose) == self.scene.num_envs and batch_size != self.scene.num_envs:
            pose = Pose.create(
                pose.raw_pose.index_select(
                    0,
                    env_idx.to(device=pose.device, dtype=torch.long),
                ),
                device=self.device,
            )
        elif len(pose) != batch_size:
            if len(pose) < batch_size:
                pose = Pose.create(
                    pose.raw_pose[0:1].repeat(batch_size, 1), device=self.device
                )
            else:
                pose = Pose.create(pose.raw_pose[:batch_size], device=self.device)
        return pose

    def _initialize_success_counters(self, env_idx: torch.Tensor):
        env_idx = env_idx.to(device=self.device, dtype=torch.long)
        if self._success_counts is None or len(self._success_counts) != self.num_envs:
            self._success_counts = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
        elif self._success_counts.device != self.device:
            self._success_counts = self._success_counts.to(self.device)
        self._success_counts.index_fill_(0, env_idx, 0)

    def _randomize_marker_pose_in_base(
        self, marker_pose_base: Pose, env_idx: torch.Tensor
    ) -> Pose:
        batch_size = len(marker_pose_base)
        rand_x_world = torch.rand((batch_size,), device=self.device) * 0.2 - 0.35
        rand_y_world = torch.rand((batch_size,), device=self.device) * 0.2 - 0.1
        rand_z_world = torch.full(
            (batch_size,), DEFAULT_MARKER_POSITION[2], device=self.device
        )
        randomized_positions_world = torch.stack(
            (rand_x_world, rand_y_world, rand_z_world), dim=-1
        )
        marker_pose_world = self._convert_pose_base_to_world(marker_pose_base, env_idx)
        base_quats_world = marker_pose_world.q
        # Randomize yaw about the world z-axis while keeping roll/pitch fixed
        rand_yaw = torch.rand((batch_size,), device=self.device) * 2 * torch.pi - torch.pi
        half_yaw = rand_yaw * 0.5
        sin_half = torch.sin(half_yaw)
        cos_half = torch.cos(half_yaw)
        zeros = torch.zeros_like(sin_half)
        yaw_quats_world = torch.stack(
            (cos_half, zeros, zeros, sin_half),
            dim=-1,
        )
        w1, x1, y1, z1 = yaw_quats_world.unbind(dim=-1)
        w2, x2, y2, z2 = base_quats_world.unbind(dim=-1)
        randomized_orientations_world = torch.stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            dim=-1,
        )
        quat_norm = torch.linalg.norm(randomized_orientations_world, dim=-1, keepdim=True)
        randomized_orientations_world = randomized_orientations_world / torch.clamp(
            quat_norm, min=1e-6
        )
        randomized_world_pose = Pose.create_from_pq(
            p=randomized_positions_world,
            q=randomized_orientations_world,
            device=self.device,
        )
        return self._convert_pose_world_to_base(randomized_world_pose, env_idx)

    def _set_marker_pose_world(self, pose_world: Pose, env_idx: torch.Tensor):
        pose_world = Pose.create(pose_world, device=self.device)
        if env_idx.ndim == 0:
            env_idx = env_idx.unsqueeze(0)
        env_idx = env_idx.to(device=self.device, dtype=torch.long)
        reset_mask = getattr(self.scene, "_reset_mask", None)
        prev_reset_mask = None
        if reset_mask is not None:
            prev_reset_mask = reset_mask.clone()
            reset_mask[:] = False
            reset_mask[env_idx] = True
        try:
            self.marker.set_pose(pose_world)
            if self.scene.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene._gpu_fetch_all()
        finally:
            if reset_mask is not None and prev_reset_mask is not None:
                reset_mask.copy_(prev_reset_mask)

    def _respawn_marker(self, env_idx: torch.Tensor):
        if env_idx.numel() == 0:
            return
        env_idx = env_idx.to(device=self.device, dtype=torch.long)
        marker_pose_base = self._get_marker_pose_in_base_frame()
        selected_pose_base = Pose.create(
            marker_pose_base.raw_pose.index_select(0, env_idx), device=self.device
        )
        randomized_pose_base = self._randomize_marker_pose_in_base(
            selected_pose_base, env_idx
        )
        randomized_pose_world = self._convert_pose_base_to_world(
            randomized_pose_base, env_idx
        )
        self._set_marker_pose_world(randomized_pose_world, env_idx)

    def _get_marker_pose_on_device(self) -> Pose:
        marker_pose = self.marker.pose
        if marker_pose.device != self.device:
            marker_pose = marker_pose.to(self.device)
        return marker_pose

    def _get_marker_pose_in_base_frame(self) -> Pose:
        return self._convert_pose_world_to_base(self._get_marker_pose_on_device())

    def _get_tcp_pose_on_device(self) -> Pose:
        tcp_pose = self.agent.tcp.pose
        if tcp_pose.device != self.device:
            tcp_pose = tcp_pose.to(self.device)
        return tcp_pose

    def _marker_frame_axes(self):
        marker_pose = self._get_marker_pose_on_device()
        quat = marker_pose.q
        quat_norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
        safe_quat = quat / torch.clamp(quat_norm, min=1e-6)
        if torch.any(torch.lt(quat_norm, 1e-6)):
            identity = torch.tensor(
                [1.0, 0.0, 0.0, 0.0],
                dtype=safe_quat.dtype,
                device=safe_quat.device,
            )
            mask = torch.lt(quat_norm.squeeze(-1), 1e-6)
            safe_quat[mask] = identity
        rot_pose = Pose.create_from_pq(q=safe_quat, device=safe_quat.device)
        rotation_matrix = rot_pose.to_transformation_matrix()[..., :3, :3]
        x_axis = rotation_matrix[..., :, 0]
        y_axis = rotation_matrix[..., :, 1]
        normal = F.normalize(torch.cross(x_axis, y_axis, dim=-1), dim=-1, eps=1e-6)
        return marker_pose, x_axis, y_axis, normal

    def evaluate(self):
        marker_pose, _, _, normal = self._marker_frame_axes()
        target_point = marker_pose.p + normal * MARKER_NORMAL_OFFSET
        tcp_pose = self._get_tcp_pose_on_device()
        tcp_position = tcp_pose.p
        distance = torch.linalg.norm(tcp_position - target_point, dim=1)
        success = distance < ALIGNMENT_TOLERANCE
        return {
            "success": success,
            "target_distance": distance,
        }

    def _get_obs_agent(self):
        obs = super()._get_obs_agent()
        indices = getattr(self.agent, "_obs_joint_indices", None)
        if indices is None:
            indices = getattr(self, "_obs_joint_indices", None)
        if indices is None:
            return obs
        if "qpos" in obs:
            dim = obs["qpos"].dim() - 1
            idx = indices.to(device=obs["qpos"].device, dtype=torch.long)
            obs["qpos"] = obs["qpos"].index_select(dim, idx)
        if "qvel" in obs:
            dim = obs["qvel"].dim() - 1
            idx = indices.to(device=obs["qvel"].device, dtype=torch.long)
            obs["qvel"] = obs["qvel"].index_select(dim, idx)
        return obs

    def _get_obs_extra(self, info: Dict):
        marker_pose_base = self._get_marker_pose_in_base_frame()
        rotation_matrix = marker_pose_base.to_transformation_matrix()[..., :3, :3]
        x_axis = rotation_matrix[..., :, 0]
        y_axis = rotation_matrix[..., :, 1]
        # Position plus first two axes of marker frame in base coordinates
        marker_obs = torch.cat([marker_pose_base.p, x_axis, y_axis], dim=-1)
        return {
            "marker_pose": marker_obs,
        }

    def _get_observed_qvel(self) -> torch.Tensor:
        qvel = self.agent.robot.get_qvel()
        if qvel.ndim == 1:
            qvel = qvel.unsqueeze(0)
        if hasattr(self.agent, "_obs_joint_indices"):
            idx = self.agent._obs_joint_indices.to(device=qvel.device, dtype=torch.long)
            dim = qvel.dim() - 1
            qvel = qvel.index_select(dim, idx)
        return qvel

    def step(self, action):
        action = self._step_action(action)
        self._elapsed_steps += 1
        info = self.get_info()
        raw_obs = self.get_obs(info, unflattened=True)
        reward = self.get_reward(obs=raw_obs, action=action, info=info)

        success = info.get("success")
        if success is not None:
            if not isinstance(success, torch.Tensor):
                success_tensor = torch.as_tensor(success, device=self.device)
                if success_tensor.ndim == 0:
                    success_tensor = success_tensor.unsqueeze(0)
            else:
                success_tensor = success.to(device=self.device)
            if success_tensor.ndim == 0:
                success_tensor = success_tensor.unsqueeze(0)
            success_envs = torch.nonzero(success_tensor, as_tuple=False).squeeze(-1)
            if success_envs.numel() > 0:
                success_envs = success_envs.to(device=self.device, dtype=torch.long)
                if self._success_counts is None:
                    # Should not happen, but guard to avoid crashes.
                    self._success_counts = torch.zeros(
                        self.num_envs, dtype=torch.long, device=self.device
                    )
                increment = torch.ones(success_envs.shape, dtype=torch.long, device=self.device)
                self._success_counts.index_add_(0, success_envs, increment)
                completed_mask = self._success_counts.index_select(0, success_envs) >= MAX_SUCCESSES_PER_EPISODE
                ongoing_envs = success_envs[~completed_mask]
                if ongoing_envs.numel() > 0:
                    # Keep the episode running by respawning the marker when success is achieved.
                    self._respawn_marker(ongoing_envs)
                    raw_obs = self.get_obs(info, unflattened=True)
                # Let termination logic below handle env shutdown after required successes.

        obs = self._flatten_raw_obs(raw_obs)
        if "fail" in info:
            terminated = info["fail"].clone()
        else:
            terminated = torch.zeros(self.num_envs, dtype=bool, device=self.device)
        if self._success_counts is not None:
            success_limit_reached = self._success_counts >= MAX_SUCCESSES_PER_EPISODE
            if terminated.device != success_limit_reached.device:
                success_limit_reached = success_limit_reached.to(terminated.device)
            terminated = torch.logical_or(terminated, success_limit_reached)
        truncated = torch.zeros(self.num_envs, dtype=bool, device=self.device)
        self._last_obs = obs
        return obs, reward, terminated, truncated, info

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        marker_pose, _, _, normal = self._marker_frame_axes()
        target_point = marker_pose.p + normal * MARKER_NORMAL_OFFSET
        tcp_pose = self._get_tcp_pose_on_device()
        tcp_position = tcp_pose.p
        alignment_error = torch.linalg.norm(tcp_position - target_point, dim=1)
        alignment_reward = torch.exp(
            -alignment_error / POSITION_REWARD_LENGTH_SCALE
        )
        reward = alignment_reward
        if "success" in info:
            reward = reward + info["success"].to(reward.dtype)

        observed_qvel = self._get_observed_qvel()
        speed = torch.linalg.norm(observed_qvel, dim=1)
        excess_speed = torch.clamp(speed - VELOCITY_PENALTY_THRESHOLD, min=0.0)
        exponent = torch.clamp(
            excess_speed * VELOCITY_PENALTY_SCALE, max=VELOCITY_PENALTY_EXP_MAX
        )
        velocity_penalty = torch.expm1(exponent)
        reward = reward - velocity_penalty

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        max_reward = 1.0 + 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
