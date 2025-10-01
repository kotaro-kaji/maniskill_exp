import sapien
import sapien.render
import torch
import torch.nn.functional as F

from typing import Any, Dict, Optional

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from robotagents.my_xarm7_over import Xarm7ReducedProprio
import robotagents.my_xarm7_mjcf
from scenebuilders.xarm7_joint_hold_scene_builder import Xarm7JointHoldSceneBuilder


MARKER_NORMAL_OFFSET = 0.2
ALIGNMENT_TOLERANCE = 0.02
POSITION_REWARD_LENGTH_SCALE = 0.05

VELOCITY_PENALTY_THRESHOLD = 0.45
VELOCITY_PENALTY_SCALE = 0.155
VELOCITY_PENALTY_EXP_MAX = 20.0

DEFAULT_MARKER_POSITION = torch.tensor([-0.15, 0.0, 0.0], dtype=torch.float32)
DEFAULT_MARKER_ORIENTATION = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
DEFAULT_MARKER_POSE = torch.cat((DEFAULT_MARKER_POSITION, DEFAULT_MARKER_ORIENTATION))

MARKER_BASE_HALF_SIZE = (0.04, 0.04, 0.002)
MARKER_GREEN_HALF_SIZE = (MARKER_BASE_HALF_SIZE[0], 0.003, 0.001)
MARKER_GREEN_Y_OFFSET = MARKER_BASE_HALF_SIZE[1] - MARKER_GREEN_HALF_SIZE[1]
MARKER_GREEN_Z_OFFSET = MARKER_BASE_HALF_SIZE[2] + MARKER_GREEN_HALF_SIZE[2]


@register_env("MyEEAlignMarker-v0", max_episode_steps=100)
class MyEEAlignMarkerEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["my_xarm7_over", "my_xarm7_mjcf"]
    agent: Xarm7ReducedProprio

    def __init__(
        self,
        *args,
        robot_uids: str = "my_xarm7_over",
        marker_pose: Optional[Any] = None,
        **kwargs,
    ):
        if marker_pose is None:
            self._configured_marker_pose: Optional[Pose] = None
        else:
            self._configured_marker_pose = Pose.create(marker_pose)
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        base_pose = sapien.Pose(p=[0.0, 0.0, 0.0])
        super()._load_agent(options, base_pose)

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

    def _clear(self):
        self._close_viewer()
        self.agent = None
        self._sensors = dict()
        self._human_render_cameras = dict()
        self.scene = None
        self._hidden_objects = []
        try:
            import gc as _gc

            _gc.collect()
        except Exception:
            pass

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)
            marker_pose = self._resolve_marker_pose(len(env_idx), options)
            if (
                self._configured_marker_pose is None
                and (options is None or options.get("marker_pose") is None)
            ):
                batch_size = len(env_idx)
                rand_x = torch.rand((batch_size,), device=self.device) * 0.1 - 0.35
                rand_y = torch.rand((batch_size,), device=self.device) * 0.1 - 0.05
                rand_z = torch.full(
                    (batch_size,),
                    DEFAULT_MARKER_POSITION[2],
                    device=self.device,
                )
                randomized_positions = torch.stack((rand_x, rand_y, rand_z), dim=-1)
                marker_pose.p = randomized_positions
            self.marker.set_pose(marker_pose)

    def _resolve_marker_pose(self, batch_size: int, options: Optional[dict]) -> Pose:
        if options is not None and options.get("marker_pose") is not None:
            base_value = options["marker_pose"]
        elif self._configured_marker_pose is not None:
            base_value = self._configured_marker_pose
        else:
            base_value = DEFAULT_MARKER_POSE

        pose = Pose.create(base_value, device=self.device)
        if len(pose) == 1 and batch_size > 1:
            pose = Pose.create(pose.raw_pose.repeat(batch_size, 1), device=self.device)
        elif len(pose) != batch_size:
            if len(pose) < batch_size:
                pose = Pose.create(
                    pose.raw_pose[0:1].repeat(batch_size, 1), device=self.device
                )
            else:
                pose = Pose.create(pose.raw_pose[:batch_size], device=self.device)
        return pose

    def _get_marker_pose_on_device(self) -> Pose:
        marker_pose = self.marker.pose
        if marker_pose.device != self.device:
            marker_pose = marker_pose.to(self.device)
        return marker_pose

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

    def _get_obs_extra(self, info: Dict):
        marker_pose, x_axis, y_axis, _ = self._marker_frame_axes()
        marker_obs = torch.cat([marker_pose.p, x_axis, y_axis], dim=-1)
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
