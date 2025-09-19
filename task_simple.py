import sapien
import torch

from typing import Any, Dict

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array

from my_xarm7 import Xarm7
import my_xarm7_mjcf
from xarm7_table_scene_builder import Xarm7TableSceneBuilder


@register_env("MySimpleReach-v0", max_episode_steps=50)
class MySimpleReachEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["my_xarm7", "my_xarm7_mjcf"]
    agent: Xarm7
    goal_radius = 0.02

    def __init__(self, *args, robot_uids="my_xarm7", target_position=None, **kwargs):
        if target_position is None:
            self._default_target_position = [0.15, 0.0, 0.25]
        else:
            self._default_target_position = [float(x) for x in target_position]
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
        self.target_site = actors.build_sphere(
            self.scene,
            radius=self.goal_radius,
            color=[0.1, 0.8, 0.2, 1.0],
            name="target_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=self._default_target_position),
        )
        self.table_scene = Xarm7TableSceneBuilder(env=self)
        self.table_scene.build()

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
            target_override = None if options is None else options.get("target_pos")
            if target_override is None:
                target = torch.tensor(
                    self._default_target_position,
                    dtype=torch.float32,
                    device=self.device,
                )
            else:
                target = torch.as_tensor(
                    target_override,
                    dtype=torch.float32,
                    device=self.device,
                )
            if target.ndim == 1:
                target = target.unsqueeze(0).repeat(len(env_idx), 1)
            elif target.shape[0] != len(env_idx):
                target = target[0:1].repeat(len(env_idx), 1)
            self.target_site.set_pose(Pose.create_from_pq(p=target))

    def evaluate(self):
        tcp_pos = self.agent.tcp.pose.p
        target_pos = self.target_site.pose.p
        dist = torch.linalg.norm(tcp_pos - target_pos, dim=1)
        success = dist < self.goal_radius
        return {
            "success": success,
            "tcp_to_target_dist": dist,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(target_pos=self.target_site.pose.p)
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        tcp_pos = self.agent.tcp.pose.p
        target_pos = self.target_site.pose.p
        dist = torch.linalg.norm(tcp_pos - target_pos, dim=1)
        reaching_reward = 1 - torch.tanh(5.0 * dist)
        reward = reaching_reward
        if "success" in info:
            reward = reward + info["success"].to(reward.dtype)
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 2.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
