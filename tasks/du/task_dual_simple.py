from typing import Any, Dict

import torch
import sapien

from mani_skill.agents.multi_agent import MultiAgent

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from robotagents.xarm_ball_ee import Xarm7BallEE

from scenebuilders.dual_xarm7_table_scene_builder import (
    DualXarm7TableSceneBuilder,
)


@register_env("MyDualSimple-v0", max_episode_steps=200)
class MyDualSimpleEnv(BaseEnv):
    SUPPORTED_ROBOTS = [("xarm7_ball_ee", "xarm7_ball_ee")]
    agent: MultiAgent[Tuple[Xarm7BallEE, Xarm7BallEE]]

    def __init__(self, *args, robot_uids=("xarm7_ball_ee", "xarm7_ball_ee"), **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: Dict[str, Any]):
        super()._load_agent(options)

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.2, 1.0, 1.0], [0.0, 0.0, 0.4])
        return CameraConfig(
            "render_camera", pose=pose, width=800, height=800, fov=1.0, near=0.01, far=5.0
        )

    def _load_scene(self, options: Dict[str, Any]):
        self.table_scene = DualXarm7TableSceneBuilder(env=self)
        self.table_scene.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict[str, Any]):
        with torch.device(self.device):
            self.table_scene.initialize(env_idx)


    def compute_normalized_dense_reward(self, obs, action, info):
        return 0.0
