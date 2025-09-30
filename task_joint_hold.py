import sapien
import torch

from typing import Any, Dict

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env

from robotagents.my_xarm7_over import Xarm7ReducedProprio
import robotagents.my_xarm7_mjcf
from scenebuilders.xarm7_table_scene_builder import Xarm7TableSceneBuilder
from scenebuilders.xarm7_joint_hold_scene_builder import Xarm7JointHoldSceneBuilder

HOME_TARGET_QPOS = torch.tensor(
    [
        0.0,
        -0.477,
        0.0,
        0.8571976,
        0.0,
        1.2771976,
        -1.5707964,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ],
    dtype=torch.float32,
)

#許容速度
VELOCITY_PENALTY_THRESHOLD = 0.45
#超過時のペナルティの鋭さ
VELOCITY_PENALTY_SCALE = 0.035
#ペナルティの最大値
VELOCITY_PENALTY_EXP_MAX = 1.5

@register_env("MyJointHold-v0", max_episode_steps=100)
class MyJointHoldEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["my_xarm7_over", "my_xarm7_mjcf"]
    agent: Xarm7ReducedProprio
    joint_tolerance = 0.05

    def __init__(self, *args, robot_uids="my_xarm7_over", target_qpos=None, **kwargs):
        if target_qpos is None:
            self._configured_target_qpos = None
        else:
            self._configured_target_qpos = [float(x) for x in target_qpos]
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
        self.table_scene: Xarm7TableSceneBuilder = Xarm7JointHoldSceneBuilder(env=self)
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

            target_override = None if options is None else options.get("target_qpos")
            if target_override is not None:
                target = torch.as_tensor(target_override, dtype=torch.float32, device=self.device)
            elif self._configured_target_qpos is not None:
                target = torch.tensor(self._configured_target_qpos, dtype=torch.float32, device=self.device)
            else:
                target = HOME_TARGET_QPOS.to(self.device)

            if target.ndim == 1:
                target = target.unsqueeze(0)
            if target.shape[0] != len(env_idx):
                target = target[0:1].repeat(len(env_idx), 1)
            self.target_qpos = target.clone()

    def _get_aligned_target_qpos(self, batch_size: int) -> torch.Tensor:
        target = self.target_qpos
        if target.ndim == 1:
            target = target.unsqueeze(0)
        if target.shape[0] != batch_size:
            target = target[0:1].repeat(batch_size, 1)
        return target

    def _get_current_qpos(self) -> torch.Tensor:
        qpos = self.agent.robot.get_qpos()
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        return qpos

    def _get_current_qvel(self) -> torch.Tensor:
        qvel = self.agent.robot.get_qvel()
        if qvel.ndim == 1:
            qvel = qvel.unsqueeze(0)
        return qvel

    def evaluate(self):
        current_qpos = self._get_current_qpos()
        target_qpos = self._get_aligned_target_qpos(len(current_qpos))
        joint_error = torch.linalg.norm(current_qpos - target_qpos, dim=1)
        success = joint_error < self.joint_tolerance
        return {
            "success": success,
            "joint_error_norm": joint_error,
        }

    def _get_obs_extra(self, info: Dict):
        current_qpos = self._get_current_qpos()
        obs: Dict[str, torch.Tensor] = dict()
        if hasattr(self, "target_qpos"):
            target_qpos = self._get_aligned_target_qpos(current_qpos.shape[0])
            obs_indices = getattr(self.agent, "_obs_joint_indices", None)
            if obs_indices is not None:
                index_tensor = obs_indices.to(device=target_qpos.device, dtype=torch.long)
                dim = target_qpos.dim() - 1
                target_qpos = target_qpos.index_select(dim, index_tensor)
            obs["target_qpos"] = target_qpos
        return obs

    reward_length_scale = 1.0

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        current_qpos = self._get_current_qpos()
        target_qpos = self._get_aligned_target_qpos(len(current_qpos))
        joint_error = torch.linalg.norm(current_qpos - target_qpos, dim=1)
        holding_reward = torch.exp(-joint_error / self.reward_length_scale)
        reward = holding_reward
        if "success" in info:
            reward = reward + info["success"].to(reward.dtype)

        if hasattr(self.agent, "_obs_joint_indices"):
            qvel = self._get_current_qvel()
            idx = self.agent._obs_joint_indices.to(device=qvel.device, dtype=torch.long)
            dim = qvel.dim() - 1
            observed_qvel = qvel.index_select(dim, idx)
        else:
            observed_qvel = self._get_current_qvel()
        speed = torch.linalg.norm(observed_qvel, dim=1)
        excess_speed = torch.clamp(speed - VELOCITY_PENALTY_THRESHOLD, min=0.0)
        exponent = torch.clamp(
            excess_speed * VELOCITY_PENALTY_SCALE, max=VELOCITY_PENALTY_EXP_MAX
        )
        velocity_penalty = torch.expm1(exponent)
        reward = reward - velocity_penalty
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        max_reward = 1.0 + 1.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
