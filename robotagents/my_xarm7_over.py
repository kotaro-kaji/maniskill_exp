import torch

from mani_skill.agents.registration import register_agent
from mani_skill.agents.utils import get_active_joint_indices

from .my_xarm7 import Xarm7


@register_agent()
class Xarm7ReducedProprio(Xarm7):
    """Xarm7 variant that hides mimic joints from proprioceptive observations."""

    uid = "my_xarm7_over"

    def _after_init(self):
        super()._after_init()
        observed_joint_names = self.arm_joint_names + ["drive_joint"]
        self._obs_joint_indices = get_active_joint_indices(
            self.robot, observed_joint_names
        ).long()

    def get_proprioception(self):
        obs = super().get_proprioception()
        idx = self._obs_joint_indices.to(device=obs["qpos"].device)
        dim = obs["qpos"].dim() - 1
        obs["qpos"] = obs["qpos"].index_select(dim, idx)
        obs["qvel"] = obs["qvel"].index_select(dim, idx)
        return obs
