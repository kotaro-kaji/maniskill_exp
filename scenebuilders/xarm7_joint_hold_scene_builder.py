import torch

from scenebuilders.xarm7_table_scene_builder import Xarm7TableSceneBuilder


class Xarm7JointHoldSceneBuilder(Xarm7TableSceneBuilder):
    """Scene builder that initializes Xarm7 to the joint-hold fixture pose."""

    # Desired joint configuration (arm joints 1-7, gripper drive + mimics)
    _JOINT_HOLD_QPOS = torch.tensor(
        [
            -0.00451699561347621,
            -0.4779577590016519,
            -0.0059982858387227518,
            0.8576097778375098,
            -0.032158957391327556,
            1.27989111575085,
            0.05005294852914137,
            0.85,
            0.85,
            0.85,
            0.85,
            0.85,
            0.85,
        ],
        dtype=torch.float32,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_qpos = self._JOINT_HOLD_QPOS.clone()

    def initialize(self, env_idx: torch.Tensor):
        super().initialize(env_idx)
        try:
            b = len(env_idx)
            target = self.initial_qpos.to(self.env.device)
            if target.ndim == 1 and b > 1:
                qpos = target.unsqueeze(0).repeat(b, 1)
            else:
                qpos = target
            self.env.agent.reset(qpos)
        except Exception:
            pass
