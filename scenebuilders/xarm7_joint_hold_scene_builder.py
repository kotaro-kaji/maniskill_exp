import numpy as np
import torch

from scenebuilders.xarm7_table_scene_builder import Xarm7TableSceneBuilder


class Xarm7JointHoldSceneBuilder(Xarm7TableSceneBuilder):
    """Scene builder that initializes Xarm7 to the joint-hold fixture pose.

    At every reset the first eight joints (seven arm joints + gripper drive)
    receive independent uniform offsets sampled from per-joint hard-coded bands.
    The remaining mimic joints stay aligned with the drive joint.
    """

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
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
        ],
        dtype=torch.float32,
    )

    _OFFSET_LOW = torch.tensor(
        [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.01],
        dtype=torch.float32,
    )
    _OFFSET_HIGH = torch.tensor(
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.05, 0.01],
        dtype=torch.float32,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_qpos = self._JOINT_HOLD_QPOS.clone()
        self._offset_low_np = self._OFFSET_LOW.detach().cpu().numpy()
        self._offset_high_np = self._OFFSET_HIGH.detach().cpu().numpy()

    def initialize(self, env_idx: torch.Tensor):
        super().initialize(env_idx)
        try:
            b = len(env_idx)
            base = self.initial_qpos.to(self.env.device)
            if base.ndim == 1:
                base = base.unsqueeze(0)
            if base.shape[0] == 1 and b > 1:
                base = base.repeat(b, 1)

            offsets = self._sample_joint_offsets(env_idx, b)
            if offsets.shape[0] == 1 and b > 1:
                offsets = offsets.repeat(b, 1)
            qpos = base + offsets
            qpos[:, 8:] = qpos[:, 7:8]
            self.env.agent.reset(qpos)
        except Exception:
            pass

    def _sample_joint_offsets(self, env_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        if getattr(self.env, "_enhanced_determinism", False):
            samples = self.env._batched_episode_rng[env_idx].uniform(
                low=self._offset_low_np, high=self._offset_high_np
            )
            samples = np.asarray(samples)
            if samples.ndim == 1:
                samples = samples[np.newaxis, :]
        else:
            samples = self.env._episode_rng.uniform(
                low=self._offset_low_np,
                high=self._offset_high_np,
                size=(batch_size, self._offset_low_np.shape[0]),
            )
        offsets = torch.zeros(
            (samples.shape[0], self._JOINT_HOLD_QPOS.numel()),
            device=self.env.device,
            dtype=torch.float32,
        )
        offsets[:, : self._OFFSET_LOW.numel()] = torch.as_tensor(
            samples, device=self.env.device, dtype=torch.float32
        )
        return offsets
