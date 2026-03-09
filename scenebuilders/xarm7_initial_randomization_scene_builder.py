import numpy as np
import torch

from scenebuilders.xarm7_table_scene_builder import Xarm7TableSceneBuilder


class Xarm7InitialRandomizationSceneBuilder(Xarm7TableSceneBuilder):
    """Scene builder that initializes Xarm7 to a randomized baseline pose.

    At every reset the first eight joints (seven arm joints + gripper drive)
    receive independent uniform offsets sampled from per-joint hard-coded bands.
    The remaining mimic joints stay aligned with the drive joint.
    """

    # Desired joint configuration (arm joints 1-7, gripper drive + mimics)
    _RESET_STATE_OF_ROBOMANIPBASELINES = torch.tensor(
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

    # Desired joint configuration (arm joints 1-7 only)
    _JOINT_ONLY_RESET_STATE_OF_ROBOMANIPBASELINES_ = torch.tensor(
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

    _OFFSET_LOW = torch.tensor(
        [-0.05, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.01],
        dtype=torch.float32,
    )
    _OFFSET_HIGH = torch.tensor(
        [0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.01],
        dtype=torch.float32,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_qpos = self._RESET_STATE_OF_ROBOMANIPBASELINES.clone()
        self._offset_low_np = self._OFFSET_LOW.detach().cpu().numpy()
        self._offset_high_np = self._OFFSET_HIGH.detach().cpu().numpy()
        self.noise_scale = 1.0

    def initialize(self, env_idx: torch.Tensor):
        super().initialize(env_idx)
        agents = self._get_agent_sequence()
        if not agents:
            return

        try:
            if self.env.scene.gpu_sim_enabled:
                self.env.scene._gpu_apply_all()
                self.env.scene._gpu_fetch_all()

            for agent in agents:
                controller = getattr(agent, "controller", None)
                self._reset_controller(controller)

            for agent in agents:
                controller = getattr(agent, "controller", None)
                current_qpos = agent.robot.get_qpos()
                self._set_controller_drive_targets(controller, current_qpos)

            for agent in agents:
                controller = getattr(agent, "controller", None)
                self._zero_controller_action(controller)
        except Exception:
            pass

    def set_noise_scale(self, scale: float):
        """
        Set multiplicative scale for joint offset sampling.
        """
        self.noise_scale = float(scale)

    def _sample_joint_offsets(self, env_idx: torch.Tensor, batch_size: int) -> torch.Tensor:
        low = self._offset_low_np * self.noise_scale
        high = self._offset_high_np * self.noise_scale
        if getattr(self.env, "_enhanced_determinism", False):
            samples = self.env._batched_episode_rng[env_idx].uniform(
                low=low, high=high
            )
            samples = np.asarray(samples)
            if samples.ndim == 1:
                samples = samples[np.newaxis, :]
        else:
            samples = self.env._episode_rng.uniform(
                low=low,
                high=high,
                size=(batch_size, self._offset_low_np.shape[0]),
            )
        offsets = torch.zeros(
            (samples.shape[0], self._RESET_STATE_OF_ROBOMANIPBASELINES.numel()),
            device=self.env.device,
            dtype=torch.float32,
        )
        offsets[:, : self._OFFSET_LOW.numel()] = torch.as_tensor(
            samples, device=self.env.device, dtype=torch.float32
        )
        return offsets

    def _baseline_qpos_tensor(self) -> torch.Tensor:
        baseline = self.initial_qpos.to(self.env.device)
        return baseline.clone()

    def _compute_initial_qpos(self, agents, env_idx: torch.Tensor):
        batch_size = len(env_idx)

        base = self._baseline_qpos_tensor()
        if base.ndim == 1:
            base = base.unsqueeze(0)
        if base.shape[0] == 1 and batch_size > 1:
            base = base.repeat(batch_size, 1)

        # Sample independent offsets per agent so left/right arms do not share the same noise realization.
        qpos_per_agent = []
        for _ in agents:
            offsets = self._sample_joint_offsets(env_idx, batch_size)
            if offsets.shape[0] == 1 and batch_size > 1:
                offsets = offsets.repeat(batch_size, 1)
            qpos = base + offsets.to(self.env.device)
            qpos[:, 8:] = qpos[:, 7:8]
            qpos_per_agent.append(qpos.clone())
        return qpos_per_agent

    def _reset_controller(self, controller):
        if controller is None:
            return
        if isinstance(controller, dict):
            for ctrl in controller.values():
                self._reset_controller(ctrl)
            return
        reset_fn = getattr(controller, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def _set_controller_drive_targets(self, controller, current_qpos):
        if controller is None:
            return
        if isinstance(controller, dict):
            for ctrl in controller.values():
                self._set_controller_drive_targets(ctrl, current_qpos)
            return
        if hasattr(controller, "controllers"):
            for ctrl in controller.controllers.values():
                self._set_controller_drive_targets(ctrl, current_qpos)
            return
        set_targets = getattr(controller, "set_drive_targets", None)
        if callable(set_targets):
            set_targets(current_qpos)

    def _zero_controller_action(self, controller):
        if controller is None:
            return
        if isinstance(controller, dict):
            for ctrl in controller.values():
                self._zero_controller_action(ctrl)
            return
        if not hasattr(controller, "action_space") or not hasattr(
            controller, "set_action"
        ):
            return
        try:
            sample = controller.action_space.sample()
            controller.set_action(self._zero_like(sample))
        except Exception:
            pass

    def _zero_like(self, sample):
        if isinstance(sample, dict):
            return {k: self._zero_like(v) for k, v in sample.items()}
        return np.zeros_like(sample)
