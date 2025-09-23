from os import path

import numpy as np

from .MujocoXarm7EnvBase import MujocoXarm7EnvBase


class MujocoXarm7RingEnv(MujocoXarm7EnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoXarm7EnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/xarm7/env_xarm7_ring.xml",
            ),
            np.array([0.0, 0.0, 0.0, 0.0, np.pi, np.pi / 2, np.pi, *[0.0] * 6]),
            **kwargs,
        )

        self.original_pole_pos = self.model.body("pole").pos.copy()
        self.pole_pos_offsets = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.04, 0.0],
                [0.0, 0.08, 0.0],
                [0.0, 0.12, 0.0],
                [0.0, 0.16, 0.0],
                [0.0, 0.20, 0.0],
            ]
        )  # [m]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.pole_pos_offsets)
        self.model.body("pole").pos = (
            self.original_pole_pos + self.pole_pos_offsets[world_idx]
        )
        return world_idx
