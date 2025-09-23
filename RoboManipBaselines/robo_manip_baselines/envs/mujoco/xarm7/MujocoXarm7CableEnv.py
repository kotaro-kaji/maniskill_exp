from os import path

import numpy as np

from .MujocoXarm7EnvBase import MujocoXarm7EnvBase


class MujocoXarm7CableEnv(MujocoXarm7EnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoXarm7EnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/xarm7/env_xarm7_cable.xml",
            ),
            np.array([0.0, 0.0, 0.0, 0.8, 0.0, 0.8, 0.0, *[0.0] * 6]),
            **kwargs,
        )

        self.original_pole_pos = self.model.body("poles").pos.copy()
        self.pole_pos_offsets = np.array(
            [
                [-0.03, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [0.06, 0.0, 0.0],
                [0.09, 0.0, 0.0],
                [0.12, 0.0, 0.0],
            ]
        )  # [m]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.pole_pos_offsets)
        self.model.body("poles").pos = (
            self.original_pole_pos + self.pole_pos_offsets[world_idx]
        )
        return world_idx
