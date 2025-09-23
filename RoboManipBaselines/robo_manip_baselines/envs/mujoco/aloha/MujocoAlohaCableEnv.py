from os import path

import numpy as np

from .MujocoAlohaEnvBase import MujocoAlohaEnvBase


class MujocoAlohaCableEnv(MujocoAlohaEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoAlohaEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/aloha/env_aloha_cable.xml",
            ),
            np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.037, 0.037] * 2),
            **kwargs,
        )

        self.original_pole_pos = self.model.geom("pole1").pos.copy()
        self.pole_pos_offsets = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.015, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [0.045, 0.0, 0.0],
                [0.06, 0.0, 0.0],
                [0.075, 0.0, 0.0],
            ]
        )  # [m]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.pole_pos_offsets)
        self.model.geom("pole1").pos = (
            self.original_pole_pos + self.pole_pos_offsets[world_idx]
        )
        self.model.geom("pole2").pos = self.model.geom("pole1").pos
        self.model.geom("pole2").pos[0] *= -1
        return world_idx
