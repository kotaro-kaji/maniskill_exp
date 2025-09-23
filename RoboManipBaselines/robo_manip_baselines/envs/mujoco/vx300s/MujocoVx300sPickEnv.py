from os import path

import numpy as np

from .MujocoVx300sEnvBase import MujocoVx300sEnvBase


class MujocoVx300sPickEnv(MujocoVx300sEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoVx300sEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/vx300s/env_vx300s_pick.xml",
            ),
            np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.037, 0.037]),
            **kwargs,
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = 0

        return world_idx
