from os import path

import numpy as np

from .MujocoG1EnvBase import MujocoG1EnvBase


class MujocoG1BottlesEnv(MujocoG1EnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoG1EnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/g1/env_g1_bottles.xml",
            ),
            np.array(
                [
                    # left arm
                    0.626,
                    0.332,
                    0.209,
                    -0.466,
                    -0.375,
                    -0.242,
                    -0.339,
                    0.0,
                    # right arm
                    0.626,
                    -0.332,
                    -0.209,
                    -0.466,
                    0.375,
                    -0.242,
                    0.339,
                    0.0,
                ]
            ),
            **kwargs,
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = 0
        return world_idx
