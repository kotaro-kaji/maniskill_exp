import numpy as np
import torch
import sapien

from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder


class Xarm7TableSceneBuilder(TableSceneBuilder):
    """Table scene builder with Xarm7 defaults.

    Reuses ManiSkill's TableSceneBuilder for geometry/ground, and only
    customizes the robot initialization for the custom agent uid "my_xarm7".
    """

    def initialize(self, env_idx: torch.Tensor):
        # Let the base class place the table and ground consistently
        super().initialize(env_idx)

        # If this env uses our custom Xarm7, set its qpos and base pose
        if self.env.robot_uids == "my_xarm7":
            b = len(env_idx)

            # Start from the provided keyframe in my_xarm7.py
            qpos = self.env.agent.keyframes["home"].qpos

            # Optional joint noise, matching TableSceneBuilder style
            if getattr(self.env, "_enhanced_determinism", False):
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )

            # Apply initial state and place the base slightly behind the table
            self.env.agent.reset(qpos)
            # Similar to Panda placement in TableSceneBuilder
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
