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

        # If this env uses our custom Xarm7 variants, set qpos and base pose
        if self.env.robot_uids in ("my_xarm7", "my_xarm7_mjcf"):
            b = len(env_idx)

            # Use the exact initial joint configuration defined in my_xarm7.py
            # without noise so it matches precisely.
            base_qpos = self.env.agent.keyframes["home"].qpos
            if base_qpos.ndim == 1 and b > 1:
                qpos = np.tile(base_qpos, (b, 1))
            else:
                qpos = base_qpos

            # Apply initial state and place the base slightly behind the table
            self.env.agent.reset(qpos)
            # Similar to Panda placement in TableSceneBuilder
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
