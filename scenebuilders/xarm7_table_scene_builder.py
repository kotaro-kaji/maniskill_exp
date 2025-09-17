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

        # Apply unified initial joint configuration from the agent's 'home' keyframe
        # for all custom Xarm7 variants.
        try:
            b = len(env_idx)
            home_qpos = self.env.agent.keyframes["home"].qpos
            # Convert to torch on correct device and batch if necessary
            if isinstance(home_qpos, np.ndarray):
                home_qpos = torch.tensor(home_qpos, device=self.env.device)
            else:
                home_qpos = home_qpos.to(self.env.device)
            if home_qpos.ndim == 1 and b > 1:
                qpos = home_qpos.unsqueeze(0).repeat(b, 1)
            else:
                qpos = home_qpos
            # Reset agent state with exact qpos and place base behind the table
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        except Exception:
            # If no keyframe is defined, fall back to base behavior
            pass
