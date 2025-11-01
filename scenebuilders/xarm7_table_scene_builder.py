import numpy as np
import torch
import sapien
import sapien.render

from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder


PEDESTAL_HEIGHT = 0.085344
PEDESTAL_HALF_EXTENT_X = 0.09
ROBOT_BASE_X_OFFSET = -0.615


class Xarm7TableSceneBuilder(TableSceneBuilder):
    """Table scene builder with Xarm7 defaults.

    Reuses ManiSkill's TableSceneBuilder for geometry/ground, and only
    customizes the robot initialization for the custom agent uid "my_xarm7".
    """

    def build(self):
        super().build()

        pedestal_half_size = (
            PEDESTAL_HALF_EXTENT_X,
            self.table_width / 2,
            PEDESTAL_HEIGHT / 2,
        )

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=pedestal_half_size)
        builder.add_box_visual(
            half_size=pedestal_half_size,
            material=sapien.render.RenderMaterial(
                base_color=[0.75, 0.75, 0.8, 1.0],
                metallic=0.9,
                roughness=0.3,
            ),
        )
        builder.initial_pose = sapien.Pose(p=[ROBOT_BASE_X_OFFSET, 0.0, PEDESTAL_HEIGHT / 2])
        self.robot_pedestal = builder.build_static(name="robot_pedestal")
        self.scene_objects.append(self.robot_pedestal)

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
            self.env.agent.robot.set_pose(
                sapien.Pose([ROBOT_BASE_X_OFFSET, 0, PEDESTAL_HEIGHT])
            )
        except Exception:
            # If no keyframe is defined, fall back to base behavior
            pass
