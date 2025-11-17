import sapien
import sapien.render

from scenebuilders.xarm7_initial_randomization_scene_builder import (
    Xarm7InitialRandomizationSceneBuilder,
)
from scenebuilders.xarm7_table_scene_builder import (
    PEDESTAL_HEIGHT,
    PEDESTAL_HALF_EXTENT_X,
    ROBOT_BASE_X_OFFSET,
)


LEFT_ARM_Y_OFFSET = 0.3291
RIGHT_ARM_Y_OFFSET = -0.3291

# Maintain legacy constant names that reflect agent indices.
PRIMARY_ARM_Y_OFFSET = LEFT_ARM_Y_OFFSET
SECONDARY_ARM_Y_OFFSET = RIGHT_ARM_Y_OFFSET

class DualXarm7TableSceneBuilder(Xarm7InitialRandomizationSceneBuilder):
    """Table scene builder that positions two xArm7 Ball-EE robots on the table."""

    def build(self):
        self._primary_pedestal_pose = sapien.Pose(
            [ROBOT_BASE_X_OFFSET, PRIMARY_ARM_Y_OFFSET, PEDESTAL_HEIGHT / 2.0]
        )
        super().build()

    def _initial_agent_pose(self, agent_index: int) -> sapien.Pose:
        if agent_index == 0:
            y_offset = PRIMARY_ARM_Y_OFFSET
        elif agent_index == 1:
            y_offset = SECONDARY_ARM_Y_OFFSET
        else:
            y_offset = 0.0
        return sapien.Pose([ROBOT_BASE_X_OFFSET, y_offset, PEDESTAL_HEIGHT])
