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


PRIMARY_ARM_Y_OFFSET = -0.3291
SECONDARY_ARM_Y_OFFSET = 0.3291

class DualXarm7TableSceneBuilder(Xarm7InitialRandomizationSceneBuilder):
    """Table scene builder that positions two xArm7 Ball-EE robots on the table."""

    def build(self):
        self._primary_pedestal_pose = sapien.Pose(
            [ROBOT_BASE_X_OFFSET, PRIMARY_ARM_Y_OFFSET, PEDESTAL_HEIGHT / 2.0]
        )
        super().build()
        self.secondary_pedestal = self._build_additional_pedestal(
            SECONDARY_ARM_Y_OFFSET, name="robot_pedestal_secondary"
        )

    def _build_additional_pedestal(self, y_offset: float, name: str):
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
        builder.initial_pose = sapien.Pose(
            p=[ROBOT_BASE_X_OFFSET, y_offset, PEDESTAL_HEIGHT / 2.0]
        )
        pedestal = builder.build_static(name=name)
        self.scene_objects.append(pedestal)
        return pedestal

    def _initial_agent_pose(self, agent_index: int) -> sapien.Pose:
        if agent_index == 0:
            y_offset = PRIMARY_ARM_Y_OFFSET
        elif agent_index == 1:
            y_offset = SECONDARY_ARM_Y_OFFSET
        else:
            y_offset = 0.0
        return sapien.Pose([ROBOT_BASE_X_OFFSET, y_offset, PEDESTAL_HEIGHT])
