import os

import numpy as np
import torch
import sapien

from scenebuilders.xarm7_table_scene_builder import (
    Xarm7TableSceneBuilder,
    PEDESTAL_HEIGHT,
    ROBOT_BASE_X_OFFSET,
)


BALL_EE_URDF_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "xarm7_ball_ee.urdf")
)
PRIMARY_ARM_Y_OFFSET = -0.3
SECONDARY_ARM_Y_OFFSET = 0.3


class DualXarm7TableSceneBuilder(Xarm7TableSceneBuilder):
    """Table scene builder that spawns a second xArm7 Ball-EE beside the agent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.secondary_robot = None

    def build(self):
        super().build()
        self._ensure_secondary_robot_created()

    def initialize(self, env_idx: torch.Tensor):
        super().initialize(env_idx)
        self._ensure_secondary_robot_created()
        self._place_primary_agent()
        self._initialize_secondary_robot()

    def _place_primary_agent(self):
        agent = getattr(self.env, "agent", None)
        if agent is None or getattr(agent, "robot", None) is None:
            return
        agent.robot.set_pose(
            sapien.Pose(
                p=[ROBOT_BASE_X_OFFSET, PRIMARY_ARM_Y_OFFSET, PEDESTAL_HEIGHT],
                q=[1, 0, 0, 0],
            )
        )

    def _initialize_secondary_robot(self):
        if self.secondary_robot is None:
            return
        self.secondary_robot.set_pose(
            sapien.Pose(
                p=[ROBOT_BASE_X_OFFSET, SECONDARY_ARM_Y_OFFSET, PEDESTAL_HEIGHT],
                q=[1, 0, 0, 0],
            )
        )
        try:
            zero_qpos = np.zeros(self.secondary_robot.dof, dtype=np.float32)
            self.secondary_robot.set_qpos(zero_qpos)
        except Exception:
            pass

    def _ensure_secondary_robot_created(self):
        if self.secondary_robot is None:
            self.secondary_robot = self._load_ball_ee_robot("xarm7_ball_ee_right")

    def _load_ball_ee_robot(self, name: str):
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.name = name
        try:
            robot = loader.load(BALL_EE_URDF_PATH)
        except FileNotFoundError:
            return None
        if robot is not None:
            robot.set_pose(
                sapien.Pose(
                    p=[ROBOT_BASE_X_OFFSET, SECONDARY_ARM_Y_OFFSET, PEDESTAL_HEIGHT],
                    q=[1, 0, 0, 0],
                )
            )
            try:
                zero_qpos = np.zeros(robot.dof, dtype=np.float32)
                robot.set_qpos(zero_qpos)
            except Exception:
                pass
            self.scene_objects.append(robot)
        return robot
