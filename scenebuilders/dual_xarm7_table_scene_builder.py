import os

import torch
import sapien

from scenebuilders.xarm7_table_scene_builder import (
    Xarm7TableSceneBuilder,
    PEDESTAL_HEIGHT,
    ROBOT_BASE_X_OFFSET,
)
from scenebuilders.xarm7_initial_randomization_scene_builder import (
    _RESET_STATE_OF_ROBOMANIPBASELINES,
)


BASELINE_QPOS = (
    _RESET_STATE_OF_ROBOMANIPBASELINES.detach()
    if hasattr(_RESET_STATE_OF_ROBOMANIPBASELINES, "detach")
    else _RESET_STATE_OF_ROBOMANIPBASELINES
)
BASELINE_QPOS = (
    BASELINE_QPOS.cpu().numpy()
    if hasattr(BASELINE_QPOS, "cpu")
    else BASELINE_QPOS.numpy()
    if hasattr(BASELINE_QPOS, "numpy")
    else BASELINE_QPOS
)
if BASELINE_QPOS.ndim > 1:
    BASELINE_QPOS = BASELINE_QPOS[0]
BASELINE_QPOS = BASELINE_QPOS.astype("float32")


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
        self._set_pose_and_qpos(agent.robot, PRIMARY_ARM_Y_OFFSET)

    def _initialize_secondary_robot(self):
        if self.secondary_robot is None:
            return
        self._set_pose_and_qpos(self.secondary_robot, SECONDARY_ARM_Y_OFFSET)

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
            self._set_pose_and_qpos(robot, SECONDARY_ARM_Y_OFFSET)
            self.scene_objects.append(robot)
        return robot

    def _set_pose_and_qpos(self, robot, y_offset: float):
        robot.set_pose(
            sapien.Pose(
                p=[ROBOT_BASE_X_OFFSET, y_offset, PEDESTAL_HEIGHT],
                q=[1, 0, 0, 0],
            )
        )
        try:
            if robot.dof == BASELINE_QPOS.shape[-1]:
                robot.set_qpos(BASELINE_QPOS)
        except Exception:
            pass
