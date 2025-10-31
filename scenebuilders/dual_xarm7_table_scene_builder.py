import os

import sapien
import torch

from scenebuilders.xarm7_table_scene_builder import (
    Xarm7TableSceneBuilder,
    PEDESTAL_HEIGHT,
    ROBOT_BASE_X_OFFSET,
)
from scenebuilders.xarm7_initial_randomization_scene_builder import (
    Xarm7InitialRandomizationSceneBuilder,
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
        self._reset_agent_to_baseline()
        agent.robot.set_pose(
            sapien.Pose(
                p=[ROBOT_BASE_X_OFFSET, PRIMARY_ARM_Y_OFFSET, PEDESTAL_HEIGHT],
                q=[1, 0, 0, 0],
            )
        )

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
            baseline = self._baseline_qpos_tensor()
            if baseline.shape[-1] == robot.dof:
                robot.set_qpos(baseline.detach().cpu().numpy())
        except Exception:
            pass

    def _baseline_qpos_tensor(self) -> torch.Tensor:
        baseline = Xarm7InitialRandomizationSceneBuilder._RESET_STATE_OF_ROBOMANIPBASELINES
        if baseline.ndim > 1:
            baseline = baseline[0]
        return baseline.clone()

    def _reset_agent_to_baseline(self):
        baseline = self._baseline_qpos_tensor()
        agent = getattr(self.env, "agent", None)
        if agent is None:
            return
        device = getattr(self.env, "device", torch.device("cpu"))
        baseline = baseline.to(device)
        agent.reset(baseline.unsqueeze(0))
