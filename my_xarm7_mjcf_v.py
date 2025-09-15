import numpy as np
import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils


@register_agent()
class Xarm7MJCF(BaseAgent):
    uid = "my_xarm7_mjcf"
    # Use a combined MJCF that includes only the robot
    mjcf_path = "assets/xarm7_mjcf_combined.xml"
    # In the MJCF, there is a site named "link_tcp" but not a link by that name.
    # We use the gripper base link as the TCP base.
    ee_link_name = "xarm_gripper_base_link"

    # 7 arm joints + 6 gripper joints (13 DOF)
    init_qpos = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            1.0471976,
            0.0,
            1.0471976,
            -1.5707964,
            # Gripper: start slightly open; driver+followers will sync
            0.045355614,  # left_driver_joint (control)
            0.045355614,  # left_inner_knuckle_joint
            0.045355614,  # left_finger_joint
            0.045355614,  # right_driver_joint
            0.045355614,  # right_inner_knuckle_joint
            0.045355614,  # right_finger_joint
        ],
        dtype=np.float32,
    )

    # Define an additional keyframe with the gripper closed
    _closed_qpos = init_qpos.copy()
    # Note: For this MJCF, larger angles close the gripper fingers.
    # Use a conservative close value within our controller upper bound (0.6).
    _closed_qpos[-6:] = 0.8
    keyframes = dict(
        home=Keyframe(qpos=init_qpos.copy(), pose=sapien.Pose([0, 0, 0])),
        grip_close=Keyframe(qpos=_closed_qpos, pose=sapien.Pose([0, 0, 0])),
    )

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        # MJCF joint names for the gripper
        self.gripper_joint_names = [
            "left_driver_joint",
            "left_inner_knuckle_joint",
            "left_finger_joint",
            "right_driver_joint",
            "right_inner_knuckle_joint",
            "right_finger_joint",
        ]

        # Reasonable defaults
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 30

        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 100

        super().__init__(*args, **kwargs)

    def _after_init(self):
        # Cache end-effector link and allow optional offset
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        self._tcp_offset = sapien.Pose([0, 0, 0], [1, 0, 0, 0])

    @property
    def tcp_pose(self) -> "sapien.Pose":
        return self.tcp.pose * self._tcp_offset

    @property
    def _controller_configs(self):
        # Arm controllers
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )

        # Treat the gripper as a 1-DOF driver using mimic: map all other joints to left_driver_joint
        gripper_mimic_map = {
            "left_inner_knuckle_joint": {"joint": "left_driver_joint"},
            "left_finger_joint": {"joint": "left_driver_joint"},
            "right_driver_joint": {"joint": "left_driver_joint"},
            "right_inner_knuckle_joint": {"joint": "left_driver_joint"},
            "right_finger_joint": {"joint": "left_driver_joint"},
        }
        # Limit how wide the gripper can command-open
        # Reduce upper bound from ~0.85 to 0.6 to avoid over-opening
        gripper_pd_joint_pos_mimic = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            0.0,
            0.81,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            normalize_action=False,
        )
        gripper_pd_joint_pos_mimic.mimic = gripper_mimic_map

        gripper_pd_joint_delta_pos_mimic = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -0.05,
            0.05,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            use_delta=True,
        )
        gripper_pd_joint_delta_pos_mimic.mimic = gripper_mimic_map

        controller_configs = dict(
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos_mimic,
            ),
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_delta_pos_mimic,
            ),
        )
        return deepcopy_dict(controller_configs)
