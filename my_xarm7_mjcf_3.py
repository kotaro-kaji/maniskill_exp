import numpy as np
import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils


def _controller_dict(controller_configs: dict):
    from mani_skill.agents.controllers import deepcopy_dict

    return deepcopy_dict(controller_configs)


@register_agent()
class Xarm7MJCFv3(BaseAgent):
    uid = "my_xarm7_mjcf_3"
    mjcf_path = "assets/xarm7_mjcf_combined.xml"
    ee_link_name = "xarm_gripper_base_link"

    init_qpos = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            1.0471976,
            0.0,
            1.0471976,
            -1.5707964,
            0.045355614,
            0.045355614,
            0.045355614,
            0.045355614,
            0.045355614,
            0.045355614,
        ],
        dtype=np.float32,
    )

    _closed_qpos = init_qpos.copy()
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
        self.gripper_joint_names = [
            "left_driver_joint",
            "left_inner_knuckle_joint",
            "left_finger_joint",
            "right_driver_joint",
            "right_inner_knuckle_joint",
            "right_finger_joint",
        ]

        self.arm_stiffness = 1500.0
        self.arm_damping = 120.0
        self.arm_force_limit = 40.0

        # More responsive
        self.gripper_stiffness = 1200.0
        self.gripper_damping = 80.0
        self.gripper_force_limit = 35.0
        self.gripper_friction = 0.7

        super().__init__(*args, **kwargs)

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        self._tcp_offset = sapien.Pose([0, 0, 0], [1, 0, 0, 0])

    @property
    def tcp_pose(self) -> "sapien.Pose":
        return self.tcp.pose * self._tcp_offset

    @property
    def _controller_configs(self):
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
            interpolate=True,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
            interpolate=True,
        )

        mimic_map = {
            "left_inner_knuckle_joint": {"joint": "left_driver_joint"},
            "left_finger_joint": {"joint": "left_driver_joint"},
            "right_driver_joint": {"joint": "left_driver_joint"},
            "right_inner_knuckle_joint": {"joint": "left_driver_joint"},
            "right_finger_joint": {"joint": "left_driver_joint"},
        }

        gripper_abs = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            0.0,
            0.85,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=True,
            interpolate=True,
        )
        gripper_abs.mimic = mimic_map

        gripper_delta = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -0.03,
            0.03,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            friction=self.gripper_friction,
            use_delta=True,
            interpolate=True,
        )
        gripper_delta.mimic = mimic_map

        return _controller_dict(
            dict(
                pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_abs),
                pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, gripper=gripper_delta),
            )
        )
