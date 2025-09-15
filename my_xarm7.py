import numpy as np
import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.utils.structs import Pose as MSPose
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils


@register_agent()
class Xarm7(BaseAgent):
    uid = "my_xarm7"
    urdf_path = "xarm7.urdf"
    # Default TCP link defined in the URDF (fixed joint from gripper base)
    ee_link_name = "link_tcp"

    # 初期位置これで。
    # Note: This articulation has 13 active DOFs (7 arm + 6 gripper joints).
    # We set all gripper joints to the same opening value.
    init_qpos = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            np.pi / 3,
            0.0,
            np.pi / 3,
            -np.pi / 2,
            # Gripper DOFs (6):
            0.0453556139430441,  # drive_joint
            0.0453556139430441,  # left_inner_knuckle_joint
            0.0453556139430441,  # right_outer_knuckle_joint
            0.0453556139430441,  # right_inner_knuckle_joint
            0.0453556139430441,  # left_finger_joint
            0.0453556139430441,  # right_finger_joint
        ],
        dtype=np.float32,
    )

    # Provide a convenient keyframe for test.py visualization
    keyframes = dict(
        home=Keyframe(qpos=init_qpos.copy(), pose=sapien.Pose([0, 0, 0]))
    )

    def __init__(self, *args, **kwargs):
        # Joint names as defined in xarm7.urdf
        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        # Active gripper joints in this URDF
        self.gripper_joint_names = [
            "drive_joint",
            "left_inner_knuckle_joint",
            "right_outer_knuckle_joint",
            "right_inner_knuckle_joint",
            "left_finger_joint",
            "right_finger_joint",
        ]

        # PD parameters (reasonable defaults)
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 500

        self.gripper_stiffness = 1e3
        self.gripper_damping = 1e2
        self.gripper_force_limit = 50

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # TCP (Tool Center Point)
    # ------------------------------------------------------------------
    def _after_init(self):
        # Resolve the physical TCP link from the articulation
        # and prepare an optional offset pose you can customize.
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        # Local offset from the TCP link frame (can be changed via set_tcp_offset)
        self._tcp_offset = sapien.Pose([0, 0, 0], [1, 0, 0, 0])

    def set_tcp_link(self, link_name: str):
        """Change the TCP base link by name (must exist in the robot)."""
        self.ee_link_name = link_name
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)

    def set_tcp_offset(self, p=None, q=None, pose: "sapien.Pose" = None):
        """Set local TCP offset relative to the TCP link.

        Args:
            p: iterable of 3 floats (xyz), world units
            q: iterable of 4 floats (xyzw) quaternion
            pose: alternatively pass a sapien.Pose
        """
        if pose is not None:
            self._tcp_offset = pose
            return
        if p is None and q is None:
            # Reset to identity offset
            self._tcp_offset = sapien.Pose([0, 0, 0], [1, 0, 0, 0])
            return
        if p is None:
            p = [0, 0, 0]
        if q is None:
            q = [1, 0, 0, 0]
        self._tcp_offset = sapien.Pose(p, q)

    @property
    def tcp_pose(self) -> "sapien.Pose":
        """World pose of the TCP (link pose composed with the local offset)."""
        return self.tcp.pose * self._tcp_offset

    @property
    def tcp_pos(self):
        return self.tcp_pose.p

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

        # 1-DOF gripper via mimic controller
        gripper_mimic_map = {
            # mimic_joint: {"joint": control_joint}
            "left_inner_knuckle_joint": {"joint": "drive_joint"},
            "right_outer_knuckle_joint": {"joint": "drive_joint"},
            "right_inner_knuckle_joint": {"joint": "drive_joint"},
            "left_finger_joint": {"joint": "drive_joint"},
            "right_finger_joint": {"joint": "drive_joint"},
        }
        gripper_pd_joint_pos_mimic = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            None,
            None,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            normalize_action=False,
        )
        gripper_pd_joint_pos_mimic.mimic = gripper_mimic_map
        gripper_pd_joint_delta_pos_mimic = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            -0.1,
            0.1,
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
