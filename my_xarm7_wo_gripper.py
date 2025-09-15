import numpy as np
import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.utils.structs import Pose as MSPose
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils


@register_agent()
class Xarm7NoGripper(BaseAgent):
    uid = "my_xarm7_wo_gripper"
    urdf_path = "xarm7_wo_gripper.urdf"
    # TCP is defined in the URDF as a fixed child of link7
    ee_link_name = "link_tcp"

    # 7-DOF arm only
    init_qpos = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            1.0471976,
            0.0,
            1.0471976,
            -1.5707964,
        ],
        dtype=np.float32,
    )

    keyframes = dict(
        home=Keyframe(qpos=init_qpos.copy(), pose=sapien.Pose([0, 0, 0]))
    )

    def __init__(self, *args, **kwargs):
        # Joint names as defined in xarm7_wo_gripper.urdf
        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]

        # PD parameters (reasonable defaults)
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 30

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # TCP (Tool Center Point)
    # ------------------------------------------------------------------
    def _after_init(self):
        # Resolve the physical TCP link from the articulation
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        # Local offset from the TCP link frame (can be changed via set_tcp_offset)
        self._tcp_offset = sapien.Pose([0, 0, 0], [1, 0, 0, 0])

    def set_tcp_link(self, link_name: str):
        self.ee_link_name = link_name
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)

    def set_tcp_offset(self, p=None, q=None, pose: "sapien.Pose" = None):
        if pose is not None:
            self._tcp_offset = pose
            return
        if p is None and q is None:
            self._tcp_offset = sapien.Pose([0, 0, 0], [1, 0, 0, 0])
            return
        if p is None:
            p = [0, 0, 0]
        if q is None:
            q = [1, 0, 0, 0]
        self._tcp_offset = sapien.Pose(p, q)

    @property
    def tcp_pose(self) -> "sapien.Pose":
        return self.tcp.pose * self._tcp_offset

    @property
    def tcp_pos(self):
        return self.tcp_pose.p

    @property
    def _controller_configs(self):
        # Arm controllers only (no gripper)
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

        controller_configs = dict(
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
            ),
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
            ),
        )

        return deepcopy_dict(controller_configs)

