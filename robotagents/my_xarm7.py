import os
import numpy as np
import sapien
import torch

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.utils.structs import Pose as MSPose
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.actor import Actor


@register_agent()
class Xarm7(BaseAgent):
    uid = "my_xarm7"
    arm_delta_pos_limit = 0.06
    urdf_path = "robotagents/assets/xarm7/xarm7_1305_left.urdf"
    # urdf_path = "robotagents/assets/xarm7/xarm7_1305_gripper_realsense_sn_kinematics.urdf"
    urdf_config = dict(
        _materials=dict(
            gripper=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            left_finger=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
            right_finger=dict(material="gripper", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )
    # Default TCP link defined in the URDF (fixed joint from gripper base)
    ee_link_name = "link_tcp"

    # 初期位置これで。
    # Note: This articulation has 13 active DOFs (7 arm + 6 gripper joints).
    # We set all gripper joints to the same opening value.
    init_qpos = np.asarray(
        [
            -0.00001,
            -0.5236051678657532,
            0.00,
            0.7853981852531433,
            -0.00001,
            1.30899178981781,
            -0.000001,
            # Gripper DOFs (6):
            0.0,  # left_drive_joint (0 rad = fully open, 0.85 rad = fully closed)
            0.0,  # left_inner_knuckle_joint
            0.0,  # right_drive_joint
            0.0,  # right_inner_knuckle_joint
            0.0,  # left_finger_joint
            0.0,  # right_finger_joint
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
        self.gripper_joint_names = ["left_drive_joint", "right_drive_joint"]

        # PD parameters (defaults) — overridable via env vars for quick tuning
        # Arm
        self.arm_stiffness = float(os.getenv("XARM_ARM_KP", 11000))
        self.arm_damping = float(os.getenv("XARM_ARM_KD", 80))
        self.arm_force_limit = float(os.getenv("XARM_ARM_FMAX", 10000))
        # Gripper
        self.gripper_stiffness = float( 1e3)
        self.gripper_damping = float( 5e2)
        self.gripper_force_limit = float( 50.1)
        self.gripper_friction = float(1.0)

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # TCP (Tool Center Point)
    # ------------------------------------------------------------------
    def _after_init(self):
        # Resolve the physical TCP link from the articulation
        # and prepare an optional offset pose you can customize.
        self.finger1_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "left_finger")
        self.finger2_link = sapien_utils.get_obj_by_name(self.robot.get_links(), "right_finger")
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        # Local offset from the TCP link frame (can be changed via set_tcp_offset)
        self._tcp_offset = sapien.Pose([0, 0, 0], [1, 0, 0, 0])

    def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
        l_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger1_link, object
        )
        r_contact_forces = self.scene.get_pairwise_contact_forces(
            self.finger2_link, object
        )
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(
            lforce >= min_force, torch.rad2deg(langle) <= max_angle
        )
        rflag = torch.logical_and(
            rforce >= min_force, torch.rad2deg(rangle) <= max_angle
        )
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2):
        qvel = self.robot.get_qvel()[..., :7]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

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

        #以下のように制限を設けましたが、ただのpd_joint_pos controllerのための制限であり、pd_joint_delta_pos controllerの制限は実質的にはURDFの関節角度制限のほうがずっと支配的です。
        #URDFのjoint limitを正しく設定すべきです。reset条件に、関節角度のはみ出しを設けるのも選択肢だと思います。
        arm_joint_lower = np.array(
        [
            -2 * np.pi,
            np.deg2rad(-118),
            -2 * np.pi,
            np.deg2rad(-11),
            -2 * np.pi,
            np.deg2rad(-97),
            -2 * np.pi,
        ],
        dtype=np.float32,
        )
        arm_joint_upper = np.array(
            [
                2 * np.pi,
                np.deg2rad(120),
                2 * np.pi,
                np.deg2rad(225),
                2 * np.pi,
                np.pi,
                2 * np.pi,
            ],
            dtype=np.float32,
        )


        # Arm controllers
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower = arm_joint_lower, 
            upper = arm_joint_upper,
            stiffness = self.arm_stiffness,
            damping =  self.arm_damping,
            force_limit = self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower = -self.arm_delta_pos_limit,
            upper = self.arm_delta_pos_limit,
            stiffness = self.arm_stiffness,
            damping =  self.arm_damping,
            force_limit = self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_ee_delta_pos = PDEEPosControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.10,
            pos_upper=0.10,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            delta_solver_config=dict(type="levenberg_marquardt", alpha=0.005),
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-1e-2,
            pos_upper=1e-2,
            rot_lower=-1.5e-2,
            rot_upper=1.5e-2,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )

        gripper_mimic_map = {
            "right_drive_joint": {"joint": "left_drive_joint"},
        }
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            0.05,
            0.84,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            friction=self.gripper_friction,
            normalize_action=False,
        )
        gripper_pd_joint_pos.mimic = gripper_mimic_map
        gripper_pd_joint_delta_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower = -0.1,
            upper = 0.1,
            stiffness = self.gripper_stiffness,
            damping =  self.gripper_damping,
            force_limit = self.gripper_force_limit,
            friction=self.gripper_friction,
            use_delta=True,
        )
        gripper_pd_joint_delta_pos.mimic = gripper_mimic_map

        controller_configs = dict(
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos,
                #balance_passive_force=False
            ),
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_delta_pos,
                #balance_passive_force=False
            ),
            pd_ee_delta_pos=dict(
                arm=arm_pd_ee_delta_pos,
                gripper=gripper_pd_joint_delta_pos,
                #balance_passive_force=False
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                gripper=gripper_pd_joint_delta_pos,
                #balance_passive_force=False
            ),
        )

        return deepcopy_dict(controller_configs)
