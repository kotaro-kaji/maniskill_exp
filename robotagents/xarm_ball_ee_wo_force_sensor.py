import os
import numpy as np
import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.utils.structs import Pose as MSPose
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils


@register_agent()
class Xarm7BallEEWoForceSensor(BaseAgent):
    uid = "xarm7_ball_ee_wo_force_sensor"
    urdf_path = "robotagents/assets/xarm7/xarm7_1305_left_ball_ee_wo_force_sensor_kinematics.urdf"
    # Use the ball link we added in the URDF as the TCP
    ee_link_name = "link_tcp_ball"
    urdf_config = dict(
        _materials=dict(
            ball_contact=dict(static_friction=0.01, dynamic_friction=0.005, restitution=0.1)
        ),
        link=dict(
            link_tcp_ball=dict(
                material="ball_contact",
            ),
        ),
    )

    # ここでinit_qposを設定。これに基づいてhome姿勢を設定。 
    init_qpos = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            # Gripper DOFs (6):
            0.85,  # drive_joint (0 rad = fully open, 0.85 rad = fully closed)
            0.85,  # left_inner_knuckle_joint
            0.85,  # right_outer_knuckle_joint
            0.85,  # right_inner_knuckle_joint
            0.85,  # left_finger_joint
            0.85,  # right_finger_joint
        ],
        dtype=np.float32,
    )

    #RoboManipBaselineにおける初期姿勢との関連性はなし。
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
        
        self.gripper_joint_names = [
            "drive_joint",
            "left_inner_knuckle_joint",
            "right_outer_knuckle_joint",
            "right_inner_knuckle_joint",
            "left_finger_joint",
            "right_finger_joint",
        ]


        self.contact_unallowed_links = [
            "link_base",
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
            "link7",
            "link_eef", #realsenseカメラのリンク
            "xarm_gripper_base_link",
            "left_finger",
            "right_finger",
            "link_tcp_stick"
        ]
        # PD parameters (defaults) — overridable via env vars for quick tuning
        # Arm
        self.arm_stiffness = float(os.getenv("XARM_ARM_KP", 1100))
        self.arm_damping = float(os.getenv("XARM_ARM_KD", 80))
        self.arm_force_limit = float(os.getenv("XARM_ARM_FMAX", 1000))
        # Gripper (driver + mimics)
        self.gripper_stiffness = float(os.getenv("XARM_GRIP_KP", 1.5))
        self.gripper_damping = float(os.getenv("XARM_GRIP_KD", 0.5))
        self.gripper_force_limit = float(os.getenv("XARM_GRIP_FMAX", 0.3)) #when it's bigger than 1.0, the robot arm goes out of control.

        super().__init__(*args, **kwargs)
        
        #for contact detection
        self.body_query: Optional[
            Tuple[physx.PhysxGpuContactBodyImpulseQuery, Tuple[int, int, int]]
        ] = None

    # ------------------------------------------------------------------
    # TCP (Tool Center Point)
    # ------------------------------------------------------------------
    def _after_init(self):
        # Resolve the physical TCP link from the articulation
        # and prepare an optional offset pose you can customize.
        self.tcp = sapien_utils.get_obj_by_name(self.robot.get_links(), self.ee_link_name)
        # Local offset from the TCP link frame (can be changed via set_tcp_offset)
        self._tcp_offset = sapien.Pose([0, 0, 0], [1, 0, 0, 0])
        self._set_link_roughness(
            link_names=["link_tcp_stick", "link_tcp_ball", "link_eef"],
            roughness=0.2,
        )
        self._set_link_roughness(
            link_names=["link7"],
            roughness=0.3,
        )

        self._contact_unallowed_links: list[Actor] = sapien_utils.get_objs_by_names(
            self.robot.get_links(),
            self.contact_unallowed_links,
        )

    def _set_link_roughness(self, link_names, roughness: float):
        for link_name in link_names:
            link = self.robot.links_map.get(link_name)
            assert link is not None, f"Link not found: {link_name}"
            for obj in link._objs:
                render_comp = obj.entity.find_component_by_type(
                    sapien.render.RenderBodyComponent
                )
                assert render_comp is not None, f"Render component not found: {link_name}"
                for shape in render_comp.render_shapes:
                    for part in shape.parts:
                        sapien_utils.set_render_material(
                            part.material, roughness=roughness
                        )
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
            lower = -0.06,
            upper = 0.06,
            stiffness = self.arm_stiffness,
            damping =  self.arm_damping,
            force_limit = self.arm_force_limit,
            use_delta=True,
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
            0.05,
            0.84,
            self.gripper_stiffness,
            self.gripper_damping,
            self.gripper_force_limit,
            normalize_action=False,
        )
        gripper_pd_joint_pos_mimic.mimic = gripper_mimic_map
        gripper_pd_joint_delta_pos_mimic = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower = -0.1,
            upper = 0.1,
            stiffness = self.gripper_stiffness,
            damping =  self.gripper_damping,
            force_limit = self.gripper_force_limit,
            use_delta=True,
        )
        gripper_pd_joint_delta_pos_mimic.mimic = gripper_mimic_map

        controller_configs = dict(
            pd_joint_pos=dict(
                arm=arm_pd_joint_pos,
                gripper=gripper_pd_joint_pos_mimic,
                #balance_passive_force=False
            ),
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos,
                gripper=gripper_pd_joint_delta_pos_mimic,
                #balance_passive_force=False
            ),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose,
                gripper=gripper_pd_joint_delta_pos_mimic,
                #balance_passive_force=False
            ),
        )

        return deepcopy_dict(controller_configs)

    
    def get_contact_detection(self):
        if self.scene.gpu_sim_enabled:
            px: physx.PhysxGpuSystem = self.scene.px
            # Create contact query if it is not existed
            if self.body_query is None:
                # Convert the order of links so that the link from the same sub-scene will come together
                # It makes life easier for reshape
                bodies = list(zip(*[link._bodies for link in self.contact_unallowed_links]))
                bodies = list(itertools.chain(*bodies))

                query = px.gpu_create_contact_body_impulse_query(bodies)
                self.body_query = (
                    query,
                    (len(self.contact_unallowed_links[0]._bodies), len(self.contact_unallowed_links), 3),
                )

            # Query contact buffer
            query, contacts_shape = self.body_query
            px.gpu_query_contact_body_impulses(query)
            contacts = (
                query.cuda_impulses.torch().clone().reshape(*contacts_shape)
            )  # [n, len(contact_unallowed_links), 3]

            return contacts
        else:
            raise ValueError("Scene is not GPU enabled")
