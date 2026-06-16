from mani_skill.agents.registration import register_agent

from .my_xarm7 import Xarm7


@register_agent()
class Xarm7SmallDelta(Xarm7):
    uid = "my_xarm7_delta003"
    arm_delta_pos_limit = 0.003
    arm_stiffness_default = 22000
    arm_force_limit_default = 20000


@register_agent()
class Xarm7Delta01Stiff(Xarm7):
    uid = "my_xarm7_delta01"
    arm_delta_pos_limit = 0.01
    arm_stiffness_default = 6600
    arm_force_limit_default = 6000


@register_agent()
class Xarm7Delta02Soft(Xarm7):
    uid = "my_xarm7_delta02"
    arm_delta_pos_limit = 0.02
    arm_stiffness_default = 3300
    arm_force_limit_default = 3000


@register_agent()
class Xarm7Delta001Stiff(Xarm7):
    uid = "my_xarm7_delta001"
    arm_delta_pos_limit = 0.001
    arm_stiffness_default = 66000
    arm_force_limit_default = 60000


@register_agent()
class Xarm7Delta03Soft(Xarm7):
    uid = "my_xarm7_delta03"
    arm_delta_pos_limit = 0.03
    arm_stiffness_default = 2200
    arm_force_limit_default = 2000


@register_agent()
class Xarm7KinematicsDelta001(Xarm7Delta001Stiff):
    uid = "my_xarm7_kinematics_delta001"
    urdf_path = "robotagents/assets/xarm7/xarm7_1305_gripper_realsense_sn_kinematics.urdf"


@register_agent()
class Xarm7KinematicsDelta01(Xarm7Delta01Stiff):
    uid = "my_xarm7_kinematics_delta01"
    urdf_path = "robotagents/assets/xarm7/xarm7_1305_gripper_realsense_sn_kinematics.urdf"


@register_agent()
class Xarm7KinematicsDelta02(Xarm7Delta02Soft):
    uid = "my_xarm7_kinematics_delta02"
    urdf_path = "robotagents/assets/xarm7/xarm7_1305_gripper_realsense_sn_kinematics.urdf"


@register_agent()
class Xarm7KinematicsDelta003(Xarm7SmallDelta):
    uid = "my_xarm7_kinematics_delta003"
    urdf_path = "robotagents/assets/xarm7/xarm7_1305_gripper_realsense_sn_kinematics.urdf"


@register_agent()
class Xarm7KinematicsDelta03(Xarm7Delta03Soft):
    uid = "my_xarm7_kinematics_delta03"
    urdf_path = "robotagents/assets/xarm7/xarm7_1305_gripper_realsense_sn_kinematics.urdf"
