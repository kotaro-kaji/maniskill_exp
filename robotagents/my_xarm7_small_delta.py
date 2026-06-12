from mani_skill.agents.registration import register_agent

from .my_xarm7 import Xarm7


@register_agent()
class Xarm7SmallDelta(Xarm7):
    uid = "my_xarm7_delta003"
    arm_delta_pos_limit = 0.003
    arm_stiffness_default = 21000
    arm_force_limit_default = 20000


@register_agent()
class Xarm7Delta001Stiff(Xarm7):
    uid = "my_xarm7_delta001"
    arm_delta_pos_limit = 0.001
    arm_stiffness_default = 11000
    arm_force_limit_default = 10000


@register_agent()
class Xarm7Delta03Soft(Xarm7):
    uid = "my_xarm7_delta03"
    arm_delta_pos_limit = 0.03
    arm_stiffness_default = 5500
    arm_force_limit_default = 5000
