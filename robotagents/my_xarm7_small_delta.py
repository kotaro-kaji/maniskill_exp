from mani_skill.agents.registration import register_agent

from .my_xarm7 import Xarm7


@register_agent()
class Xarm7SmallDelta(Xarm7):
    uid = "my_xarm7_small_delta"
    arm_delta_pos_limit = 0.003
    arm_stiffness_default = 21000
    arm_force_limit_default = 20000


@register_agent()
class Xarm7SmallDeltaSub(Xarm7):
    uid = "my_xarm7_small_delta_sub"
    arm_delta_pos_limit = 0.001
    arm_stiffness_default = 11000
    arm_force_limit_default = 10000


@register_agent()
class Xarm7SmallDeltaSub2(Xarm7):
    uid = "my_xarm7_small_delta_sub2"
    arm_delta_pos_limit = 0.03
    arm_stiffness_default = 5500
    arm_force_limit_default = 5000
