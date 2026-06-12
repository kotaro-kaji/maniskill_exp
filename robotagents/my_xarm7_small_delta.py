from mani_skill.agents.registration import register_agent

from .my_xarm7 import Xarm7


@register_agent()
class Xarm7SmallDelta(Xarm7):
    uid = "my_xarm7_small_delta"
    arm_delta_pos_limit = 0.003
