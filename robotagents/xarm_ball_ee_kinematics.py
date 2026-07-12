from mani_skill.agents.registration import register_agent

from robotagents.xarm_ball_ee_wo_force_sensor import Xarm7BallEEWoForceSensor


@register_agent()
class Xarm7BallEEKinematics(Xarm7BallEEWoForceSensor):
    uid = "xarm7_ball_ee_kinematics"
    urdf_path = "xarm7_1305_left_ball_ee_kinematics.urdf"
