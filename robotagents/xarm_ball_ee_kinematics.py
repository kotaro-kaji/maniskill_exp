from mani_skill.agents.registration import register_agent

from robotagents.xarm_ball_ee_wo_force_sensor import Xarm7BallEEWoForceSensor


@register_agent()
class Xarm7BallEEKinematicsLeft(Xarm7BallEEWoForceSensor):
    uid = "xarm7_ball_ee_kinematics_left"
    urdf_path = "robotagents/assets/xarm7/xarm7_1305_left_ball_ee_wo_force_sensor_kinematics.urdf"


@register_agent()
class Xarm7BallEEKinematicsRight(Xarm7BallEEWoForceSensor):
    uid = "xarm7_ball_ee_kinematics_right"
    urdf_path = "robotagents/assets/xarm7/xarm7_1305_right_ball_ee_wo_force_sensor_kinematics.urdf"
