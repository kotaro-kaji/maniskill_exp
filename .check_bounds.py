import gymnasium as gym
import mani_skill.envs
import my_xarm7_mjcf

env = gym.make(
    "Empty-v1",
    obs_mode="none",
    reward_mode="none",
    control_mode="pd_joint_pos",
    robot_uids="my_xarm7_mjcf",
    sim_backend="cpu",
    render_backend="cpu",
    render_mode=None,
)
agent = env.unwrapped.agent
ctrl = agent.controller
print('Mode:', agent.control_mode)
print('Action space:', ctrl.action_space)
print('shape =', ctrl.action_space.shape)
