import gymnasium as gym
import mani_skill.envs
import my_xarm7_mjcf

print('Making env...')
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
print('Reset...')
obs, info = env.reset(seed=0)
agent = env.unwrapped.agent
print('Agent UID:', agent.uid)
print('Active joints:', [j.name for j in agent.robot.active_joints])
print('Keyframes:', list(agent.keyframes.keys()))
print('Done ok')
