import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers.gymnasium import CPUGymWrapper
env = gym.make("PickCube-v1", num_envs=1, obs_mode="state")
env = CPUGymWrapper(env)
obs, _ = env.reset(seed=0)
for _ in range(200):
    action = env.action_space.sample()
    obs, rew, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("Episode finished:", info)
        obs, _ = env.reset()














