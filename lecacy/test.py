import mani_skill.envs
import gymnasium as gym
N = 4
env = gym.make("PickCube-v1", num_envs=N)
env.action_space # shape (N, D)
env.observation_space # shape (N, ...)
env.reset()
obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
# obs (N, ...), rew (N, ), terminated (N, ), truncated (N, )
