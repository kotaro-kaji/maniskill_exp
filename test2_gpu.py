import gymnasium as gym
import mani_skill.envs  # registers built-in envs
import my_xarm7  # registers your custom robot
import task_pushcube  # registers MyPushCube-v1 from your file
import my_xarm7_mjcf
import numpy as np


def main():
    env = gym.make(
        "MyPushCube-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        robot_uids="my_xarm7_mjcf",
        render_mode="human",
        sim_backend="gpu",
        render_backend="gpu",
    )

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        # Example 1: random valid action
        # action = env.action_space.sample()

        # Example 2: manual action vector (must be a numpy array)
        action = np.array([
            0.9079149,
            0.7303872,
            -0.9322209,
            0.9849459,
            -0.10654075,
            0.3548803,
            -0.02951217,
            -0.625985,
        ], dtype=np.float32)

        # Clip to valid bounds (similar to PPO script)
        low, high = env.single_action_space.low, env.single_action_space.high
        action = np.clip(action, low, high)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
    env.close()


if __name__ == "__main__":
    main()
