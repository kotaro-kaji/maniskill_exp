import argparse

import gymnasium as gym
import mani_skill.envs  # registers built-in envs
import my_xarm7  # registers your custom robot
import task_pushcube  # registers MyPushCube-v1 from your file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-mode", type=str, default="state")
    parser.add_argument("--control-mode", type=str, default="pd_joint_delta_pos")
    # Default to GPU for both physics sim and rendering. Accepts aliases like
    # "gpu", "cuda", or explicit device ids e.g., "cuda:0".
    parser.add_argument("--sim-backend", type=str, default="gpu")
    parser.add_argument("--render-backend", type=str, default="gpu")
    parser.add_argument("--robot-uid", type=str, default="my_xarm7")
    return parser.parse_args()


def main():
    args = parse_args()

    env = gym.make(
        "MyPushCube-v1",
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        robot_uids=args.robot_uid,
        render_mode="human",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
    )

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
    env.close()


if __name__ == "__main__":
    main()

