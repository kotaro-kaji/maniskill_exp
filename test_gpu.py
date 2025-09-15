import argparse

import gymnasium as gym
import mani_skill.envs  # registers built-in envs
import my_xarm7  # registers your custom robot


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-uid", type=str, default="my_xarm7")
    parser.add_argument("--control-mode", type=str, default="pd_joint_pos")
    # Default to GPU backends (maps "gpu"/"cuda" to PhysX/SAPIEN CUDA)
    parser.add_argument("--sim-backend", type=str, default="gpu")
    parser.add_argument("--render-backend", type=str, default="gpu")
    return parser.parse_args()


def main():
    args = parse_args()

    env = gym.make(
        "Empty-v1",
        obs_mode="none",
        reward_mode="none",
        control_mode=args.control_mode,
        robot_uids=args.robot_uid,
        render_mode="human",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
    )

    env.reset(seed=0)

    # Simple viewer loop with zero actions
    for _ in range(300):
        env.step(env.action_space.sample() * 0)
        env.render()

    env.close()


if __name__ == "__main__":
    main()

