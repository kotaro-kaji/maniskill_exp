import argparse
from collections import OrderedDict
from pathlib import Path
import sys

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import task_pushcube_beatiful  # noqa: F401
import tasks.dual.task_dual_box_rotation  # noqa: F401
import tasks.dual.task_dual_cardboard_cabinet  # noqa: F401
import tasks.dual.task_dual_simple  # noqa: F401
import tasks.single_arm.pick_cube  # noqa: F401
import tasks.single_arm.push_cube  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="MyDualBoxRotation-v0")
    parser.add_argument("--obs-mode", type=str, default="state")
    parser.add_argument("--control-mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--render-mode", type=str, default="human")
    parser.add_argument("--robot-init-noise-scale", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--random-action",
        action="store_true",
        help="Use random actions instead of zero actions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        robot_init_noise_scale=args.robot_init_noise_scale,
    )

    print(f"Environment ID: {args.env_id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, _ = env.reset(seed=args.seed)
    done = False
    while True:
        action = env.action_space.sample()
        if not args.random_action:
            action = OrderedDict((k, np.zeros_like(v)) for k, v in action.items())
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        # if done:
        #     break

    print(f"Episode finished: {info}")
    env.close()


if __name__ == "__main__":
    main()
