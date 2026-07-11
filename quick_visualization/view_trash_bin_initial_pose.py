import argparse
from collections.abc import Mapping
from pathlib import Path
import sys

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tasks.dual.task_dual_trash_bin_rolling  # noqa: F401
from scenebuilders.xarm7_table_scene_builder import ROBOT_BASE_X_OFFSET
from tasks.dual.task_dual_trash_bin_rolling import MyDualTrashBinRollingEnv


def parse_xy(value: str) -> tuple[float, float]:
    tokens = [token.strip() for token in value.split(",")]
    assert len(tokens) == 2, value
    return float(tokens[0]), float(tokens[1])


def zero_action(env):
    action = env.action_space.sample()
    if isinstance(action, Mapping):
        return {key: np.zeros_like(value) for key, value in action.items()}
    return np.zeros_like(action)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-bin-xy", type=parse_xy, default=(0.34, 0.0))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=0)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--sim-backend", type=str, default="physx_cuda")
    parser.add_argument("--robot-init-noise-scale", type=float, default=0.0)
    return parser.parse_args()


def print_initial_geometry(initial_bin_xy: tuple[float, float]):
    center_world_x = ROBOT_BASE_X_OFFSET + initial_bin_xy[0]
    edge_world_x = center_world_x + MyDualTrashBinRollingEnv.BIN_HEIGHT / 2.0
    print(f"initial_bin_xy bimanual_center frame: {initial_bin_xy}")
    print(f"bin center world x: {center_world_x:.6f}")
    print(f"bin +x edge world x: {edge_world_x:.6f}")


def main():
    args = parse_args()
    print_initial_geometry(args.initial_bin_xy)

    env = gym.make(
        "MyDualTrashBinRolling-v0",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
        sim_backend=args.sim_backend,
        robot_init_noise_scale=args.robot_init_noise_scale,
        initial_bin_xy=args.initial_bin_xy,
    )
    env.reset(seed=args.seed)

    action = zero_action(env)
    for _ in range(args.settle_steps):
        env.step(action)

    print("Sapien viewer opened.")
    print("Close the viewer window or press Ctrl+C in this terminal to quit.")
    print("Use --settle-steps 20 to inspect the post-fall pose.")
    print("Use --play to continue stepping with zero action.")

    try:
        while True:
            viewer = env.unwrapped.render_human()
            if hasattr(viewer, "closed") and viewer.closed:
                break
            if args.play:
                env.step(action)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
