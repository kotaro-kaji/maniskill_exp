"""
Simple keyboard teleop for MyDualBoxRotation-v0.

- Runs the dual xArm7 ball EE environment with pd_joint_delta_pos control.
- Use number keys 1-7 to pick an arm joint, 8 to pick the gripper.
- TAB switches active arm (Left/Right). Active arm and joint are printed on change.
- Press '=' to increment the selected joint, '-' to decrement (hold to continue).
- 'r' resets the environment, 'esc' quits.
- Contact forces are monitored each step; if box <-> stick or box <-> gripper base
  exceeds the threshold, a big alert is printed.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from typing import Dict, Iterable, Tuple

import os
import sys

# Ensure local ManiSkill checkout is on the path if the package is not installed
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MANISKILL_LOCAL = os.path.join(PROJECT_ROOT, "reference", "ManiSkill")
if MANISKILL_LOCAL not in sys.path:
    sys.path.insert(0, MANISKILL_LOCAL)

import numpy as np
import torch
import mani_skill.envs  # noqa: F401 (registers envs)
import gymnasium as gym

# Registers the custom task
import tasks.dual.task_dual_box_rotation  # noqa: F401


def build_zero_action(env) -> OrderedDict:
    """Create a zero action with the same structure as the env action_space."""
    sample = env.action_space.sample()
    return OrderedDict((k, np.zeros_like(v)) for k, v in sample.items())


def format_arm_joint(joint_idx: int, arm_dof: int) -> str:
    if joint_idx < arm_dof:
        return f"joint{joint_idx + 1}"
    return "gripper"


def print_contact_alert(
    info: Dict,
    threshold: float,
    highlight_pairs: Iterable[str],
    verbose: bool = False,
):
    contact = info.get("contact/force", None)
    if contact is None:
        return
    forces, names = contact
    if isinstance(forces, torch.Tensor):
        forces = forces.detach().cpu().numpy()
    forces = np.asarray(forces)
    if forces.ndim == 1:
        forces = forces[None, :]

    alerted = False
    for pair in highlight_pairs:
        if pair not in names:
            continue
        idx = names.index(pair)
        vals = forces[:, idx]
        max_val = float(np.max(vals))
        if max_val > threshold:
            alerted = True
            print("\n" + "=" * 80)
            print(f"CONTACT ALERT: {pair} > {threshold}")
            for env_idx, val in enumerate(vals):
                if val > threshold:
                    print(f"  env {env_idx}: {val:.6f}")
            print("=" * 80 + "\n")

    if verbose:
        mask = forces > threshold
        if mask.any():
            print("\nContact pairs above threshold:")
            for name, col in zip(names, forces.T):
                for env_idx, val in enumerate(col):
                    if val > threshold:
                        print(f"  env {env_idx}: {name} -> {val:.6f}")

    return alerted


def parse_args():
    parser = argparse.ArgumentParser(description="Keyboard teleop for MyDualBoxRotation-v0")
    parser.add_argument("--env-id", default="MyDualBoxRotation-v0")
    parser.add_argument("--arm-step", type=float, default=0.02, help="Delta (rad) per step for arm joints")
    parser.add_argument(
        "--gripper-step", type=float, default=0.02, help="Delta per step for gripper joint"
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=0.01,
        help="Force threshold to trigger contact alert",
    )
    parser.add_argument(
        "--verbose-contacts",
        action="store_true",
        help="Print all contact pairs above threshold, not just highlights",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
    )
    viewer = env.render_human()
    window = viewer.window

    uids: Tuple[str, ...] = tuple(env.unwrapped.agent.agents_dict.keys())
    arm_dof = 7  # xArm7
    action_dim = env.action_space[uids[0]].shape[0]
    gripper_index = action_dim - 1

    active_arm = 0  # 0: left, 1: right
    selected_joint = 0
    print("Controls: TAB switch arm | 1-8 select joint | '=' / '-' increase/decrease | r reset | esc quit")
    print(f"Active arm: {uids[active_arm]}, selected joint: {format_arm_joint(selected_joint, arm_dof)}")
    highlight_pairs = (
        "L_link_tcp_stick|box",
        "R_link_tcp_stick|box",
        "L_xarm_gripper_base_link|box",
        "R_xarm_gripper_base_link|box",
    )

    try:
        while True:
            viewer = env.render_human()
            window = viewer.window

            if hasattr(viewer, "closed") and viewer.closed:
                break
            if window.key_press("escape"):
                break
            if window.key_press("tab"):
                active_arm = 1 - active_arm
                print(
                    f"Switched active arm to {uids[active_arm]}, selected joint: {format_arm_joint(selected_joint, arm_dof)}"
                )
            for i in range(action_dim):
                key = str(i + 1)
                if window.key_press(key):
                    selected_joint = i
                    print(f"Selected joint {format_arm_joint(selected_joint, arm_dof)} on {uids[active_arm]}")

            if window.key_press("r"):
                env.reset()
                print("Environment reset.")
                continue

            delta = 0.0
            if window.key_down("="):
                delta += args.arm_step if selected_joint < arm_dof else args.gripper_step
            if window.key_down("-"):
                delta -= args.arm_step if selected_joint < arm_dof else args.gripper_step

            action = build_zero_action(env)
            for idx, uid in enumerate(uids):
                arr = action[uid]
                if idx == active_arm and delta != 0.0:
                    target_idx = selected_joint if selected_joint < action_dim else gripper_index
                    target_idx = min(target_idx, action_dim - 1)
                    arr[target_idx] = delta

            _, _, terminated, truncated, info = env.step(action)
            print_contact_alert(
                info,
                threshold=args.contact_threshold,
                highlight_pairs=highlight_pairs,
                verbose=args.verbose_contacts,
            )
            if terminated or truncated:
                env.reset()

    finally:
        env.close()


if __name__ == "__main__":
    main()
