from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import tkinter as tk
from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tasks.dual.task_dual_box_rotation  # noqa: F401
import tasks.dual.task_dual_box_rotation_regrasp  # noqa: F401
import tasks.dual.task_dual_box_rotation_sandwitch  # noqa: F401
import tasks.dual.task_dual_box_uprighting_task  # noqa: F401
import tasks.dual.task_dual_simple  # noqa: F401
import tasks.dual.task_dual_trash_bin_rolling  # noqa: F401
import tasks.single.task_single_cardboard_cabinet  # noqa: F401
import tasks.single_arm.pick_cube  # noqa: F401
import tasks.single_arm.push_cube  # noqa: F401


KEY_NAMES = (
    "w",
    "s",
    "a",
    "d",
    "q",
    "e",
    "i",
    "k",
    "j",
    "l",
    "u",
    "o",
    "t",
    "g",
    "z",
    "x",
)


def is_joint_delta_control(control_mode: str) -> bool:
    return control_mode == "pd_joint_delta_pos"


class KeyboardInputWindow:
    def __init__(self, control_mode: str):
        self.keys = {key: False for key in KEY_NAMES}
        self.reset_requested = False
        self.switch_requested = False
        self.quit_requested = False

        self.root = tk.Tk()
        self.root.title("teleop keyboard input")
        self.root.geometry("420x220")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        if is_joint_delta_control(control_mode):
            text = (
                "Focus this window for teleop input\n\n"
                "W/S: joint1    A/D: joint2    Q/E: joint3\n"
                "J/L: joint4    I/K: joint5    U/O: joint6\n"
                "T/G: joint7    Z: close gripper    X: open gripper\n"
                "TAB: switch arm    R: reset    ESC: quit"
            )
        else:
            text = (
                "Focus this window for teleop input\n\n"
                "W/S: +/-X    A/D: +/-Y    Q/E: +/-Z\n"
                "J/L: roll    I/K: pitch   U/O: yaw\n"
                "Z: close gripper    X: open gripper\n"
                "TAB: switch arm    R: reset    ESC: quit"
            )
        label = tk.Label(self.root, text=text, justify="left", padx=16, pady=16)
        label.pack(fill="both", expand=True)

        self.root.bind("<KeyPress>", self._on_press)
        self.root.bind("<KeyRelease>", self._on_release)
        self.root.focus_force()

    def _key_name(self, event):
        return event.keysym.lower()

    def _on_press(self, event):
        key = self._key_name(event)
        if key in self.keys:
            self.keys[key] = True
        elif key == "r":
            self.reset_requested = True
        elif key == "tab":
            self.switch_requested = True
        elif key == "escape":
            self.quit_requested = True

    def _on_release(self, event):
        key = self._key_name(event)
        if key in self.keys:
            self.keys[key] = False

    def update(self):
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.quit_requested = True

    def pop_reset(self):
        value = self.reset_requested
        self.reset_requested = False
        return value

    def pop_switch(self):
        value = self.switch_requested
        self.switch_requested = False
        return value

    def pressed_keys(self):
        return dict(self.keys)

    def close(self):
        self.quit_requested = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keyboard teleop for xArm7 ManiSkill tasks"
    )
    parser.add_argument("--env-id", default="MyDualCardboardCabinet-v1")
    parser.add_argument("--control-mode", default="pd_ee_delta_pose")
    parser.add_argument("--position-action", type=float, default=1.0)
    parser.add_argument("--rotation-action", type=float, default=1.0)
    parser.add_argument("--joint-action", type=float, default=0.4)
    parser.add_argument("--yaw-multiplier", type=float, default=2.0)
    parser.add_argument("--gripper-action", type=float, default=0.4)
    parser.add_argument("--control-hz", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-dir", default="teleop_logs")
    return parser.parse_args()


def zero_action(env) -> OrderedDict:
    if isinstance(env.action_space, gym.spaces.Dict):
        return OrderedDict(
            (uid, np.zeros(space.shape, dtype=np.float32))
            for uid, space in env.action_space.items()
        )
    return OrderedDict(
        [(env.unwrapped.agent.uid, np.zeros(env.action_space.shape, dtype=np.float32))]
    )


def step_action(env, action_by_uid):
    if isinstance(env.action_space, gym.spaces.Dict):
        return action_by_uid
    return next(iter(action_by_uid.values()))


def agent_uids(env):
    if isinstance(env.action_space, gym.spaces.Dict):
        return tuple(env.action_space.keys())
    return (env.unwrapped.agent.uid,)


def sub_agents(env):
    agent = env.unwrapped.agent
    return agent.agents if hasattr(agent, "agents") else (agent,)


def action_shape(env, uid):
    if isinstance(env.action_space, gym.spaces.Dict):
        return env.action_space[uid].shape
    return env.action_space.shape


def render_human(env):
    return env.unwrapped.render_human()


def keyboard_to_joint_delta_action(keys, shape, args) -> np.ndarray:
    assert len(shape) == 1, shape
    action = np.zeros(shape, dtype=np.float32)
    assert action.shape[0] == 8, action.shape

    key_pairs = (
        ("w", "s", 0),
        ("a", "d", 1),
        ("q", "e", 2),
        ("j", "l", 3),
        ("i", "k", 4),
        ("u", "o", 5),
        ("t", "g", 6),
    )
    for positive_key, negative_key, joint_idx in key_pairs:
        if keys[positive_key]:
            action[joint_idx] += args.joint_action
        if keys[negative_key]:
            action[joint_idx] -= args.joint_action

    if keys["z"] and not keys["x"]:
        action[7] = args.gripper_action
    elif keys["x"] and not keys["z"]:
        action[7] = -args.gripper_action

    return np.clip(action, -1.0, 1.0)


def keyboard_to_ee_delta_action(keys, shape, args) -> np.ndarray:
    assert len(shape) == 1, shape
    action = np.zeros(shape, dtype=np.float32)
    assert action.shape[0] in (7, 8), action.shape

    if keys["w"]:
        action[0] += args.position_action
    if keys["s"]:
        action[0] -= args.position_action
    if keys["a"]:
        action[1] += args.position_action
    if keys["d"]:
        action[1] -= args.position_action
    if keys["q"]:
        action[2] += args.position_action
    if keys["e"]:
        action[2] -= args.position_action

    if keys["j"]:
        action[3] -= args.rotation_action
    if keys["l"]:
        action[3] += args.rotation_action
    if keys["i"]:
        action[4] += args.rotation_action
    if keys["k"]:
        action[4] -= args.rotation_action
    if keys["u"]:
        action[5] += args.rotation_action * args.yaw_multiplier
    if keys["o"]:
        action[5] -= args.rotation_action * args.yaw_multiplier

    if keys["z"] and not keys["x"]:
        action[6:] = args.gripper_action
    elif keys["x"] and not keys["z"]:
        action[6:] = -args.gripper_action

    return np.clip(action, -1.0, 1.0)


def keyboard_to_action(keys, shape, args) -> np.ndarray:
    if is_joint_delta_control(args.control_mode):
        return keyboard_to_joint_delta_action(keys, shape, args)
    return keyboard_to_ee_delta_action(keys, shape, args)


def tensor_row(tensor):
    return tensor.detach().cpu().numpy().reshape(-1).tolist()


def pose_row(pose):
    return tensor_row(pose.raw_pose)


def action_log_row(action, action_dim):
    row = action.tolist()
    while len(row) < action_dim:
        row.append("")
    return row


def open_log_files(log_dir: str, env_id: str, action_dim: int):
    run_name = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, f"{env_id}_keyboard_{run_name}")
    os.makedirs(run_dir, exist_ok=True)

    command_path = os.path.join(run_dir, "command_log.csv")
    state_path = os.path.join(run_dir, "state_log.csv")
    command_file = open(command_path, "w", newline="", encoding="utf-8")
    state_file = open(state_path, "w", newline="", encoding="utf-8")

    command_writer = csv.writer(command_file)
    state_writer = csv.writer(state_file)
    command_writer.writerow(
        [
            "step",
            "time",
            "uid",
            "active",
            *[f"key_{key}" for key in KEY_NAMES],
            *[f"action_{i}" for i in range(action_dim)],
        ]
    )
    state_writer.writerow(
        ["step", "time", "uid"]
        + [f"tcp_pose_{i}" for i in range(7)]
        + [f"qpos_{i}" for i in range(13)]
    )
    return run_dir, command_file, state_file, command_writer, state_writer


def log_step(
    step_idx: int,
    elapsed: float,
    uids,
    sub_agents,
    action,
    active_uid,
    keys,
    command_writer,
    state_writer,
    action_dim,
):
    key_values = [int(keys[key]) for key in KEY_NAMES]
    for uid, sub_agent in zip(uids, sub_agents):
        command_writer.writerow(
            [step_idx, elapsed, uid, int(uid == active_uid)]
            + key_values
            + action_log_row(action[uid], action_dim)
        )
        state_writer.writerow(
            [step_idx, elapsed, uid]
            + pose_row(sub_agent.tcp_pose)
            + tensor_row(sub_agent.robot.get_qpos())
        )


def print_controls(uids, active_arm, control_mode):
    print("Keyboard teleop started.")
    print("Focus the small 'teleop keyboard input' window, not the Sapien viewer.")
    if is_joint_delta_control(control_mode):
        print("W/S: joint1 | A/D: joint2 | Q/E: joint3")
        print("J/L: joint4 | I/K: joint5 | U/O: joint6 | T/G: joint7")
    else:
        print("W/S: +/-X | A/D: +/-Y | Q/E: +/-Z")
        print("J/L: roll | I/K: pitch | U/O: yaw")
    print("Z: close gripper | X: open gripper")
    print("TAB: switch active arm | r: reset | escape: quit")
    print(f"active arm: {uids[active_arm]}")


def main():
    args = parse_args()
    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode=args.control_mode,
        render_mode="human",
    )
    uids = agent_uids(env)
    assert len(uids) in (1, 2), uids
    for uid in uids:
        assert action_shape(env, uid) in ((7,), (8,)), (uid, action_shape(env, uid))

    active_arm = 0
    env.reset(seed=args.seed)
    viewer = render_human(env)
    keyboard = KeyboardInputWindow(args.control_mode)
    print_controls(uids, active_arm, args.control_mode)

    action_dim = max(action_shape(env, uid)[0] for uid in uids)
    run_dir, command_file, state_file, command_writer, state_writer = open_log_files(
        args.log_dir, args.env_id, action_dim
    )
    print(f"logging to: {run_dir}")

    step_idx = 0
    start_time = time.time()
    dt = 1.0 / args.control_hz
    try:
        while True:
            viewer = render_human(env)
            keyboard.update()
            if hasattr(viewer, "closed") and viewer.closed:
                break
            if keyboard.quit_requested:
                break
            if keyboard.pop_reset():
                env.reset()
            if len(uids) == 2 and keyboard.pop_switch():
                active_arm = 1 - active_arm
                print(f"active arm: {uids[active_arm]}")

            keys = keyboard.pressed_keys()
            action = zero_action(env)
            active_uid = uids[active_arm]
            action[active_uid] = keyboard_to_action(
                keys, action_shape(env, active_uid), args
            )

            env.step(step_action(env, action))
            elapsed = time.time() - start_time
            log_step(
                step_idx,
                elapsed,
                uids,
                sub_agents(env),
                action,
                active_uid,
                keys,
                command_writer,
                state_writer,
                action_dim,
            )
            command_file.flush()
            state_file.flush()
            step_idx += 1
            time.sleep(dt)
    finally:
        command_file.close()
        state_file.close()
        keyboard.close()
        env.close()


if __name__ == "__main__":
    main()
