from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import pyspacemouse


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tasks.dual.task_dual_cardboard_cabinet  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description="SpaceMouse teleop for dual xArm7 ManiSkill tasks"
    )
    parser.add_argument("--env-id", default="MyDualCardboardCabinet-v1")
    parser.add_argument("--left-device-path", default="")
    parser.add_argument("--right-device-path", default="")
    parser.add_argument("--left-device-index", type=int, default=0)
    parser.add_argument("--right-device-index", type=int, default=1)
    parser.add_argument("--deadzone", type=float, default=0.0)
    parser.add_argument("--gripper-action", type=float, default=0.4)
    parser.add_argument("--control-hz", type=float, default=60.0)
    parser.add_argument("--log-dir", default="teleop_logs")
    parser.add_argument("--print-devices", action="store_true")
    return parser.parse_args()


def open_spacemouse(path: str, device_index: int):
    if path:
        return pyspacemouse.open_by_path(path)
    return pyspacemouse.open(device_index=device_index)


def read_latest(device):
    state = None
    for _ in range(10):
        state = device.read()
    assert state is not None
    return state


def apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0
    return float(value)


def spacemouse_to_action(state, deadzone: float, gripper_action: float) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)

    # Same axis convention as RoboManipBaselines SpacemouseInputDevice.
    action[:3] = np.array(
        [
            -apply_deadzone(state.y, deadzone),
            apply_deadzone(state.x, deadzone),
            apply_deadzone(state.z, deadzone),
        ],
        dtype=np.float32,
    )
    action[3:6] = np.array(
        [
            4.0 * apply_deadzone(state.roll, deadzone),
            4.0 * apply_deadzone(state.pitch, deadzone),
            4.0 * apply_deadzone(state.yaw, deadzone),
        ],
        dtype=np.float32,
    )

    if len(state.buttons) > 0 and state.buttons[0] > 0 and state.buttons[-1] <= 0:
        action[6] = gripper_action
    elif len(state.buttons) > 0 and state.buttons[-1] > 0 and state.buttons[0] <= 0:
        action[6] = -gripper_action

    return np.clip(action, -1.0, 1.0)


def zero_action(env) -> OrderedDict:
    return OrderedDict((uid, np.zeros(space.shape, dtype=np.float32)) for uid, space in env.action_space.items())


def state_values(state):
    if state is None:
        return [""] * 8
    buttons = "".join(str(int(v)) for v in state.buttons)
    return [
        state.t,
        state.x,
        state.y,
        state.z,
        state.roll,
        state.pitch,
        state.yaw,
        buttons,
    ]


def tensor_row(tensor):
    return tensor.detach().cpu().numpy().reshape(-1).tolist()


def pose_row(pose):
    return tensor_row(pose.raw_pose)


def open_log_files(log_dir: str, env_id: str):
    run_name = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(log_dir, f"{env_id}_{run_name}")
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
            "source",
            "active",
            "device_t",
            "spacemouse_x",
            "spacemouse_y",
            "spacemouse_z",
            "spacemouse_roll",
            "spacemouse_pitch",
            "spacemouse_yaw",
            "buttons",
            "action_x",
            "action_y",
            "action_z",
            "action_roll",
            "action_pitch",
            "action_yaw",
            "action_gripper",
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
    states_by_uid,
    sources_by_uid,
    command_writer,
    state_writer,
):
    for uid, sub_agent in zip(uids, sub_agents):
        source = sources_by_uid[uid]
        state = states_by_uid[uid]
        command_writer.writerow(
            [step_idx, elapsed, uid, source, int(source != "none")]
            + state_values(state)
            + action[uid].tolist()
        )
        state_writer.writerow(
            [step_idx, elapsed, uid]
            + pose_row(sub_agent.tcp_pose)
            + tensor_row(sub_agent.robot.get_qpos())
        )


def main():
    args = parse_args()
    if args.print_devices:
        print(pyspacemouse.get_connected_devices())
        return

    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode="pd_ee_delta_pose",
        render_mode="human",
    )
    uids = tuple(env.action_space.keys())
    assert len(uids) == 2, uids
    for uid in uids:
        assert env.action_space[uid].shape == (7,), (uid, env.action_space[uid])

    devices = []
    left_device = open_spacemouse(args.left_device_path, args.left_device_index)
    devices.append(left_device)
    right_device = None
    if args.right_device_path:
        right_device = open_spacemouse(args.right_device_path, args.right_device_index)
        devices.append(right_device)

    active_arm = 0
    env.reset(seed=0)
    viewer = env.render_human()
    print("SpaceMouse teleop started.")
    print("TAB: switch active arm when using one SpaceMouse | r: reset | escape: quit")
    print("button 0: close gripper | last button: open gripper")
    print(f"left device: {left_device.describe_connection()}")
    if right_device is not None:
        print(f"right device: {right_device.describe_connection()}")
    else:
        print(f"one-device mode: controlling {uids[active_arm]}")

    run_dir, command_file, state_file, command_writer, state_writer = open_log_files(
        args.log_dir, args.env_id
    )
    print(f"logging to: {run_dir}")

    step_idx = 0
    start_time = time.time()
    dt = 1.0 / args.control_hz
    try:
        while True:
            viewer = env.render_human()
            window = viewer.window
            if hasattr(viewer, "closed") and viewer.closed:
                break
            if window.key_press("escape"):
                break
            if window.key_press("r"):
                env.reset()
            if right_device is None and window.key_press("tab"):
                active_arm = 1 - active_arm
                print(f"active arm: {uids[active_arm]}")

            action = zero_action(env)
            states_by_uid = {uid: None for uid in uids}
            sources_by_uid = {uid: "none" for uid in uids}
            left_state = read_latest(left_device)
            if right_device is None:
                action[uids[active_arm]] = spacemouse_to_action(
                    left_state, args.deadzone, args.gripper_action
                )
                states_by_uid[uids[active_arm]] = left_state
                sources_by_uid[uids[active_arm]] = "left"
            else:
                right_state = read_latest(right_device)
                action[uids[0]] = spacemouse_to_action(
                    left_state, args.deadzone, args.gripper_action
                )
                action[uids[1]] = spacemouse_to_action(
                    right_state, args.deadzone, args.gripper_action
                )
                states_by_uid[uids[0]] = left_state
                states_by_uid[uids[1]] = right_state
                sources_by_uid[uids[0]] = "left"
                sources_by_uid[uids[1]] = "right"

            env.step(action)
            elapsed = time.time() - start_time
            log_step(
                step_idx,
                elapsed,
                uids,
                env.unwrapped.agent.agents,
                action,
                states_by_uid,
                sources_by_uid,
                command_writer,
                state_writer,
            )
            command_file.flush()
            state_file.flush()
            step_idx += 1
            time.sleep(dt)
    finally:
        command_file.close()
        state_file.close()
        for device in devices:
            device.close()
        env.close()


if __name__ == "__main__":
    main()
