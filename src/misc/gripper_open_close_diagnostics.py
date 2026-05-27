from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ARM_JOINTS = 7
GRIPPER_JOINT_NAMES = [
    "drive_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "left_finger_joint",
    "right_finger_joint",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="MyDualCardboardCabinet-v1")
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--steps", type=int, default=480)
    parser.add_argument("--phase-steps", type=int, default=60)
    parser.add_argument("--action-mag", type=float, default=1.0)
    parser.add_argument("--kp", type=float, default=None)
    parser.add_argument("--kd", type=float, default=None)
    parser.add_argument("--fmax", type=float, default=None)
    parser.add_argument("--friction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="tmp_note/gripper_diagnostics")
    parser.add_argument("--name", default="")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def set_gain_env(args):
    if args.kp is not None:
        os.environ["XARM_GRIP_KP"] = str(args.kp)
    if args.kd is not None:
        os.environ["XARM_GRIP_KD"] = str(args.kd)
    if args.fmax is not None:
        os.environ["XARM_GRIP_FMAX"] = str(args.fmax)
    if args.friction is not None:
        os.environ["XARM_GRIP_FRICTION"] = str(args.friction)


def make_action(env, gripper_cmd: float):
    action = {}
    for uid, space in env.action_space.items():
        arr = np.zeros(space.shape, dtype=np.float32)
        arr[-1] = gripper_cmd
        action[uid] = arr
    return action


def phase_command(step: int, phase_steps: int, action_mag: float) -> float:
    phase = (step // phase_steps) % 2
    if phase == 0:
        return action_mag
    return -action_mag


def tensor_np(tensor):
    return tensor.detach().cpu().numpy().reshape(-1)


def collect_row(step, elapsed, uid, agent, command):
    qpos = tensor_np(agent.robot.get_qpos())
    qvel = tensor_np(agent.robot.get_qvel())
    tcp_pose = tensor_np(agent.tcp_pose.raw_pose)

    arm_qpos = qpos[:ARM_JOINTS]
    arm_qvel = qvel[:ARM_JOINTS]
    gripper_qpos = qpos[ARM_JOINTS : ARM_JOINTS + len(GRIPPER_JOINT_NAMES)]
    gripper_qvel = qvel[ARM_JOINTS : ARM_JOINTS + len(GRIPPER_JOINT_NAMES)]

    return (
        [step, elapsed, uid, command]
        + tcp_pose.tolist()
        + arm_qpos.tolist()
        + arm_qvel.tolist()
        + gripper_qpos.tolist()
        + gripper_qvel.tolist()
    )


def summarize(rows, header):
    arr = np.asarray(rows, dtype=object)
    numeric = arr[:, 3:].astype(np.float64)
    uid_col = arr[:, 2]
    summary = []
    for uid in sorted(set(uid_col.tolist())):
        mask = uid_col == uid
        data = numeric[mask]
        tcp = data[:, 1:4]
        arm_qpos = data[:, 8:15]
        arm_qvel = data[:, 15:22]
        grip_qpos = data[:, 22:28]
        grip_qvel = data[:, 28:34]
        summary.append(
            {
                "uid": uid,
                "tcp_drift_max": float(np.max(np.linalg.norm(tcp - tcp[0], axis=1))),
                "arm_qpos_drift_max": float(
                    np.max(np.linalg.norm(arm_qpos - arm_qpos[0], axis=1))
                ),
                "arm_qvel_abs_max": float(np.max(np.abs(arm_qvel))),
                "gripper_qvel_abs_max": float(np.max(np.abs(grip_qvel))),
                "gripper_qpos_min": float(np.min(grip_qpos)),
                "gripper_qpos_max": float(np.max(grip_qpos)),
            }
        )
    return summary


def write_summary(path: Path, summary):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def plot_csv(csv_path: Path, plot_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    for uid in sorted(set(data["uid"].tolist())):
        mask = data["uid"] == uid
        t = data["step"][mask]
        tcp = np.vstack([data["tcp_x"][mask], data["tcp_y"][mask], data["tcp_z"][mask]]).T
        tcp_drift = np.linalg.norm(tcp - tcp[0], axis=1)
        arm_qvel = np.vstack([data[f"arm_qvel_{i}"][mask] for i in range(7)]).T
        gripper_qpos = np.vstack(
            [data[f"gripper_qpos_{name}"][mask] for name in GRIPPER_JOINT_NAMES]
        ).T
        gripper_qvel = np.vstack(
            [data[f"gripper_qvel_{name}"][mask] for name in GRIPPER_JOINT_NAMES]
        ).T

        axes[0].plot(t, tcp_drift, label=uid)
        axes[1].plot(t, np.max(np.abs(arm_qvel), axis=1), label=uid)
        axes[2].plot(t, gripper_qpos[:, 0], label=f"{uid} drive")
        axes[3].plot(t, np.max(np.abs(gripper_qvel), axis=1), label=uid)

    axes[0].set_ylabel("TCP drift [m]")
    axes[1].set_ylabel("max |arm qvel|")
    axes[2].set_ylabel("drive qpos")
    axes[3].set_ylabel("max |gripper qvel|")
    axes[3].set_xlabel("step")
    for ax in axes:
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)


def main():
    args = parse_args()
    set_gain_env(args)

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401
    import tasks.single.task_single_cardboard_cabinet  # noqa: F401

    run_name = args.name or time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode=args.control_mode,
        render_mode="rgb_array",
    )
    env.reset(seed=args.seed)

    header = (
        ["step", "time", "uid", "command"]
        + ["tcp_x", "tcp_y", "tcp_z", "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz"]
        + [f"arm_qpos_{i}" for i in range(ARM_JOINTS)]
        + [f"arm_qvel_{i}" for i in range(ARM_JOINTS)]
        + [f"gripper_qpos_{name}" for name in GRIPPER_JOINT_NAMES]
        + [f"gripper_qvel_{name}" for name in GRIPPER_JOINT_NAMES]
    )
    rows = []
    csv_path = out_dir / "timeseries.csv"
    start = time.time()
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for step in range(args.steps):
            command = phase_command(step, args.phase_steps, args.action_mag)
            action = make_action(env, command)
            env.step(action)
            elapsed = time.time() - start
            for uid, agent in zip(env.action_space.keys(), env.unwrapped.agent.agents):
                row = collect_row(step, elapsed, uid, agent, command)
                writer.writerow(row)
                rows.append(row)

    env.close()
    summary = summarize(rows, header)
    write_summary(out_dir / "summary.csv", summary)
    if args.plot:
        plot_csv(csv_path, out_dir / "plot.png")

    for item in summary:
        print(item)
    print(out_dir)


if __name__ == "__main__":
    main()
