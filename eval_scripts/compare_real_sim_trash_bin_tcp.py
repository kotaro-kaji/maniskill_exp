from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_kinematics as pk
import torch
from mani_skill.utils.geometry.rotation_conversions import (
    matrix_to_euler_angles,
    quaternion_to_matrix,
)


ARM_DOF = 7
ROBOT_BASE_X_OFFSET = -0.615
LEFT_ARM_Y_OFFSET = 0.3291
PEDESTAL_HEIGHT = 0.0880
ROOT_P = torch.tensor([ROBOT_BASE_X_OFFSET, LEFT_ARM_Y_OFFSET, PEDESTAL_HEIGHT])
ROOT_Q = torch.tensor([1.0, 0.0, 0.0, 0.0])

PLOTS = [
    ("x", "tcp x [m]", "current_tcp_x", "commanded_tcp_x", "delta_tcp_x"),
    ("y", "tcp y [m]", "current_tcp_y", "commanded_tcp_y", "delta_tcp_y"),
    ("z", "tcp z [m]", "current_tcp_z", "commanded_tcp_z", "delta_tcp_z"),
    ("roll", "tcp roll [rad]", "current_tcp_roll", "commanded_tcp_roll", "delta_tcp_roll"),
    ("pitch", "tcp pitch [rad]", "current_tcp_pitch", "commanded_tcp_pitch", "delta_tcp_pitch"),
    ("yaw", "tcp yaw [rad]", "current_tcp_yaw", "commanded_tcp_yaw", "delta_tcp_yaw"),
]

MARKER_FRAME_LABELS = [
    "pos_x",
    "pos_y",
    "pos_z",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
]


def read_float_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None, path
        rows = [
            {key: float(value) for key, value in row.items() if value != ""}
            for row in reader
        ]
    assert rows, f"empty CSV: {path}"
    return rows


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    assert rows, path
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fk_world(chain, qpos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    local_matrix = chain.forward_kinematics(qpos).get_matrix()
    root_rotation = quaternion_to_matrix(ROOT_Q.reshape(1, 4)).repeat(qpos.shape[0], 1, 1)
    local_position = local_matrix[:, :3, 3]
    local_rotation = local_matrix[:, :3, :3]
    world_position = ROOT_P.reshape(1, 3) + torch.bmm(
        root_rotation,
        local_position.unsqueeze(-1),
    ).squeeze(-1)
    world_rotation = torch.bmm(root_rotation, local_rotation)
    return world_position, matrix_to_euler_angles(world_rotation, "XYZ")


def real_rows_to_tcp_trace(
    real_rows: list[dict[str, float]],
    urdf_path: Path,
    ee_link: str,
) -> list[dict[str, float]]:
    with urdf_path.open("rb") as f:
        chain = pk.build_serial_chain_from_urdf(f.read(), ee_link)
    joint_names = chain.get_joint_parameter_names()
    assert joint_names == [f"joint{i}" for i in range(1, ARM_DOF + 1)], joint_names

    trace_rows = []
    for step, row in enumerate(real_rows):
        current_qpos = torch.tensor(
            [[row[f"state_{idx}"] for idx in range(ARM_DOF)]],
            dtype=torch.float32,
        )
        commanded_qpos = torch.tensor(
            [[row[f"action_command_joint_pos_{idx}"] for idx in range(ARM_DOF)]],
            dtype=torch.float32,
        )
        current_tcp, current_rpy = fk_world(chain, current_qpos)
        commanded_tcp, commanded_rpy = fk_world(chain, commanded_qpos)
        delta_tcp = commanded_tcp - current_tcp
        delta_rpy = commanded_rpy - current_rpy
        trace_rows.append(
            {
                "step": row.get("rollout_step", float(step)),
                "time": row.get("time", float(step) * 0.05),
                "current_tcp_x": float(current_tcp[0, 0].item()),
                "current_tcp_y": float(current_tcp[0, 1].item()),
                "current_tcp_z": float(current_tcp[0, 2].item()),
                "commanded_tcp_x": float(commanded_tcp[0, 0].item()),
                "commanded_tcp_y": float(commanded_tcp[0, 1].item()),
                "commanded_tcp_z": float(commanded_tcp[0, 2].item()),
                "delta_tcp_x": float(delta_tcp[0, 0].item()),
                "delta_tcp_y": float(delta_tcp[0, 1].item()),
                "delta_tcp_z": float(delta_tcp[0, 2].item()),
                "current_tcp_roll": float(current_rpy[0, 0].item()),
                "current_tcp_pitch": float(current_rpy[0, 1].item()),
                "current_tcp_yaw": float(current_rpy[0, 2].item()),
                "commanded_tcp_roll": float(commanded_rpy[0, 0].item()),
                "commanded_tcp_pitch": float(commanded_rpy[0, 1].item()),
                "commanded_tcp_yaw": float(commanded_rpy[0, 2].item()),
                "delta_tcp_roll": float(delta_rpy[0, 0].item()),
                "delta_tcp_pitch": float(delta_rpy[0, 1].item()),
                "delta_tcp_yaw": float(delta_rpy[0, 2].item()),
            }
        )
    return trace_rows


def comparison_summary(
    real_rows: list[dict[str, float]],
    sim_rows: list[dict[str, float]],
) -> list[dict[str, float | str]]:
    n = min(len(real_rows), len(sim_rows))
    keys = [key for _, _, a, b, c in PLOTS for key in (a, b, c)]
    summary = []
    for key in keys:
        diff = torch.tensor(
            [real_rows[i][key] - sim_rows[i][key] for i in range(n)],
            dtype=torch.float32,
        )
        summary.append(
            {
                "key": key,
                "real_start": real_rows[0][key],
                "sim_start": sim_rows[0][key],
                "mean_diff_real_minus_sim": float(diff.mean().item()),
                "mean_abs_diff": float(diff.abs().mean().item()),
                "rmse": float(torch.sqrt(torch.mean(diff * diff)).item()),
                "max_abs_diff": float(diff.abs().max().item()),
            }
        )
    return summary


def plot_overlay(
    real_rows: list[dict[str, float]],
    sim_rows: list[dict[str, float]],
    output_dir: Path,
) -> None:
    n = min(len(real_rows), len(sim_rows))
    steps = list(range(n))
    for suffix, ylabel, current_key, commanded_key, delta_key in PLOTS:
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        axes[0].plot(steps, [real_rows[i][current_key] for i in range(n)], label=f"real {current_key}")
        axes[0].plot(steps, [sim_rows[i][current_key] for i in range(n)], label=f"sim {current_key}")
        axes[0].plot(
            steps,
            [real_rows[i][commanded_key] for i in range(n)],
            label=f"real {commanded_key}",
            linewidth=1.0,
            alpha=0.75,
        )
        axes[0].plot(
            steps,
            [sim_rows[i][commanded_key] for i in range(n)],
            label=f"sim {commanded_key}",
            linewidth=1.0,
            alpha=0.75,
        )
        axes[0].set_ylabel(ylabel)
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best", fontsize=8)

        axes[1].plot(steps, [real_rows[i][delta_key] for i in range(n)], label=f"real {delta_key}")
        axes[1].plot(steps, [sim_rows[i][delta_key] for i in range(n)], label=f"sim {delta_key}")
        axes[1].set_ylabel("command - current")
        axes[1].set_xlabel("aligned step")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"real_sim_tcp_{suffix}.png", dpi=160)
        plt.close(fig)


def marker_frame_step_rows(
    real_rows: list[dict[str, float]],
    sim_rows: list[dict[str, float]],
    steps: list[int],
) -> list[dict[str, float | int | str]]:
    rows = []
    for step in steps:
        real_row = next(row for row in real_rows if int(row["rollout_step"]) == step)
        sim_row = next(row for row in sim_rows if int(row["step"]) == step)
        for idx, label in enumerate(MARKER_FRAME_LABELS):
            real_value = real_row[f"state_{32 + idx}"]
            sim_value = sim_row[f"observation_{32 + idx:03d}"]
            diff = real_value - sim_value
            rows.append(
                {
                    "step": step,
                    "dim": 32 + idx,
                    "label": label,
                    "real_state": real_value,
                    "sim_obs": sim_value,
                    "diff_real_minus_sim": diff,
                    "abs_diff": abs(diff),
                }
            )
    return rows


def marker_frame_summary_rows(
    real_rows: list[dict[str, float]],
    sim_rows: list[dict[str, float]],
) -> list[dict[str, float | int | str]]:
    n = min(120, len(real_rows), len(sim_rows))
    rows = []
    for idx, label in enumerate(MARKER_FRAME_LABELS):
        diffs = [
            real_rows[step][f"state_{32 + idx}"] - sim_rows[step][f"observation_{32 + idx:03d}"]
            for step in range(n)
        ]
        rows.append(
            {
                "dim": 32 + idx,
                "label": label,
                "mean_diff": sum(diffs) / n,
                "mean_abs": sum(abs(diff) for diff in diffs) / n,
                "rmse": math.sqrt(sum(diff * diff for diff in diffs) / n),
                "max_abs": max(abs(diff) for diff in diffs),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-csv", type=Path, required=True)
    parser.add_argument("--sim-csv", type=Path, required=True)
    parser.add_argument("--sim-rollout-log", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=Path(
            "robotagents/assets/xarm7/xarm7_1305_left_ball_ee_wo_force_sensor_kinematics.urdf"
        ),
    )
    parser.add_argument("--ee-link", default="link_tcp_ball")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    real_input_rows = read_float_rows(args.real_csv)
    real_tcp_rows = real_rows_to_tcp_trace(real_input_rows, args.urdf_path, args.ee_link)
    sim_tcp_rows = read_float_rows(args.sim_csv)
    write_rows(args.output_dir / "real_tcp_trace_left.csv", real_tcp_rows)
    write_rows(args.output_dir / "sim_tcp_trace_left.csv", sim_tcp_rows)
    write_rows(args.output_dir / "summary.csv", comparison_summary(real_tcp_rows, sim_tcp_rows))
    plot_overlay(real_tcp_rows, sim_tcp_rows, args.output_dir)

    if args.sim_rollout_log is not None:
        sim_obs_rows = read_float_rows(args.sim_rollout_log)
        write_rows(
            args.output_dir / "marker_frame_step_0_10.csv",
            marker_frame_step_rows(real_input_rows, sim_obs_rows, [0, 10]),
        )
        write_rows(
            args.output_dir / "marker_frame_summary_120.csv",
            marker_frame_summary_rows(real_input_rows, sim_obs_rows),
        )

    aligned_rows = min(len(real_tcp_rows), len(sim_tcp_rows))
    print(f"real_rows={len(real_tcp_rows)} sim_rows={len(sim_tcp_rows)} aligned_rows={aligned_rows}")
    print(args.output_dir)


if __name__ == "__main__":
    main()
