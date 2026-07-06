from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_kinematics as pk
import torch
from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles


PLOTS = [
    ("x", "tcp x [m]", "current_tcp_x", "commanded_tcp_x", "delta_tcp_x"),
    ("y", "tcp y [m]", "current_tcp_y", "commanded_tcp_y", "delta_tcp_y"),
    ("z", "tcp z [m]", "current_tcp_z", "commanded_tcp_z", "delta_tcp_z"),
    (
        "roll",
        "tcp roll [rad]",
        "current_tcp_roll",
        "commanded_tcp_roll",
        "delta_tcp_roll",
    ),
    (
        "pitch",
        "tcp pitch [rad]",
        "current_tcp_pitch",
        "commanded_tcp_pitch",
        "delta_tcp_pitch",
    ),
    ("yaw", "tcp yaw [rad]", "current_tcp_yaw", "commanded_tcp_yaw", "delta_tcp_yaw"),
]


ARM_DOF = 7
DEFAULT_URDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "robotagents/assets/xarm7/xarm7_1305_left.urdf"
)
DEFAULT_EE_LINK = "link_tcp"
LEFT_ARM_Y_OFFSET = 0.3291
DEFAULT_ROOT_MINUS_TRACE_ORIGIN = torch.tensor([0.0, LEFT_ARM_Y_OFFSET, 0.0])


def read_csv_rows(csv_path: Path) -> list[dict[str, float]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None, csv_path
        rows = [
            {key: float(value) for key, value in row.items() if value != ""}
            for row in reader
        ]
    assert rows, f"empty CSV: {csv_path}"
    return rows


def has_tcp_trace_columns(row: dict[str, float]) -> bool:
    return "step" in row and all(key in row for _, _, a, b, c in PLOTS for key in (a, b, c))


def has_real_state_action_columns(row: dict[str, float]) -> bool:
    required = ["rollout_step"]
    required += [f"state_agent_qpos_{i}" for i in range(ARM_DOF)]
    required += [f"action_command_joint_pos_{i}" for i in range(ARM_DOF)]
    return all(key in row for key in required)


def fk_rows_from_real_state_action(
    rows: list[dict[str, float]],
    urdf_path: Path,
    ee_link: str,
) -> list[dict[str, float]]:
    with urdf_path.open("rb") as f:
        chain = pk.build_serial_chain_from_urdf(f.read(), ee_link)
    joint_names = chain.get_joint_parameter_names()
    assert joint_names == [f"joint{i}" for i in range(1, ARM_DOF + 1)], joint_names

    output_rows = []
    for row in rows:
        current_qpos = torch.tensor(
            [[row[f"state_agent_qpos_{i}"] for i in range(ARM_DOF)]],
            dtype=torch.float32,
        )
        commanded_qpos = torch.tensor(
            [[row[f"action_command_joint_pos_{i}"] for i in range(ARM_DOF)]],
            dtype=torch.float32,
        )

        current_matrix = chain.forward_kinematics(current_qpos).get_matrix()
        commanded_matrix = chain.forward_kinematics(commanded_qpos).get_matrix()
        current_tcp = current_matrix[:, :3, 3] + DEFAULT_ROOT_MINUS_TRACE_ORIGIN
        commanded_tcp = commanded_matrix[:, :3, 3] + DEFAULT_ROOT_MINUS_TRACE_ORIGIN
        delta_tcp = commanded_tcp - current_tcp
        current_rpy = matrix_to_euler_angles(current_matrix[:, :3, :3], "XYZ")
        commanded_rpy = matrix_to_euler_angles(commanded_matrix[:, :3, :3], "XYZ")
        delta_rpy = commanded_rpy - current_rpy

        output_rows.append(
            {
                "step": row["rollout_step"],
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
    return output_rows


def tcp_trace_rows(csv_path: Path, urdf_path: Path, ee_link: str) -> list[dict[str, float]]:
    rows = read_csv_rows(csv_path)
    if has_tcp_trace_columns(rows[0]):
        return rows
    if has_real_state_action_columns(rows[0]):
        return fk_rows_from_real_state_action(rows, urdf_path, ee_link)

    columns = ", ".join(rows[0].keys())
    raise AssertionError(f"unsupported CSV columns: {columns}")


def save_plot(
    rows: list[dict[str, float]],
    output_path: Path,
    suffix: str,
    ylabel: str,
    current_key: str,
    commanded_key: str,
    delta_key: str,
) -> None:
    missing = [
        key
        for key in (current_key, commanded_key, delta_key)
        if key not in rows[0]
    ]
    assert not missing, f"missing columns for {suffix}: {missing}"

    steps = [row["step"] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    axes[0].plot(steps, [row[current_key] for row in rows], label=current_key)
    axes[0].plot(
        steps,
        [row[commanded_key] for row in rows],
        label=commanded_key,
        linewidth=1.0,
    )
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(
        steps,
        [row[delta_key] for row in rows],
        label=delta_key,
        color="tab:red",
    )
    axes[1].set_ylabel("command - current")
    axes[1].set_xlabel("step")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot real-robot TCP trace PNGs from a rollout_tcp_trace-style CSV."
    )
    parser.add_argument("csv_path", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--prefix", default="real_rollout_tcp_trace")
    parser.add_argument("--urdf-path", type=Path, default=DEFAULT_URDF_PATH)
    parser.add_argument("--ee-link", default=DEFAULT_EE_LINK)
    args = parser.parse_args()

    assert args.csv_path.is_file(), args.csv_path
    assert args.urdf_path.is_file(), args.urdf_path
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = tcp_trace_rows(args.csv_path, args.urdf_path, args.ee_link)
    output_paths = []
    for suffix, ylabel, current_key, commanded_key, delta_key in PLOTS:
        output_path = args.output_dir / f"{args.prefix}_{suffix}.png"
        save_plot(
            rows,
            output_path,
            suffix,
            ylabel,
            current_key,
            commanded_key,
            delta_key,
        )
        output_paths.append(output_path)

    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
