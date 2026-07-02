from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


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


def read_rows(csv_path: Path) -> list[dict[str, float]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None, csv_path
        rows = [
            {key: float(value) for key, value in row.items() if value != ""}
            for row in reader
        ]
    assert rows, f"empty CSV: {csv_path}"
    assert "step" in rows[0], f"CSV must have a step column: {csv_path}"
    return rows


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
    args = parser.parse_args()

    assert args.csv_path.is_file(), args.csv_path
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.csv_path)
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
