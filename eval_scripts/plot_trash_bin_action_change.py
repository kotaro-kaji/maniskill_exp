from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LEFT_ARM_INDICES = tuple(range(7))
RIGHT_ARM_INDICES = tuple(range(8, 15))
ARM_INDICES = LEFT_ARM_INDICES + RIGHT_ARM_INDICES


def read_actions(csv_path: Path, env_index: int) -> np.ndarray:
    with csv_path.open() as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if int(row["env_index"]) == env_index
        ]
    assert rows, (csv_path, env_index)
    actions = np.array(
        [
            [float(row[f"clipped_action_{index:03d}"]) for index in range(16)]
            for row in rows
        ],
        dtype=np.float32,
    )
    return actions[:, ARM_INDICES]


def action_change_penalty(
    actions: np.ndarray,
    no_penalty_threshold: float,
    reference_delta: float,
    reference_penalty: float,
) -> np.ndarray:
    absolute_change = np.abs(np.diff(actions, axis=0))
    max_change = absolute_change.max(axis=1)
    penalty = np.maximum(max_change - no_penalty_threshold, 0.0)
    penalty *= reference_penalty / (reference_delta - no_penalty_threshold)
    return penalty


def summarize(
    actions: np.ndarray,
    no_penalty_threshold: float,
    reference_delta: float,
    reference_penalty: float,
) -> dict[str, float]:
    absolute_change = np.abs(np.diff(actions, axis=0))
    step_max = absolute_change.max(axis=1)
    penalty = action_change_penalty(
        actions,
        no_penalty_threshold,
        reference_delta,
        reference_penalty,
    )
    return {
        "absolute_change_mean": float(absolute_change.mean()),
        "absolute_change_p95": float(np.percentile(absolute_change, 95)),
        "absolute_change_p99": float(np.percentile(absolute_change, 99)),
        "absolute_change_max": float(absolute_change.max()),
        "step_max_mean": float(step_max.mean()),
        "step_max_p95": float(np.percentile(step_max, 95)),
        "step_max_max": float(step_max.max()),
        "fraction_over_0.25": float(np.mean(absolute_change > 0.25)),
        "fraction_over_0.5": float(np.mean(absolute_change > 0.5)),
        "fraction_over_1.0": float(np.mean(absolute_change > 1.0)),
        "penalty_mean": float(penalty.mean()),
        "penalty_p95": float(np.percentile(penalty, 95)),
        "penalty_max": float(penalty.max()),
    }


def plot_action_traces(
    gpu_actions: np.ndarray,
    cpu_actions: np.ndarray,
    policy_label: str,
    no_penalty_threshold: float,
    reference_delta: float,
    output_path: Path,
):
    datasets = (("PhysX CUDA", gpu_actions), ("PhysX CPU", cpu_actions))
    fig, axes = plt.subplots(4, 2, figsize=(18, 12), sharex="col")
    colors = plt.get_cmap("tab10").colors

    for column, (backend_label, actions) in enumerate(datasets):
        steps = np.arange(actions.shape[0])
        delta_steps = steps[1:]
        signed_change = np.diff(actions, axis=0)

        for joint_index in range(7):
            color = colors[joint_index]
            axes[0, column].plot(
                steps,
                actions[:, joint_index],
                color=color,
                label=f"joint {joint_index + 1}",
            )
            axes[1, column].plot(
                delta_steps,
                signed_change[:, joint_index],
                color=color,
            )
            axes[2, column].plot(
                steps,
                actions[:, 7 + joint_index],
                color=color,
                label=f"joint {joint_index + 1}",
            )
            axes[3, column].plot(
                delta_steps,
                signed_change[:, 7 + joint_index],
                color=color,
            )

        axes[0, column].set_title(backend_label)
        axes[0, column].set_ylabel("left arm action")
        axes[1, column].set_ylabel("left Δaction")
        axes[2, column].set_ylabel("right arm action")
        axes[3, column].set_ylabel("right Δaction")
        axes[3, column].set_xlabel("step")
        axes[0, column].set_ylim(-1.05, 1.05)
        axes[2, column].set_ylim(-1.05, 1.05)

        change_limit = max(
            reference_delta * 1.1,
            float(np.abs(signed_change).max()) * 1.05,
        )
        axes[1, column].set_ylim(-change_limit, change_limit)
        axes[3, column].set_ylim(-change_limit, change_limit)
        for row in (1, 3):
            axes[row, column].axhline(
                reference_delta,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label=f"reference Δ ±{reference_delta:g}",
            )
            axes[row, column].axhline(
                -reference_delta,
                color="black",
                linestyle="--",
                linewidth=1.0,
            )
            axes[row, column].axhline(
                no_penalty_threshold,
                color="gray",
                linestyle=":",
                linewidth=1.0,
                label=f"zero penalty ±{no_penalty_threshold:g}",
            )
            axes[row, column].axhline(
                -no_penalty_threshold,
                color="gray",
                linestyle=":",
                linewidth=1.0,
            )

        for row in range(4):
            axes[row, column].grid(True, alpha=0.25)

    axes[0, 1].legend(loc="upper right", ncol=2)
    axes[1, 1].legend(loc="upper right")
    fig.suptitle(
        f"TrashBin arm action change: {policy_label} policy, seed 1",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_scores(
    datasets: dict[str, np.ndarray],
    no_penalty_threshold: float,
    reference_delta: float,
    reference_penalty: float,
    output_path: Path,
):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    groups = (
        ("deterministic", ("GPU deterministic", "CPU deterministic")),
        ("stochastic", ("GPU stochastic", "CPU stochastic")),
    )
    for axis, (policy_label, names) in zip(axes, groups):
        for name in names:
            penalty = action_change_penalty(
                datasets[name],
                no_penalty_threshold,
                reference_delta,
                reference_penalty,
            )
            axis.plot(
                np.arange(1, penalty.shape[0] + 1),
                penalty,
                label=f"{name} penalty",
            )
        axis.set_ylabel("reward penalty")
        axis.set_ylim(bottom=0.0)
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper right")
        axis.set_title(policy_label)
    axes[-1].set_xlabel("step")
    fig.suptitle(
        "Stage 3 max-joint action-change penalty "
        f"(Δ={no_penalty_threshold:g} → 0, "
        f"Δ={reference_delta:g} → {reference_penalty:g}, no upper limit)",
        fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_summary(
    datasets: dict[str, np.ndarray],
    no_penalty_threshold: float,
    reference_delta: float,
    reference_penalty: float,
    output_path: Path,
):
    rows = []
    for name, actions in datasets.items():
        row = {"condition": name}
        row.update(
            summarize(
                actions,
                no_penalty_threshold,
                reference_delta,
                reference_penalty,
            )
        )
        rows.append(row)

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-deterministic", type=Path, required=True)
    parser.add_argument("--cpu-deterministic", type=Path, required=True)
    parser.add_argument("--gpu-stochastic", type=Path, required=True)
    parser.add_argument("--cpu-stochastic", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument("--no-penalty-threshold", type=float, default=0.01)
    parser.add_argument("--reference-delta", type=float, default=0.2)
    parser.add_argument("--reference-penalty", type=float, default=0.25)
    args = parser.parse_args()

    assert 0.0 <= args.no_penalty_threshold < args.reference_delta
    assert args.reference_delta <= 2.0, args.reference_delta
    assert args.reference_penalty > 0.0, args.reference_penalty
    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "GPU deterministic": read_actions(
            args.gpu_deterministic, args.env_index
        ),
        "CPU deterministic": read_actions(
            args.cpu_deterministic, args.env_index
        ),
        "GPU stochastic": read_actions(args.gpu_stochastic, args.env_index),
        "CPU stochastic": read_actions(args.cpu_stochastic, args.env_index),
    }
    lengths = {actions.shape[0] for actions in datasets.values()}
    assert len(lengths) == 1, lengths

    plot_action_traces(
        datasets["GPU deterministic"],
        datasets["CPU deterministic"],
        "deterministic",
        args.no_penalty_threshold,
        args.reference_delta,
        args.output_dir / "trash_bin_action_change_deterministic.png",
    )
    plot_action_traces(
        datasets["GPU stochastic"],
        datasets["CPU stochastic"],
        "stochastic",
        args.no_penalty_threshold,
        args.reference_delta,
        args.output_dir / "trash_bin_action_change_stochastic.png",
    )
    plot_scores(
        datasets,
        args.no_penalty_threshold,
        args.reference_delta,
        args.reference_penalty,
        args.output_dir / "trash_bin_action_change_candidate_score.png",
    )
    write_summary(
        datasets,
        args.no_penalty_threshold,
        args.reference_delta,
        args.reference_penalty,
        args.output_dir / "trash_bin_action_change_summary.csv",
    )


if __name__ == "__main__":
    main()
