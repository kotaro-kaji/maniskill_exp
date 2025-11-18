#!/usr/bin/env python3
"""
Plot commanded actions vs measured joint positions for simulation CSV logs
and real robot trajectories (HDF5), one figure per joint with four traces:
sim measured, sim command, real measured, real command.

Usage:
    python plot_mesured_q.py --csv-root runs/.../info/rollout --h5 path/to/main.rmb.hdf5 --out plots

Notes:
- CSV root must contain subfolders "actions" and "mesured_q" with env_XXX.csv.
- HDF5 must provide datasets "time", "measured_joint_pos", "command_joint_pos".
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import pandas as pd


def load_env_csvs(root: Path, subdir: str) -> Dict[str, pd.DataFrame]:
    """Load env_*.csv under root/subdir indexed by step."""
    folder = root / subdir
    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {folder}")
    data: Dict[str, pd.DataFrame] = {}
    for csv_path in sorted(folder.glob("env_*.csv")):
        df = pd.read_csv(csv_path)
        if "step" not in df.columns:
            raise ValueError(f"Missing 'step' column in {csv_path}")
        data[csv_path.stem] = df.set_index("step")
    if not data:
        raise FileNotFoundError(f"No env_*.csv files found in {folder}")
    return data


def load_real_h5(h5_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load real measured/command joint positions and time from HDF5."""
    with h5py.File(h5_path, "r") as f:
        time = pd.Series(f["time"][:], name="time")
        measured = pd.DataFrame(f["measured_joint_pos"][:])
        command = pd.DataFrame(f["command_joint_pos"][:])
    return measured, command, time


def plot_joint(
    joint_idx: int,
    sim_steps,
    sim_measured,
    sim_command,
    real_time,
    real_measured,
    real_command,
    out_dir: Path,
    env_name: str,
) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(sim_steps, sim_measured, label="sim measured q", linewidth=1.5)
    plt.plot(sim_steps, sim_command, label="sim command q", linestyle="--", linewidth=1.5)
    plt.plot(real_time, real_measured, label="real measured q", linewidth=1.5)
    plt.plot(real_time, real_command, label="real command q", linestyle="--", linewidth=1.5)
    plt.title(f"{env_name} - Joint {joint_idx}")
    plt.xlabel("step (sim) / time (real)")
    plt.ylabel("joint position")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{env_name}_joint_{joint_idx}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sim vs real joint trajectories.")
    parser.add_argument(
        "--csv-root",
        type=Path,
        required=True,
        help="Path to info/rollout containing actions/ and mesured_q/ folders.",
    )
    parser.add_argument(
        "--h5",
        type=Path,
        required=True,
        help="Path to HDF5 file with real data (time, measured_joint_pos, command_joint_pos).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plots"),
        help="Output directory for PNG figures.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Optional env name (e.g., env_000). If omitted, first common env is used.",
    )
    args = parser.parse_args()

    sim_measured = load_env_csvs(args.csv_root, "mesured_q")
    sim_actions = load_env_csvs(args.csv_root, "actions")
    common_envs = sorted(set(sim_measured) & set(sim_actions))
    if not common_envs:
        raise RuntimeError("No common env_*.csv between mesured_q and actions.")
    env_name = args.env or common_envs[0]
    if env_name not in common_envs:
        raise ValueError(f"Env {env_name} not found; available: {common_envs}")

    real_measured, real_command, real_time = load_real_h5(args.h5)

    sim_measured_df = sim_measured[env_name]
    sim_actions_df = sim_actions[env_name]
    sim_steps = sim_measured_df.index.to_numpy()

    joints: List[int] = sorted(
        int(c.split("_")[-1]) for c in sim_measured_df.columns if c.startswith("mesured_q_")
    )
    for j in joints:
        m_col = f"mesured_q_{j}"
        a_col = f"actions_{j}"
        if a_col not in sim_actions_df.columns:
            print(f"Skip joint {j}: {a_col} not in actions")
            continue
        # real data may have different joint count; only plot if exists
        if j >= real_measured.shape[1] or j >= real_command.shape[1]:
            print(f"Skip joint {j}: not present in real data")
            continue
        plot_joint(
            j,
            sim_steps,
            sim_measured_df[m_col].to_numpy(),
            sim_actions_df[a_col].to_numpy(),
            real_time.to_numpy(),
            real_measured.iloc[:, j].to_numpy(),
            real_command.iloc[:, j].to_numpy(),
            args.out,
            env_name,
        )


if __name__ == "__main__":
    main()
