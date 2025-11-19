from __future__ import annotations

import argparse
import csv
import os
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def _parse_obs_csv(path: str, max_rows: Optional[int]) -> np.ndarray:
    """Parse obs_# columns from rollout_debug_log_expanded.csv style file."""
    obs_rows: List[List[float]] = []
    with open(os.path.abspath(path), "r") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV is missing headers")

        obs_columns = [name for name in reader.fieldnames if name and name.startswith("obs_")]
        obs_columns = sorted(
            obs_columns,
            key=lambda name: int(name.rsplit("_", maxsplit=1)[1]) if "_" in name else 0,
        )
        if not obs_columns:
            raise KeyError("No obs_# columns found")

        for row in reader:
            obs_rows.append([float(row[col]) for col in obs_columns])
            if max_rows is not None and len(obs_rows) >= max_rows:
                break
    return np.asarray(obs_rows, dtype=float)


def _parse_env_obs_csv(path: str, max_rows: Optional[int]) -> tuple[np.ndarray, Sequence[str]]:
    """Parse env_000.csv style file; first column is step, rest are obs values."""
    obs_rows: List[List[float]] = []
    column_names: Sequence[str] = []
    with open(os.path.abspath(path), "r") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if headers is None:
            raise ValueError("Empty CSV")
        if len(headers) < 2:
            raise ValueError("Expected at least step plus one observation column")
        column_names = headers[1:]
        for row in reader:
            if len(row) < len(headers):
                continue
            obs_rows.append([float(value) for value in row[1:]])
            if max_rows is not None and len(obs_rows) >= max_rows:
                break
    return np.asarray(obs_rows, dtype=float), column_names


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare observations between rollout_debug_log_expanded.csv and env obs CSV."
    )
    parser.add_argument(
        "--csv",
        default="rollout_debug_log_expanded.csv",
        help="Path to rollout_debug_log_expanded.csv (with obs_# columns).",
    )
    parser.add_argument(
        "--env-csv",
        default="runs/MyDualBoxRotationAblated-v0__ppo_dual_xarm7__1__1763351759/info/rollout/obs_flat/env_000.csv",
        help="Path to env observation CSV to compare.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit number of rows to load (min rows across both files are used).",
    )
    parser.add_argument(
        "--out-dir",
        default="graphs",
        help="Directory to save comparison plots.",
    )
    args = parser.parse_args()

    obs_arr = _parse_obs_csv(args.csv, args.max_rows)
    env_obs_arr, env_headers = _parse_env_obs_csv(args.env_csv, args.max_rows)

    if obs_arr.shape[1] != env_obs_arr.shape[1]:
        raise ValueError(
            f"Dimension mismatch: obs columns {obs_arr.shape[1]} vs env columns {env_obs_arr.shape[1]}"
        )

    num_steps = min(obs_arr.shape[0], env_obs_arr.shape[0])
    obs_arr = obs_arr[:num_steps]
    env_obs_arr = env_obs_arr[:num_steps]

    os.makedirs(args.out_dir, exist_ok=True)

    for idx in range(obs_arr.shape[1]):
        plt.figure(figsize=(10, 4))
        plt.plot(obs_arr[:, idx], label=f"csv_obs_{idx}", linewidth=1.2)
        plt.plot(env_obs_arr[:, idx], label=f"env_obs_{idx}", linewidth=1.0)
        title = f"obs_{idx}"
        if idx < len(env_headers):
            title = f"obs_{idx} ({env_headers[idx]})"
        plt.title(title)
        plt.xlabel("step")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(args.out_dir, f"obs_{idx}.png")
        plt.savefig(out_path)
        plt.close()
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
