from __future__ import annotations

import csv
import json
import os
import statistics
import sys
import time
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch

from ppo_dual_xarm7 import Agent


# Joint bounds per arm (7 joints + gripper drive joint).
_SINGLE_ARM_JOINT_POSITION_LOW = torch.tensor(
    [
        -6.283185307179586,
        -2.059,
        -6.283185307179586,
        -0.19198,
        -6.283185307179586,
        -1.69297,
        -6.283185307179586,
        0.05,
    ],
    dtype=torch.float32,
)
_SINGLE_ARM_JOINT_POSITION_HIGH = torch.tensor(
    [
        6.283185307179586,
        2.0944,
        6.283185307179586,
        3.927,
        6.283185307179586,
        3.141592653589793,
        6.283185307179586,
        0.84,
    ],
    dtype=torch.float32,
)

# Dual-arm bounds are just per-arm bounds concatenated.
JOINT_POSITION_LOW = torch.cat(
    [_SINGLE_ARM_JOINT_POSITION_LOW, _SINGLE_ARM_JOINT_POSITION_LOW]
)
JOINT_POSITION_HIGH = torch.cat(
    [_SINGLE_ARM_JOINT_POSITION_HIGH, _SINGLE_ARM_JOINT_POSITION_HIGH]
)


class _DummySpace:
    def __init__(self, shape: Sequence[int]):
        self.shape = tuple(shape)


class _DummyEnv:
    def __init__(self, obs_shape: Sequence[int], action_shape: Sequence[int]):
        self.single_observation_space = _DummySpace(obs_shape)
        self.single_action_space = _DummySpace(action_shape)


def gripper_q_maniskill_to_robomanip(q_maniskill: torch.Tensor) -> torch.Tensor:
    return q_maniskill * (-1000.0) + 840.0


def gripper_q_robomanip_to_maniskill(q_robomanip: torch.Tensor) -> torch.Tensor:
    return (q_robomanip - 840.0) / (-1000.0)


# ---------------------------------------------------------------------------
# Configuration (edit as needed)
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "runs/MyDualBoxRotationAblated-v0__ppo_dual_xarm7__1__1763351759/ckpt_latest.pt"
CSV_PATH = "rollout_debug_log_expanded.csv"
USE_CUDA = True
DETERMINISTIC_POLICY = True
PRINT_JSON = False
CONVERT_GRIPPER = True
MAX_ROWS: Optional[int] = None  # set to an int to process fewer rows
RAW_ACTION_PLOT = "raw_action_plot.png"
SHOW_PLOT = False  # Set True to open a window; requires a GUI environment.
GRAPH_DIR = "graph"

# Physical delta bounds (pd_joint_delta_pos controller for dual arm).
_DEFAULT_ARM_JOINT_DELTA_LIMIT = 0.06
_DEFAULT_GRIPPER_JOINT_DELTA_LIMIT = 0.1
_GRIPPER_JOINT_INDICES = (7, 15)


def _print_duration_stats(label: str, durations: Iterable[float]) -> None:
    durations = list(durations)
    if not durations:
        return

    mean_duration = statistics.fmean(durations)
    if len(durations) >= 2:
        variance = statistics.pvariance(durations)
        quartiles = statistics.quantiles(durations, n=4, method="inclusive")
    else:
        variance = 0.0
        quartiles = [durations[0]] * 3

    min_duration = min(durations)
    max_duration = max(durations)
    median_duration = statistics.median(durations)
    print(
        f"{label} "
        f"mean={mean_duration:.6f} "
        f"variance={variance:.6f} "
        f"min={min_duration:.6f} "
        f"max={max_duration:.6f} "
        f"median={median_duration:.6f} "
        f"q1={quartiles[0]:.6f} "
        f"q2={quartiles[1]:.6f} "
        f"q3={quartiles[2]:.6f}"
    )


def _parse_observations(
    csv_path: str, max_rows: Optional[int]
) -> tuple[List[List[float]], Optional[List[List[float]]]]:
    observations: List[List[float]] = []
    logged_raw_actions: List[List[float]] = []
    with open(os.path.abspath(csv_path), "r") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV is missing headers")

        obs_columns = [name for name in reader.fieldnames if name and name.startswith("obs_")]
        obs_columns = sorted(
            obs_columns,
            key=lambda name: int(name.rsplit("_", maxsplit=1)[1]) if "_" in name else 0,
        )

        raw_action_columns = [
            name for name in reader.fieldnames if name and name.startswith("raw_action_")
        ]
        raw_action_columns = sorted(
            raw_action_columns,
            key=lambda name: int(name.rsplit("_", maxsplit=1)[1]) if "_" in name else 0,
        )

        use_json_observation = False
        if "observation" in reader.fieldnames and not obs_columns:
            use_json_observation = True

        if not obs_columns and not use_json_observation:
            raise KeyError("CSV must contain either 'observation' or 'obs_#' columns")

        for row_idx, row in enumerate(reader):
            if use_json_observation:
                obs = json.loads(row["observation"])
                if not isinstance(obs, list):
                    raise ValueError("observation column must be a JSON array")
            else:
                obs = [float(row[col]) for col in obs_columns]
            observations.append(obs)

            if raw_action_columns:
                logged_raw_actions.append([float(row[col]) for col in raw_action_columns])

            if max_rows is not None and len(observations) >= max_rows:
                break

    if not observations:
        raise ValueError("No observations parsed from CSV")
    if raw_action_columns and len(logged_raw_actions) != len(observations):
        raise ValueError("raw_action rows count does not match observation rows")
    return observations, (logged_raw_actions if raw_action_columns else None)


def _maybe_plot_raw_actions(raw_actions: List[List[float]], output_path: str, show: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - best-effort plotting
        print(f"Skipping raw action plot (matplotlib not available): {exc}")
        return

    raw_actions_arr = np.asarray(raw_actions, dtype=float)
    plt.figure(figsize=(12, 6))
    for dim in range(raw_actions_arr.shape[1]):
        plt.plot(raw_actions_arr[:, dim], label=f"action_{dim}")
    plt.xlabel("step")
    plt.ylabel("raw_action")
    plt.title("Raw action outputs (policy network)")
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved raw action plot to {output_path}")
    if show:
        plt.show()
    plt.close()


def main() -> None:
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else CHECKPOINT_PATH
    csv_path = sys.argv[2] if len(sys.argv) > 2 else CSV_PATH

    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")

    observations, logged_raw_actions = _parse_observations(csv_path, MAX_ROWS)
    obs_dim = len(observations[0])

    state_dict = torch.load(checkpoint_path, map_location=device)
    if "actor_logstd" not in state_dict:
        raise KeyError("Checkpoint missing 'actor_logstd' parameter")
    action_dim = int(state_dict["actor_logstd"].shape[-1])

    dummy_env = _DummyEnv((obs_dim,), (action_dim,))
    agent = Agent(dummy_env).to(device)
    agent.load_state_dict(state_dict)
    agent.eval()

    normalized_low = torch.full((action_dim,), -1.0, device=device)
    normalized_high = torch.full((action_dim,), 1.0, device=device)

    delta_limit = torch.full(
        (action_dim,), _DEFAULT_ARM_JOINT_DELTA_LIMIT, dtype=torch.float32, device=device
    )
    for idx in _GRIPPER_JOINT_INDICES:
        if 0 <= idx < action_dim:
            delta_limit[idx] = _DEFAULT_GRIPPER_JOINT_DELTA_LIMIT
    physical_low = -delta_limit
    physical_high = delta_limit

    if action_dim != len(JOINT_POSITION_LOW):
        raise ValueError(
            f"Checkpoint action dim {action_dim} does not match hard-coded bounds {len(JOINT_POSITION_LOW)}"
        )
    joint_position_low = JOINT_POSITION_LOW.to(device=device)
    joint_position_high = JOINT_POSITION_HIGH.to(device=device)

    inference_durations: List[float] = []
    pipeline_durations: List[float] = []
    inferred_raw_actions: List[List[float]] = []

    for step_idx, obs_values in enumerate(observations):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pipeline_start = time.perf_counter()

        obs_tensor = torch.tensor(obs_values, dtype=torch.float32, device=device).unsqueeze(0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        with torch.no_grad():
            action = agent.get_action(obs_tensor, deterministic=DETERMINISTIC_POLICY)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_durations.append(time.perf_counter() - start_time)

        inferred_raw_actions.append(action.squeeze(0).cpu().numpy().tolist())

        clipped_action = torch.clamp(action, normalized_low, normalized_high)

        scale = (clipped_action - normalized_low) / (normalized_high - normalized_low)
        denormalized_delta = physical_low + scale * (physical_high - physical_low)
        current_joint_pos = obs_tensor[..., : action_dim]
        direct_joint_command = current_joint_pos + denormalized_delta

        direct_joint_command = torch.max(
            torch.min(direct_joint_command, joint_position_high), joint_position_low
        )

        if CONVERT_GRIPPER and direct_joint_command.shape[-1] > 0:
            direct_joint_command = direct_joint_command.clone()
            for idx in _GRIPPER_JOINT_INDICES:
                gripper_q = direct_joint_command[..., idx]
                direct_joint_command[..., idx] = gripper_q_maniskill_to_robomanip(gripper_q)

        direct_joint_np = direct_joint_command.squeeze(0).cpu().numpy().tolist()

        if PRINT_JSON:
            payload = {
                "step": step_idx,
                "raw_action": inferred_raw_actions[-1],
                "direct_joint_command": direct_joint_np,
            }
            print(json.dumps(payload))
        else:
            print(
                f"step={step_idx} raw_action={inferred_raw_actions[-1]} direct_joint_command={direct_joint_np}"
            )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        pipeline_durations.append(time.perf_counter() - pipeline_start)

    _print_duration_stats("inference_timing_seconds", inference_durations)
    _print_duration_stats("pipeline_timing_seconds", pipeline_durations)

    if inferred_raw_actions:
        _maybe_plot_raw_actions(inferred_raw_actions, RAW_ACTION_PLOT, SHOW_PLOT)

    if logged_raw_actions is not None:
        logged_arr = np.asarray(logged_raw_actions, dtype=float)
        inferred_arr = np.asarray(inferred_raw_actions, dtype=float)
        if logged_arr.shape != inferred_arr.shape:
            print(
                f"Mismatch between logged raw_action shape {logged_arr.shape} and inferred {inferred_arr.shape}; "
                "comparison plots will be skipped."
            )
        else:
            os.makedirs(GRAPH_DIR, exist_ok=True)
            for idx in range(logged_arr.shape[1]):
                output_path = os.path.join(GRAPH_DIR, f"raw_action_{idx}.png")
                try:
                    import matplotlib.pyplot as plt
                except Exception as exc:  # pragma: no cover
                    print(f"Skipping comparison plots (matplotlib not available): {exc}")
                    break
                plt.figure(figsize=(10, 4))
                plt.plot(logged_arr[:, idx], label="csv_raw_action", linewidth=1.2)
                plt.plot(inferred_arr[:, idx], label="inferred_raw_action", linewidth=1.0)
                plt.xlabel("step")
                plt.ylabel(f"raw_action_{idx}")
                plt.title(f"raw_action_{idx} (csv vs inferred)")
                plt.legend()
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                print(f"Saved comparison plot to {output_path}")


if __name__ == "__main__":
    main()
