from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import List, Sequence

import torch

from ppo_xarm7 import Agent


JOINT_POSITION_LOW = torch.tensor(
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

JOINT_POSITION_HIGH = torch.tensor(
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

_DELTA_PHYSICAL_LOW = torch.tensor([-0.1] * 8, dtype=torch.float32)
_DELTA_PHYSICAL_HIGH = torch.tensor([0.1] * 8, dtype=torch.float32)


class _DummySpace:
    def __init__(self, shape: Sequence[int]):
        self.shape = tuple(shape)


class _DummyEnv:
    def __init__(self, obs_shape: Sequence[int], action_shape: Sequence[int]):
        self.single_observation_space = _DummySpace(obs_shape)
        self.single_action_space = _DummySpace(action_shape)


@dataclass
class RolloutRow:
    step_idx: int
    observation: List[float]
    reference_action: List[float] | None


def gripper_q_maniskill_to_robomanip(q_maniskill: torch.Tensor) -> torch.Tensor:
    return q_maniskill * (-1000.0) + 840.0


def _infer_indexed_columns(fieldnames: Sequence[str], prefix: str) -> List[str]:
    columns = []
    for name in fieldnames:
        if name.startswith(prefix):
            try:
                index = int(name[len(prefix) :])
            except ValueError:
                continue
            columns.append((index, name))
    columns.sort(key=lambda item: item[0])
    return [name for _, name in columns]


def load_rollout_rows(csv_path: str) -> List[RolloutRow]:
    with open(os.path.abspath(csv_path), "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or missing a header row")

        obs_columns = _infer_indexed_columns(reader.fieldnames, "obs_")
        action_columns = _infer_indexed_columns(reader.fieldnames, "direct_joint_command_")

        if not obs_columns:
            raise ValueError("No observation columns found (expected obs_0, obs_1, ...)" )

        rows: List[RolloutRow] = []
        for row in reader:
            step_idx_str = row.get("step_idx")
            step_idx = int(step_idx_str) if step_idx_str is not None else len(rows)
            observation = [float(row[col]) for col in obs_columns]

            reference_action = None
            if action_columns:
                reference_action = [float(row[col]) for col in action_columns]

            rows.append(RolloutRow(step_idx=step_idx, observation=observation, reference_action=reference_action))

    return rows


def evaluate_rollout(
    checkpoint_path: str,
    rows: Sequence[RolloutRow],
    device: torch.device,
    deterministic: bool,
    convert_gripper: bool,
) -> List[torch.Tensor]:
    if not rows:
        raise ValueError("No rows provided for evaluation")

    obs_dim = len(rows[0].observation)

    state_dict = torch.load(checkpoint_path, map_location=device)
    action_dim = int(state_dict["actor_logstd"].shape[-1])

    dummy_env = _DummyEnv((obs_dim,), (action_dim,))
    agent = Agent(dummy_env).to(device)
    agent.load_state_dict(state_dict)
    agent.eval()

    normalized_low = torch.full((action_dim,), -1.0, device=device)
    normalized_high = torch.full((action_dim,), 1.0, device=device)

    if action_dim != len(_DELTA_PHYSICAL_LOW):
        raise ValueError(
            f"Action dimension {action_dim} does not match hard-coded delta bounds {len(_DELTA_PHYSICAL_LOW)}"
        )

    physical_low = _DELTA_PHYSICAL_LOW.to(device)
    physical_high = _DELTA_PHYSICAL_HIGH.to(device)
    joint_position_low = JOINT_POSITION_LOW.to(device=device)
    joint_position_high = JOINT_POSITION_HIGH.to(device=device)

    outputs: List[torch.Tensor] = []

    for rollout_row in rows:
        obs_tensor = torch.tensor(rollout_row.observation, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = agent.get_action(obs_tensor, deterministic=deterministic)
        clipped_action = torch.clamp(action, normalized_low, normalized_high)

        scale = (clipped_action - normalized_low) / (normalized_high - normalized_low)
        denormalized_delta = physical_low + scale * (physical_high - physical_low)
        current_joint_pos = obs_tensor[..., : action_dim]
        direct_joint_command = current_joint_pos + denormalized_delta

        direct_joint_command = torch.max(torch.min(direct_joint_command, joint_position_high), joint_position_low)

        if convert_gripper and direct_joint_command.shape[-1] > 0:
            direct_joint_command = direct_joint_command.clone()
            direct_joint_command[..., -1] = gripper_q_maniskill_to_robomanip(direct_joint_command[..., -1])

        outputs.append(direct_joint_command.squeeze(0).cpu())

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay PPO policy on recorded observations")
    parser.add_argument("--csv", default="rollout_debug_log_expanded.csv", help="CSV file produced from robot logs")
    parser.add_argument(
        "--checkpoint",
        default="runs/MyJointHold-v0__ppo_xarm7__1__1758689027/ckpt_76.pt",
        help="Policy checkpoint to load",
    )
    parser.add_argument("--cuda", action="store_true", help="Force CUDA usage when available")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional limit on number of rows to process")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy sampling instead of deterministic")
    parser.add_argument(
        "--no-gripper-convert",
        action="store_true",
        help="Skip converting gripper command to RoboManip units",
    )
    parser.add_argument(
        "--report-diff",
        action="store_true",
        help="Compare replayed commands to direct_joint_command columns and report stats",
    )
    args = parser.parse_args()

    if args.cpu and args.cuda:
        raise SystemExit("--cpu and --cuda are mutually exclusive")

    if args.cpu:
        device = torch.device("cpu")
    elif args.cuda or (not args.cpu and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rows = load_rollout_rows(args.csv)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    outputs = evaluate_rollout(
        checkpoint_path=os.path.abspath(args.checkpoint),
        rows=rows,
        device=device,
        deterministic=not args.stochastic,
        convert_gripper=not args.no_gripper_convert,
    )

    for rollout_row, command in zip(rows, outputs):
        command_list = command.tolist()
        print(f"step={rollout_row.step_idx} direct_joint_command={command_list}")

    if args.report_diff:
        diffs = []
        for rollout_row, command in zip(rows, outputs):
            if rollout_row.reference_action is None:
                continue
            reference = torch.tensor(rollout_row.reference_action, dtype=command.dtype)
            if reference.shape != command.shape:
                raise ValueError(
                    f"Reference action length {reference.shape[0]} does not match replay output {command.shape[0]}"
                )
            diffs.append(torch.abs(reference - command))

        if diffs:
            stacked = torch.stack(diffs)
            max_abs = stacked.max(dim=0).values
            mean_abs = stacked.mean(dim=0)
            overall_max = stacked.max().item()
            overall_mean = stacked.mean().item()
            print("--- Difference summary (replay vs reference) ---")
            print(f"overall_max_abs={overall_max:.6e} overall_mean_abs={overall_mean:.6e}")
            print(f"per_joint_max_abs={[round(v, 6) for v in max_abs.tolist()]}")
            print(f"per_joint_mean_abs={[round(v, 6) for v in mean_abs.tolist()]}")
        else:
            print("CSV did not contain direct_joint_command_* columns; nothing to compare")


if __name__ == "__main__":
    main()
