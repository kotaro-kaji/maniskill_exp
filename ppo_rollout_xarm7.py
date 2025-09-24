from __future__ import annotations

import os
import random
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import tyro
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from ppo_xarm7 import Agent


CSV_LOG_FILENAME = "rollout_log.csv"
INFO_LOG_FILENAME = "rollout_info.jsonl"


def _to_serializable(obj: Any):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(value) for value in obj]
    return repr(obj)


@dataclass
class RolloutArgs:
    checkpoint: str
    """Path to the trained PPO checkpoint."""
    env_id: str = "MyPushCube-v1"
    """Environment id registered with ManiSkill."""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """Control mode forwarded to the environment."""
    sim_backend: str = "physx_cuda"
    """Simulation backend (e.g. physx_cuda or cpu)."""
    obs_mode: str = "state"
    """Observation mode passed to the environment."""
    render_mode: Optional[str] = "rgb_array"
    """Render mode for the environment."""
    seed: int = 1
    """Random seed."""
    torch_deterministic: bool = True
    """If true, sets torch.backends.cudnn.deterministic."""
    cuda: bool = True
    """Run policy on CUDA when available."""
    deterministic_policy: bool = True
    """If true, use the mean action during rollout."""
    num_eval_envs: int = 1
    """Number of parallel evaluation environments."""
    num_eval_steps: int = 200
    """Number of environment steps to roll out."""
    reconfiguration_freq: Optional[int] = 1
    """How often the environment reconfigures objects."""
    partial_reset: bool = False
    """Whether to ignore terminations during rollout."""
    capture_video: bool = True
    """Capture rollout videos via ManiSkill's RecordEpisode wrapper."""
    record_dir: Optional[str] = None
    """Directory to store rollout videos (defaults next to checkpoint)."""
    print_actions: bool = False
    """Print actions each step."""


def _build_env(args: RolloutArgs):
    env_kwargs = dict(obs_mode=args.obs_mode, render_mode=args.render_mode, sim_backend=args.sim_backend)
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode

    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )
    if isinstance(eval_envs.action_space, gym.spaces.Dict):
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    record_dir = args.record_dir
    if args.capture_video:
        if record_dir is None:
            ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
            record_dir = os.path.join(ckpt_dir, "rollout_videos")
        os.makedirs(record_dir, exist_ok=True)
        print(f"Saving rollout videos to {record_dir}")
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=record_dir,
            save_trajectory=False,
            trajectory_name="rollout",
            max_steps_per_video=args.num_eval_steps,
            video_fps=30,
        )
    return ManiSkillVectorEnv(
        eval_envs,
        args.num_eval_envs,
        ignore_terminations=not args.partial_reset,
        record_metrics=True,
    )


def run_rollout(args: RolloutArgs) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    eval_envs = _build_env(args)

    csv_path = os.path.abspath(CSV_LOG_FILENAME)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "step",
            "env_index",
            "observation",
            "action",
            "clipped_action",
            "physical_action",
        ],
    )
    csv_writer.writeheader()

    info_path = os.path.abspath(INFO_LOG_FILENAME)
    info_file = open(info_path, "w")

    agent = Agent(eval_envs).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()

    obs, _ = eval_envs.reset(seed=args.seed)
    obs = obs.to(device)

    normalized_low = torch.from_numpy(eval_envs.single_action_space.low).to(device)
    normalized_high = torch.from_numpy(eval_envs.single_action_space.high).to(device)

    controller = eval_envs.base_env.agent.controller
    has_physical_bounds = hasattr(controller, "action_space_low")
    if has_physical_bounds:
        physical_low = controller.action_space_low.to(device)
        physical_high = controller.action_space_high.to(device)
    else:
        physical_low = normalized_low
        physical_high = normalized_high

    metrics = defaultdict(list)
    for step in range(args.num_eval_steps):
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=args.deterministic_policy)
        if args.print_actions:
            print(f"step={step} action={action.detach().cpu().numpy()}")
        clipped_action = torch.clamp(action, normalized_low, normalized_high)
        if has_physical_bounds:
            physical_action = gym_utils.clip_and_scale_action(
                clipped_action, physical_low, physical_high
            )
        else:
            physical_action = clipped_action
        obs_cpu = obs.detach().cpu()
        action_cpu = action.detach().cpu()
        clipped_cpu = clipped_action.detach().cpu()
        physical_cpu = physical_action.detach().cpu()
        for env_index in range(args.num_eval_envs):
            csv_writer.writerow(
                {
                    "step": step,
                    "env_index": env_index,
                    "observation": json.dumps(obs_cpu[env_index].tolist()),
                    "action": json.dumps(action_cpu[env_index].tolist()),
                    "clipped_action": json.dumps(clipped_cpu[env_index].tolist()),
                    "physical_action": json.dumps(
                        physical_cpu[env_index].tolist()
                    ),
                }
            )
        obs, reward, terminations, truncations, info = eval_envs.step(clipped_action)
        obs = obs.to(device)

        payload = {"step": step, "info": _to_serializable(info)}
        info_file.write(json.dumps(payload) + "\n")

        if "final_info" in info:
            mask = info["_final_info"]
            for key, value in info["final_info"]["episode"].items():
                metrics[key].append(value[mask])

    eval_envs.close()

    csv_file.close()
    info_file.close()

    if metrics:
        print("Rollout metrics:")
        for key, values in metrics.items():
            tensor = torch.cat([v.float().cpu() for v in values])
            mean = tensor.mean().item()
            std = tensor.std(unbiased=False).item() if tensor.numel() > 1 else 0.0
            print(f"  {key}_mean={mean:.4f}, std={std:.4f}")
    else:
        print("No completed episodes during rollout.")


if __name__ == "__main__":
    run_rollout(tyro.cli(RolloutArgs))
