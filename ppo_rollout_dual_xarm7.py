from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, List, Optional

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import tyro
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# Ensure dual-arm envs are registered with ManiSkill.
from tasks.dual.task_dual_box_rotation import (  # noqa: F401
    MyDualBoxRotationEnv,
    MyDualBoxRotationAblatedEnv,
)
from tasks.single.task_single_cardboard_cabinet import (  # noqa: F401
    MyDualCardboardCabinetEnv,
)
from tasks.single.task_single_cardboard_cabinet_just_return import (  # noqa: F401
    MyDualCardboardCabinetEnv as MySingleCardboardCabinetJustReturnEnv,
)
from tasks.dual.task_dual_simple import MyDualSimpleEnv  # noqa: F401

from ppo_dual_xarm7 import Agent, InfoDirectoryLogger

_DEFAULT_ARM_JOINT_DELTA_LIMIT = 0.06
_DEFAULT_GRIPPER_JOINT_DELTA_LIMIT = 0.1
_NORMALIZED_ACTION_LOW = torch.tensor(-1.0, dtype=torch.float32)
_NORMALIZED_ACTION_HIGH = torch.tensor(1.0, dtype=torch.float32)


CSV_LOG_FILENAME = "rollout_dual_log.csv"
INFO_LOG_FILENAME = "rollout_dual_info.jsonl"


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
    """Path to the trained PPO checkpoint (.pt) produced by ppo_dual_xarm7."""
    env_id: str = "MyDualBoxRotation-v0"
    """Environment id registered with ManiSkill."""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """Control mode forwarded to the environment."""
    sim_backend: str = "physx_cuda"
    """Simulation backend (e.g. physx_cuda or cpu)."""
    obs_mode: str = "state"
    """Observation mode passed to the environment."""
    render_mode: Optional[str] = "rgb_array"
    """Render mode for the environment."""
    render_width: int = 1280
    """Width of rendered frames for video capture (default: high-res)."""
    render_height: int = 1280
    """Height of rendered frames for video capture (default: high-res)."""
    robot_init_noise_scale: float = 0.05
    """Scale for robot initial joint randomization (normalized internally; 1.0 = training-level high randomness, ~10x legacy offsets)."""
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
    video_fps: int = 20
    """Frames per second for recorded videos."""
    print_actions: bool = False
    """Print raw actions each step."""
    gripper_joint_indices: Optional[str] = "7,15"
    """Comma-separated joint indices that correspond to grippers (use gripper delta limit and 119.0 override)."""


def _build_env(args: RolloutArgs):
    env_kwargs = dict(
        obs_mode=args.obs_mode,
        render_mode=args.render_mode,
        sim_backend=args.sim_backend,
        robot_init_noise_scale=args.robot_init_noise_scale,
        human_render_camera_configs={
            "render_camera": {"width": args.render_width, "height": args.render_height}
        },
    )
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

    info_output_root = None
    record_dir = args.record_dir
    if args.capture_video:
        if record_dir is None:
            ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
            record_dir = os.path.join(ckpt_dir, "dual_rollout_videos")
        os.makedirs(record_dir, exist_ok=True)
        print(f"Saving rollout videos to {record_dir}")
        info_output_root = os.path.join(os.path.dirname(record_dir), "info")
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=record_dir,
            save_trajectory=False,
            trajectory_name="rollout",
            max_steps_per_video=args.num_eval_steps,
            video_fps=args.video_fps,
        )
    return ManiSkillVectorEnv(
        eval_envs,
        args.num_eval_envs,
        ignore_terminations=not args.partial_reset,
        record_metrics=True,
    ), info_output_root


def run_rollout(args: RolloutArgs) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    eval_envs, info_output_root = _build_env(args)

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
    action_dim = eval_envs.single_action_space.shape[0]
    control_timestep = float(eval_envs.base_env.control_timestep)
    obs_dim = int(obs.shape[-1])

    csv_path = os.path.abspath(CSV_LOG_FILENAME)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=(
            ["step", "env_index", "time"]
            + [f"observation_{idx:03d}" for idx in range(obs_dim)]
            + [f"action_{idx:03d}" for idx in range(action_dim)]
            + [f"clipped_action_{idx:03d}" for idx in range(action_dim)]
        ),
    )
    csv_writer.writeheader()

    # parse gripper indices if provided
    gripper_indices: List[int] = []
    if args.gripper_joint_indices:
        for token in args.gripper_joint_indices.split(","):
            token = token.strip()
            if token:
                gripper_indices.append(int(token))
    gripper_idx_tensor = (
        torch.tensor(gripper_indices, dtype=torch.long, device=device) if gripper_indices else None
    )

    delta_limit = torch.full(
        (action_dim,), _DEFAULT_ARM_JOINT_DELTA_LIMIT, dtype=torch.float32, device=device
    )
    if gripper_idx_tensor is not None and gripper_idx_tensor.numel() > 0:
        delta_limit[gripper_idx_tensor] = _DEFAULT_GRIPPER_JOINT_DELTA_LIMIT
    action_delta_low = -delta_limit
    action_delta_high = delta_limit

    info_logger = None
    if info_output_root is not None:
        # Save info next to captured rollout videos (same layout as training runs).
        rollout_info_dir = os.path.join(info_output_root, "rollout")
        info_logger = InfoDirectoryLogger(rollout_info_dir, args.num_eval_envs)

    metrics = defaultdict(list)
    for step in range(args.num_eval_steps):

        with torch.no_grad():
            action = agent.get_action(obs, deterministic=args.deterministic_policy)
        if args.print_actions:
            print(f"step={step} action={action.detach().cpu().numpy()}")
        clipped_action = torch.clamp(action, normalized_low, normalized_high)
        obs_cpu = obs.detach().cpu()

        action_cpu = action.detach().cpu()
        clipped_cpu = clipped_action.detach().cpu()
        time_sec = step * control_timestep
        for env_index in range(args.num_eval_envs):
            row = {
                "step": step,
                "env_index": env_index,
                "time": time_sec,
            }
            row.update(
                {
                    f"observation_{idx:03d}": value
                    for idx, value in enumerate(obs_cpu[env_index].tolist())
                }
            )
            row.update(
                {
                    f"action_{idx:03d}": value
                    for idx, value in enumerate(action_cpu[env_index].tolist())
                }
            )
            row.update(
                {
                    f"clipped_action_{idx:03d}": value
                    for idx, value in enumerate(clipped_cpu[env_index].tolist())
                }
            )
            csv_writer.writerow(
                row
            )
        obs, reward, terminations, truncations, info = eval_envs.step(clipped_action)
        obs = obs.to(device)

        if info_logger is not None:
            info_logger.log({"actions": action}, step)
            info_logger.log(
                {
                    "rewards": reward,
                    "terminations": terminations,
                    "truncations": truncations,
                    "next_obs": obs,
                },
                step,
            )
            info_logger.log(info, step)

            # direct joint command: convert NN action -> delta -> add to measured_q -> clip
            if "mesured_q" in info:
                measured_q = torch.as_tensor(info["mesured_q"], device=device, dtype=torch.float32)
                # ensure shape [num_envs, action_dim]
                if measured_q.ndim == 1:
                    measured_q = measured_q.unsqueeze(0)
                normalized_span = _NORMALIZED_ACTION_HIGH - _NORMALIZED_ACTION_LOW
                delta_scale = (clipped_action - _NORMALIZED_ACTION_LOW) / normalized_span
                denorm_delta = action_delta_low + delta_scale * (action_delta_high - action_delta_low)
                direct_joint_command = measured_q + denorm_delta
                if gripper_idx_tensor is not None and gripper_idx_tensor.numel() > 0:
                    direct_joint_command = direct_joint_command.clone()
                    direct_joint_command[:, gripper_idx_tensor] = 119.0
                info["direct_joint_command"] = direct_joint_command.detach().cpu()
                info_logger.log({"direct_joint_command": direct_joint_command}, step)

        payload = {"step": step, "info": _to_serializable(info)}
        info_file.write(json.dumps(payload) + "\n")

        if "final_info" in info:
            mask = info["_final_info"]
            for key, value in info["final_info"]["episode"].items():
                metrics[key].append(value[mask])

    eval_envs.close()
    if info_logger is not None:
        info_logger.close()
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

if __name__ == "__main__2":
    obs  = [-0.2320682853460312,-0.28943148255348206,-0.23436734080314636,0.5481565594673157,-0.23629631102085114,1.5453053712844849,-0.15090152621269226,116.0,-0.11660993099212646,0.11978746205568314,-0.118472620844841,-0.120170958340168,-0.12091055512428284,0.11833566427230835,-0.07020701467990875,0.0,0.2320452779531479,-0.7240120768547058,0.16682040691375732,0.5529502034187317,-0.2341429889202118,1.0770443677902222,-0.2342177778482437,117.0,0.11507595330476761,-0.10176318883895874,0.082725390791893,-0.11702082306146622,-0.11839044839143753,-0.11608947813510895,-0.11765085160732269,0.0,0.4399999976158142,-0.009999999776482582,0.03999999910593033,1.0,0.0,0.0,0.0,1.0, 0.0]
    print(len(obs))
