import argparse
import os
from pathlib import Path
import time
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
import h5py
import videoio

import mani_skill.envs  # noqa: F401
from mani_skill.utils.structs.types import SimConfig
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

from sac import Actor

#デフォルトはSIM_FREQUENCY_HZ=100, CONTROL_FREQUENCY_HZ=20
SIM_FREQUENCY_HZ = 90
CONTROL_FREQUENCY_HZ = 90

_LEGACY_JITTERS = [
    # (-0.06, 0.0, 0.0),
    # (-0.03, 0.0, 0.0),
    # (0.0, 0.0, 0.0),
    # (0.03, 0.0, 0.0),
    # (0.06, 0.0, 0.0),
    # (0.0, 0.0, -15.0),
    # (0.0, 0.0, 15.0),
]

X_JITTERS = [-0.03, -0.015, 0.0, 0.015, 0.03]
Y_JITTERS = [-0.06, -0.045, -0.03, -0.015, 0.0, 0.015, 0.03, 0.045, 0.06]
THETA_JITTERS = [0.0, 15.0, -15.0]

DEFAULT_JITTERS = [(x, y, t) for x in X_JITTERS for y in Y_JITTERS for t in THETA_JITTERS]
assert len(DEFAULT_JITTERS) == 135, f"Expected 135 jitters, got {len(DEFAULT_JITTERS)}"

_DEFAULT_ARM_JOINT_DELTA_LIMIT = 0.045
_DEFAULT_GRIPPER_JOINT_DELTA_LIMIT = 0.1
_GRIPPER_JOINT_INDICES = [7, 15]


def _parse_jitters(text: str) -> List[Tuple[float, float, float]]:
    if not text:
        return DEFAULT_JITTERS
    out = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid jitter tuple: '{chunk}'")
        out.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not out:
        return DEFAULT_JITTERS
    return out


def _save_rmb_episode(
    out_dir: Path,
    times: List[float],
    rewards: List[float],
    measured_q: List[np.ndarray],
    command_abs: List[np.ndarray],
    rgb_frames: List[np.ndarray],
    camera_name: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "main.rmb.hdf5"
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("time", data=np.asarray(times, dtype=np.float32))
        f.create_dataset("reward", data=np.asarray(rewards, dtype=np.float32))
        f.create_dataset("measured_joint_pos", data=np.asarray(measured_q, dtype=np.float32))
        f.create_dataset("command_joint_pos", data=np.asarray(command_abs, dtype=np.float32))
        f.attrs["format"] = "RmbData-Compact"
    if rgb_frames:
        video_path = out_dir / f"{camera_name}_rgb_image.rmb.mp4"
        images = np.asarray(rgb_frames, dtype=np.uint8)
        if images.ndim == 5 and images.shape[1] == 1:
            images = images[:, 0]
        if images.shape[-1] == 4:
            images = images[..., :3]

        videoio.videosave(str(video_path), images)


def _build_measured_q_from_info(info, action_dim: int) -> np.ndarray:
    q = info.get("mesured_q")
    if q is None:
        return np.zeros((action_dim,), dtype=np.float32)
    if torch.is_tensor(q):
        q = q.detach().cpu().numpy()
    q = np.asarray(q)
    if q.ndim >= 2:
        q = q[0]
    q = q.reshape(-1)
    if q.shape[0] != action_dim:
        raise RuntimeError(
            f"mesured_q dim mismatch: expected {action_dim}, got {q.shape[0]}"
        )
    return q.astype(np.float32)


def _convert_gripper_tensor_to_robomanip(
    tensor: torch.Tensor, gripper_joint_indices: List[int]
) -> torch.Tensor:
    if not gripper_joint_indices:
        return tensor
    for idx in gripper_joint_indices:
        tensor[idx] = tensor[idx].new_tensor(
            float(tensor[idx].item()) * (-1000.0) + 850.0
        )
    return tensor


def _get_joint_limits(env) -> Tuple[torch.Tensor, torch.Tensor]:
    low = torch.tensor(env.single_action_space.low, dtype=torch.float32, device=env.device)
    high = torch.tensor(env.single_action_space.high, dtype=torch.float32, device=env.device)
    return low, high


def _process_action_like_rollout_sac(
    raw_action: torch.Tensor,
    current_joint_pos: torch.Tensor,
    joint_low: torch.Tensor,
    joint_high: torch.Tensor,
) -> torch.Tensor:
    action_dim = raw_action.numel()
    normalized_low = raw_action.new_full((action_dim,), -1.0)
    normalized_high = raw_action.new_full((action_dim,), 1.0)
    clipped_action = torch.clamp(raw_action, normalized_low, normalized_high)

    delta_limit = raw_action.new_full((action_dim,), _DEFAULT_ARM_JOINT_DELTA_LIMIT)
    if _GRIPPER_JOINT_INDICES:
        idx = torch.tensor(_GRIPPER_JOINT_INDICES, dtype=torch.long, device=raw_action.device)
        delta_limit[idx] = _DEFAULT_GRIPPER_JOINT_DELTA_LIMIT
    action_delta_low = -delta_limit
    action_delta_high = delta_limit

    normalized_span = normalized_high - normalized_low
    delta_scale = (clipped_action - normalized_low) / normalized_span
    denormalized_delta = action_delta_low + delta_scale * (action_delta_high - action_delta_low)

    direct_joint_command = current_joint_pos + denormalized_delta
    direct_joint_command = torch.max(torch.min(direct_joint_command, joint_high), joint_low)
    direct_joint_command = direct_joint_command.clone()
    direct_joint_command = _convert_gripper_tensor_to_robomanip(
        direct_joint_command, _GRIPPER_JOINT_INDICES
    )
    if _GRIPPER_JOINT_INDICES:
        idx = torch.tensor(_GRIPPER_JOINT_INDICES, dtype=torch.long, device=raw_action.device)
        direct_joint_command[idx] = 119.0

    return direct_joint_command


def _extract_ckpt_number(path: Path) -> int:
    stem = path.stem
    if not stem.startswith("ckpt_"):
        raise ValueError(f"Invalid checkpoint name: {path}")
    return int(stem.split("_", 1)[1])


def _list_ckpt_paths(base_ckpt_path: Path) -> List[Path]:
    ckpt_dir = base_ckpt_path.parent
    ckpt_paths = sorted(
        ckpt_dir.glob("ckpt_*.pt"), key=lambda p: _extract_ckpt_number(p)
    )
    if not ckpt_paths:
        raise RuntimeError(f"No ckpt_*.pt files found in {ckpt_dir}")
    return ckpt_paths


def _load_actor_from_ckpt(actor: Actor, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["actor"] if isinstance(ckpt, dict) and "actor" in ckpt else ckpt
    actor.load_state_dict(state)
    actor.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--env_id", type=str, default="MyDualBoxRotation-v0")
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--camera_name", type=str, default="top")
    parser.add_argument("--output_dir", type=str, default="dataset")
    parser.add_argument(
        "--jitters",
        type=str,
        default="",
        help="Semicolon-separated x,y,theta tuples. Example: '-0.06,0,0;0,0,0'",
    )
    parser.add_argument("--control_mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--sim_backend", type=str, default="gpu")
    parser.add_argument("--robot_init_noise_scale", type=float, default=0.1)
    parser.add_argument("--cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic (mean) action if True.",
    )
    args = parser.parse_args()

    jitters = _parse_jitters(args.jitters)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        sim_config=SimConfig(
            sim_freq=SIM_FREQUENCY_HZ,
            control_freq=CONTROL_FREQUENCY_HZ,
        ),
        robot_init_noise_scale=args.robot_init_noise_scale,
        control_mode=args.control_mode,
        collect_rmb_data=True,
    )
    env = gym.make(
        args.env_id,
        num_envs=1,
        reconfiguration_freq=1,
        max_episode_steps=300,
        **env_kwargs,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=False, record_metrics=True)
    dt = 1.0 / CONTROL_FREQUENCY_HZ

    actor = Actor(env).to(device)
    base_ckpt_path = Path(args.checkpoint)
    ckpt_paths = _list_ckpt_paths(base_ckpt_path)
    base_ckpt_num = _extract_ckpt_number(base_ckpt_path)
    ckpt_nums = [_extract_ckpt_number(p) for p in ckpt_paths]
    if base_ckpt_num not in ckpt_nums:
        raise RuntimeError(f"Base checkpoint not found in directory: {base_ckpt_path}")
    base_ckpt_idx = ckpt_nums.index(base_ckpt_num)
    _load_actor_from_ckpt(actor, base_ckpt_path, device)

    ckpt_tag = base_ckpt_path.stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / f"{timestamp}_{ckpt_tag}"
    output_root.mkdir(parents=True, exist_ok=True)
    failure_root = Path(args.output_dir) / f"{timestamp}_{ckpt_tag}_failures"
    failure_root.mkdir(parents=True, exist_ok=True)

    joint_low, joint_high = _get_joint_limits(env)

    for idx, (x_j, y_j, t_j) in enumerate(jitters):
        attempt_ckpt_idx = base_ckpt_idx
        while True:
            ckpt_path = ckpt_paths[attempt_ckpt_idx]
            ckpt_num = _extract_ckpt_number(ckpt_path)
            if attempt_ckpt_idx != base_ckpt_idx:
                _load_actor_from_ckpt(actor, ckpt_path, device)

            options = {
                "box_jitter_xy": (x_j, y_j),
                "box_jitter_theta_deg": t_j,
            }
            obs, info = env.reset(seed=args.seed, options=options)
            obs = obs.to(device)

            times: List[float] = []
            rewards: List[float] = []
            measured_q: List[np.ndarray] = []
            command_abs_list: List[np.ndarray] = []
            rgb_frames: List[np.ndarray] = []
            episode_return = 0.0

            elapsed_time = 0.0
            for step in range(args.num_steps):
                with torch.no_grad():
                    if args.deterministic:
                        action = actor.get_eval_action(obs)
                    else:
                        action, _, _ = actor.get_action(obs)
                next_obs, reward, term, trunc, info = env.step(action)
                obs = next_obs.to(device)

                reward_scalar = float(reward.detach().cpu().numpy().reshape(-1)[0])
                episode_return += reward_scalar

                q = _build_measured_q_from_info(
                    info, action.detach().cpu().numpy().reshape(-1).shape[0]
                )
                if _GRIPPER_JOINT_INDICES:
                    q = q.copy()
                    q[_GRIPPER_JOINT_INDICES] = 119.0
                current_q = torch.tensor(q, device=device, dtype=torch.float32)
                command_abs_tensor = _process_action_like_rollout_sac(
                    action.squeeze(0),
                    current_q,
                    joint_low.to(device),
                    joint_high.to(device),
                )

                rgb = None
                if "rgb_images" in info:
                    rgb = info["rgb_images"].get(args.camera_name)
                if rgb is None:
                    raise RuntimeError(
                        f"RGB image for camera '{args.camera_name}' is missing in info['rgb_images']."
                    )
                if torch.is_tensor(rgb):
                    rgb = rgb.detach().cpu().numpy()

                times.append(elapsed_time)
                rewards.append(reward_scalar)
                measured_q.append(q)
                command_abs_list.append(
                    command_abs_tensor.detach().cpu().numpy().reshape(-1)
                )
                rgb_frames.append(rgb)
                elapsed_time += dt

                # done = bool(term.detach().cpu().numpy().reshape(-1)[0]) or bool(
                #     trunc.detach().cpu().numpy().reshape(-1)[0]
                # )
                # print(f"done={done}")
                # if done:
                #     break

            episode_name = (
                f"episode_{idx:03d}_x{x_j:+.2f}_y{y_j:+.2f}_t{t_j:+.1f}_ckpt{ckpt_num}.rmb"
            )
            if episode_return >= 400.0:
                out_dir = output_root / episode_name
                _save_rmb_episode(
                    out_dir,
                    times,
                    rewards,
                    measured_q,
                    command_abs_list,
                    rgb_frames,
                    args.camera_name,
                )
                print(
                    f"episode {idx}: jitter=({x_j},{y_j},{t_j}) ckpt={ckpt_num} return={episode_return:.4f} saved={out_dir}"
                )
                if attempt_ckpt_idx != base_ckpt_idx:
                    _load_actor_from_ckpt(actor, base_ckpt_path, device)
                break

            failure_out_dir = failure_root / episode_name
            _save_rmb_episode(
                failure_out_dir,
                times,
                rewards,
                measured_q,
                command_abs_list,
                rgb_frames,
                args.camera_name,
            )
            print(
                f"[FAILURE] return {episode_return:.4f} < 400.0. saved to {failure_out_dir}"
            )

            attempt_ckpt_idx += 1
            if attempt_ckpt_idx >= len(ckpt_paths):
                message = (
                    f"[ERROR] episode {idx} return {episode_return:.4f} < 400.0 and no more ckpts."
                )
                print(message)
                raise RuntimeError(message)


if __name__ == "__main__":
    main()
