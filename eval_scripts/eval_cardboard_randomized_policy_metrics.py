import argparse
import csv
import sys
from pathlib import Path

import gymnasium as gym
import torch

cwd = Path.cwd().resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if (cwd / "ppo_dual_xarm7.py").exists():
    sys.path.insert(0, str(cwd))

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import tasks.single.task_single_cardboard_cabinet  # noqa: F401
from ppo_dual_xarm7 import Agent


def _reset_with_seeds(env, seed: int, num_envs: int):
    seeds = [seed + i for i in range(num_envs)]
    try:
        return env.reset(seed=seeds)
    except TypeError:
        return env.reset(seed=seed)


def _yaw_from_quaternion(q: torch.Tensor) -> torch.Tensor:
    if q.ndim == 1:
        q = q.unsqueeze(0)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _initial_box_offset_and_yaw(base_env, num_envs: int):
    center = torch.tensor(
        base_env.workspace_center_pose.p,
        dtype=torch.float32,
        device=base_env.device,
    )
    if hasattr(base_env, "_initial_box_reference_positions"):
        box_position = base_env._initial_box_reference_positions[:num_envs]
    else:
        box_position = base_env.cardboard_cabinet.pose.p
        if box_position.ndim == 1:
            box_position = box_position.unsqueeze(0)
    box_offset = box_position - center
    yaw_deg = torch.rad2deg(_yaw_from_quaternion(base_env.cardboard_cabinet.pose.q))
    yaw_noise_deg = yaw_deg - float(base_env.CABINET_SPEC.yaw_deg)
    return box_offset, yaw_deg, yaw_noise_deg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--env-id",
        default="MySingleCardboardCabinetRandomized-v2",
    )
    parser.add_argument("--num-envs", type=int, default=160)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--robot-uid", default=None)
    parser.add_argument("--sim-backend", default="physx_cuda")
    parser.add_argument("--robot-init-noise-scale", type=float, default=0.25)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        obs_mode="state",
        render_mode=None,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        robot_init_noise_scale=args.robot_init_noise_scale,
        robot_uids=args.robot_uid if args.robot_uid is not None else "my_xarm7",
        reconfiguration_freq=1,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(
        env,
        args.num_envs,
        ignore_terminations=True,
        record_metrics=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    action_low = torch.as_tensor(env.single_action_space.low, device=device)
    action_high = torch.as_tensor(env.single_action_space.high, device=device)

    obs, _ = _reset_with_seeds(env, args.seed, args.num_envs)
    obs = obs.to(device)
    base_env = env.base_env.unwrapped
    box_offset, box_yaw_deg, box_yaw_noise_deg = _initial_box_offset_and_yaw(
        base_env,
        args.num_envs,
    )

    episode_return = torch.zeros(args.num_envs, device=device)
    max_open = torch.zeros(args.num_envs, device=device)
    max_shift = torch.zeros(args.num_envs, device=device)
    first_success_step = torch.full(
        (args.num_envs,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    final_info = None
    panel_y_alignment_reward_sum = torch.zeros(args.num_envs, device=device)
    panel_y_abs_error_sum = torch.zeros(args.num_envs, device=device)

    for step in range(args.num_steps):
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        action = torch.clamp(action, action_low, action_high)
        obs, reward, _, _, info = env.step(action)
        obs = obs.to(device)
        reward = reward.reshape(-1).to(device)
        episode_return += reward
        panel_y_alignment_reward = info[
            "inner_marker_panel_tcp_y_alignment_reward"
        ].reshape(-1).to(device)
        panel_y_abs_error = info["inner_marker_panel_tcp_y_abs_error"].reshape(
            -1
        ).to(device)
        panel_y_alignment_reward_sum += panel_y_alignment_reward
        panel_y_abs_error_sum += panel_y_abs_error

        open_amount = info["inner_box_open_amount"].reshape(-1).to(device)
        outer_shift = info["outer_box_shift"].reshape(-1).to(device)
        max_open = torch.maximum(max_open, open_amount)
        max_shift = torch.maximum(max_shift, outer_shift)
        success_now = (open_amount >= 0.12) & (outer_shift <= 0.02)
        first_success_step[(first_success_step < 0) & success_now] = step
        final_info = info

    assert final_info is not None
    final_open = final_info["inner_box_open_amount"].reshape(-1).to(device)
    final_shift = final_info["outer_box_shift"].reshape(-1).to(device)
    final_tcp_position_error = final_info["return_tcp_position_error"].reshape(
        -1
    ).to(device)
    final_tcp_roll_abs_error = final_info["return_tcp_roll_abs_error"].reshape(
        -1
    ).to(device)
    final_tcp_pitch_abs_error = final_info["return_tcp_pitch_abs_error"].reshape(
        -1
    ).to(device)
    final_tcp_yaw_abs_error = final_info["return_tcp_yaw_abs_error"].reshape(
        -1
    ).to(device)
    final_panel_y_alignment_reward = final_info[
        "inner_marker_panel_tcp_y_alignment_reward"
    ].reshape(-1).to(device)
    final_panel_y_abs_error = final_info["inner_marker_panel_tcp_y_abs_error"].reshape(
        -1
    ).to(device)
    panel_y_alignment_reward_mean = panel_y_alignment_reward_sum / args.num_steps
    panel_y_abs_error_mean = panel_y_abs_error_sum / args.num_steps
    panel_y_alignment_return_contribution = panel_y_alignment_reward_sum / 9.05

    reached_open_success = (max_open >= 0.12) & (max_shift <= 0.02)
    final_open_success = (final_open >= 0.12) & (final_shift <= 0.02)

    rows = []
    for env_idx in range(args.num_envs):
        rows.append(
            {
                "env_idx": env_idx,
                "base_seed": args.seed,
                "seed_list_entry": args.seed + env_idx,
                "replay_num_envs": args.num_envs,
                "box_x_from_base_center": float(box_offset[env_idx, 0].item()),
                "box_y_from_base_center": float(box_offset[env_idx, 1].item()),
                "box_z_from_base_center": float(box_offset[env_idx, 2].item()),
                "box_yaw_deg": float(box_yaw_deg[env_idx].item()),
                "box_yaw_noise_deg": float(box_yaw_noise_deg[env_idx].item()),
                "episode_return": float(episode_return[env_idx].item()),
                "panel_y_alignment_reward_mean": float(
                    panel_y_alignment_reward_mean[env_idx].item()
                ),
                "panel_y_alignment_reward_sum": float(
                    panel_y_alignment_reward_sum[env_idx].item()
                ),
                "panel_y_alignment_return_contribution": float(
                    panel_y_alignment_return_contribution[env_idx].item()
                ),
                "panel_y_abs_error_mean": float(
                    panel_y_abs_error_mean[env_idx].item()
                ),
                "final_panel_y_alignment_reward": float(
                    final_panel_y_alignment_reward[env_idx].item()
                ),
                "final_panel_y_abs_error": float(
                    final_panel_y_abs_error[env_idx].item()
                ),
                "max_open": float(max_open[env_idx].item()),
                "max_shift": float(max_shift[env_idx].item()),
                "final_open": float(final_open[env_idx].item()),
                "final_shift": float(final_shift[env_idx].item()),
                "reached_open_success": bool(reached_open_success[env_idx].item()),
                "final_open_success": bool(final_open_success[env_idx].item()),
                "first_success_step": int(first_success_step[env_idx].item()),
                "final_tcp_position_error": float(
                    final_tcp_position_error[env_idx].item()
                ),
                "final_tcp_roll_abs_error": float(
                    final_tcp_roll_abs_error[env_idx].item()
                ),
                "final_tcp_pitch_abs_error": float(
                    final_tcp_pitch_abs_error[env_idx].item()
                ),
                "final_tcp_yaw_abs_error": float(
                    final_tcp_yaw_abs_error[env_idx].item()
                ),
            }
        )

    env.close()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"env_id {args.env_id}")
    print(f"robot_uid {args.robot_uid if args.robot_uid is not None else 'my_xarm7'}")
    print(f"num_envs {args.num_envs}")
    print("randomized_center_x 0.337300")
    print("randomized_x_range [0.307300, 0.367300]")
    print("randomized_y_range [-0.060000, 0.060000]")
    print("randomized_yaw_noise_range_deg [-5.000, 5.000]")
    print(f"eval_return_mean {episode_return.mean().item():.6f}")
    print(f"eval_return_std {episode_return.std(unbiased=False).item():.6f}")
    print(
        "panel_y_alignment_reward_mean "
        f"{panel_y_alignment_reward_mean.mean().item():.6f}"
    )
    print(
        "panel_y_alignment_return_contribution_mean "
        f"{panel_y_alignment_return_contribution.mean().item():.6f}"
    )
    print(f"panel_y_abs_error_mean {panel_y_abs_error_mean.mean().item():.6f}")
    print(
        "final_panel_y_alignment_reward_mean "
        f"{final_panel_y_alignment_reward.mean().item():.6f}"
    )
    print(f"final_panel_y_abs_error_mean {final_panel_y_abs_error.mean().item():.6f}")
    print(f"reached_open_success_mean {reached_open_success.float().mean().item():.6f}")
    print(f"final_open_success_mean {final_open_success.float().mean().item():.6f}")
    print(f"max_open_mean {max_open.mean().item():.6f}")
    print(f"final_open_mean {final_open.mean().item():.6f}")
    print(f"final_shift_mean {final_shift.mean().item():.6f}")
    print(f"final_tcp_position_error_mean {final_tcp_position_error.mean().item():.6f}")
    print(f"output_csv {output_path}")


if __name__ == "__main__":
    main()
