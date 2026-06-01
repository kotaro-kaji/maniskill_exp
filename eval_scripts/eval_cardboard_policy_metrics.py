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
import tasks.single.task_single_cardboard_cabinet  # noqa: F401
from ppo_dual_xarm7 import Agent


def reset_with_seeds(env, seed: int, num_envs: int):
    seeds = [seed + i for i in range(num_envs)]
    try:
        return env.reset(seed=seeds)
    except TypeError:
        return env.reset(seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="MyDualCardboardCabinet-v1")
    parser.add_argument("--num-envs", type=int, default=160)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--sim-backend", default="physx_cuda")
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        obs_mode="state",
        render_mode=None,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        robot_init_noise_scale=0.0,
        reconfiguration_freq=1,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    action_low = torch.as_tensor(env.single_action_space.low, device=device)
    action_high = torch.as_tensor(env.single_action_space.high, device=device)

    obs, _ = reset_with_seeds(env, args.seed, args.num_envs)
    obs = obs.to(device)
    base_env = env.base_env.unwrapped
    target = base_env.RETURN_TARGET_QPOS.to(device=base_env.device).reshape(1, -1)[
        :, :7
    ]

    episode_return = torch.zeros(args.num_envs, device=device)
    max_open = torch.zeros(args.num_envs, device=device)
    max_shift = torch.zeros(args.num_envs, device=device)
    first_success_seen = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    first_success_arm_abs_error = torch.full((args.num_envs, 7), float("nan"), device=device)
    first_success_tcp_position_error = torch.full((args.num_envs,), float("nan"), device=device)
    first_success_tcp_roll_abs_error = torch.full((args.num_envs,), float("nan"), device=device)
    first_success_tcp_pitch_abs_error = torch.full((args.num_envs,), float("nan"), device=device)
    first_success_tcp_yaw_abs_error = torch.full((args.num_envs,), float("nan"), device=device)
    final_info = None
    final_arm_score = None
    final_arm_mean_abs_error = None
    final_arm_max_abs_error = None
    final_arm_abs_error = None

    for _ in range(args.num_steps):
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        action = torch.clamp(action, action_low, action_high)
        obs, reward, _, _, info = env.step(action)
        obs = obs.to(device)
        reward = reward.reshape(-1).to(device)
        episode_return += reward

        open_amount = info["inner_box_open_amount"].reshape(-1).to(device)
        outer_shift = info["outer_box_shift"].reshape(-1).to(device)
        max_open = torch.maximum(max_open, open_amount)
        max_shift = torch.maximum(max_shift, outer_shift)
        final_info = info

        qpos = base_env.agent.robot.get_qpos()
        if qpos.ndim == 1:
            qpos = qpos.unsqueeze(0)
        arm_abs_error = torch.abs(qpos[:, :7] - target)
        success_now = (open_amount >= 0.12) & (outer_shift <= 0.02)
        first_success_now = success_now & (~first_success_seen)
        first_success_arm_abs_error[first_success_now] = arm_abs_error[
            first_success_now
        ]
        first_success_tcp_position_error[first_success_now] = info[
            "return_tcp_position_error"
        ].reshape(-1).to(device)[first_success_now]
        first_success_tcp_roll_abs_error[first_success_now] = info[
            "return_tcp_roll_abs_error"
        ].reshape(-1).to(device)[first_success_now]
        first_success_tcp_pitch_abs_error[first_success_now] = info[
            "return_tcp_pitch_abs_error"
        ].reshape(-1).to(device)[first_success_now]
        first_success_tcp_yaw_abs_error[first_success_now] = info[
            "return_tcp_yaw_abs_error"
        ].reshape(-1).to(device)[first_success_now]
        first_success_seen = first_success_seen | success_now

        final_arm_abs_error = arm_abs_error
        final_arm_score = (1.0 - torch.tanh(arm_abs_error)).mean(dim=-1)
        final_arm_mean_abs_error = arm_abs_error.mean(dim=-1)
        final_arm_max_abs_error = arm_abs_error.max(dim=-1).values

    assert final_info is not None
    assert final_arm_abs_error is not None
    assert final_arm_abs_error.shape == (args.num_envs, 7), final_arm_abs_error.shape
    final_open = final_info["inner_box_open_amount"].reshape(-1).to(device)
    final_shift = final_info["outer_box_shift"].reshape(-1).to(device)
    final_tcp_position_error = final_info["return_tcp_position_error"].reshape(-1).to(device)
    final_tcp_roll_abs_error = final_info["return_tcp_roll_abs_error"].reshape(-1).to(device)
    final_tcp_pitch_abs_error = final_info["return_tcp_pitch_abs_error"].reshape(-1).to(device)
    final_tcp_yaw_abs_error = final_info["return_tcp_yaw_abs_error"].reshape(-1).to(device)
    reached_open_success = (max_open >= 0.12) & (max_shift <= 0.02)
    final_open_success = (final_open >= 0.12) & (final_shift <= 0.02)

    rows = []
    for env_idx in range(args.num_envs):
        rows.append(
            {
                "env_idx": env_idx,
                "seed": args.seed + env_idx,
                "episode_return": float(episode_return[env_idx].item()),
                "max_open": float(max_open[env_idx].item()),
                "max_shift": float(max_shift[env_idx].item()),
                "final_open": float(final_open[env_idx].item()),
                "final_shift": float(final_shift[env_idx].item()),
                "reached_open_success": bool(reached_open_success[env_idx].item()),
                "final_open_success": bool(final_open_success[env_idx].item()),
                "first_success_seen": bool(first_success_seen[env_idx].item()),
                "final_return_arm_score": float(final_arm_score[env_idx].item()),
                "first_success_return_arm_mean_abs_error": float(
                    first_success_arm_abs_error[env_idx].mean().item()
                ),
                "first_success_return_arm_max_abs_error": float(
                    first_success_arm_abs_error[env_idx].max().item()
                ),
                "first_success_tcp_position_error": float(
                    first_success_tcp_position_error[env_idx].item()
                ),
                "first_success_tcp_roll_abs_error": float(
                    first_success_tcp_roll_abs_error[env_idx].item()
                ),
                "first_success_tcp_pitch_abs_error": float(
                    first_success_tcp_pitch_abs_error[env_idx].item()
                ),
                "first_success_tcp_yaw_abs_error": float(
                    first_success_tcp_yaw_abs_error[env_idx].item()
                ),
                "final_return_arm_mean_abs_error": float(
                    final_arm_mean_abs_error[env_idx].item()
                ),
                "final_return_arm_max_abs_error": float(
                    final_arm_max_abs_error[env_idx].item()
                ),
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
                "final_return_arm_joint_0_abs_error": float(
                    final_arm_abs_error[env_idx, 0].item()
                ),
                "first_success_return_arm_joint_0_abs_error": float(
                    first_success_arm_abs_error[env_idx, 0].item()
                ),
                "final_return_arm_joint_1_abs_error": float(
                    final_arm_abs_error[env_idx, 1].item()
                ),
                "first_success_return_arm_joint_1_abs_error": float(
                    first_success_arm_abs_error[env_idx, 1].item()
                ),
                "final_return_arm_joint_2_abs_error": float(
                    final_arm_abs_error[env_idx, 2].item()
                ),
                "first_success_return_arm_joint_2_abs_error": float(
                    first_success_arm_abs_error[env_idx, 2].item()
                ),
                "final_return_arm_joint_3_abs_error": float(
                    final_arm_abs_error[env_idx, 3].item()
                ),
                "first_success_return_arm_joint_3_abs_error": float(
                    first_success_arm_abs_error[env_idx, 3].item()
                ),
                "final_return_arm_joint_4_abs_error": float(
                    final_arm_abs_error[env_idx, 4].item()
                ),
                "first_success_return_arm_joint_4_abs_error": float(
                    first_success_arm_abs_error[env_idx, 4].item()
                ),
                "final_return_arm_joint_5_abs_error": float(
                    final_arm_abs_error[env_idx, 5].item()
                ),
                "first_success_return_arm_joint_5_abs_error": float(
                    first_success_arm_abs_error[env_idx, 5].item()
                ),
                "final_return_arm_joint_6_abs_error": float(
                    final_arm_abs_error[env_idx, 6].item()
                ),
                "first_success_return_arm_joint_6_abs_error": float(
                    first_success_arm_abs_error[env_idx, 6].item()
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

    eval_return_mean = episode_return.mean().item()
    eval_return_std = episode_return.std(unbiased=False).item()
    print(f"num_envs {args.num_envs}")
    print(f"eval_return_mean {eval_return_mean:.6f}")
    print(f"eval_return_std {eval_return_std:.6f}")
    print(f"reached_open_success_mean {reached_open_success.float().mean().item():.6f}")
    print(f"final_open_success_mean {final_open_success.float().mean().item():.6f}")
    print(f"final_open_mean {final_open.mean().item():.6f}")
    print(f"final_shift_mean {final_shift.mean().item():.6f}")
    print(f"final_return_arm_score_mean {final_arm_score.mean().item():.6f}")
    print(f"first_success_seen_mean {first_success_seen.float().mean().item():.6f}")
    print(
        "first_success_return_arm_mean_abs_error_mean "
        f"{torch.nanmean(first_success_arm_abs_error.mean(dim=1)).item():.6f}"
    )
    print(
        "first_success_return_arm_max_abs_error_mean "
        f"{torch.nanmean(first_success_arm_abs_error.max(dim=1).values).item():.6f}"
    )
    print(
        "first_success_tcp_position_error_mean "
        f"{torch.nanmean(first_success_tcp_position_error).item():.6f}"
    )
    print(
        "first_success_tcp_roll_abs_error_mean "
        f"{torch.nanmean(first_success_tcp_roll_abs_error).item():.6f}"
    )
    print(
        "first_success_tcp_pitch_abs_error_mean "
        f"{torch.nanmean(first_success_tcp_pitch_abs_error).item():.6f}"
    )
    print(
        "first_success_tcp_yaw_abs_error_mean "
        f"{torch.nanmean(first_success_tcp_yaw_abs_error).item():.6f}"
    )
    print(
        "final_return_arm_mean_abs_error_mean "
        f"{final_arm_mean_abs_error.mean().item():.6f}"
    )
    print(
        "final_return_arm_max_abs_error_mean "
        f"{final_arm_max_abs_error.mean().item():.6f}"
    )
    print(f"final_tcp_position_error_mean {final_tcp_position_error.mean().item():.6f}")
    print(f"final_tcp_roll_abs_error_mean {final_tcp_roll_abs_error.mean().item():.6f}")
    print(f"final_tcp_pitch_abs_error_mean {final_tcp_pitch_abs_error.mean().item():.6f}")
    print(f"final_tcp_yaw_abs_error_mean {final_tcp_yaw_abs_error.mean().item():.6f}")
    joint_abs_error_mean = final_arm_abs_error.mean(dim=0)
    joint_abs_error_mean_deg = torch.rad2deg(joint_abs_error_mean)
    first_success_joint_abs_error_mean = torch.nanmean(
        first_success_arm_abs_error, dim=0
    )
    first_success_joint_abs_error_mean_deg = torch.rad2deg(
        first_success_joint_abs_error_mean
    )
    for joint_idx in range(7):
        print(
            f"first_success_return_arm_joint_{joint_idx}_abs_error_mean "
            f"{first_success_joint_abs_error_mean[joint_idx].item():.6f}"
        )
        print(
            f"first_success_return_arm_joint_{joint_idx}_abs_error_mean_deg "
            f"{first_success_joint_abs_error_mean_deg[joint_idx].item():.3f}"
        )
        print(
            f"final_return_arm_joint_{joint_idx}_abs_error_mean "
            f"{joint_abs_error_mean[joint_idx].item():.6f}"
        )
        print(
            f"final_return_arm_joint_{joint_idx}_abs_error_mean_deg "
            f"{joint_abs_error_mean_deg[joint_idx].item():.3f}"
        )
    print(f"output_csv {output_path}")


if __name__ == "__main__":
    main()
