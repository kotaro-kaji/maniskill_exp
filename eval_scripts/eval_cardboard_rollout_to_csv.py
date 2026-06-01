import argparse
import csv
import sys
from pathlib import Path

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import tasks.single.task_single_cardboard_cabinet  # noqa: F401
from ppo_dual_xarm7 import Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="MyDualCardboardCabinet-v1")
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-episodes", type=int, default=16)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--sim-backend", default="cpu")
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        num_envs=1,
        obs_mode="state",
        render_mode=None,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        robot_init_noise_scale=0.0,
        reconfiguration_freq=1,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    action_low = torch.as_tensor(env.single_action_space.low, device=device)
    action_high = torch.as_tensor(env.single_action_space.high, device=device)
    rows = []
    summaries = []

    for episode in range(args.num_episodes):
        seed = args.seed + episode
        obs, _ = env.reset(seed=seed)
        obs = obs.to(device)
        base_env = env.base_env.unwrapped
        target = base_env.RETURN_TARGET_QPOS.to(device=base_env.device)
        episode_rows = []

        for step in range(args.num_steps):
            with torch.no_grad():
                action = agent.get_action(obs, deterministic=True)
            action = torch.clamp(action, action_low, action_high)
            obs, _, _, _, info = env.step(action)
            obs = obs.to(device)

            qpos = base_env.agent.robot.get_qpos()
            if qpos.ndim == 1:
                qpos = qpos.unsqueeze(0)
            arm_qpos = qpos[:, :7]
            arm_abs_error = torch.abs(arm_qpos - target.reshape(1, 7))
            arm_return_score = 1.0 - torch.tanh(arm_abs_error)
            row = {
                "episode": episode,
                "seed": seed,
                "step": step,
                "inner_box_open_amount": float(info["inner_box_open_amount"][0].item()),
                "outer_box_shift": float(info["outer_box_shift"][0].item()),
                "drawer_open_success_reached": bool(
                    info["drawer_open_success_reached"][0].item()
                ),
                "return_arm_score": float(arm_return_score.mean(dim=-1)[0].item()),
                "return_arm_mean_abs_error": float(arm_abs_error.mean(dim=-1)[0].item()),
                "return_arm_max_abs_error": float(arm_abs_error.max(dim=-1)[0].item()),
            }
            for joint_idx in range(7):
                row[f"arm_qpos_{joint_idx}"] = float(arm_qpos[0, joint_idx].item())
                row[f"arm_error_{joint_idx}"] = float(arm_abs_error[0, joint_idx].item())
            rows.append(row)
            episode_rows.append(row)

        final = episode_rows[-1]
        max_open = max(row["inner_box_open_amount"] for row in episode_rows)
        max_shift = max(row["outer_box_shift"] for row in episode_rows)
        final_open_success = (
            final["inner_box_open_amount"] >= 0.12
            and final["outer_box_shift"] <= 0.02
        )
        reached_open_success = max_open >= 0.12 and max_shift <= 0.02
        summaries.append(
            {
                "episode": episode,
                "seed": seed,
                "max_open": max_open,
                "max_shift": max_shift,
                "final_open": final["inner_box_open_amount"],
                "final_shift": final["outer_box_shift"],
                "final_open_success": final_open_success,
                "reached_open_success": reached_open_success,
                "final_return_arm_score": final["return_arm_score"],
                "final_return_arm_mean_abs_error": final["return_arm_mean_abs_error"],
                "final_return_arm_max_abs_error": final["return_arm_max_abs_error"],
            }
        )

    env.close()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    final_open_success_count = sum(row["final_open_success"] for row in summaries)
    reached_open_success_count = sum(row["reached_open_success"] for row in summaries)
    mean_final_return = sum(row["final_return_arm_score"] for row in summaries) / len(
        summaries
    )
    print(f"num_episodes {args.num_episodes}")
    print(f"reached_open_success_episodes {reached_open_success_count}")
    print(f"final_open_success_episodes {final_open_success_count}")
    print(f"mean_final_return_arm_score {mean_final_return:.6f}")
    for summary in summaries:
        print(
            "episode_summary "
            f"episode={summary['episode']} seed={summary['seed']} "
            f"max_open={summary['max_open']:.6f} "
            f"final_open={summary['final_open']:.6f} "
            f"final_shift={summary['final_shift']:.6f} "
            f"final_open_success={summary['final_open_success']} "
            f"final_return_arm_score={summary['final_return_arm_score']:.6f} "
            f"final_return_arm_mean_abs_error="
            f"{summary['final_return_arm_mean_abs_error']:.6f} "
            f"final_return_arm_max_abs_error="
            f"{summary['final_return_arm_max_abs_error']:.6f}"
        )
    print(f"output_csv {output_path}")


if __name__ == "__main__":
    main()
