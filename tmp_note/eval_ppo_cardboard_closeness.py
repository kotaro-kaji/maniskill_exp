import argparse
import csv
import sys
from pathlib import Path

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import tasks.dual.task_dual_cardboard_cabinet  # noqa: F401
from ppo_dual_xarm7 import Agent


def parse_thresholds(value: str):
    return [float(item) for item in value.split(",") if item]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="MyDualCardboardCabinet-v1")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--robot-init-noise-scale", type=float, default=0.0)
    parser.add_argument("--sim-backend", default="gpu")
    parser.add_argument("--thresholds", default="0.005,0.01,0.015,0.02")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    env = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        obs_mode="state",
        render_mode=None,
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        robot_init_noise_scale=args.robot_init_noise_scale,
        reconfiguration_freq=1,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    obs, _ = env.reset(seed=args.seed)
    best_distance = torch.full((args.num_envs,), float("inf"), device=device)
    action_low = torch.as_tensor(env.single_action_space.low, device=device)
    action_high = torch.as_tensor(env.single_action_space.high, device=device)

    for _ in range(args.num_steps):
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        action = torch.clamp(action, action_low, action_high)
        obs, _, _, _, info = env.step(action)
        best_distance = torch.minimum(best_distance, info["tcp_to_target_distance"])

    best_distance_cpu = best_distance.detach().cpu()
    env.close()

    print(f"num_envs {args.num_envs}")
    print(f"num_steps {args.num_steps}")
    print(f"mean_best_tcp_to_target_distance {best_distance_cpu.mean().item():.6f}")
    print(f"median_best_tcp_to_target_distance {best_distance_cpu.median().item():.6f}")
    print(f"min_best_tcp_to_target_distance {best_distance_cpu.min().item():.6f}")
    print(f"max_best_tcp_to_target_distance {best_distance_cpu.max().item():.6f}")
    for threshold in thresholds:
        rate = (best_distance_cpu <= threshold).float().mean().item()
        print(f"success_rate_at_{threshold:.3f}m {rate:.6f}")

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["env_id", "best_tcp_to_target_distance"])
            for env_id, distance in enumerate(best_distance_cpu.tolist()):
                writer.writerow([env_id, distance])
        print(f"output_csv {output_path}")


if __name__ == "__main__":
    main()
