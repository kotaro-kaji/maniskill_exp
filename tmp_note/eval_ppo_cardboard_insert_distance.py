import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

import tasks.dual.task_dual_cardboard_cabinet  # noqa: F401
from ppo_dual_xarm7 import Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--robot-init-noise-scale", type=float, default=1.0)
    args = parser.parse_args()

    env = gym.make(
        "MyDualCardboardCabinet-v0",
        num_envs=args.num_envs,
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend="gpu",
        control_mode=args.control_mode,
        robot_init_noise_scale=args.robot_init_noise_scale,
        reconfiguration_freq=1,
        sim_config=dict(
            gpu_memory_config=dict(
                max_rigid_contact_count=2**22,
                max_rigid_patch_count=2**21,
            )
        ),
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = ManiSkillVectorEnv(
        env, args.num_envs, ignore_terminations=True, record_metrics=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    obs, _ = env.reset(seed=1)
    best_distance = torch.full((args.num_envs,), float("inf"), device=device)
    best_delta = torch.zeros((args.num_envs, 3), device=device)
    max_box_shift = torch.zeros((args.num_envs,), device=device)

    for _ in range(args.num_steps):
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
        distance = info["insert_marker_distance"].to(device)
        improved = distance < best_distance
        best_distance = torch.minimum(best_distance, distance)
        obs_delta = obs[:, -4:-1]
        best_delta[improved] = obs_delta[improved]
        max_box_shift = torch.maximum(max_box_shift, info["box_position_shift"].to(device))

    best = best_distance.detach().cpu()
    delta = best_delta.detach().cpu()
    shift = max_box_shift.detach().cpu()
    finite_best = best[torch.isfinite(best)]
    finite_shift = shift[torch.isfinite(shift)]
    print("checkpoint", args.checkpoint)
    print("num_envs", args.num_envs)
    print("num_steps", args.num_steps)
    print("finite", int(finite_best.numel()), "/", int(best.numel()))
    print("mean", float(finite_best.mean()))
    print("median", float(finite_best.median()))
    print("min", float(finite_best.min()))
    print("max", float(finite_best.max()))
    print("best_delta_mean", [float(v) for v in delta.mean(dim=0)])
    print("best_abs_delta_mean", [float(v) for v in delta.abs().mean(dim=0)])
    print("best_abs_delta_median", [float(v) for v in delta.abs().median(dim=0).values])
    print("max_box_shift_mean", float(finite_shift.mean()))
    print("max_box_shift_median", float(finite_shift.median()))
    print("max_box_shift_max", float(finite_shift.max()))
    for threshold in (0.005, 0.01, 0.02, 0.03):
        print(
            f"under_{threshold:.3f}",
            int((finite_best < threshold).sum()),
            "/",
            int(finite_best.numel()),
        )

    env.close()


if __name__ == "__main__":
    main()
