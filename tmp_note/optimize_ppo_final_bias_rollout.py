import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import tasks.dual.task_dual_cardboard_cabinet  # noqa: F401
from ppo_dual_xarm7 import Agent


def make_env(args, num_envs):
    env = gym.make(
        args.env_id,
        num_envs=num_envs,
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
    return ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)


def evaluate_candidates(args, agent, candidates):
    num_candidates = candidates.shape[0]
    env = make_env(args, num_candidates * args.repeats)
    device = candidates.device
    obs, _ = env.reset(seed=args.seed)
    rollout_bias = candidates.repeat_interleave(args.repeats, dim=0)
    rollout_count = rollout_bias.shape[0]
    best_distance = torch.full((rollout_count,), float("inf"), device=device)
    best_shift = torch.zeros((rollout_count,), device=device)
    max_shift = torch.zeros((rollout_count,), device=device)
    best_eef_x_dot = torch.zeros((rollout_count,), device=device)
    best_gripper_qpos = torch.zeros((rollout_count,), device=device)

    for _ in range(args.num_steps):
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        obs, _, _, _, info = env.step(torch.clamp(action + rollout_bias, -1.0, 1.0))
        distance = info["insert_marker_distance"].to(device)
        shift = info["box_position_shift"].to(device)
        max_shift = torch.maximum(max_shift, shift)
        improved = distance < best_distance
        best_distance = torch.minimum(best_distance, distance)
        best_shift[improved] = max_shift[improved]
        best_eef_x_dot[improved] = info["eef_x_axis_world"].to(device)[improved, 0]
        best_gripper_qpos[improved] = info["gripper_drive_qpos"].to(device)[improved]
    env.close()

    shift_over = torch.clamp(best_shift - args.target_box_shift, min=0.0)
    eef_under = torch.clamp(args.target_eef_x_dot - best_eef_x_dot, min=0.0)
    gripper_under = torch.clamp(args.min_gripper_qpos - best_gripper_qpos, min=0.0)
    score = (
        best_distance
        + args.box_shift_weight * shift_over
        + args.eef_weight * eef_under
        + args.gripper_weight * gripper_under
    )
    shape = (num_candidates, args.repeats)
    return (
        score.reshape(shape).mean(dim=1),
        best_distance.reshape(shape).mean(dim=1),
        best_shift.reshape(shape).mean(dim=1),
        best_eef_x_dot.reshape(shape).mean(dim=1),
        best_gripper_qpos.reshape(shape).mean(dim=1),
        (best_distance.reshape(shape) < 0.003).sum(dim=1),
        ((best_distance.reshape(shape) < 0.003) & (best_shift.reshape(shape) <= args.target_box_shift)).sum(dim=1),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--robot-init-noise-scale", type=float, default=0.0)
    parser.add_argument("--num-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--population", type=int, default=192)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--elite-frac", type=float, default=0.15)
    parser.add_argument("--dims", nargs="+", type=int, default=[5, 8, 13, 14])
    parser.add_argument("--mean", nargs="+", type=float, default=[-0.08, -0.12, 0.0, -0.12])
    parser.add_argument("--std", nargs="+", type=float, default=[0.04, 0.04, 0.04, 0.04])
    parser.add_argument("--max-abs", type=float, default=0.18)
    parser.add_argument("--target-box-shift", type=float, default=0.005)
    parser.add_argument("--target-eef-x-dot", type=float, default=0.965)
    parser.add_argument("--min-gripper-qpos", type=float, default=0.40)
    parser.add_argument("--box-shift-weight", type=float, default=12.0)
    parser.add_argument("--eef-weight", type=float, default=0.035)
    parser.add_argument("--gripper-weight", type=float, default=0.01)
    args = parser.parse_args()
    assert len(args.dims) == len(args.mean) == len(args.std)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_env = make_env(args, 1)
    agent = Agent(probe_env).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    agent.load_state_dict(state)
    agent.eval()
    action_dim = probe_env.single_action_space.shape[0]
    probe_env.close()

    mean = torch.tensor(args.mean, dtype=torch.float32, device=device)
    std = torch.tensor(args.std, dtype=torch.float32, device=device)
    elite_count = max(2, int(args.population * args.elite_frac))
    best_row = None
    for iteration in range(args.iterations):
        values = mean + std * torch.randn((args.population, len(args.dims)), device=device)
        values = torch.clamp(values, -args.max_abs, args.max_abs)
        values[0] = mean
        candidates = torch.zeros((args.population, action_dim), device=device)
        for col, dim in enumerate(args.dims):
            candidates[:, dim] = values[:, col]
        score, distance, shift, eef, gripper, under3, valid_under3 = evaluate_candidates(
            args, agent, candidates
        )
        order = torch.argsort(score)
        elite_values = values[order[:elite_count]]
        mean = elite_values.mean(dim=0)
        std = elite_values.std(dim=0).clamp(min=0.005, max=args.max_abs)
        idx = order[0]
        best_row = (
            float(score[idx]),
            values[idx].detach().cpu().tolist(),
            float(distance[idx]),
            float(shift[idx]),
            float(eef[idx]),
            float(gripper[idx]),
            int(under3[idx]),
            int(valid_under3[idx]),
        )
        print(
            "iter",
            iteration,
            "best",
            best_row,
            "next_mean",
            mean.detach().cpu().tolist(),
            "next_std",
            std.detach().cpu().tolist(),
        )

    assert best_row is not None
    best_values = torch.tensor(best_row[1], dtype=torch.float32)
    tuned = dict(state)
    bias_key = "actor_mean.6.bias"
    tuned[bias_key] = tuned[bias_key].detach().cpu().clone()
    for dim, value in zip(args.dims, best_values):
        tuned[bias_key][dim] += value

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tuned, output)
    print("output", output)
    print("dims", args.dims)
    print("best_score_values_distance_shift_eef_gripper", best_row)


if __name__ == "__main__":
    main()
