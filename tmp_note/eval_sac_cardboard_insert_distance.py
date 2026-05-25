import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

import tasks.dual.task_dual_cardboard_cabinet  # noqa: F401
from sac import Actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="MyDualCardboardCabinet-v0")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=120)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--robot-init-noise-scale", type=float, default=1.0)
    parser.add_argument("--valid-box-shift", type=float, default=0.005)
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
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
    env = ManiSkillVectorEnv(env, args.num_envs, ignore_terminations=True, record_metrics=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(env).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    obs, _ = env.reset(seed=1)
    best_distance = torch.full((args.num_envs,), float("inf"), device=device)
    best_delta = torch.zeros((args.num_envs, 3), device=device)
    valid_best_distance = torch.full((args.num_envs,), float("inf"), device=device)
    valid_best_delta = torch.zeros((args.num_envs, 3), device=device)
    max_box_shift = torch.zeros((args.num_envs,), device=device)
    final_distance = torch.full((args.num_envs,), float("inf"), device=device)
    final_delta = torch.zeros((args.num_envs, 3), device=device)
    final_box_shift = torch.zeros((args.num_envs,), device=device)

    for _ in range(args.num_steps):
        with torch.no_grad():
            action = actor.get_eval_action(obs)
        obs, _, _, _, info = env.step(action)
        distance = info["insert_marker_distance"].to(device)
        box_shift = info["box_position_shift"].to(device)
        obs_delta = obs[:, -4:-1]
        final_distance = distance
        final_delta = obs_delta
        final_box_shift = box_shift
        improved = distance < best_distance
        best_distance = torch.minimum(best_distance, distance)
        best_delta[improved] = obs_delta[improved]
        max_box_shift = torch.maximum(max_box_shift, box_shift)
        valid = max_box_shift <= args.valid_box_shift
        valid_improved = valid & (distance < valid_best_distance)
        valid_best_distance = torch.where(
            valid_improved, distance, valid_best_distance
        )
        valid_best_delta[valid_improved] = obs_delta[valid_improved]

    best = best_distance.detach().cpu()
    delta = best_delta.detach().cpu()
    valid_best = valid_best_distance.detach().cpu()
    valid_delta = valid_best_delta.detach().cpu()
    shift = max_box_shift.detach().cpu()
    final = final_distance.detach().cpu()
    final_delta = final_delta.detach().cpu()
    final_shift = final_box_shift.detach().cpu()
    finite_best = best[torch.isfinite(best)]
    finite_valid_best = valid_best[torch.isfinite(valid_best)]
    finite_shift = shift[torch.isfinite(shift)]
    finite_final = final[torch.isfinite(final)]
    finite_final_shift = final_shift[torch.isfinite(final_shift)]
    print("checkpoint", args.checkpoint)
    print("env_id", args.env_id)
    print("num_envs", args.num_envs)
    print("num_steps", args.num_steps)
    print("finite", int(finite_best.numel()), "/", int(best.numel()))
    print("mean", float(finite_best.mean()))
    print("median", float(finite_best.median()))
    print("min", float(finite_best.min()))
    print("max", float(finite_best.max()))
    print("best_delta_mean", [float(v) for v in delta.mean(dim=0)])
    print("best_abs_delta_mean", [float(v) for v in delta.abs().mean(dim=0)])
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
    print("valid_box_shift_threshold", args.valid_box_shift)
    print("valid_finite", int(finite_valid_best.numel()), "/", int(valid_best.numel()))
    if finite_valid_best.numel() > 0:
        finite_valid_delta = valid_delta[torch.isfinite(valid_best)]
        print("valid_mean", float(finite_valid_best.mean()))
        print("valid_median", float(finite_valid_best.median()))
        print("valid_min", float(finite_valid_best.min()))
        print("valid_max", float(finite_valid_best.max()))
        print("valid_best_delta_mean", [float(v) for v in finite_valid_delta.mean(dim=0)])
        print(
            "valid_best_abs_delta_mean",
            [float(v) for v in finite_valid_delta.abs().mean(dim=0)],
        )
        for threshold in (0.005, 0.01, 0.02, 0.03):
            print(
                f"valid_under_{threshold:.3f}",
                int((finite_valid_best < threshold).sum()),
                "/",
                int(finite_valid_best.numel()),
            )
    print("final_mean", float(finite_final.mean()))
    print("final_median", float(finite_final.median()))
    print("final_min", float(finite_final.min()))
    print("final_max", float(finite_final.max()))
    print("final_delta_mean", [float(v) for v in final_delta.mean(dim=0)])
    print("final_abs_delta_mean", [float(v) for v in final_delta.abs().mean(dim=0)])
    print("final_box_shift_mean", float(finite_final_shift.mean()))
    print("final_box_shift_median", float(finite_final_shift.median()))
    print("final_box_shift_max", float(finite_final_shift.max()))
    final_valid = (final_shift <= args.valid_box_shift) & torch.isfinite(final)
    final_valid_distance = final[final_valid]
    print("final_valid_finite", int(final_valid_distance.numel()), "/", int(final.numel()))
    if final_valid_distance.numel() > 0:
        print("final_valid_mean", float(final_valid_distance.mean()))
        print("final_valid_median", float(final_valid_distance.median()))
        for threshold in (0.005, 0.01, 0.02, 0.03):
            print(
                f"final_valid_under_{threshold:.3f}",
                int((final_valid_distance < threshold).sum()),
                "/",
                int(final_valid_distance.numel()),
            )

    env.close()


if __name__ == "__main__":
    main()
