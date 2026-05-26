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
    parser.add_argument("--env-id", default="MyDualCardboardCabinet-v0")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--robot-init-noise-scale", type=float, default=1.0)
    parser.add_argument("--valid-box-shift", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=1)
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
    env = ManiSkillVectorEnv(
        env, args.num_envs, ignore_terminations=True, record_metrics=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    obs, _ = env.reset(seed=args.seed)
    best_distance = torch.full((args.num_envs,), float("inf"), device=device)
    best_delta = torch.zeros((args.num_envs, 3), device=device)
    best_axis_alignment = torch.zeros((args.num_envs, 3), device=device)
    best_eef_x_dot = torch.zeros((args.num_envs,), device=device)
    best_eef_x_error = torch.zeros((args.num_envs,), device=device)
    best_eef_x_roll = torch.zeros((args.num_envs,), device=device)
    best_gripper_qpos = torch.zeros((args.num_envs,), device=device)
    valid_best_distance = torch.full((args.num_envs,), float("inf"), device=device)
    valid_best_delta = torch.zeros((args.num_envs, 3), device=device)
    valid_best_axis_alignment = torch.zeros((args.num_envs, 3), device=device)
    max_box_shift = torch.zeros((args.num_envs,), device=device)
    final_distance = torch.full((args.num_envs,), float("inf"), device=device)
    final_delta = torch.zeros((args.num_envs, 3), device=device)
    final_box_shift = torch.zeros((args.num_envs,), device=device)
    valid_close_streak = torch.zeros((args.num_envs,), dtype=torch.long, device=device)
    max_valid_close_streak = torch.zeros(
        (args.num_envs,), dtype=torch.long, device=device
    )

    for _ in range(args.num_steps):
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
        distance = info["insert_marker_distance"].to(device)
        box_shift = info["box_position_shift"].to(device)
        axis_alignment = info["finger_axis_alignment"].to(device)
        eef_x_dot = info["eef_x_axis_world"].to(device)[:, 0]
        eef_x_error = info["eef_x_world_error"].to(device)
        eef_x_roll = info["eef_x_roll_world"].to(device)
        gripper_qpos = info["gripper_drive_qpos"].to(device)
        obs_delta = obs[:, -4:-1]
        final_distance = distance
        final_delta = obs_delta
        final_box_shift = box_shift
        improved = distance < best_distance
        best_distance = torch.minimum(best_distance, distance)
        best_delta[improved] = obs_delta[improved]
        best_axis_alignment[improved] = axis_alignment[improved]
        best_eef_x_dot[improved] = eef_x_dot[improved]
        best_eef_x_error[improved] = eef_x_error[improved]
        best_eef_x_roll[improved] = eef_x_roll[improved]
        best_gripper_qpos[improved] = gripper_qpos[improved]
        max_box_shift = torch.maximum(max_box_shift, box_shift)
        valid = max_box_shift <= args.valid_box_shift
        valid_close = valid & (distance < 0.005)
        valid_close_streak = torch.where(
            valid_close,
            valid_close_streak + 1,
            torch.zeros_like(valid_close_streak),
        )
        max_valid_close_streak = torch.maximum(
            max_valid_close_streak,
            valid_close_streak,
        )
        valid_improved = valid & (distance < valid_best_distance)
        valid_best_distance = torch.where(valid_improved, distance, valid_best_distance)
        valid_best_delta[valid_improved] = obs_delta[valid_improved]
        valid_best_axis_alignment[valid_improved] = axis_alignment[valid_improved]

    best = best_distance.detach().cpu()
    delta = best_delta.detach().cpu()
    axis = best_axis_alignment.detach().cpu()
    eef_x_dot = best_eef_x_dot.detach().cpu()
    eef_x_error = best_eef_x_error.detach().cpu()
    eef_x_roll = torch.rad2deg(best_eef_x_roll.detach().cpu())
    gripper_qpos = best_gripper_qpos.detach().cpu()
    valid_best = valid_best_distance.detach().cpu()
    valid_delta = valid_best_delta.detach().cpu()
    valid_axis = valid_best_axis_alignment.detach().cpu()
    shift = max_box_shift.detach().cpu()
    final = final_distance.detach().cpu()
    final_delta = final_delta.detach().cpu()
    final_shift = final_box_shift.detach().cpu()
    streak = max_valid_close_streak.detach().cpu()
    finite_best = best[torch.isfinite(best)]
    finite_valid_best = valid_best[torch.isfinite(valid_best)]
    finite_shift = shift[torch.isfinite(shift)]
    finite_final = final[torch.isfinite(final)]
    finite_final_shift = final_shift[torch.isfinite(final_shift)]
    print("checkpoint", args.checkpoint)
    print("env_id", args.env_id)
    print("num_envs", args.num_envs)
    print("num_steps", args.num_steps)
    print("seed", args.seed)
    print("finite", int(finite_best.numel()), "/", int(best.numel()))
    print("mean", float(finite_best.mean()))
    print("median", float(finite_best.median()))
    print("min", float(finite_best.min()))
    print("max", float(finite_best.max()))
    print("best_delta_mean", [float(v) for v in delta.mean(dim=0)])
    print("best_abs_delta_mean", [float(v) for v in delta.abs().mean(dim=0)])
    print("best_abs_delta_median", [float(v) for v in delta.abs().median(dim=0).values])
    print("best_axis_alignment_mean", [float(v) for v in axis.mean(dim=0)])
    print("best_abs_axis_alignment_mean", [float(v) for v in axis.abs().mean(dim=0)])
    print("best_eef_x_dot_mean", float(eef_x_dot.mean()))
    print("best_eef_x_error_mean", float(eef_x_error.mean()))
    print("best_eef_x_roll_deg_mean", float(eef_x_roll.mean()))
    print("best_eef_x_roll_deg_abs_mean", float(eef_x_roll.abs().mean()))
    print("best_gripper_qpos_mean", float(gripper_qpos.mean()))
    print("max_box_shift_mean", float(finite_shift.mean()))
    print("max_box_shift_median", float(finite_shift.median()))
    print("max_box_shift_max", float(finite_shift.max()))
    for threshold in (0.003, 0.005, 0.01, 0.02, 0.03):
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
        finite_valid_axis = valid_axis[torch.isfinite(valid_best)]
        print("valid_mean", float(finite_valid_best.mean()))
        print("valid_median", float(finite_valid_best.median()))
        print("valid_min", float(finite_valid_best.min()))
        print("valid_max", float(finite_valid_best.max()))
        print("valid_best_delta_mean", [float(v) for v in finite_valid_delta.mean(dim=0)])
        print(
            "valid_best_abs_delta_mean",
            [float(v) for v in finite_valid_delta.abs().mean(dim=0)],
        )
        print(
            "valid_best_axis_alignment_mean",
            [float(v) for v in finite_valid_axis.mean(dim=0)],
        )
        print(
            "valid_best_abs_axis_alignment_mean",
            [float(v) for v in finite_valid_axis.abs().mean(dim=0)],
        )
        for threshold in (0.003, 0.005, 0.01, 0.02, 0.03):
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
    print(
        "final_abs_delta_median",
        [float(v) for v in final_delta.abs().median(dim=0).values],
    )
    print("final_box_shift_mean", float(finite_final_shift.mean()))
    print("final_box_shift_median", float(finite_final_shift.median()))
    print("final_box_shift_max", float(finite_final_shift.max()))
    final_valid = (final_shift <= args.valid_box_shift) & torch.isfinite(final)
    final_valid_distance = final[final_valid]
    print("final_valid_finite", int(final_valid_distance.numel()), "/", int(final.numel()))
    if final_valid_distance.numel() > 0:
        print("final_valid_mean", float(final_valid_distance.mean()))
        print("final_valid_median", float(final_valid_distance.median()))
        for threshold in (0.003, 0.005, 0.01, 0.02, 0.03):
            print(
                f"final_valid_under_{threshold:.3f}",
                int((final_valid_distance < threshold).sum()),
                "/",
                int(final_valid_distance.numel()),
            )
    print("max_valid_close_streak_mean", float(streak.float().mean()))
    print("max_valid_close_streak_median", float(streak.float().median()))
    for steps in (1, 3, 5, 10):
        print(
            f"valid_close_streak_ge_{steps}",
            int((streak >= steps).sum()),
            "/",
            int(streak.numel()),
        )

    env.close()


if __name__ == "__main__":
    main()
