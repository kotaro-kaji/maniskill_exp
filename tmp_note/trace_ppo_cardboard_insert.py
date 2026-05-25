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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--num-steps", type=int, default=400)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--robot-init-noise-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        num_envs=1,
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
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    obs, _ = env.reset(seed=args.seed)
    base_env = env.base_env.unwrapped
    initial_box_position = base_env._initial_inner_box_world_position().detach().clone()
    best = None
    max_shift = None
    rows = []
    for step in range(args.num_steps):
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        obs, _, _, _, info = env.step(action)
        distance = float(info["insert_marker_distance"][0])
        box_shift = float(info["box_position_shift"][0])
        box_position = base_env.cardboard_inner_box.pose.p[0].detach()
        box_delta = (box_position - initial_box_position).detach().cpu()
        delta = obs[0, -4:-1].detach().cpu()
        action_cpu = action[0].detach().cpu()
        eef_x_dot = float(info["eef_x_axis_world"][0, 0])
        gripper_qpos = float(info["gripper_drive_qpos"][0])
        row = (
            step,
            distance,
            float(delta[0]),
            float(delta[1]),
            float(delta[2]),
            box_shift,
            float(box_delta[0]),
            float(box_delta[1]),
            float(box_delta[2]),
            float(action_cpu.abs().mean()),
            float(action_cpu.abs().max()),
            eef_x_dot,
            gripper_qpos,
            float(action_cpu[7]),
        )
        rows.append(row)
        if best is None or distance < best[1]:
            best = row
        if max_shift is None or box_shift > max_shift[5]:
            max_shift = row

    print("checkpoint", args.checkpoint)
    print("env_id", args.env_id)
    print("seed", args.seed)
    print(
        "columns step distance dx dy dz box_shift box_dx box_dy box_dz "
        "action_abs_mean action_abs_max eef_x_dot gripper_qpos left_gripper_action"
    )
    print("best", " ".join(str(v) for v in best))
    print("max_shift", " ".join(str(v) for v in max_shift))
    print("first", " ".join(str(v) for v in rows[0]))
    print("last", " ".join(str(v) for v in rows[-1]))
    for row in rows[::25]:
        print("trace", " ".join(str(v) for v in row))

    env.close()


if __name__ == "__main__":
    main()
