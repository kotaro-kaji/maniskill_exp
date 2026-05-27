import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import tasks.single.task_single_cardboard_cabinet  # noqa: F401
from ppo_dual_xarm7 import Agent


def camera_config(camera_name: str):
    if camera_name == "slot_close":
        return {
            "render_camera": {
                "pose": sapien_utils.look_at(
                    eye=[-0.43, -0.43, 0.17],
                    target=[-0.315, -0.15, 0.105],
                ),
                "width": 960,
                "height": 960,
                "fov": 0.72,
            }
        }
    if camera_name == "slot_side":
        return {
            "render_camera": {
                "pose": sapien_utils.look_at(
                    eye=[-0.20, -0.34, 0.18],
                    target=[-0.315, -0.15, 0.105],
                ),
                "width": 960,
                "height": 960,
                "fov": 0.78,
            }
        }
    if camera_name == "slot_open":
        return {
            "render_camera": {
                "pose": sapien_utils.look_at(
                    eye=[-0.44, 0.42, 0.15],
                    target=[-0.315, 0.156, 0.105],
                ),
                "width": 960,
                "height": 960,
                "fov": 0.58,
            }
        }
    if camera_name == "slot_open_low":
        return {
            "render_camera": {
                "pose": sapien_utils.look_at(
                    eye=[-0.40, 0.34, 0.115],
                    target=[-0.315, 0.156, 0.100],
                ),
                "width": 960,
                "height": 960,
                "fov": 0.52,
            }
        }
    if camera_name == "slot_open_wide":
        return {
            "render_camera": {
                "pose": sapien_utils.look_at(
                    eye=[-0.52, 0.50, 0.19],
                    target=[-0.315, 0.156, 0.105],
                ),
                "width": 960,
                "height": 960,
                "fov": 0.74,
            }
        }
    if camera_name == "slot_headon":
        return {
            "render_camera": {
                "pose": sapien_utils.look_at(
                    eye=[-0.315, 0.58, 0.112],
                    target=[-0.315, 0.135, 0.104],
                ),
                "width": 960,
                "height": 960,
                "fov": 0.50,
            }
        }
    if camera_name == "slot_headon_high":
        return {
            "render_camera": {
                "pose": sapien_utils.look_at(
                    eye=[-0.335, 0.56, 0.145],
                    target=[-0.315, 0.135, 0.105],
                ),
                "width": 960,
                "height": 960,
                "fov": 0.54,
            }
        }
    if camera_name == "slot_peek_right":
        return {
            "render_camera": {
                "pose": sapien_utils.look_at(
                    eye=[-0.18, 0.45, 0.135],
                    target=[-0.315, 0.145, 0.104],
                ),
                "width": 960,
                "height": 960,
                "fov": 0.54,
            }
        }
    if camera_name == "slot_peek_left":
        return {
            "render_camera": {
                "pose": sapien_utils.look_at(
                    eye=[-0.51, 0.43, 0.13],
                    target=[-0.315, 0.145, 0.104],
                ),
                "width": 960,
                "height": 960,
                "fov": 0.54,
            }
        }
    assert camera_name == "default", camera_name
    return {"render_camera": {"width": 960, "height": 960}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--robot-init-noise-scale", type=float, default=0.0)
    parser.add_argument(
        "--camera",
        choices=(
            "default",
            "slot_close",
            "slot_side",
            "slot_open",
            "slot_open_low",
            "slot_open_wide",
            "slot_headon",
            "slot_headon_high",
            "slot_peek_right",
            "slot_peek_left",
        ),
        default="slot_close",
    )
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
        human_render_camera_configs=camera_config(args.camera),
        sim_config=dict(
            gpu_memory_config=dict(
                max_rigid_contact_count=2**22,
                max_rigid_patch_count=2**21,
            )
        ),
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = RecordEpisode(
        env,
        output_dir=args.output_dir,
        save_trajectory=False,
        trajectory_name="rollout",
        max_steps_per_video=args.num_steps,
        video_fps=20,
    )
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    obs, _ = env.reset(seed=args.seed)
    best_distance = float("inf")
    for _ in range(args.num_steps):
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        obs, _, _, _, info = env.step(torch.clamp(action, -1.0, 1.0))
        best_distance = min(best_distance, float(info["tcp_to_target_distance"][0]))

    env.close()
    print("best_tcp_to_target_distance", best_distance)
    print("output_dir", args.output_dir)


if __name__ == "__main__":
    main()
