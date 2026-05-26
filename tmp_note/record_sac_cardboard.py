import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from record_ppo_cardboard import camera_config
from sac import Actor

import tasks.dual.task_dual_cardboard_cabinet  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--robot-init-noise-scale", type=float, default=0.0)
    parser.add_argument("--camera", default="slot_close")
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
    actor = Actor(env).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    obs, _ = env.reset(seed=args.seed)
    best_distance = float("inf")
    for _ in range(args.num_steps):
        with torch.no_grad():
            action = actor.get_eval_action(obs)
        obs, _, _, _, info = env.step(torch.clamp(action, -1.0, 1.0))
        best_distance = min(best_distance, float(info["tcp_to_target_distance"][0]))

    env.close()
    print("best_tcp_to_target_distance", best_distance)
    print("output_dir", args.output_dir)


if __name__ == "__main__":
    main()
