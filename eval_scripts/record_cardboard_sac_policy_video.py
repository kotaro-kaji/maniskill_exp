import argparse
import sys
from pathlib import Path

import gymnasium as gym
import torch

cwd = Path.cwd().resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if (cwd / "sac.py").exists():
    sys.path.insert(0, str(cwd))

from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode

import tasks.single.task_single_cardboard_cabinet  # noqa: F401
from sac import Actor


def camera_config(camera_name: str):
    if camera_name == "overview":
        return {
            "render_camera": {
                "pose": sapien_utils.look_at([1.2, 1.0, 0.9], [0.0, 0.0, 0.22]),
                "width": 960,
                "height": 960,
                "fov": 1.0,
            }
        }
    assert camera_name == "default", camera_name
    return {"render_camera": {"width": 960, "height": 960}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="MyDualCardboardCabinet-v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--sim-backend", default="physx_cuda")
    parser.add_argument("--camera", choices=("overview", "default"), default="overview")
    args = parser.parse_args()

    env = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        robot_init_noise_scale=0.0,
        reconfiguration_freq=1,
        human_render_camera_configs=camera_config(args.camera),
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(env).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    obs, _ = env.reset(seed=[args.seed + i for i in range(args.num_envs)])
    obs = obs.to(device)
    episode_return = torch.zeros(args.num_envs, device=device)
    max_open = torch.zeros(args.num_envs, device=device)
    max_shift = torch.zeros(args.num_envs, device=device)
    final_info = None

    for _ in range(args.num_steps):
        with torch.no_grad():
            action = actor.get_eval_action(obs)
        obs, reward, _, _, info = env.step(action)
        obs = obs.to(device)
        episode_return += reward.reshape(-1).to(device)
        open_amount = info["inner_box_open_amount"].reshape(-1).to(device)
        outer_shift = info["outer_box_shift"].reshape(-1).to(device)
        max_open = torch.maximum(max_open, open_amount)
        max_shift = torch.maximum(max_shift, outer_shift)
        final_info = info

    assert final_info is not None
    final_open = final_info["inner_box_open_amount"].reshape(-1).to(device)
    final_shift = final_info["outer_box_shift"].reshape(-1).to(device)
    reached_open_success = (max_open >= 0.12) & (max_shift <= 0.02)
    final_open_success = (final_open >= 0.12) & (final_shift <= 0.02)

    env.close()

    print(f"num_envs {args.num_envs}")
    print(f"episode_return_mean {episode_return.mean().item():.6f}")
    print(f"max_open_mean {max_open.mean().item():.6f}")
    print(f"max_shift_mean {max_shift.mean().item():.6f}")
    print(f"final_open_mean {final_open.mean().item():.6f}")
    print(f"final_shift_mean {final_shift.mean().item():.6f}")
    print(f"reached_open_success_mean {reached_open_success.float().mean().item():.6f}")
    print(f"final_open_success_mean {final_open_success.float().mean().item():.6f}")
    print(f"output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
