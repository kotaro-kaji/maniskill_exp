import argparse
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import torch

cwd = Path.cwd().resolve()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if (cwd / "ppo_dual_xarm7.py").exists():
    sys.path.insert(0, str(cwd))

from mani_skill.utils import sapien_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import tasks.single.task_single_cardboard_cabinet as cardboard_envs
from ppo_dual_xarm7 import Agent


def camera_config(camera_name: str, width: int, height: int):
    if camera_name == "slot_open":
        return {
            "render_camera": {
                "pose": sapien_utils.look_at(
                    eye=[-0.44, 0.42, 0.15],
                    target=[-0.315, 0.156, 0.105],
                ),
                "width": width,
                "height": height,
                "fov": 0.58,
            }
        }
    if camera_name == "overview":
        return {
            "render_camera": {
                "pose": sapien_utils.look_at([1.2, 1.0, 0.9], [0.0, 0.0, 0.22]),
                "width": width,
                "height": height,
                "fov": 1.0,
            }
        }
    assert camera_name == "default", camera_name
    return {"render_camera": {"width": width, "height": height}}


def parse_env_indices(value: str | None) -> list[int]:
    if value is None:
        return []
    indices = []
    for token in value.split(","):
        token = token.strip()
        if token:
            indices.append(int(token))
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env-id", default="MyDualCardboardCabinet-v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--robot-uid", default=None)
    parser.add_argument("--sim-backend", default="physx_cuda")
    parser.add_argument("--robot-init-noise-scale", type=float, default=0.05)
    parser.add_argument("--camera", choices=("slot_open", "overview", "default"), default="slot_open")
    parser.add_argument("--record-env-index", type=int, default=None)
    parser.add_argument("--record-env-indices", default=None)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=960)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--disable-panel-obs-step-noise", action="store_true")
    args = parser.parse_args()
    assert not (
        args.record_env_index is not None and args.record_env_indices is not None
    )
    record_env_indices = parse_env_indices(args.record_env_indices)
    if args.record_env_index is not None:
        record_env_indices = [args.record_env_index]
    for env_idx in record_env_indices:
        assert 0 <= env_idx < args.num_envs, (env_idx, args.num_envs)
    if args.disable_panel_obs_step_noise:
        cardboard_envs.MySingleCardboardCabinetRandomizedEnv.PANEL_OBS_POSITION_STEP_NOISE_M = 0.0

    env = gym.make(
        args.env_id,
        num_envs=args.num_envs,
        obs_mode="state",
        render_mode=None if args.skip_video else "rgb_array",
        sim_backend=args.sim_backend,
        control_mode=args.control_mode,
        robot_init_noise_scale=args.robot_init_noise_scale,
        robot_uids=args.robot_uid if args.robot_uid is not None else "my_xarm7",
        reconfiguration_freq=1,
        human_render_camera_configs=camera_config(args.camera, args.render_width, args.render_height),
    )

    writers = {}
    if record_env_indices and not args.skip_video:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        for env_idx in record_env_indices:
            writers[env_idx] = imageio.get_writer(
                str(Path(args.output_dir) / f"env_{env_idx}.mp4"),
                fps=args.video_fps,
                quality=5,
            )

    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    if not record_env_indices and not args.skip_video:
        env = RecordEpisode(
            env,
            output_dir=args.output_dir,
            save_trajectory=False,
            trajectory_name="rollout",
            max_steps_per_video=args.num_steps,
            video_fps=args.video_fps,
        )
    env = ManiSkillVectorEnv(
        env,
        args.num_envs,
        ignore_terminations=True,
        record_metrics=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    action_low = torch.as_tensor(env.single_action_space.low, device=device)
    action_high = torch.as_tensor(env.single_action_space.high, device=device)

    obs, _ = env.reset(seed=[args.seed + i for i in range(args.num_envs)])
    obs = obs.to(device)

    def record_selected_env_frame():
        if not record_env_indices or args.skip_video:
            return
        images = env.base_env.render()
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        for env_idx in record_env_indices:
            writers[env_idx].append_data(images[env_idx])

    record_selected_env_frame()
    episode_return = torch.zeros(args.num_envs, device=device)
    max_open = torch.zeros(args.num_envs, device=device)
    max_shift = torch.zeros(args.num_envs, device=device)
    final_info = None

    for _ in range(args.num_steps):
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        action = torch.clamp(action, action_low, action_high)
        obs, reward, _, _, info = env.step(action)
        obs = obs.to(device)
        record_selected_env_frame()
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

    for writer in writers.values():
        writer.close()

    env.close()

    print(f"num_envs {args.num_envs}")
    print(f"robot_uid {args.robot_uid if args.robot_uid is not None else 'my_xarm7'}")
    print(f"panel_obs_step_noise_m {cardboard_envs.MySingleCardboardCabinetRandomizedEnv.PANEL_OBS_POSITION_STEP_NOISE_M:.6f}")
    print(f"episode_return_mean {episode_return.mean().item():.6f}")
    print(f"max_open_mean {max_open.mean().item():.6f}")
    print(f"max_shift_mean {max_shift.mean().item():.6f}")
    print(f"final_open_mean {final_open.mean().item():.6f}")
    print(f"final_shift_mean {final_shift.mean().item():.6f}")
    print(f"reached_open_success_mean {reached_open_success.float().mean().item():.6f}")
    print(f"final_open_success_mean {final_open_success.float().mean().item():.6f}")
    env_indices_to_print = record_env_indices if record_env_indices else range(args.num_envs)
    for env_idx in env_indices_to_print:
        print(
            "env "
            f"{env_idx} return={episode_return[env_idx].item():.6f} "
            f"max_open={max_open[env_idx].item():.6f} "
            f"final_open={final_open[env_idx].item():.6f} "
            f"final_shift={final_shift[env_idx].item():.6f} "
            f"final_open_success={bool(final_open_success[env_idx].item())}"
        )
    print(f"output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
