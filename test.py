import argparse

import gymnasium as gym
import mani_skill.envs  # registers envs
import my_xarm7  # registers your custom robot
from mani_skill.agents.controllers.base_controller import DictController
from mani_skill.envs.sapien_env import BaseEnv
import my_xarm7_wo_gripper
import my_xarm7_mjcf  # registers MJCF-based Xarm7


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--robot-uid", type=str, default="panda")
    parser.add_argument("-b", "--sim-backend", type=str, default="cpu")
    parser.add_argument("-c", "--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument("-k", "--keyframe", type=str)
    parser.add_argument("--keyframe-actions", action="store_true")
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--none-actions", action="store_true")
    parser.add_argument("--zero-actions", action="store_true")
    parser.add_argument("--shader", default="default", type=str)
    parser.add_argument("--sim-freq", type=int, default=100)
    parser.add_argument("--control-freq", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make(
        "Empty-v1",
        obs_mode="none",
        reward_mode="none",
        enable_shadow=True,
        control_mode=args.control_mode,
        robot_uids=args.robot_uid,
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
        viewer_camera_configs=dict(shader_pack=args.shader),
        render_mode="human",
        sim_config=dict(sim_freq=args.sim_freq, control_freq=args.control_freq),
        sim_backend=args.sim_backend,
        render_backend="cpu",  # Force CPU rendering to avoid CUDA requirement
    )
    env.reset(seed=0)
    env: BaseEnv = env.unwrapped

    print(f"Selected robot {args.robot_uid}. Control mode: {args.control_mode}")
    print("Selected Robot has the following keyframes to view:")
    print(env.agent.keyframes.keys())

    # Reset base pose
    env.agent.robot.set_qpos(env.agent.robot.qpos * 0)

    # Load first or specified keyframe if available
    kf = None
    if len(env.agent.keyframes) > 0:
        kf_name = None
        if args.keyframe is not None:
            kf_name = args.keyframe
            kf = env.agent.keyframes[kf_name]
        else:
            for kf_name, kf in env.agent.keyframes.items():
                break
        if kf.qpos is not None:
            env.agent.robot.set_qpos(kf.qpos)
            env.agent.controller.reset()
        if kf.qvel is not None:
            env.agent.robot.set_qvel(kf.qvel)
        env.agent.robot.set_pose(kf.pose)
        if kf_name is not None:
            print(f"Viewing keyframe {kf_name}")

    viewer = env.render()
    viewer.paused = True
    viewer = env.render()
    while True:
        if args.random_actions:
            env.step(env.action_space.sample())
        elif args.none_actions:
            env.step(None)
        elif args.zero_actions:
            env.step(env.action_space.sample() * 0)
        elif args.keyframe_actions:
            assert kf is not None, "this robot has no keyframes, cannot use it to set actions"
            if isinstance(env.agent.controller, DictController):
                env.step(env.agent.controller.from_qpos(kf.qpos))
            else:
                env.step(kf.qpos)
        viewer = env.render()


if __name__ == "__main__":
    main()
