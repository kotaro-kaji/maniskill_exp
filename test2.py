import argparse
import importlib
from typing import Dict, Tuple

import gymnasium as gym
import mani_skill.envs  # registers built-in envs
import robotagents.my_xarm7  # registers custom robots
import robotagents.my_xarm7_mjcf
import robotagents.my_xarm7_official
import robotagents.my_xarm7_wo_gripper
import robotagents.xarm_ball_ee
import sapien
import sapien.render

# Import custom task modules so their envs become available.
CUSTOM_ENV_MODULES: Tuple[str, ...] = (
    "task_pushcube_beatiful",
    "task_joint_hold",
    "task_marker_align_official",
    "tasks.du.task_dual_simple",
)
for _module_name in CUSTOM_ENV_MODULES:
    importlib.import_module(_module_name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-id",
        type=str,
        default="MyDualSimple-v0",
        help="Environment ID to visualize.",
    )
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="List imported custom env IDs and exit.",
    )
    parser.add_argument("--obs-mode", type=str, default="state")
    parser.add_argument("--control-mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--sim-backend", type=str, default="cpu")
    parser.add_argument("--render-backend", type=str, default="cpu")
    parser.add_argument("--robot-uid", type=str, default="xarm7_ball_ee")
    return parser.parse_args()


def add_frame(
    scene,
    pose: sapien.Pose,
    size: float = 0.1,
    thickness: float = 0.004,
    name: str = "debug_frame",
):
    """Draw a small XYZ frame using three colored box visuals."""

    def make_mat(color):
        mat = sapien.render.RenderMaterial(base_color=color + [1.0])
        mat.roughness = 0.2
        return mat

    builder = scene.create_actor_builder()
    half = size * 0.5
    builder.add_box_visual(
        half_size=[half, thickness, thickness],
        pose=sapien.Pose([half, 0, 0]),
        material=make_mat([1.0, 0.0, 0.0]),  # X axis
    )
    builder.add_box_visual(
        half_size=[thickness, half, thickness],
        pose=sapien.Pose([0, half, 0]),
        material=make_mat([0.0, 1.0, 0.0]),  # Y axis
    )
    builder.add_box_visual(
        half_size=[thickness, thickness, half],
        pose=sapien.Pose([0, 0, half]),
        material=make_mat([0.0, 0.0, 1.0]),  # Z axis
    )
    builder.initial_pose = pose
    frame = builder.build_static(name=name)
    return frame


def main():
    args = parse_args()
    if args.list_envs:
        print("Custom environment modules imported:")
        for name in CUSTOM_ENV_MODULES:
            print(f"  - {name}")
        print("Try one of these env IDs (if registered):")
        print(
            "  MyPushCube-v1, MyJointHold-v0, MyEEAlignMarker-v0, MyDualSimple-v0, Empty-v1"
        )
        return

    env_kwargs: Dict[str, object] = dict(
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode="human",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
    )
    if args.robot_uid:
        env_kwargs["robot_uids"] = args.robot_uid

    try:
        env = gym.make(args.env_id, **env_kwargs)
    except TypeError:
        # Some envs may not accept robot_uids/control_mode.
        fallback_kwargs = dict(env_kwargs)
        fallback_kwargs.pop("robot_uids", None)
        try:
            env = gym.make(args.env_id, **fallback_kwargs)
        except Exception as exc:
            raise RuntimeError(f"Failed to create env '{args.env_id}': {exc}") from exc

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    obs, _ = env.reset(seed=0)
    scene = env.unwrapped.scene
    add_frame(scene, sapien.Pose(), size=0.15, name="world_frame")
    agent = getattr(env.unwrapped, "agent", None)
    if agent is not None:
        add_frame(scene, agent.robot.pose, size=0.1, name="robot_base_frame")

    done = False
    while not done:
        #action = env.action_space.sample()
        #obs, reward, terminated, truncated, info = env.step(action)
        #done = terminated or truncated
        env.render()
    env.close()


if __name__ == "__main__":
    main()
