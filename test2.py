import argparse

import gymnasium as gym
import mani_skill.envs  # registers built-in envs
import robotagents.my_xarm7_official  # registers your custom robot
import sapien
import sapien.render
import task_marker_align_official  # registers MyPushCube-v1 from your file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs-mode", type=str, default="state")
    parser.add_argument("--control-mode", type=str, default="pd_joint_delta_pos")
    parser.add_argument("--sim-backend", type=str, default="cpu")
    parser.add_argument("--render-backend", type=str, default="cpu")
    parser.add_argument("--robot-uid", type=str, default="my_xarm7_official")
    return parser.parse_args()


def add_frame(scene, pose: sapien.Pose, size: float = 0.1, thickness: float = 0.004):
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
    frame = builder.build_static(name="debug_frame")
    frame.set_pose(pose)
    return frame


def main():
    args = parse_args()

    env = gym.make(
        "MyEEAlignMarker-v0",
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        robot_uids=args.robot_uid,
        render_mode="human",
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,  # force CPU rendering
    )

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    obs, _ = env.reset(seed=0)
    scene = env.unwrapped.scene
    add_frame(scene, sapien.Pose(), size=0.15)
    agent = getattr(env.unwrapped, "agent", None)
    if agent is not None:
        add_frame(scene, agent.robot.pose, size=0.1)

    done = False
    while not done:
        #action = env.action_space.sample()
        #obs, reward, terminated, truncated, info = env.step(action)
        #done = terminated or truncated
        env.render()
    env.close()


if __name__ == "__main__":
    main()
