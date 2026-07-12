import argparse
from pathlib import Path
import sys

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tasks.dual.task_dual_box_rotation  # noqa: F401
from scenebuilders.xarm7_table_scene_builder import PEDESTAL_HEIGHT


BALL_RADIUS_M = 0.02
DEFAULT_ARM_QPOS_DEG = np.array(
    [-0.1, 4.7, -0.3, 21.6, -2.2, 17.0, 1.1],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", type=str, default="MyDualBoxRotation-v0")
    parser.add_argument(
        "--robot-uid",
        type=str,
        default="xarm7_ball_ee_kinematics_left",
    )
    parser.add_argument("--agent-index", type=int, default=0)
    parser.add_argument(
        "--arm-qpos-deg",
        type=float,
        nargs=7,
        default=DEFAULT_ARM_QPOS_DEG.tolist(),
    )
    parser.add_argument(
        "--ball-compression-m",
        type=float,
        default=0.0,
        help="Positive value means the real ball center is lower by this amount.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--render-mode",
        type=str,
        default=None,
        choices=[None, "human"],
        help="Use 'human' to open the Sapien viewer after setting the FK pose.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    robot_uids = (args.robot_uid, args.robot_uid)
    env = gym.make(
        args.env_id,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode=args.render_mode,
        robot_uids=robot_uids,
        robot_init_noise_scale=0.0,
    )

    try:
        env.reset(seed=args.seed)
        base_env = env.unwrapped
        agent = base_env.agent.agents[args.agent_index]

        qpos = agent.robot.get_qpos().clone()
        arm_qpos_rad = torch.tensor(
            np.deg2rad(np.asarray(args.arm_qpos_deg, dtype=np.float32)),
            device=qpos.device,
            dtype=qpos.dtype,
        )
        qpos[..., :7] = arm_qpos_rad
        agent.robot.set_qpos(qpos)

        if base_env.scene.gpu_sim_enabled:
            base_env.scene._gpu_apply_all()
            base_env.scene._gpu_fetch_all()

        tcp_pos = np.asarray(agent.tcp_pose.p.detach().cpu()).reshape(-1, 3)[0]
        ball_bottom_z = tcp_pos[2] - BALL_RADIUS_M
        target_tcp_center_z = BALL_RADIUS_M - args.ball_compression_m
        pedestal_delta = target_tcp_center_z - tcp_pos[2]
        calibrated_pedestal_height = PEDESTAL_HEIGHT + pedestal_delta

        print("FK input")
        print(f"  env_id: {args.env_id}")
        print(f"  robot_uid: {args.robot_uid}")
        print(f"  agent_index: {args.agent_index}")
        print(f"  arm_qpos_deg: {np.asarray(args.arm_qpos_deg)}")
        print(f"  arm_qpos_rad: {arm_qpos_rad.detach().cpu().numpy()}")
        print()
        print("Current FK result")
        print(f"  pedestal_height_m: {PEDESTAL_HEIGHT:.9f}")
        print(f"  tcp_center_xyz_m: {tcp_pos}")
        print(f"  ball_radius_m: {BALL_RADIUS_M:.9f}")
        print(f"  ball_bottom_z_m: {ball_bottom_z:.9f}")
        print()
        print("Pedestal calibration estimate")
        print(f"  target_tcp_center_z_m: {target_tcp_center_z:.9f}")
        print(f"  pedestal_delta_m: {pedestal_delta:.9f}")
        print(f"  calibrated_pedestal_height_m: {calibrated_pedestal_height:.9f}")

        if args.render_mode == "human":
            print()
            print("Rendering. Close the viewer or interrupt the command to stop.")
            while True:
                env.render()
    finally:
        env.close()


if __name__ == "__main__":
    main()
