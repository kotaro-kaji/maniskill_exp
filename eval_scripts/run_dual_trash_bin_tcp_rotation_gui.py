import math
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tasks.dual.task_dual_trash_bin_rolling  # noqa: F401
from mani_skill.utils.structs.pose import Pose


def quat_wxyz_to_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_rotvec(R):
    cos_angle = (np.trace(R) - 1.0) / 2.0
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = math.acos(cos_angle)
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    )
    axis = axis / (2.0 * math.sin(angle))
    return axis * angle


def tcp_rotation(agent, device):
    pose = Pose.create(agent.tcp_pose, device=device)
    quat = pose.q.detach().cpu().numpy()[0]
    return quat_wxyz_to_matrix(quat)


def render_or_stop(env):
    try:
        env.render()
        return True
    except AttributeError as exc:
        if "should_close" in str(exc):
            return False
        raise


target_angle_rad = math.radians(35.0)
c = math.cos(target_angle_rad)
s = math.sin(target_angle_rad)
target_left = np.array(
    [
        [0.0, 0.0, 1.0],
        [-c, -s, 0.0],
        [s, -c, 0.0],
    ],
    dtype=np.float64,
)
target_right = np.array(
    [
        [0.0, 0.0, 1.0],
        [-c, s, 0.0],
        [-s, -c, 0.0],
    ],
    dtype=np.float64,
)

env = gym.make(
    "MyDualTrashBinRolling-v0",
    num_envs=1,
    obs_mode="state",
    render_mode="human",
    control_mode="pd_ee_delta_pose",
    robot_init_noise_scale=0.0,
)
env.reset(seed=0, options={"robot_init_noise_scale": 0.0})
render_or_stop(env)

keys = list(env.action_space.keys())
agents = env.unwrapped.agent.agents
targets = [target_left, target_right]
max_rot_step = 1.5e-2

for step in range(500):
    action = {}
    for key, agent, target_R in zip(keys, agents, targets):
        current_R = tcp_rotation(agent, env.unwrapped.device)
        error_R = target_R @ current_R.T
        rotvec = matrix_to_rotvec(error_R)
        command = np.zeros(7, dtype=np.float32)
        command[3:6] = -np.clip(rotvec / max_rot_step, -1.0, 1.0)
        action[key] = command
    env.step(action)
    if not render_or_stop(env):
        break
    time.sleep(1.0 / 60.0)

for _ in range(600):
    if not render_or_stop(env):
        break
    time.sleep(1.0 / 60.0)

env.close()
