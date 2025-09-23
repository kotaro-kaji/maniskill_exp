import gymnasium as gym
import mani_skill.envs  # registers built-in envs
import my_xarm7  # registers your custom robot
import task_pushcube_beatiful  # registers MyPushCube-v1 from your file
import my_xarm7_mjcf
import task_joint_hold
import numpy as np


def main():
    env = gym.make(
        "MyJointHold-v0",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        robot_uids="my_xarm7",
        render_mode="human",
        sim_backend="gpu",
        render_backend="gpu",
    )

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    obs, _ = env.reset(seed=0)
    done = False

    # Keep arm still (zeros) and smoothly open/close the gripper with small deltas.
    # Action space is normalized [-1, 1] for pd_joint_delta_pos, so choose a small step.
    step_norm = 0.2  # normalized delta; maps to ~0.01 rad with [-0.05,0.05]
    direction = -1.0  # + open, - close
    # RoboManipBaselines準拠のレンジ（実効関節角）: 約0.0〜0.85 [rad]
    gripper_min, gripper_max = 0.0, 0.85

    while not done:
        # Read current gripper driver joint position to flip direction at bounds
        
        action = np.zeros(8, dtype=np.float32)
        #action[-1] = direction * step_norm  # only gripper moves; arm deltas are zero

        # Clip to valid bounds (normalized [-1,1])
        low, high = env.single_action_space.low, env.single_action_space.high
        action = np.clip(action, low, high)

        obs, reward, terminated, truncated, info = env.step(action)
        #done = terminated or truncated
        env.render()
    env.close()


if __name__ == "__main__":
    main()
