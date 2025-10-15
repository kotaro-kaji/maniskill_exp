import os
import time
from typing import List, Optional

import gymnasium as gym
import mani_skill.envs  # registers built-in envs
import robotagents.my_xarm7_official  # registers custom URDF robot
import numpy as np
import task_marker_align_official

CONTROL_FILE = "control_delta_input.txt"
STATE_FILE = "joint_state_delta.txt"


def _write_control_file_template(
    path: str,
    arm_joint_names: List[str],
    arm_delta_limits: Optional[np.ndarray],
    gripper_delta_range: Optional[List[float]],
    current_action: np.ndarray,
):
    """Write a human-editable template describing delta commands."""

    lines = []
    lines.append("# XArm7 file control (pd_joint_delta_pos)")
    lines.append("# Edit the last line with 8 numbers then save. The robot applies the delta once per control step.")
    lines.append("# Format: 7 ARM joint DELTAS [rad], then 1 GRIPPER delta (rad).")
    lines.append("# Positive deltas increase joint angles; negative deltas decrease them.")
    lines.append("#")
    lines.append("# Arm joint delta limits (rad/step):")
    for i, name in enumerate(arm_joint_names):
        if arm_delta_limits is not None and i < len(arm_delta_limits):
            lo, hi = arm_delta_limits[i]
            lines.append(f"#   {i}: {name:>9s}  min={lo: .4f}  max={hi: .4f}")
        else:
            lines.append(f"#   {i}: {name:>9s}  min=unknown  max=unknown")
    lines.append("#")
    if gripper_delta_range is not None:
        lines.append(
            "# Gripper delta range: set in "
            f"[{gripper_delta_range[0]:.4f}, {gripper_delta_range[1]:.4f}] rad"
        )
        lines.append("# The drive joint mimics to all fingers; positive values open, negative close.")
    lines.append("#")
    lines.append("# Tips:")
    lines.append("# - Deltas accumulate each control step. After applying, set values back to 0 to hold pose.")
    lines.append("# - The script polls this file continuously; invalid lines are ignored.")
    lines.append("# - Stop the script with Ctrl+C in the terminal.")
    lines.append("#")
    lines.append("# Current delta command (used if the edited line is invalid):")
    lines.append("# " + " ".join(f"{v:.6f}" for v in current_action.tolist()))
    lines.append("# --- Edit the line below: ---")
    lines.append(" ".join(str(float(v)) for v in current_action.tolist()))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _read_control_file(path: str, expected_dim: int) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return None

    candidate = None
    for raw in lines:
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        candidate = stripped

    if candidate is None:
        return None

    parts = [p for p in candidate.replace(",", " ").split() if p]
    try:
        vals = np.asarray([float(p) for p in parts], dtype=np.float32)
    except ValueError:
        return None

    if vals.shape[0] != expected_dim:
        return None
    return vals


def _write_joint_state(path: str, joint_names: List[str], qpos: np.ndarray):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Current joint positions (rad)\n")
            for idx, (name, val) in enumerate(zip(joint_names, qpos.tolist())):
                f.write(f"{idx:02d} {name:>24s}: {val:.6f}\n")
    except Exception:
        pass


def main():
    robot_uid = os.environ.get("ROBOT_UID", "my_xarm7_official")
    print(f"Using robot_uids='{robot_uid}' (set ROBOT_UID to override)")

    env = gym.make(
        "MyEEAlignMarker-v0",
        obs_mode="state",
        control_mode="rate_limited_pd_joint_delta_pos",
        robot_uids=robot_uid,
        render_mode="human",
        sim_backend="gpu",
        render_backend="gpu",
    )

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    env.reset(seed=0)

    action_low = env.single_action_space.low
    action_high = env.single_action_space.high

    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]

    robot = env.unwrapped.agent.robot

    # Map action indices using the controller action mapping for stability
    mapping = env.unwrapped.agent.controller.action_mapping
    arm_start, arm_end = mapping.get("arm", (0, 7))
    gripper_start, gripper_end = mapping.get("gripper", (arm_end, arm_end + 1))

    arm_delta_limits = np.stack(
        [action_low[arm_start:arm_end], action_high[arm_start:arm_end]], axis=1
    )
    gripper_delta_range = [
        float(action_low[gripper_start]),
        float(action_high[gripper_start]),
    ]

    # Initial command is zero delta (hold pose)
    current_action = np.zeros_like(action_low, dtype=np.float32)

    _write_control_file_template(
        CONTROL_FILE,
        arm_joint_names,
        arm_delta_limits,
        gripper_delta_range,
        current_action,
    )
    print(f"Wrote delta control template to '{CONTROL_FILE}'.")

    last_state_write = 0.0
    try:
        while True:
            try:
                qpos_full = robot.get_qpos()[0].detach().cpu().numpy()
                joint_names_out = arm_joint_names + ["drive_joint"]
                indices = [int(robot.joints_map[name].active_index[0]) for name in joint_names_out]
                qpos_out = qpos_full[indices]
                now = time.time()
                if now - last_state_write > 0.1:
                    _write_joint_state(STATE_FILE, joint_names_out, qpos_out)
                    last_state_write = now
            except Exception:
                pass

            new_action = _read_control_file(CONTROL_FILE, 8)
            if new_action is not None:
                flat = np.zeros_like(action_low, dtype=np.float32)
                flat[arm_start:arm_end] = new_action[:7]
                flat[gripper_start:gripper_end] = new_action[7]
                flat = np.clip(flat, action_low, action_high)
                current_action = flat.astype(np.float32)

            env.step(current_action)
            env.render()
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nStopping on user request (Ctrl+C).")
    finally:
        env.close()


if __name__ == "__main__":
    main()

