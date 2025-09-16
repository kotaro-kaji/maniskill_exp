import os
import time
import json
from typing import List, Optional

import gymnasium as gym
import mani_skill.envs  # registers built-in envs
import my_xarm7  # registers custom robot (legacy)
import my_xarm7_mjcf  # registers my_xarm7_mjcf (baseline)
# Optional variants for quick A/B testing
try:
    import my_xarm7_mjcf_1  # registers my_xarm7_mjcf_1 (safe)
    import my_xarm7_mjcf_2  # registers my_xarm7_mjcf_2 (balanced)
    import my_xarm7_mjcf_3  # registers my_xarm7_mjcf_3 (responsive)
except Exception:
    pass
import task_pushcube  # registers MyPushCube-v1
import numpy as np


CONTROL_FILE = "control_input.txt"
STATE_FILE = "joint_state.txt"


def _write_control_file_template(
    path: str,
    arm_joint_names: List[str],
    arm_limits: Optional[np.ndarray],
    gripper_range: Optional[List[float]],
    current_action: np.ndarray,
):
    """
    Create or overwrite a human-editable control file with comments describing
    the expected format and bounds. Lines starting with '#' are comments.

    Format: 8 values per line -> 7 arm joint absolute positions (rad),
    then 1 gripper command. For this controller:
      - Arm: absolute target joint positions [rad] (not normalized).
      - Gripper: normalized in [-1, 1], internally mapped to [0.00, 0.85] rad.
    """
    lines = []
    lines.append("# XArm7 file control (pd_joint_pos)")
    lines.append("# Edit the last line with 8 numbers then save. The robot follows it.")
    lines.append("# Format: 7 arm joint ABS positions [rad], then 1 GRIPPER value (normalized)")
    lines.append("# Example: 0 0 0 1.0472 0 1.0472 -1.5708 0.0")
    lines.append("#")
    lines.append("# Arm inputs (abs rad):")
    for i, name in enumerate(arm_joint_names):
        lim = None
        if arm_limits is not None and i < len(arm_limits):
            lim = arm_limits[i]
        if lim is not None and lim.shape[0] == 2:
            lines.append(
                f"#   {i}: {name:>9s}  min={lim[0]: .4f}  max={lim[1]: .4f}"
            )
        else:
            lines.append(f"#   {i}: {name:>9s}  min=unknown  max=unknown")
    lines.append("#")
    if gripper_range is not None:
        lines.append(
            f"# Gripper (normalized): set in [-1, 1]  -> mapped to [{gripper_range[0]:.2f}, {gripper_range[1]:.2f}] rad"
        )
    else:
        lines.append("# Gripper (absolute): set radians directly (e.g., 0.0 open, 0.85 close)")
    lines.append("#")
    lines.append("# Tips:")
    lines.append("# - Save the file to apply new targets. The script polls this file.")
    lines.append("# - Leave values unchanged to keep the robot at its current pose.")
    lines.append("# - Stop the script with Ctrl+C in the terminal.")
    lines.append("#")
    lines.append("# Current target (will be used if line below is invalid):")
    lines.append("# " + " ".join(f"{v:.6f}" for v in current_action.tolist()))
    lines.append("# --- Edit the line below: ---")
    lines.append(" ".join(str(float(v)) for v in current_action.tolist()))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _read_control_file(path: str, expected_dim: int) -> Optional[np.ndarray]:
    """
    Read the control file. Ignores comment/empty lines. Parses the last
    non-comment line into an array. Returns None if parsing fails.
    Accepts space- or comma-separated numbers.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return None

    candidate = None
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        candidate = s

    if candidate is None:
        return None

    # Split by comma or whitespace
    parts = [p for p in candidate.replace(",", " ").split() if p]
    try:
        vals = np.asarray([float(p) for p in parts], dtype=np.float32)
    except ValueError:
        return None

    if vals.shape[0] != expected_dim:
        return None
    return vals


def _write_joint_state(path: str, joint_names: List[str], qpos: np.ndarray):
    """Write current joint positions to a text file for user reference."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Current joint positions (rad)\n")
            for i, (name, v) in enumerate(zip(joint_names, qpos.tolist())):
                f.write(f"{i:02d} {name:>24s}: {v:.6f}\n")
    except Exception:
        pass


def main():
    # Choose robot uid via env var or default
    robot_uid = os.environ.get("ROBOT_UID", "my_xarm7_mjcf")
    print(f"Using robot_uids='{robot_uid}' (set ROBOT_UID to override)")

    env = gym.make(
        "MyPushCube-v1",
        obs_mode="state",
        control_mode="pd_joint_pos",  # absolute joint position control
        robot_uids=robot_uid,
        render_mode="human",
        sim_backend="gpu",
        render_backend="gpu",
    )

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    obs, _ = env.reset(seed=0)

    # Determine action dimension and helpful metadata
    action_low = env.single_action_space.low
    action_high = env.single_action_space.high
    action_dim = env.single_action_space.shape[0]
    print(f"Action dim: {action_dim}")

    # For my_xarm7_mjcf with pd_joint_pos: 7 arm DOF (abs rad) + 1 gripper (normalized)
    arm_joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    gripper_normalized_range = [-1.0, 1.0]

    # Query arm joint limits from the underlying articulation (if available)
    try:
        robot = env.unwrapped.agent.robot
        # Active joints include gripper; get for arm names specifically
        qlimits = robot.get_qlimits()[0].cpu().numpy()  # shape [n_dofs, 2]
        # Build a map name -> limits when possible
        arm_limits = []
        for jname in arm_joint_names:
            j = robot.joints_map[jname]
            idx = int(j.active_index[0]) if j.active_index is not None else None
            if idx is None:
                arm_limits.append(np.array([np.nan, np.nan]))
            else:
                arm_limits.append(qlimits[idx])
        arm_limits = np.asarray(arm_limits)
    except Exception:
        arm_limits = None

    # Initialize current target from current joints to keep robot still
    # We extract current arm joint positions (7) and set gripper target to 0.0 (neutral)
    # Note: arm is absolute; gripper is normalized [-1, 1]
    try:
        # Fetch current qpos via controllers when possible
        # Access combined controller through env.unwrapped.agent.controller
        current_full_qpos = robot.get_qpos()[0].detach().cpu().numpy()
        # Map to arm joints
        current_arm = []
        for jname in arm_joint_names:
            j = robot.joints_map[jname]
            idx = int(j.active_index[0])
            current_arm.append(float(current_full_qpos[idx]))
        current_arm = np.asarray(current_arm, dtype=np.float32)
    except Exception:
        # Fallback to zeros if mapping fails
        current_arm = np.zeros(7, dtype=np.float32)

    # Start with a neutral gripper command (normalized by default)
    current_gripper = np.array([0.0], dtype=np.float32)
    current_action = np.concatenate([current_arm, current_gripper], axis=0)

    # Create control file template for the user
    # Determine gripper UI hint by robot uid
    if robot_uid == "my_xarm7_mjcf_3":
        # v3 experimental: absolute radians for gripper
        gr_range = None
    else:
        gr_range = [0.0, 0.85]
    _write_control_file_template(CONTROL_FILE, arm_joint_names, arm_limits, gr_range, current_action)
    print(f"Wrote control template to '{CONTROL_FILE}'. Edit and save to command.")

    # Main loop: run forever until user stops (Ctrl+C)
    last_state_write_t = 0.0
    try:
        while True:
            # Always get joint states (arm + gripper driver) for user reference
            try:
                qpos_full = robot.get_qpos()[0].detach().cpu().numpy()
                # Build ordered names for export: 7 arm + left_driver_joint (gripper)
                joint_names_out = arm_joint_names + ["left_driver_joint"]
                indices = [
                    int(robot.joints_map[n].active_index[0]) for n in joint_names_out
                ]
                qpos_out = qpos_full[indices]
                # Throttle file writes to ~10 Hz
                now = time.time()
                if now - last_state_write_t > 0.1:
                    _write_joint_state(STATE_FILE, joint_names_out, qpos_out)
                    last_state_write_t = now
            except Exception:
                pass

            # Check for new control values
            new_action = _read_control_file(CONTROL_FILE, 8)
            if new_action is not None:
                # We always parse as [7 arm abs, 1 gripper]. Map into flat action
                # according to the underlying controller's action mapping to avoid
                # relying on controller order.
                try:
                    mapping = env.unwrapped.agent.controller.action_mapping
                    arm_start, arm_end = mapping.get("arm", (0, 7))
                    grip_start, grip_end = mapping.get("gripper", (arm_end, arm_end + 1))
                    flat = np.zeros_like(action_low, dtype=np.float32)
                    # Arm absolute positions
                    flat[arm_start:arm_end] = new_action[:7]
                    # Gripper value
                    flat[grip_start:grip_end] = new_action[7]
                    # Clip to action space bounds for safety
                    flat = np.clip(flat, action_low, action_high)
                    current_action = flat.astype(np.float32)
                except Exception:
                    # Fallback: assume concatenated [arm(7), gripper(1)] ordering
                    flat = np.clip(new_action, action_low, action_high)
                    current_action = flat.astype(np.float32)

            # Step with current action; no termination handling (user stops manually)
            obs, reward, terminated, truncated, info = env.step(current_action)
            env.render()
            # Small sleep to avoid busy loop if render is fast
            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nStopping on user request (Ctrl+C).")
    finally:
        env.close()


if __name__ == "__main__":
    main()
