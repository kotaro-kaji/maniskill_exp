# ManiSkill xArm7 Position-Control Reference

This note captures every parameter needed to reproduce the xArm7 position-control stack (no force control) inside ManiSkill. All values come from the ROS controller YAMLs, URDF, and SDK utilities bundled with the official xArm repositories.

## Control Architecture Snapshot
- The default ROS1 setup loads `xarm7_traj_controller`, a `position_controllers/JointTrajectoryController` covering all seven joints with 0.5 s goal time tolerance, 0.01 rad goal error, and 0.2 s stop duration.`xarm_ros/xarm_controller/config/xarm7/xarm7_controllers.yaml:7`
- For demos there are per-joint `JointPositionController` instances with hand-tuned PID gains per axis (table below).`xarm_ros/xarm_controller/config/xarm7/xarm7_controllers.yaml:72`
- ROS1 also exposes a velocity-space trajectory controller, but it is disabled unless explicitly requested; the real robot ships with the position trajectory interface only.`xarm_ros/ReadMe.md:202`
- ROS2 provides a single `joint_trajectory_controller/JointTrajectoryController` with combined position/velocity interfaces and the same tolerances, running at 150 Hz update rate.`xarm_ros2/xarm_controller/config/xarm7_controllers.yaml:1`

## ROS1 Joint-Position PID Gains (simulation defaults)
Source: `xarm_ros/xarm_controller/config/xarm7/xarm7_controllers.yaml:72`

| Joint | P | I | D |
| --- | --- | --- | --- |
| joint1 | 1200.0 | 5.0 | 10.0 |
| joint2 | 1400.0 | 5.0 | 10.0 |
| joint3 | 1200.0 | 5.0 | 5.0 |
| joint4 | 850.0 | 3.0 | 5.0 |
| joint5 | 500.0 | 3.0 | 1.0 |
| joint6 | 500.0 | 1.0 | 1.0 |
| joint7 | 300.0 | 0.05 | 1.0 |

These gains are intended for Gazebo; hardware relies on embedded servo loops instead.

## ROS2 Controller Parameters
- `controller_manager` runs at 150 Hz, broadcasting joint states and commanding the same trajectory controller with position/velocity interfaces.`xarm_ros2/xarm_controller/config/xarm7_controllers.yaml:1`
- Tolerances mirror the ROS1 config: `goal_time = 0.5 s`, trajectory error limit 1 rad, and goal error 0.01 rad per joint.`xarm_ros2/xarm_controller/config/xarm7_controllers.yaml:9`

## Joint Limits and Dynamics
- URDF lower/upper bounds are baked into the xacro macro: joint1 ±2π, joint2 ∈ [−2.059, 2.0944], joint3 ±2π, joint4 ∈ [−0.19198, 3.927], joint5 ±2π, joint6 ∈ [−1.69297, π], joint7 ±2π.`xarm_ros/xarm_description/urdf/xarm7/xarm7.urdf.xacro:5`
- Each revolute joint shares a 3.14 rad/s velocity cap at the URDF level, with efforts of 50 Nm (joints 1–2), 30 Nm (joints 3–5), and 20 Nm (joints 6–7). Damping/friction is 10/1 for joints 1–2, 5/1 for joints 3–5, and 2/1 for joints 6–7.`xarm_ros/xarm_description/urdf/xarm7/xarm7.urdf.xacro:94`
- MoveIt stacks override velocities to 2.14 rad/s for all joints; ROS1 disables acceleration limits whereas ROS2 sets ±10 rad/s². Adjust if ManiSkill should mimic either behavior.`xarm_ros/xarm7_moveit_config/config/joint_limits.yaml:1` `xarm_ros2/xarm_moveit_config/config/xarm7/joint_limits.yaml:1`

## Enforcement and Hardware Notes
- The hardware interface enforces URDF-based limits unless the `enforce_limits` parameter is explicitly turned off (defaults to true).`xarm_ros/xarm_controller/src/xarm_hw.cpp:239`
- Real hardware exposes only the trajectory controller; effort controllers in the repo are simulation examples and should not be ported to ManiSkill.`xarm_ros/ReadMe.md:202`
- Servo-level PID coefficients can be queried directly via the SDK (`get_servo_all_pids`) if you need the firmware gains for a higher-fidelity simulation.`xArm-Python-SDK/xarm/x3/servo.py:460`

## ManiSkill Implementation Tips
- Mirror the JointTrajectoryController contract: accept joint-space trajectories and enforce per-joint tolerances similar to ROS (goal error 0.01 rad, stopped velocity tolerance 0.05 rad/s in ROS1, 0 in ROS2).
- Apply the URDF velocity/effort caps before integrating commands; optionally clamp to MoveIt’s 2.14 rad/s if replicating planner behavior.
- Keep all seven joints revolute; there are no prismatic axes in the kinematic chain.`xarm_ros/xarm_description/urdf/xarm7/xarm7.urdf.xacro:94`
- If you need initial joint PID seeds for your simulator, start with the Gazebo gains above and tune to match ManiSkill’s dynamics.

## Quick Checklist
- [ ] Load seven-DOF revolute chain with URDF limit ranges noted above.
- [ ] Implement trajectory tracking with 0.5 s goal-time tolerance and 0.01 rad goal accuracy.
- [ ] Clamp joint velocities to ≤3.14 rad/s (or 2.14 rad/s for MoveIt parity) and respect effort tiers.
- [ ] Include optional per-joint PID loop for simulation with the Gazebo gains as defaults.
- [ ] Expose `enforce_limits`-style guard to toggle saturation if needed.
