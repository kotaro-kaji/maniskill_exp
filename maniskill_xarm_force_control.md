# ManiSkill xArm7 Force-Control Reference

This note extracts everything needed to reproduce the xArm7 six-axis force controller inside ManiSkill. Values and semantics are taken directly from the official Python SDK examples and API specification.

## Control Architecture Overview
- The SDK exposes `set_ft_sensor_force_parameters` to configure both the task-space force objective and the outer PID loop that turns measured force error into an adjustment velocity for the TCP (tool).`xArm-Python-SDK/doc/api/xarm_api.md:2493`
- Force control runs in the task frame selected by `coord` (0 = base, 1 = tool). Only axes flagged with `1` in `c_axis` are compliant; the rest remain stiff in position control.`xArm-Python-SDK/doc/api/xarm_api.md:2506`
- The controller computes a 6D correction velocity in Cartesian space and saturates it using `xe_limit` before mapping it through the Jacobian to joint velocities. This means the velocity limits apply at the TCP, not per joint. The joints remain standard revolute joints; limits on joint speeds come from their own actuators and trajectory manager.`xArm-Python-SDK/doc/api/xarm_api.md:2513` `xarm_ros/xarm_description/urdf/xarm7/xarm7.urdf.xacro:94`

## Gain and Limit Parameters
All are 6-element vectors matching `[X, Y, Z, Rx, Ry, Rz]` of the chosen task frame.

| Parameter | Meaning / Units | Valid Range | Notes |
| --- | --- | --- | --- |
| `kp` | Converts force error to linear (mm/s) or angular (rad/s) correction velocity per Newton/N·m | `[0, 0.05]` | Low values avoid oscillation; sample uses `0.005`. `xArm-Python-SDK/xarm/x3/ft_sensor.py:143`
| `ki` | Integral term gain | `[0, 0.0005]` | Optional; example uses `0.00005`. `xArm-Python-SDK/xarm/x3/ft_sensor.py:143`
| `kd` | Derivative gain | `[0, 0.05]` | Example uses `0.05`. `xArm-Python-SDK/xarm/x3/ft_sensor.py:143`
| `xe_limit` | Max TCP correction speed on each compliant axis (mm/s for XYZ, rad/s for rotations) | `[0, 200]` | Acts as position-correction velocity limit. `xArm-Python-SDK/xarm/x3/ft_sensor.py:143`
| `f_ref` | Desired steady-state force/torque (N / N·m) | `±[150, 150, 200, 4, 4, 4]` | Saturated when `params_limit=True`. `xArm-Python-SDK/xarm/x3/ft_sensor.py:128`
| `limits` | Reserved; still treated as TCP velocity caps in docs | - | Example fills with zeros. `xArm-Python-SDK/example/wrapper/common/8003-force_control.py:55`

## Example Configuration (SDK Reference)
The official demo sets symmetric gains and limits for all axes before enabling the force loop:`xArm-Python-SDK/example/wrapper/common/8003-force_control.py:48`

```python
Kp = 0.005
Ki = 0.00005
Kd = 0.05
linear_v_max = 200.0      # mm/s cap for XYZ
rot_v_max = 0.35          # rad/s cap for rotations
arm.set_ft_sensor_force_parameters(
    [Kp]*6,
    [Ki]*6,
    [Kd]*6,
    [linear_v_max]*3 + [rot_v_max]*3
)
```

This is followed by the force objective configuration:

```python
ref_frame = 1  # Tool frame
force_axis = [0, 0, 1, 0, 0, 0]
force_ref = [0, 0, 5.0, 0, 0, 0]
arm.set_ft_sensor_force_parameters(ref_frame, force_axis, force_ref, [0]*6)
```
- Only Z-axis is compliant; the controller tries to achieve +5 N along tool Z while keeping other axes stiff.`xArm-Python-SDK/example/wrapper/common/8003-force_control.py:55`

## Runtime Sequence (Hardware SDK)
Mimic this ordering when scripting the ManiSkill controller state machine:
1. Connect and power servos: `motion_enable`, `clean_error`, set mode/state 0.`xArm-Python-SDK/example/wrapper/common/8003-force_control.py:36`
2. Upload PID/velocity limits with `set_ft_sensor_force_parameters(kp, ki, kd, xe_limit)`.
3. Configure the force reference and compliant axes.
4. Enable the FT sensor (`set_ft_sensor_enable(1)`) and optionally zero bias (`set_ft_sensor_zero()`).`xArm-Python-SDK/example/wrapper/common/8003-force_control.py:58`
5. Switch controller to force mode (`set_ft_sensor_mode(2)`), then start motion by clearing state to 0.
6. When done, return to non-force mode and disable the sensor.

## Notes for ManiSkill Implementation
- Represent the outer loop in task space: ManiSkill should expose a 6D wrench error (desired - measured) and integrate it through PID gains to command TCP velocity.
- Apply `xe_limit` as an element-wise clamp on the resulting correction velocity vector before converting to joint velocities with the simulated Jacobian.
- Ensure the simulated arm uses seven revolute joints to match xArm7 kinematics; there are no prismatic joints in the chain.`xarm_ros/xarm_description/urdf/xarm7/xarm7.urdf.xacro:94`
- Respect the API-specified ranges for gains and force targets to stay within firmware-stable behavior; these should become validation clamps in the ManiSkill controller layer.`xArm-Python-SDK/xarm/x3/ft_sensor.py:128`
- If admittance mode is required, the companion API `set_ft_sensor_admittance_parameters` offers MBK parameters and shares the same frame/axis conventions, but the demo controller sticks to pure force mode.`xArm-Python-SDK/doc/api/xarm_api.md:2458`

## Quick Checklist
- [ ] Task frame and compliant axes align with ManiSkill environment orientation.
- [ ] PID gains and `xe_limit` set before entering force mode.
- [ ] Force reference within `[±150, ±150, ±200, ±4, ±4, ±4]`.
- [ ] Correction velocity clamp matches units (mm/s vs rad/s).
- [ ] FT sensor state machine mirrored (enable → zero → mode → run → disable).
