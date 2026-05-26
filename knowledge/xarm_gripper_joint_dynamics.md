# xArm gripper joint dynamics tuning

## Context

For the custom xArm7 gripper in `robotagents/assets/xarm7/xarm7_1305_left.urdf`, the URDF mimic structure itself looked reasonable, but the mimic/finger joints were too weak against external contact. This caused unstable finger behavior and object penetration during SpaceMouse teleoperation.

Adding URDF joint dynamics to the six gripper joints made the gripper noticeably more stable:

```xml
<dynamics damping="0.01" friction="1.0"/>
```

Apply this to:

- `left_drive_joint`
- `left_finger_joint`
- `left_inner_knuckle_joint`
- `right_drive_joint`
- `right_finger_joint`
- `right_inner_knuckle_joint`

## Empirical result

A short 60-step close/open probe in `MyXarm7PickCube-v1` showed that damping is the value that strongly affects whether the gripper can still move.

Representative free-space close results:

```text
damping friction close_left close_right open_left open_right
none    none     0.57053    0.57054     0.00009   0.00009
0       0        0.57053    0.57054     0.00009   0.00009
0.01    0        0.51260    0.51521     0.00018   0.00000
0.03    0        0.42803    0.43490     0.00052   0.00000
0.1     0        0.27137    0.28748     0.00215   0.00001
0       3        0.57053    0.57054     0.00009   0.00009
0.03    3        0.42803    0.43490     0.00052   0.00000
```

`friction` did not noticeably reduce free-space gripper motion in this probe, even up to `3`. `damping=0.1` was already too heavy, while `damping=0.01` preserved most of the motion and still improved stability in SpaceMouse testing.

## Current recommendation

Use:

```xml
<dynamics damping="0.01" friction="1.0"/>
```

This is a practical tuned value for the current ManiSkill/SAPIEN setup, not a universal physical constant.
