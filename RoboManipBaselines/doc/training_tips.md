# Training tips

## Change the input/output of policy
The `--state_keys`, `--action_keys`, and `--camera_names` options in `Train.py` allow changing the policy input and output.

### Examples of `--state_keys`
```console
# Set measured joint position as observed state (default)
--state_keys measured_joint_pos

# Set measured end-effector pose as observed state
--state_keys measured_eef_pose measured_gripper_joint_pos

# Leave observed state empty
--state_keys # no arguments
```

### Examples of `--action_keys`
```console
# Set command joint position as action (default)
--action_keys command_joint_pos

# Set command joint position relative to previous step as action
--action_keys command_joint_pos_rel

# Set command end-effector pose as action
--action_keys command_eef_pose command_gripper_joint_pos

# Set command end-effector pose relative to previous step as action
--action_keys command_eef_pose_rel command_gripper_joint_pos
```

It is recommended that state_keys and action_keys be set to the corresponding quantities (e.g. `--state_keys measured_eef_pose` and `--action_keys command_eef_pose`).

### Examples of `--camera_names`
```console
# Use front camera image (default)
--camera_names front

# Use front and side camera images
--camera_names front side
```
