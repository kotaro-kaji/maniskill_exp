# Codex Knowledge Notes

This directory stores durable operational knowledge for this repository. Keep
notes practical, concise, and specific to experiments that should influence
future work.

## Directory Map

- `remote/`: RunPod and remote GPU workflow notes.
- `cardboard/`: Cardboard insertion, cabinet, drawer, and reward-design notes.
- `xarm/`: xArm gripper dynamics, visual mesh repair, and related thread
  references.

## Remote

- `remote/runpod_maniskill_gpu.md`: RunPod-style GPU server setup and ManiSkill
  rendering/simulation notes.
- `remote/experiment_workflow.md`: How to run long remote training/eval jobs and
  bring back artifacts.

## Cardboard

- `cardboard/insertion_policy.md`: Policy reporting rule for cardboard
  insertion; no fixed evaluation-time action offsets.
- `cardboard/insertion_user_advice.md`: User-provided orientation, gripper, and
  video-name hints for insertion work.
- `cardboard/drawer_reward_lessons.md`: Reward-design lessons from cardboard
  drawer opening experiments.
- `cardboard/arm_qpos_return_handoff.md`: Current handoff note for arm-only qpos
  return direction.
- `cardboard/cabinet_deferred_open_reward.md`: Deferred cabinet open reward idea
  and why it is not current.

## xArm

- `xarm/gripper_joint_dynamics.md`: xArm gripper mimic/joint dynamics tuning
  notes.
- `xarm/gripper_stability_thread.md`: Codex thread reference for gripper
  stability work.
- `xarm/stl_repair_and_finger_visual.md`: STL self-intersection repair notes,
  xArm gripper voxel remesh results, and URDF visual mesh replacement details.
