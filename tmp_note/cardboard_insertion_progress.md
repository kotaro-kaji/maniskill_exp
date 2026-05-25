# Cardboard insertion progress

This note keeps only the current plain policy evaluation state. Earlier
evaluation-time action-offset experiments were discarded because they changed
the policy output and made the result hard to interpret.

## Current setup

- Environment family: `MyDualCardboardCabinet-v0`
- Controller: `pd_joint_delta_pos`
- Robot initial pose noise for evaluation: `robot_init_noise_scale=0.0`
- Inner box wall thickness: `2 mm`
- Inner rear wall thickness: `4 mm`
- Insert target local z was raised by `1 mm`
- The unreachable second visual waypoint was removed
- Box shift tolerance was relaxed:
  - reward tolerance: `3 mm`
  - success max shift: `7 mm`
  - penalty scale: `180`

## Current non-biased rollout

Checkpoint:
`tmp_note/ckpt_density1500_a5_040_d0_013_d6_025.pt`

No runtime action correction is applied. The policy action is only clamped to
the environment action range before stepping the environment.

- Recorded best distance: `3.33 mm`
- Max box shift: `6.54 mm`
- Camera videos:
  - overview:
    `tmp_note/eval_videos/two_point_2mm_targetup_no_bias_default_noise0_seed1/0.mp4`
  - close inspection:
    `tmp_note/eval_videos/two_point_2mm_targetup_no_bias_slot_peek_left_noise0_seed1/0.mp4`

## Manual hint

The SpaceMouse demonstration showed that a partially open gripper and a slightly
higher target are important. The current geometry changes reflect that
observation without adding fixed offsets to policy actions.
