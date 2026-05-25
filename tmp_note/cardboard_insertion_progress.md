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

## Pose/gripper reward attempt

Added reward variant:
`MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionPoseGripBoxStrict-v0`

The variant keeps the same geometry and no-offset policy rule, then adds:

- gripper drive target reward around `qpos=0.44`, based on the SpaceMouse
  successful example;
- a strong EEF x-axis constraint using `link_tcp`: the local x axis should match
  the world x axis.

The EEF x-axis term is intentionally a hard gate on reward rather than a mild
bonus. A quick check of the previous PPO checkpoint showed:

- distance mean: `4.44 mm`
- under 5 mm: `111 / 128`
- valid under 5 mm with 7 mm box-shift threshold: `78 / 128`
- EEF x-axis error mean at best approach: `0.50`
- gripper drive qpos mean at best approach: `0.48`

This means the old checkpoint is still geometrically close, but it does not
satisfy the new EEF x-axis constraint.

The first hard-gated run was stopped. It made the PPO policy worse instead of
better:

- PPO `ckpt_1.pt`: mean `4.88 mm`, under 5 mm `22 / 32`
- PPO `ckpt_61.pt`: mean `35.98 mm`, under 5 mm `0 / 32`
- PPO `ckpt_131.pt`: mean `192.16 mm`, under 5 mm `0 / 32`
- SAC `ckpt_0.pt`: mean `168.87 mm`, under 5 mm `0 / 32`
- SAC `ckpt_100096.pt`: mean `211.73 mm`, under 5 mm `0 / 32`
- SAC `ckpt_150016.pt`: mean `297.15 mm`, under 5 mm `0 / 32`

The failure mode is not the gripper-opening term. The EEF x-axis gate was too
strict: the existing close policy has `eef_x_dot ~= 0.876` and is already within
5 mm, so punishing it as nearly zero reward destroys the useful behavior.

The current PoseGrip variant now gates only when `eef_x_dot < 0.85`. A
fixed-noise evaluation of the unchanged PPO `ckpt_1.pt` after that change gives:

- distance mean: `4.55 mm`
- under 5 mm: `52 / 64`
- valid under 5 mm with 7 mm box-shift threshold: `48 / 64`
- EEF x dot mean at best approach: `0.876`
- EEF x-axis error mean at best approach: `0.498`
- gripper drive qpos mean at best approach: `0.463`

Runs started on RunPod A6000:

- PPO fine-tune from the best close checkpoint with low LR and the relaxed
  `eef_x_dot >= 0.85` gate:
  - log: `tmp_note/logs/ppo_pose_grip_dot085_lr1e5_seed1.out`
  - pid: `tmp_note/logs/ppo_pose_grip_dot085_lr1e5_seed1.pid`
  - stopped after `ckpt_6.pt`; it still degraded from `22 / 32` under 5 mm to
    `3 / 32` under 5 mm. The gripper qpos collapsed from about `0.42` to
    `0.20`, so PPO fine-tuning with this extra term is not the current best
    path.
- SAC from scratch on a simpler gripper-only reward variant:
  `MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionGripBoxStrict-v0`
  - log: `tmp_note/logs/sac_grip_only_2m_seed1.out`
  - pid: `tmp_note/logs/sac_grip_only_2m_seed1.pid`
  - stopped after `ckpt_50176.pt`; it did not approach the slot:
    mean distance `293.23 mm`, under 5 mm `0 / 64`.
    One rollout group produced a large box shift, while valid rollouts stayed
    near the initial `298 mm` distance.

Current conclusion: the no-offset PPO checkpoint remains the best artifact.
Both adding EEF/gripper reward to PPO and starting SAC from scratch are worse
than preserving the existing close policy. The next useful direction is likely
not more scalar reward on top of the existing PPO checkpoint, but either:

- behavior-cloning or supervised warm start from the SpaceMouse state/action
  style;
- a curriculum that first reproduces the existing no-offset PPO behavior before
  adding insertion/pose constraints;
- checking whether the EEF frame target should use a different TCP/finger axis
  before treating `world x` as a hard constraint.

## Hold stability check

The apparent `final_distance ~= 298 mm` in the evaluation script is a reset
artifact at the 200-step episode boundary, not a late-episode failure. A trace
of a single rollout shows:

- step 25: `11.31 mm`
- step 50: `5.96 mm`
- step 92: best distance `4.01 mm`
- step 100: `4.05 mm`
- step 125: `5.22 mm`
- step 175: `10.33 mm`
- step 200: reset to initial distance

In the 64-env fixed-noise evaluation, the no-offset PPO checkpoint has:

- under 5 mm: `52 / 64`
- valid under 5 mm with 7 mm box-shift threshold: `48 / 64`
- valid close streak >= 5 steps: `47 / 64`
- mean max valid close streak: `26`
- median max valid close streak: `33`

So the policy is not only touching the target for one frame; most successful
rollouts hold the 5 mm condition for multiple control steps.

Updated confirmation video:

- `tmp_note/eval_videos/best_no_bias_hold_confirm_slot_peek_left_noise0_seed1/0.mp4`
- best distance: `4.01 mm`
- max box shift until best distance: `4.54 mm`
- max box shift over the 120-step video: `4.59 mm`

## Y + gripper-open fine-tune attempt

Added reward variant:
`MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripOpenBoxStrict-v0`

This keeps the same physical geometry, adds a stronger final Y-axis term, and
near the slot only penalizes gripper qpos below `0.40`. It was meant to avoid
the previous mistake of forcing the gripper to a too-small target.

PPO fine-tune from the no-offset checkpoint with `learning_rate=1e-6` and
`update_epochs=2` still degraded:

- `ckpt_1.pt`: under 5 mm `52 / 64`, valid close streak >= 5 `47 / 64`
- `ckpt_6.pt`: under 5 mm `0 / 64`, valid close streak >= 5 `0 / 64`
- gripper qpos mean at best: `0.46 -> 0.23`

This confirms that the PPO update itself is currently destructive even with a
very small learning rate and a one-sided gripper-opening reward.

An even more conservative PPO update was also tested:

- `learning_rate=1e-7`
- `update_epochs=1`
- `clip_coef=0.02`
- `target_kl=0.001`

Result:

- `ckpt_1.pt`: under 5 mm `52 / 64`, valid close streak >= 5 `47 / 64`
- `ckpt_6.pt`: under 5 mm `28 / 64`, valid close streak >= 5 `16 / 64`
- gripper qpos mean at best: `0.46 -> 0.28`

This is less destructive than the earlier PPO updates, but still worse than
the original checkpoint. It supports the current diagnosis: scalar reward
fine-tuning from this checkpoint reliably moves the policy toward closing the
gripper and away from the narrow successful insertion behavior.

## Anchor-regularized PPO attempt

Added `ppo_dual_xarm7.py` options:

- `--anchor-checkpoint`
- `--anchor-coef`

During PPO updates, the deterministic action mean is penalized by MSE against
the frozen anchor checkpoint. This is not an evaluation-time action correction;
it only constrains training updates.

Run:

- reward env:
  `MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripOpenBoxStrict-v0`
- anchor checkpoint: same no-offset PPO checkpoint
- `anchor_coef=10`
- `learning_rate=1e-6`
- `clip_coef=0.05`
- `target_kl=0.002`

Result:

- `ckpt_1.pt`: under 5 mm `52 / 64`, valid close streak >= 5 `47 / 64`
- `ckpt_6.pt`: under 5 mm `9 / 64`, valid close streak >= 5 `6 / 64`
- gripper qpos mean at best: `0.46 -> 0.25`

The anchor term as implemented is not strong enough to preserve the narrow
successful behavior. The failure is still the same: PPO quickly moves the
policy toward closing the gripper too much and losing the 5 mm approach.

## Gripper action sign correction

Tracing the original checkpoint and the failed fine-tuned checkpoints showed
that the left gripper qpos drops while the policy outputs positive gripper
actions. On this controller/task trajectory, positive `action[7]` is therefore
effectively the closing direction. The earlier gripper rewards only looked at
qpos and did not directly stop PPO from learning this closing action.

Added reward variant:
`MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripActionBoxStrict-v0`

It inherits the Y/final-distance and minimum-gripper-qpos reward, and near the
slot adds a small reward for negative `action[7]`.

Run:

- anchor checkpoint: same no-offset PPO checkpoint
- `anchor_coef=50`
- `learning_rate=1e-7`
- `update_epochs=1`
- `clip_coef=0.02`
- `target_kl=0.001`

Result:

- `ckpt_6.pt`: under 5 mm `44 / 64`, valid under 5 mm `33 / 64`
- `ckpt_11.pt`: under 5 mm `58 / 64`, valid under 5 mm `50 / 64`
- `ckpt_16.pt`: under 5 mm `34 / 64`, valid under 5 mm `33 / 64`

The best checkpoint is `ckpt_11.pt`, copied locally as:

- `tmp_note/ckpt_cardboard_y_grip_action_anchor50_ckpt11.pt`

Compared with the previous no-offset best:

- under 5 mm: `52 / 64` -> `58 / 64`
- valid under 5 mm with 7 mm box-shift threshold: `48 / 64` -> `50 / 64`
- mean distance: `4.55 mm` -> `4.50 mm`
- mean max box shift: `6.21 mm` -> `5.46 mm`
- valid close streak >= 5 steps: `47 / 64` -> `48 / 64`

Updated best video:

- `tmp_note/eval_videos/best_y_grip_action_anchor50_ckpt11_slot_peek_left_noise0_seed1/0.mp4`
- best distance in the recorded rollout: `4.06 mm`
- max box shift until best distance: `4.19 mm`
- max box shift over the 120-step video: `4.20 mm`

This is the first fine-tuning run that improves the no-offset PPO baseline.
The causal point is specific: rewarding the wrong gripper action sign lets PPO
close the gripper too far; adding the near-slot negative-action term keeps the
gripper open enough while preserving the approach policy.

## Action reward weight and EEF constraint sweep

Action reward weights were compared with the same anchor and PPO settings:

- `GRIPPER_ACTION_REWARD_WEIGHT=0.05`
  - best checked checkpoint: under 5 mm `41 / 64`, valid under 5 mm `26 / 64`
  - too weak to preserve the opening behavior.
- `GRIPPER_ACTION_REWARD_WEIGHT=0.10`
  - best checked checkpoint: under 5 mm `58 / 64`, valid under 5 mm `50 / 64`.
- `GRIPPER_ACTION_REWARD_WEIGHT=0.20`
  - `ckpt_16.pt`: under 5 mm `60 / 64`, valid under 5 mm `56 / 64`,
    valid close streak >= 5 steps `56 / 64`.
  - copied locally as:
    `tmp_note/ckpt_cardboard_y_grip_action020_anchor50_ckpt16.pt`
  - video:
    `tmp_note/eval_videos/best_y_grip_action020_anchor50_ckpt16_slot_peek_left_noise0_seed1/0.mp4`

Then a weak EEF x-axis constraint was added on top of the `0.20` action reward:

- env:
  `MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripAction020EefBoxStrict-v0`
- `EEF_X_WORLD_REWARD_WEIGHT=0.10`
- `EEF_X_WORLD_MIN_DOT=0.84`
- start checkpoint: action020 `ckpt_16.pt`
- anchor checkpoint: the same action020 `ckpt_16.pt`

Best EEF-constrained checkpoint:

- remote:
  `runs/MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripAction020EefBoxStrict-v0__ppo_dual_xarm7__1__1779713280/ckpt_11.pt`
- local:
  `tmp_note/ckpt_cardboard_y_grip_action020_eef_anchor50_ckpt11.pt`

Metrics:

- under 5 mm: `58 / 64`
- valid under 5 mm with 7 mm box-shift threshold: `57 / 64`
- mean distance: `4.46 mm`
- valid mean distance: `4.89 mm`
- mean max box shift: `5.55 mm`
- valid close streak >= 5 steps: `55 / 64`
- mean EEF x dot at best approach: `0.875`
- mean gripper qpos at best approach: `0.472`

Video:

- `tmp_note/eval_videos/best_y_grip_action020_eef_anchor50_ckpt11_slot_peek_left_noise0_seed1/0.mp4`
- recorded rollout best distance: `4.10 mm`
- max box shift until best: `5.13 mm`
- max box shift over the 120-step video: `5.19 mm`

This was the final best policy for that round because it included insertion
distance, gripper opening/action sign, and a weak EEF x-axis posture constraint.
It remains fixed-action-bias free. The action-sign reward was later removed
from the active reward design because the always-on gripper qpos reward is
clearer.

## Next goal

The next target is stable 3 mm-level insertion distance. The 5 mm target is
already mostly reached by the no-offset PPO checkpoints, but the video-level
best still sits around 4.7 to 5.3 mm and the EEF x axis is about 30 degrees off
world +x. Future PPO/SAC attempts should optimize toward:

- 3 mm-level purple-to-lime marker distance.
- EEF local x axis close to world +x.
- User-observed Spacemouse posture hypothesis: EEF rotation around its local x
  axis may need to be about 45 to 65 degrees during insertion.
- Gripper opening qpos reward should apply throughout the episode, including
  before first waypoint.
- No fixed runtime action bias.
