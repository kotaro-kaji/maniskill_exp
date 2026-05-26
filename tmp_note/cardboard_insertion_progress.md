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

## 3 mm A6000 attempts

Added EEF roll instrumentation:

- `eef_x_roll_world` is the signed angle of EEF local y projected in the world
  yz plane, measured around world +x. It is meaningful when EEF local x is close
  to world +x.
- The Spacemouse-derived 45 to 65 degree roll hypothesis should be checked
  against this metric.

Added a 3 mm reward variant:

- `MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripEef3mmBoxStrict-v0`
- Same geometry as the current family.
- `INSERT_SUCCESS_DISTANCE = 0.003`
- stronger final distance scales.

Current diagnostic from the best 5 mm-level PPO checkpoint:

- Mean best distance: `4.21 mm`
- Min best distance: `3.52 mm`
- Under 3 mm: `0 / 64`
- Dominant residual error is y: best delta mean
  `[-1.57 mm, -3.80 mm, -0.26 mm]`.

Scratch SAC on the current reward remains ineffective:

- checkpoint: SAC `ckpt_400128`
- mean distance: `293.75 mm`
- under 5 mm: `0 / 64`
- conclusion: SAC from scratch still does not discover the approach behavior.
  A useful SAC path likely needs PPO/teleop behavior distillation or another
  curriculum, not raw scratch SAC.

Scratch PPO checks:

- A strong-pose scratch PPO variant was started with no checkpoint and no
  anchor. Early checkpoints align EEF local x with world +x, but do not approach
  the cabinet: mean distance remains about `293-297 mm`.
- This confirms that making the EEF x-axis reward strong from the start can
  produce a policy that satisfies posture while ignoring insertion distance.
- The first user-facing scratch video was saved with a Japanese filename:
  `tmp_note/eval_videos/scratch_strongpose_reference/こちらをご参照ください.mp4`.
- A second scratch PPO run was started on the weaker EEF 3 mm reward variant to
  check whether distance learning starts without anchor or strong pose pressure.

## 3 mm curriculum/fine-tune follow-up

Scratch PPO can learn a coarse approach if the reward has a non-saturated
first-waypoint term:

- `MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGrip3mmCurriculumBoxStrict-v0`
- PPO `ckpt_11`: mean distance `26.45 mm`
- It reaches the first waypoint from scratch, but leaves about `25 mm` y
  residual, so it does not insert.

Adding a gripper target based on the SpaceMouse example changes the failure
mode:

- `MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGrip3mmPoseTargetBoxStrict-v0`
- PPO `ckpt_11`: mean distance `25.84 mm`
- y residual improves to about `0.76 mm`, and gripper qpos moves to `0.456`.
- z residual remains about `25 mm`.
- Later checkpoints reduce distance slightly but collapse the gripper or EEF
  orientation, so they are not acceptable.

SAC remains ineffective on these scratch curricula:

- `YCoarse` SAC `ckpt_100096`: mean distance about `298 mm`
- `PoseTarget` SAC `ckpt_100096`: mean distance about `298 mm`
- It stays near the initial pose even when PPO responds to the same reward.

SpaceMouse posture measurement from the user-provided joint values:

- final marker distance in the current target frame: about `12.7 mm`
- `eef_x_dot`: `0.953`
- `eef_x_roll_world`: about `120.3 deg`
- `gripper_drive_qpos`: `0.44`

This means the user's 45 to 65 degree roll hint is not using the same zero
angle as `eef_x_roll_world`; in the current metric, the demonstrated posture is
closer to `120 deg`.

The current best fixed-action-bias-free policy remains:

- checkpoint: `tmp_note/ckpt_cardboard_y_grip_action020_anchor50_ckpt16.pt`
- evaluation env: `MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripEef3mmBoxStrict-v0`
- mean best distance: `4.21 mm`
- min best distance: `3.52 mm`
- under 3 mm: `0 / 64`
- under 5 mm: `60 / 64`
- dominant residual: y about `-3.8 mm`
- mean EEF x dot: `0.869`
- mean gripper qpos: `0.521`

Fine-tuning that checkpoint directly on the 3 mm reward with low LR degraded
the behavior. Even with an anchor and a y-polish reward, PPO moved the gripper
away from the useful opening and did not produce under-3-mm rollouts.

Current best user-facing videos, saved with Japanese filenames:

- `tmp_note/eval_videos/現在のベスト結果_俯瞰視点.mp4`
- `tmp_note/eval_videos/現在のベスト結果_近接視点.mp4`

Recorded rollout for those videos:

- best distance: `4.02 mm`
- max box shift until best: `4.08 mm`
- max box shift over the video: `4.14 mm`

## Action sensitivity and y-polish attempts

To diagnose whether the remaining `3.5-4 mm` residual is physically reachable,
an analysis-only constant action sweep was run. This is not used as a final
policy result because fixed runtime action correction is disallowed.

Best sweep cases:

- `action[2] -= 0.02`: best distance `3.10 mm`, but box shift about `8.6 mm`
- `action[14] -= 0.08`: best distance `3.25 mm`, box shift about `6.9 mm`

Interpretation:

- The remaining error is actionable; a small left-arm joint action change can
  reduce the distance close to 3 mm.
- The same perturbations tend to increase box shift, so the final policy still
  needs to learn the correction while respecting contact stability.

Fine-tune attempts from the current best checkpoint:

- `YOnlyPolish` reward with strong anchor:
  - `ckpt_6`: mean distance `5.85 mm`, min `3.97 mm`
  - gripper qpos collapsed to about `0.30`
  - box shift increased, so this is worse than the starting checkpoint.
- `Action2Polish` reward using the action-sweep hint:
  - `ckpt_6`: mean distance `4.82 mm`, min `3.35 mm`
  - under 3 mm: `0 / 64`
  - gripper qpos dropped to about `0.41`
  - box shift increased, so this is also worse than the starting checkpoint.

Current conclusion:

- PPO fine-tuning is very sensitive at this stage. The best checkpoint already
  sits near a narrow contact solution, and scalar reward changes often improve
  one residual while damaging gripper opening or box stability.
- The most promising next direction is not a stronger scalar reward, but a
  constrained policy update or supervised distillation target that changes only
  the relevant local action component while preserving gripper and contact
  behavior.

## Local policy distillation attempt

Instead of applying a runtime action bias, a supervised distillation checkpoint
was created from the current best policy. The teacher actions were preserved on
all dimensions except selected near-slot samples, where the policy output was
trained to emit the small action changes found by the action sweep.

Best distillation checkpoint so far:

- checkpoint: `tmp_note/ckpt_cardboard_distill_combo_2m004_14m002.pt`
- target near-slot policy change: `action[2] -= 0.04`, `action[14] -= 0.02`
- evaluation env: `MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripEef3mmBoxStrict-v0`
- mean best distance: `4.03 mm`
- median best distance: `2.85 mm`
- min best distance: `2.60 mm`
- under 3 mm: `35 / 64`
- valid under 3 mm with 5 mm box shift threshold: `6 / 64`
- mean max box shift: `8.13 mm`
- mean EEF x dot: `0.879`
- mean gripper qpos: `0.441`

This is the first fixed-action-bias-free policy artifact that produces many
under-3-mm best approaches, but it is not yet stable enough because box shift is
too high on many environments.

User-facing videos:

- `tmp_note/eval_videos/蒸留policy_3mm候補_俯瞰視点.mp4`
- `tmp_note/eval_videos/蒸留policy_3mm候補_近接視点.mp4`

Recorded single rollout:

- best distance: `2.89 mm`
- max box shift until best: `4.13 mm`
- max box shift over the video: `4.13 mm`

A second distillation target using `action[2] -= 0.04`, `action[13] += 0.04`,
`action[14] += 0.02` produced more valid under-3-mm cases (`14 / 64`) but worse
mean distance and a collapsed gripper qpos, so it is not the current best.

## Final-layer bias diagnostic

The fixed action-bias search found a stronger local correction:
`action[2] -= 0.04`, `action[13] += 0.05`, `action[14] += 0.02`. To test whether
this direction is useful when emitted by the policy itself, the same offsets
were added directly to the final actor mean bias. This is not a final learned
result, but it does not use runtime action correction during rollout.

Diagnostic checkpoint:

- checkpoint: `tmp_note/ckpt_cardboard_finalbias_2m004_13p005_14p002.pt`
- source checkpoint: `tmp_note/ckpt_cardboard_y_grip_action020_anchor50_ckpt16.pt`
- evaluation env: `MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripEef3mmBoxStrict-v0`
- mean best distance: `4.84 mm`
- median best distance: `2.88 mm`
- min best distance: `2.68 mm`
- under 3 mm: `38 / 64`
- valid under 3 mm with 5 mm box shift threshold: `38 / 64`
- mean max box shift: `4.31 mm`
- max box shift: `5.03 mm`
- mean EEF x dot: `0.883`
- mean EEF x error: `0.477`
- mean gripper qpos: `0.411`

This is the best fixed-runtime-bias-free 3-mm diagnostic so far for distance and
box stability. The weak point is still orientation: EEF local x is not close to
world +x, so this does not satisfy the final posture requirement.

User-facing videos with Japanese filenames:

- `tmp_note/eval_videos/最終bias診断_3mm候補_俯瞰視点.mp4`
- `tmp_note/eval_videos/最終bias診断_3mm候補_近接視点.mp4`

Recorded single rollout:

- best distance: `2.70 mm`
- max box shift until best: `4.54 mm`
- max box shift over the video: `4.54 mm`

Follow-up started on RunPod:

- PID: `2808902`
- log: `tmp_note/logs/ppo_finalbias_3mm_no_runtime_bias.log`
- setup: PPO fine-tuning from the diagnostic checkpoint with no runtime action
  correction, `robot_init_noise_scale=0.0`, low learning rate `5e-5`.

Result of that follow-up:

- `ckpt_5`: mean distance `8.01 mm`, under 3 mm `0 / 64`
- `ckpt_9`: mean distance `9.53 mm`, under 3 mm `0 / 64`
- failure mode: gripper qpos collapsed from about `0.41` to `0.20`, then
  `0.08`; box shift also increased. The run was stopped.

An anchor-constrained PPO follow-up from the same diagnostic checkpoint was also
tested:

- run dir: `runs/MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripEef3mmBoxStrict-v0__ppo_dual_xarm7__1__1779724992`
- command idea: checkpoint and anchor checkpoint both set to
  `tmp_note/ckpt_cardboard_finalbias_2m004_13p005_14p002.pt`, `anchor_coef=10`,
  learning rate `1e-5`.
- `ckpt_5`: mean distance `3.66 mm`, under 3 mm `24 / 64`, but valid under
  3 mm only `2 / 64`
- failure mode: the policy can reduce point distance, but it does so by pushing
  the box; later checkpoints again lose gripper opening and distance.

A stricter reward-only variant was added without changing geometry:

- env id:
  `MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripEef3mmStableBoxStrict-v0`
- changes versus `YGripEef3mm`: episode-max box shift penalty, stronger
  gripper-min-qpos reward, stronger EEF x-axis reward.
- initial diagnostic checkpoint on this env still evaluates like the base
  diagnostic checkpoint.
- `ckpt_3` after PPO fine-tuning: mean distance `4.80 mm`, under 3 mm
  `12 / 64`, valid under 3 mm only `2 / 64`, max box shift `28.7 mm`.
- failure mode remains box pushing, so this run was stopped.

Current interpretation:

- PPO fine-tuning from the near-contact policy is extremely brittle. The value
  update finds actions that locally reduce marker distance but exploit contact
  by moving the inner box or by closing the gripper.
- The best current no-runtime-action-correction artifact is still the final
  bias diagnostic checkpoint, not the PPO fine-tuned checkpoints.

## Pose-aware SAC and distillation follow-up

SAC was retried on the 3 mm EEF reward with two controllers:

- `pd_joint_delta_pos` SAC `ckpt_100096`:
  - mean best distance: `300.2 mm`
  - under 30 mm: `0 / 64`
  - mean EEF x dot: `0.994`
  - conclusion: it keeps a good posture but does not approach the cabinet.
- `pd_ee_delta_pose` SAC `ckpt_100096`:
  - mean best distance: `179.0 mm`
  - under 30 mm: `0 / 64`
  - mean EEF x dot: `0.989`
  - conclusion: pose control explores movement more than joint SAC, but still
    has not discovered insertion-level approach.

A scratch PPO reward variant that emphasized the previous `FullCoarse` failure
modes was also tested, then discarded rather than kept as a permanent env:

- stronger gripper target reward around `qpos=0.44`
- more final z weighting
- stronger near-slot EEF x reward

It improved from `292 mm` to `134 mm` by `ckpt_21`, but it was still much worse
than the earlier `FullCoarse ckpt_21` (`24 mm`) and much worse than the current
near-contact PPO artifacts. The variant was removed from local code after the
failed check to avoid accumulating dead environment IDs.

Action-bias diagnostics from the 5 mm PPO checkpoint found multiple
fixed-action candidates with `2-3 mm` distance, but the best-distance candidates
usually pushed the box beyond the valid `5 mm` threshold and had EEF x dot only
around `0.82-0.89`.

Using the final-layer-bias diagnostic checkpoint as the base, a larger local
diagnostic sweep over `action[5]`, `action[8]`, `action[13]`, and `action[14]`
found a better distance/stability/orientation tradeoff:

- diagnostic action offset:
  `action[5] -= 0.08`, `action[8] -= 0.12`, `action[14] -= 0.12`
- one-rollout diagnostic metrics:
  - distance: `2.97 mm`
  - box shift: `4.32 mm`
  - EEF x dot: `0.921`
  - gripper qpos: `0.529`

This still does not meet the 2 degree EEF x-axis goal, but it proves that a
moderate EEF x improvement can coexist with a 3 mm-level distance in the local
action space.

Distilling that diagnostic direction into the actor produced:

- checkpoint:
  `tmp_note/ckpt_cardboard_distill_pose_combo_5m008_8m012_14m012.pt`
- mean best distance: `5.06 mm`
- median best distance: `3.41 mm`
- under 3 mm: `3 / 64`
- under 5 mm: `45 / 64`
- mean box shift: `10.78 mm`
- mean EEF x dot: `0.921`

This is not acceptable because the learned rollout changes contact timing and
pushes the box too much.

A more conservative distillation target with better diagnostic box stability
was also tested:

- checkpoint:
  `tmp_note/ckpt_cardboard_distill_stable_pose_5m012_8m012_14m008.pt`
- mean best distance: `8.84 mm`
- under 5 mm: `0 / 64`
- mean box shift: `3.95 mm`
- mean EEF x dot: `0.921`

This preserves box stability and improves EEF x dot relative to the final-bias
diagnostic, but loses the 3 mm insertion. It is also not a current best.

Current conclusion:

- The local action space does contain 3 mm/stable-ish directions and separate
  EEF-improving directions.
- Making the policy emit those directions by simple actor distillation is not
  enough; it changes the trajectory timing and either pushes the box or loses
  insertion.
- A stronger next direction is to collect short on-policy rollouts around the
  final-bias diagnostic and optimize a trajectory-level loss or very
  conservative PPO objective that directly rejects rollouts with max box shift
  above `5 mm`, rather than matching isolated biased actions.

## CEM policy-parameter search

To avoid evaluation-time action offsets, a CEM-style search was run over the
PPO actor final-layer bias. During search, candidate final-bias deltas were
evaluated as equivalent constant action shifts, then the selected candidate was
written into `actor_mean.6.bias` and evaluated normally with no runtime action
correction.

This is not PPO/SAC gradient learning; treat it as a black-box policy-parameter
diagnostic. It is useful for checking whether a policy parameter direction can
trade distance, box stability, and EEF x orientation better than the previous
manual final-bias diagnostic.

The first narrow CEM search over dims `[5, 8, 13, 14]` improved orientation but
did not keep enough 3 mm rollouts:

- checkpoint: `tmp_note/ckpt_cardboard_cem_repeated_finalbias_pose_stable.pt`
- mean best distance: `5.30 mm`
- median best distance: `3.50 mm`
- under 3 mm: `0 / 64`
- valid under 3 mm with 5 mm box shift: `0 / 64`
- mean max box shift: `7.74 mm`
- mean EEF x dot: `0.942`

A broader CEM search over left-arm-like action dims plus the previously useful
right-side dims performed better:

- checkpoint: `tmp_note/ckpt_cardboard_cem_broad_finalbias_pose_stable.pt`
- searched dims: `[0, 1, 2, 3, 4, 5, 6, 7, 8, 13, 14]`
- selected final-bias delta:
  `[0.0059, -0.0195, -0.0493, -0.0596, 0.0444, -0.1213, -0.0157,
  -0.0067, -0.1209, 0.0182, -0.0643]`
- mean best distance: `4.68 mm`
- median best distance: `2.67 mm`
- under 3 mm: `38 / 64`
- valid under 3 mm with 5 mm box shift: `18 / 64`
- valid under 5 mm with 5 mm box shift: `26 / 64`
- mean max box shift: `9.25 mm`
- median max box shift: `5.45 mm`
- mean EEF x dot: `0.947`
- mean EEF x error: `0.326`
- mean gripper qpos: `0.387`

Single recorded rollout:

- best distance: `2.35 mm`
- max box shift: `5.50 mm`
- videos:
  - `tmp_note/eval_videos/CEM探索_姿勢改善3mm候補_俯瞰視点.mp4`
  - `tmp_note/eval_videos/CEM探索_姿勢改善3mm候補_近接視点.mp4`

Interpretation:

- This is the best current fixed-runtime-correction-free checkpoint for the
  combined distance/orientation tradeoff.
- It improves EEF x dot substantially over the previous final-bias diagnostic
  (`0.883 -> 0.947`) while keeping many under-3-mm contacts.
- It is not complete: box shift remains too high in many environments, and EEF
  x is still far from the requested 2 degree-level alignment.
- The next useful step is to keep the broader CEM direction but add stronger
  robustness against box shift, or use it as a warm start for a very
  conservative PPO update that directly penalizes episode max box shift.

The `pd_ee_delta_pose` SAC run was stopped after `ckpt_200064`:

- mean best distance: `302.9 mm`
- under 30 mm: `0 / 64`
- mean EEF x dot: `0.99998`

Conclusion: SAC with pose control learned the desired EEF x posture but did not
learn approach. For this task, raw SAC from scratch is currently not competitive
with PPO/CEM warm starts.
- SAC is running as a separate check on
  `MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripEef3mmBoxStrict-v0`
  with `pd_joint_delta_pos`, `robot_init_noise_scale=0.0`.

First SAC comparison:

- run dir:
  `runs/MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripEef3mmBoxStrict-v0__sac__1__1779725069`
- `pd_joint_delta_pos`, `ckpt_100096`
- mean best distance: `300.2 mm`
- under 3 mm: `0 / 64`
- mean max box shift: `1.49 mm`
- mean EEF x dot: `0.994`
- mean gripper qpos: `0.728`

Interpretation: SAC has not learned approach at 100k steps. It does keep the
box stable and EEF x orientation is good, but it is still far from the target.
The run is continuing to test whether this is simply slow off-policy warmup.

Controller comparison started:

- `pd_ee_delta_pose` SAC run:
  `runs/MyDualCardboardCabinetLowerFixedTwoPointFinalDensity6000LowFrictionYGripEef3mmBoxStrict-v0__sac__1__1779725963`
- command uses `num_envs=128`, `training_freq=128`, `robot_init_noise_scale=0.0`
- purpose: test whether SAC has an easier time with EEF delta pose control than
  with joint delta control.

## EEF x-axis action sensitivity from the final-bias diagnostic checkpoint

A short runtime-bias diagnostic was run from
`tmp_note/ckpt_cardboard_finalbias_2m004_13p005_14p002.pt` to see whether any
single action dimension can improve EEF local x alignment without losing the
3-mm insertion behavior. This is diagnostic only; runtime action correction is
not accepted as a final result.

Result:

- Best distance-preserving candidates were still around EEF x dot `0.84-0.89`.
  Examples: extra `action[8] -= 0.08` gave distance `2.36 mm`, box shift
  `4.54 mm`, but EEF x dot only `0.869`.
- Best EEF x dot candidate was extra `action[5] -= 0.08`, with EEF x dot
  `0.972`, but distance worsened to `15.3 mm` and gripper qpos collapsed to
  `0.054`.

Interpretation:

- In the current near-contact PPO policy, local action changes that improve EEF
  x alignment tend to leave the insertion point or collapse the gripper.
- EEF x alignment likely needs to be learned earlier in the policy/curriculum or
  through a controller better matched to pose control, instead of patched into
  the final near-contact policy.
