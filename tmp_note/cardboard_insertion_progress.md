# Cardboard insertion progress

## Current env

The cardboard insertion task is now consolidated into a single registered env:

- `MyDualCardboardCabinet-v1`

All old experimental `MyDualCardboardCabinet...` variants were removed from
`tasks/dual/task_dual_cardboard_cabinet.py`.

Current v1 properties:

- one final insertion target in the inner-box local frame
- inner box wall thickness: `2 mm`
- inner box density: `6000`
- low friction: static `0.2`, dynamic `0.1`
- robot init noise is normally set to `0.0` for this insertion study
- controller baseline: `pd_joint_delta_pos`

## Current reward design

The reward no longer tracks the finger marker distance to the final target.

1. Gripper opening target reward around `qpos=0.44` is applied throughout the
   episode.
2. EEF local x-axis alignment toward world +x is applied throughout the episode.
3. Box shift scales the reward by `reward * (1 - penalty)`.

## Current commands

Training commands live in `some_commands2.sh` and now use `MyDualCardboardCabinet-v1`.

PPO supports separate metric and video eval env counts:

- `--num-eval-envs`: scalar metrics
- `--num-eval-video-envs`: video recording only

This avoids the previous failure where 64 eval envs were tiled into an
oversized `7680x7680` mp4.

## Historical notes kept for context

- Plain scratch SAC has repeatedly failed to discover the approach behavior.
- PPO responds better to staged rewards than SAC so far.
- CEM over the PPO actor final-layer bias showed that 3 mm-level insertion is
  locally reachable, but that result is diagnostic, not standard PPO/SAC
  gradient learning.
- Fixed runtime action offsets should not be used as final evidence.
- User-facing videos should use Japanese filenames.

## Current retained artifacts

Kept artifacts are intentionally minimal:

- `tmp_note/eval_ppo_cardboard_insert_distance.py`
- `tmp_note/eval_sac_cardboard_insert_distance.py`
- `tmp_note/optimize_ppo_final_bias_rollout.py`
- `tmp_note/ckpt_cardboard_cem_broad_finalbias_pose_stable.pt`
- `tmp_note/eval_videos/CEM探索_姿勢改善3mm候補_俯瞰視点.mp4`
- `tmp_note/eval_videos/CEM探索_姿勢改善3mm候補_近接視点.mp4`

The CEM checkpoint/video pair is retained only as a diagnostic reference for
what local policy-parameter changes can achieve. It should not be treated as the
main learned PPO/SAC result.
