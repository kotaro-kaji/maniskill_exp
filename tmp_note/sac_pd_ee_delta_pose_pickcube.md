# SAC pd_ee_delta_pose PickCube notes

Date: 2026-05-23

Goal: check whether official SAC can learn `MyXarm7PickCube-v1` with `pd_ee_delta_pose`.

## Runs

- `official_sac_pickcube_ee_pose_alpha02_fast_tf256_500k`
  - Controller: current `pd_ee_delta_pose`
  - SAC: official ManiSkill SAC, `num_envs=256`, `training_freq=256`, `utd=0.05`, fixed `alpha=0.2`
  - Result at 300k: eval success 0/32, eval return about 0.69.

- `official_sac_pickcube_ee_pose_ikalpha005_alpha02_fast_500k`
  - Change: added `delta_solver_config=dict(type="levenberg_marquardt", alpha=0.005)` to `pd_ee_delta_pose`
  - Result at 300k: eval success 0/32, eval return about 0.75.
  - Conclusion: IK alpha alone does not solve SAC learning.

- `official_sac_pickcube_ee_pose_scale005_ikalpha005_500k`
  - Change: widened `pd_ee_delta_pose` to pos +/-0.05 m, rot +/-0.05 rad, with IK alpha 0.005
  - Result at 300k: eval success 0/32, eval return about 0.74.
  - Conclusion: action range alone does not solve SAC learning.

- `official_sac_panda_pickcube_ee_pose_fast_300k`
  - Env: standard `PickCube-v1`, Panda, `pd_ee_delta_pose`
  - Same fast SAC settings
  - Result at 200k: eval success 0/32, eval return improved from about 2.07 to 5.42.
  - Conclusion: this SAC setup can pick up dense reward for Panda, while xArm remains flat around 0.7.

## Notes

- Official SAC requires `training_freq >= num_envs`. With `num_envs=256` and default `training_freq=64`, `steps_per_env = training_freq // num_envs` becomes 0 and training does not progress.
- Default official SAC autotune collapsed `alpha` to about 0.001 by 65k on xArm, so fixed `--no-autotune --alpha=0.2` is better for comparison.
- Temporary controller edits were reverted after the experiments.
