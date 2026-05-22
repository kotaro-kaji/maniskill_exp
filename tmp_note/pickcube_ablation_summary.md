# PickCube ablation summary

Date: 2026-05-21

Target performance is the policy from `runs/tmp_pickcube_ppo_h150_from_xyz46_5m`,
which is the level shown in the eval video.

## Main result

The failure is not one isolated scalar bug. The successful behavior needs a
small set of changes that each removes a different blocker:

1. Right finger grasp direction must be negated in `Xarm7.is_grasping`.
2. The task TCP used by observation/reaching reward must be near the real grasp
   point, not the original `link_tcp`.
3. `is_grasped` must open early enough during learning (`min_force=0.1` worked;
   `0.5` did not learn in 5M steps).
4. The pre-grasp reward must explicitly align finger pads and gripper width.
5. The post-grasp reward must separately reward lift and goal placement.
6. Horizon 150 is needed for the learned trajectory; horizon 100 nearly kills
   the success rate under the same shaping.

## TensorBoard scalar comparison

All ablations below used 1024 train envs, 64 eval envs, 5M steps, deterministic
eval, 150 eval steps unless the ablation itself was horizon 100.

| variant | run | best success_once | step | best success_at_end | step | last once | last end | best return |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full | tmp_pickcube_ppo_h150_from_xyz46_5m | 0.7031 | 4608000 | 0.1406 | 3225600 | 0.7031 | 0.0625 | 128.33 |
| baseline_original | tmp_pickcube_ppo_baseline_2m | 0.0000 | 0 | 0.0000 | 0 | 0.0000 | 0.0000 | 15.71 |
| h100 | tmp_ablate_h100_5m | 0.0156 | 819200 | 0.0156 | 819200 | 0.0000 | 0.0000 | 23.21 |
| min_force_0.5 | tmp_ablate_minforce05_5m | 0.0156 | 819200 | 0.0156 | 819200 | 0.0156 | 0.0000 | 46.32 |
| link_tcp | tmp_ablate_linktcp_5m | 0.0312 | 2867200 | 0.0000 | 0 | 0.0000 | 0.0000 | 49.17 |
| no_pad_reward | tmp_ablate_nopadreward_5m | 0.0469 | 4096000 | 0.0000 | 0 | 0.0000 | 0.0000 | 42.46 |
| old_grasp_direction | tmp_ablate_oldgrasp_5m | 0.0312 | 3686400 | 0.0156 | 819200 | 0.0000 | 0.0000 | 36.23 |
| no_place_lift_shaping | tmp_ablate_noplace_5m | 0.0312 | 3276800 | 0.0312 | 3276800 | 0.0156 | 0.0000 | 38.11 |
| no_pad_width | tmp_ablate_nopadwidth_5m | 0.0312 | 3686400 | 0.0000 | 0 | 0.0000 | 0.0000 | 42.05 |
| no_pad_closed | tmp_ablate_nopadclosed_5m | 0.0156 | 1228800 | 0.0000 | 0 | 0.0000 | 0.0000 | 23.71 |

## Extra predicate check

Using the successful `h150` policy for 150 eval steps over 64 envs:

- `success_once`: 0.671875
- new grasp predicate once: 1.0
- old right-finger direction predicate once: 0.0
- lift once: 0.984375

This confirms the old right-finger direction blocks the reward gate even when
the physical behavior is already a valid grasp/lift.

## Interpretation

The minimal high-performing set is not a single reward hack. It is the following
causal chain:

- Old `link_tcp` gives a misleading reaching target; the policy can put TCP near
  the cube while the finger pads are still too high/offset.
- Without pad-center and pad-width shaping, the policy improves dense reward but
  rarely creates a two-sided grasp.
- Without the right-finger direction fix, `is_grasped` almost never becomes true,
  so lift/place rewards are gated off during learning.
- With `min_force=0.5`, the early weak contacts are not enough to open the same
  reward gate; learning stalls at near/reach behavior.
- With horizon 100, the learned pick-place sequence is too short to reach the
  same success-once rate.

Current evidence says the current full change set is close to minimal for this
performance target. Further minimization would require testing smaller numeric
coefficients, but removing any whole component above collapses success from
about 70% to at most about 5%.
