# SAC PickCube probe notes

Date: 2026-05-22

Goal: check whether SAC can solve `MyXarm7PickCube-v1` with smaller or simpler
changes than the PPO reward stack.

## Commands and results

All runs were executed on the RunPod server in `/root/work/maniskill_exp` through
`tmp_note/sac_pickcube_ablation_runner.py`.

Common stable SAC settings:

```bash
uv run python tmp_note/sac_pickcube_ablation_runner.py \
  --env-id <ENV> --control-mode pd_joint_delta_pos \
  --num-envs 256 --num-eval-envs 256 --num-eval-steps 150 \
  --total-timesteps 500000 --buffer-size 1000000 --batch-size 1024 \
  --learning-starts 8192 --training-freq 256 --utd 0.05 \
  --gamma 0.99 --tau 0.01 --eval-freq 100000 --log-freq 10000 \
  --no-capture-video --no-save-trajectory --buffer-device cuda \
  --bootstrap-at-done never --no-autotune --alpha 0.2
```

| Run | Difference | Best eval/success_once | Best eval/return | Note |
| --- | --- | ---: | ---: | --- |
| `tmp_sac_fast_full_500k` | 64 env, autotune | 0.0000 | 0.1392 | alpha collapsed to about 0.005 |
| `tmp_sac_bootnever_alpha02_full_500k` | 64 env, fixed alpha 0.2 | 0.0000 | 0.8042 | reaching improved but no grasp |
| `tmp_sac_env256_alpha02_full_500k` | 256 env, fixed alpha 0.2 | 0.0039 | 0.5735 | one eval env success also appeared at step 0, so not learned |
| `tmp_sac_fixedinit_env256_alpha02_500k` | fixed cube/goal/robot init | 0.0000 | 0.8468 | randomization is not the only bottleneck |
| `tmp_sac_grasponly_env256_alpha02_500k` | success is only `is_grasped` | 0.0000 | 0.7543 | even grasp-only did not trigger contact success |
| `tmp_sac_grasponly_env256_alpha02_utd02_300k` | grasp-only, UTD 0.2 | 0.0000 at 100k | 0.2494 at 100k | same as UTD 0.05 at 100k, much slower, stopped |

## Official ManiSkill SAC check

The installed `mani-skill==3.0.0b22` site package does not ship the baseline
script files, so the official source repository was cloned at:

```text
/root/work/ManiSkill
commit ea2e7faf6b37742e0147147ad125b6d114722698
```

Official baseline script:

```text
/root/work/ManiSkill/examples/baselines/sac/sac.py
```

Panda official PickCube smoke run:

```bash
uv run python /root/work/ManiSkill/examples/baselines/sac/sac.py \
  --env_id=PickCube-v1 --num_envs=32 --num_eval_envs=32 --num_eval_steps=50 \
  --utd=0.5 --buffer_size=500000 --total_timesteps=500000 \
  --eval_freq=50000 --control-mode=pd_ee_delta_pos \
  --no-capture-video --no-save-model
```

It was stopped after the 50k eval to free the GPU:

| Env | Step | eval/success_once | eval/return | train/success_once | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| `PickCube-v1` Panda | 0 | 0.0000 | 2.1219 | 0.0000 | official env/reward/controller |
| `PickCube-v1` Panda | 50048 | 0.0000 | 20.2874 | 0.0000 | no success yet, but dense return rises strongly |

Custom xArm7 through official SAC:

```bash
uv run python tmp_note/run_official_sac_xarm7_pickcube.py \
  --env_id=MyXarm7PickCube-v1 --num_envs=32 --num_eval_envs=32 \
  --num_eval_steps=150 --utd=0.5 --buffer_size=500000 \
  --total_timesteps=500000 --eval_freq=50000 \
  --control-mode=pd_joint_delta_pos --no-capture-video --no-save-model
```

`tmp_note/run_official_sac_xarm7_pickcube.py` only imports
`tasks.single_arm.pick_cube` to register the custom env and then `runpy`s the
official SAC script. This run was stopped after the 100k eval:

| Env | Step | eval/success_once | eval/return | train/success_once | alpha |
| --- | ---: | ---: | ---: | ---: | ---: |
| `MyXarm7PickCube-v1` | 0 | 0.0000 | 0.1385 | 0.0000 | - |
| `MyXarm7PickCube-v1` | 50048 | 0.0000 | 1.5285 | 0.0000 | 0.0061 near Panda 50k, 0.0016 near xArm 94k |
| `MyXarm7PickCube-v1` | 100032 | 0.0000 | 2.0034 | 0.0000 | 0.0014 |

The official SAC algorithm alone does not close the gap. Under the current
xArm7 setup, reward improves only slowly and never reaches grasp success by
100k, while the official Panda run is already around 10x higher in dense return
by 50k.

One major remaining mismatch is controller type. The official Panda command uses
`pd_ee_delta_pos`, but `my_xarm7` currently exposes only `pd_joint_pos` and
`pd_joint_delta_pos`. This means the official SAC comparison is not controller
matched; it mainly confirms that the local SAC implementation was not the sole
cause.

## xArm7 `pd_ee_delta_pos` controller check

Added `pd_ee_delta_pos` to `robotagents/my_xarm7.py` with the same action shape
as the official Panda controller: 3D EE delta position plus the 1D mimic
gripper action.

Initial naive controller:

```python
PDEEPosControllerConfig(
    joint_names=self.arm_joint_names,
    pos_lower=-0.10,
    pos_upper=0.10,
    stiffness=self.arm_stiffness,
    damping=self.arm_damping,
    force_limit=self.arm_force_limit,
    ee_link=self.ee_link_name,
    urdf_path=self.urdf_path,
)
```

This was not stable. With random `pd_ee_delta_pos` actions on GPU, qpos/qvel
grew until the state became non-finite:

| IK alpha | Random rollout result |
| ---: | --- |
| 1.0 | obs became non-finite at step 636; qpos reached about `1.6e24`, qvel about `2.2e25` |
| 0.1 | finite for 1000 steps, but qpos still drifted to about `5.4` rad |
| 0.02 | finite for 1000 steps, but qpos still drifted to about `4.3` rad |
| 0.005 | finite for 1500 steps; qpos stayed around `2.7` rad |

So the first concrete failure cause of the naive EE controller is excessive IK
joint delta. The current controller uses:

```python
delta_solver_config=dict(type="levenberg_marquardt", alpha=0.005)
```

Official SAC with the naive `pd_ee_delta_pos` controller crashed before 50k:

| Run | Difference | Last step | Failure |
| --- | --- | ---: | --- |
| `official_sac_xarm7_pickcube_ee_500k` | default IK alpha 1.0, autotune | 27776 | Q loss exploded to about `4.7e18`, actor mean became NaN |
| `official_sac_xarm7_pickcube_ee_alpha02_500k` | default IK alpha 1.0, fixed alpha 0.2 | 14272 | actor mean became NaN |

Official SAC with the stabilized EE controller:

```bash
uv run python tmp_note/run_official_sac_xarm7_pickcube.py \
  --env_id=MyXarm7PickCube-v1 --num_envs=32 --num_eval_envs=32 \
  --num_eval_steps=150 --utd=0.5 --buffer_size=500000 \
  --total_timesteps=500000 --eval_freq=50000 \
  --control-mode=pd_ee_delta_pos --no-capture-video --no-save-model
```

| Env/controller | Step | eval/success_once | eval/return | Note |
| --- | ---: | ---: | ---: | --- |
| `MyXarm7PickCube-v1`, `pd_ee_delta_pos`, IK alpha 0.005 | 0 | 0.0000 | 0.8664 | stable initial eval |
| `MyXarm7PickCube-v1`, `pd_ee_delta_pos`, IK alpha 0.005 | 50048 | 0.0000 | 0.8453 | no learning; worse than joint delta at 50k |

This rules out a simple "add Panda-style EE delta controller" explanation. The
controller must be slowed down heavily to avoid state blow-up, but after that it
is too weak/poorly conditioned for the official SAC run to improve by 50k.

## Current interpretation

The failed cases are not explained by reward complexity alone or by local
changes in `sac.py`. SAC learns some approach/pad shaping return, but does not
discover the narrow contact state that sets `is_grasped`, even when the task is
reduced to grasp-only and the initialization is fixed.

The likely bottleneck is exploration around the grasp contact manifold: reaching
near the cube, aligning the pads, and closing with enough force must co-occur
for the success signal to appear. PPO reached this with massive on-policy
parallel exploration and the full staged reward, while these SAC probes do not
yet cross that discrete contact threshold.

## Implementation note

`sac.py` needed a small bug fix for `--no-autotune`: define `log_alpha` when
alpha is fixed, otherwise checkpoint saving crashes.
