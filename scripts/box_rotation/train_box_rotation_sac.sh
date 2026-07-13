#!/usr/bin/env bash
set -euo pipefail

python sac.py --env_id="MyDualBoxRotation-v0" --robot-uid="xarm7_ball_ee_wo_force_sensor" \
  --num_envs=256 --training_freq 256 --utd=0.5 --buffer_size=500_000 \
  --total_timesteps=25_000_000 --eval_freq=100_000 --control-mode="pd_joint_delta_pos" \
  --num_eval_envs=64 --num_eval_video_envs=4 --num_steps 120 --num_eval_steps 120 \
  --robot_init_noise_scale=1.0 --seed 1
