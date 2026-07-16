# Trash bin rolling PPO
uv run python ppo_dual_xarm7.py \
    --env-id MyDualTrashBinRolling-v0 \
    --control-mode pd_joint_delta_pos --robot-init-noise-scale 1.0 \
    --num-envs 1024 --num-steps 120 --num-eval-steps 120 \
    --num-eval-envs 8 --num-eval-video-envs 8 \
    --update-epochs 4 --num-minibatches 32 \
    --total-timesteps 20_000_000 --eval-freq 5 --gamma 0.99

# Trash bin rolling PPO on a remote GPU with logs
mkdir -p tmp_note/logs
nohup uv run python ppo_dual_xarm7.py \
    --env-id MyDualTrashBinRolling-v0 \
    --control-mode pd_joint_delta_pos --robot-init-noise-scale 1.0 \
    --num-envs 256 --num-steps 50 --num-eval-steps 120 \
    --num-eval-envs 8 --num-eval-video-envs 8 \
    --update-epochs 4 --num-minibatches 32 \
    --total-timesteps 20_000_000 --eval-freq 5 --gamma 0.99 \
    > tmp_note/logs/trash_bin_rolling_ppo_seed1.out 2>&1 &

# Trash bin rolling SAC
uv run python sac.py \
    --env-id MyDualTrashBinRolling-v0 \
    --control-mode pd_joint_delta_pos --robot-init-noise-scale 1.0 \
    --num-envs 256 --training-freq 256 --utd 0.5 \
    --buffer-size 1_000_000 --total-timesteps 20_000_000 \
    --eval-freq 100_000 --num-eval-envs 8 --num-eval-video-envs 8 \
    --num-steps 120 --num-eval-steps 120 --gamma 0.99

# Trash bin rolling SAC on a remote GPU with logs
mkdir -p tmp_note/logs
nohup uv run python sac.py \
    --env-id MyDualTrashBinRolling-v0 \
    --control-mode pd_joint_delta_pos --robot-init-noise-scale 1.0 \
    --num-envs 256 --training-freq 256 --utd 0.5 \
    --buffer-size 1_000_000 --total-timesteps 20_000_000 \
    --eval-freq 100_000 --num-eval-envs 8 --num-eval-video-envs 8 \
    --num-steps 120 --num-eval-steps 120 --gamma 0.99 \
    > tmp_note/logs/trash_bin_rolling_sac_seed1.out 2>&1 &

# Trash bin rolling Stage2 PPO from scratch
uv run python ppo_dual_xarm7.py \
    --env-id MyDualTrashBinRollingStage2-v0 \
    --control-mode pd_joint_delta_pos --robot-init-noise-scale 1.0 \
    --num-envs 1024 --num-steps 120 --num-eval-steps 120 \
    --num-eval-envs 8 --num-eval-video-envs 8 \
    --update-epochs 4 --num-minibatches 32 \
    --total-timesteps 20_000_000 --eval-freq 5 --gamma 0.99

# Trash bin rolling Stage2 PPO from scratch on a remote GPU with logs
mkdir -p tmp_note/logs
nohup uv run python ppo_dual_xarm7.py \
    --env-id MyDualTrashBinRollingStage2-v0 \
    --control-mode pd_joint_delta_pos --robot-init-noise-scale 1.0 \
    --num-envs 1024 --num-steps 120 --num-eval-steps 120 \
    --num-eval-envs 8 --num-eval-video-envs 8 \
    --update-epochs 4 --num-minibatches 32 \
    --total-timesteps 20_000_000 --eval-freq 5 --gamma 0.99 \
    > tmp_note/logs/trash_bin_stage2_ppo_scratch_seed1.out 2>&1 &
