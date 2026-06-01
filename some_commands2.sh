python ppo_xarm7.py --env_id="PickCube-v1"  --control-mode pd_joint_delta_pos \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32   --total_timesteps=25_000_000 \
    --num-steps=50 --num_eval_steps=50 --gamma=0.99 

python ppo_xarm7.py --env_id="MyXarm7PickCube-v1"  \
    --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 \
    --num_minibatches=32   --total_timesteps=25_000_000 --gamma=0.99 


python ppo_xarm7.py --env_id="MyXarm7PushCube-v1" \
    --total_timesteps=25_000_000

python sac.py --env_id="MyXarm7PushCube-v1" \
    --total_timesteps=5_000_000

#簡単な可視化
python quick_visualization/test2_gpu.py --env-id "MyXarm7PickCube-v1"

# cabinet を SpaceMouse で teleop
uv run python src/bin/teleop_spacemouse_dual_xarm7.py --env-id MyDualCardboardCabinet-v1
uv run python src/bin/teleop_keyboard_dual_xarm7.py --env-id MyDualCardboardCabinet-v1

# cardboard cabinet drawer PPO: スクラッチから箱を12cm開けて戻る現行本命
uv run python ppo_dual_xarm7.py \
    --env-id MyDualCardboardCabinet-v1 \
    --control-mode pd_joint_delta_pos --robot-init-noise-scale 0.0 \
    --num-envs 1024 --num-steps 100 --num-eval-steps 100 \
    --num-eval-envs 64 --num-eval-video-envs 4 \
    --update-epochs 8 --num-minibatches 32 \
    --total-timesteps 15_000_000 --eval-freq 5 --gamma 0.99

# cardboard cabinet drawer PPO: 固定環境ckptから小さめランダマイズ環境で継続
uv run python ppo_dual_xarm7.py \
    --checkpoint <stage1_ckpt> \
    --env-id MySingleCardboardCabinetRandomized-v1 \
    --control-mode pd_joint_delta_pos --robot-init-noise-scale 0.25 \
    --num-envs 1024 --num-steps 100 --num-eval-steps 100 \
    --num-eval-envs 64 --num-eval-video-envs 4 \
    --update-epochs 8 --num-minibatches 32 \
    --total-timesteps 15_000_000 --eval-freq 5 --gamma 0.99

# cardboard cabinet just-return PPO
uv run python ppo_dual_xarm7.py \
    --exp-name cardboard_just_return_unique_ppo_seed1 \
    --env-id MySingleCardboardCabinetJustReturn-v1 \
    --control-mode pd_joint_delta_pos --robot-init-noise-scale 0.0 \
    --num-envs 1024 --num-steps 100 --num-eval-steps 100 \
    --num-eval-envs 64 --num-eval-video-envs 4 \
    --update-epochs 8 --num-minibatches 32 \
    --total-timesteps 15_000_000 --eval-freq 5 --gamma 0.99

# single-arm checkpoint policy rollout: state と action の時系列を rollout_dual_log.csv に保存
uv run python ppo_rollout_single_xarm7.py \
    --checkpoint runs/cardboard_drawer_soft_pd_small_return_jump_ppo_seed1/ckpt_241.pt \
    --env-id MySingleCardboardCabinetRandomized-v1 \
    --control-mode pd_joint_delta_pos \
    --num-eval-envs 1 --num-eval-steps 100 \
    --sim-backend cpu

# cardboard cabinet drawer SAC: joint delta controller での比較
uv run python sac.py \
    --env-id MyDualCardboardCabinet-v1 \
    --control-mode pd_joint_delta_pos --robot-init-noise-scale 0.0 \
    --num-envs 256 --training-freq 256 --utd 0.5 --buffer-size 1_000_000 \
    --total-timesteps 15_000_000 --eval-freq 25_000 \
    --num-eval-envs 64 --num-eval-video-envs 4 --num-steps 100 --num-eval-steps 100 \
    --gamma 0.99

# cardboard cabinet drawer SAC: EEF pose controller での比較
uv run python sac.py \
    --env-id MyDualCardboardCabinet-v1 \
    --control-mode pd_ee_delta_pose --robot-init-noise-scale 0.0 \
    --num-envs 128 --training-freq 128 --utd 0.5 --buffer-size 1_000_000 \
    --total-timesteps 15_000_000 --eval-freq 25_000 \
    --num-eval-envs 64 --num-eval-video-envs 4 --num-steps 100 --num-eval-steps 100 \
    --gamma 0.99
