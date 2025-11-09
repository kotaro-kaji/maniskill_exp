source /opt/ros/humble/setup.bash


# 動画保存のみ
ppo.py（状態ベース、手軽）
python examples/baselines/ppo/ppo.py --env_id PushT-v1 --evaluate --checkpoint runs/<run_name>/final_ckpt.pt --num_eval_envs 8 --num_eval_steps 200
動画保存先: dirname(<ckpt>)/test_videos
スクリプト: mani_skill/trajectory/replay_trajectory.py
コマンド例:
python -m mani_skill.trajectory.replay_trajectory --traj_path path/to/trajectory.h5 --save_video --vis
用途: 評価時に保存された trajectory.h5 から再生・動画化。






# 確認コマンド (CPU backend で安定性チェック)
python ppo_xarm7.py \
  --env_id MyPushCube-v1 \
  --seed 0 \
  --num_envs 64 \
  --num-steps 4 \
  --update_epochs 2 \
  --num_minibatches 8 \
  --total_timesteps 256 \
  --capture_video False \
  --save_model False \
  --track False \
  --sim-backend cpu \
  --exp-name "smoke-ppo-xarm7-pushcube-cpu"






=======
# 確認コマンド
python ppo_xarm7.py --env_id MyPushCube-v1 --seed 0 --num_envs 64 --num-steps 4 --update_epochs 2 --num_minibatches 8 --total_timesteps 256 --exp-name "smoke-ppo-xarm7-pushcube"

python ppo_xarm7_mjcf.py --env_id MyPushCube-v1 --seed 0 --num
_envs 64 --num-steps 4 --update_epochs 2 --num_minibatches 8 --total_timesteps 256 --exp-name "smoke-ppo-xarm7-pushcube"

# 長時間の学習
python ppo_xarm7.py --env_id MyPushCube-v1 --seed 42 --num_envs 1024 --num-steps 8 --update_epochs 8 --num_minibatches 32 --total_timesteps 50_000_000 --num_eval_envs 16 --control-mode pd_joint_delta_pos --exp-name "ppo-MyPushCube-v1-xarm7-42"
python ppo_xarm7.py --env_id="MyPushCube-v1" \
  --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=25_000_000 --num-steps=100 --num_eval_steps=100 --gamma=0.99 

python ppo_xarm7.py --env_id="MySimpleReach-v0" \
  --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=25_000_000 --num-steps=100 --num_eval_steps=100 --gamma=0.99 

python ppo_xarm7.py --env_id="MyJointHold-v0" \
  --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=25_000_000 --num-steps=100 --num_eval_steps=100 --gamma=0.99 


python ppo_xarm7.py --env_id="MyEEAlignMarker-v0" \
  --control-mode rate_limited_pd_joint_delta_pos --num_envs=1024 --update_epochs=8 --num_minibatches=32 \
  --total_timesteps=100_000_000 --num-steps=100 --num_eval_steps=100 --gamma=0.99 --print_eval_actions






#rolloutのみ
python ppo_rollout_xarm7.py --env_id MyPushCube-v1 --checkpoint runs/MyPushCube-v1__ppo_xarm7__1__1760757372/ckpt_51.pt
python ppo_rollout_xarm7.py --env_id MySimpleReach-v0 --checkpoint runs/MySimpleReach-v0__ppo_xarm7__1__1758253720/ckpt_226.pt
python ppo_rollout_xarm7.py --env_id MyJointHold-v0 --control-mode pd_joint_delta_pos --checkpoint runs/MyJointHold-v0__ppo_xarm7__1__1759051124/final_ckpt.pt


#仮想的なrolloutのみ
python ppo_rollout_virtual_xarm7.py runs/MyJointHold-v0__ppo_xarm7__1__1759051124/final_ckpt.pt


controller check
python file_control_xarm7_delta.py --env_id MySimpleReach-v0 --control-mode rate_limited_pd_joint_delta_pos



python ppo_xarm7.py --env_id="MyEEAlignMarker-v0"   --control-mode rate_limited_pd_joint_delta_pos --num_envs=1024 --update_epochs=8 --num_minibatches=32   --total_timesteps=100_000_000 --num-steps=100 --num_eval_steps=100 --gamma=0.99 --print_eval_actions

python ppo_xarm7.py --env_id="MyEEAlignMarker-v0"   --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 --num_minibatches=32   --total_timesteps=50_000_000 --num-steps=200 --num_eval_steps=200 --gamma=0.99

python ppo_xarm7.py --env_id="MyEEAlignMarker-v0"   --control-mode official_pd_joint_delta_pos --num_envs=1024 --update_epochs=8 --num_minibatches=32   --total_timesteps=100_000_000 --num-steps=100 --num_eval_steps=100 --gamma=0.99 --print_eval_actions -checkpoint runs/MyEEAlignMarker-v0__ppo_xarm7__1__1760590273/ckpt_76.pt

python ppo_xarm7.py --env_id="MyEEAlignMarker-v0"   --control-mode local_pd_joint_delta_pos --num_envs=1024 --update_epochs=8 --num_minibatches=32   --total_timesteps=100_000_000 --num-steps=100 --num_eval_steps=100 --gamma=0.99 --print_eval_actions


python sac_xarm7.py --env_id="MyPushCube-v1" \
  --num_envs=128 --utd=0.5 --buffer_size=500_000 \
  --total_timesteps=5_000_000 --eval_freq=50_000 --control-mode="pd_joint_delta_pos" \
  --num_eval_envs 4 --num-steps=200 --num_eval_steps=200

python ppo_xarm7.py --env_id="MyEEAlignMarker-v0" \
  --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 \
  --num_minibatches=32   --total_timesteps=500_000_000 --num-steps=200 --num_eval_steps=200 --gamma=0.99


python ppo_dual_xarm7.py --env_id="MyDualSimple-v0" \
  --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 \
  --num_minibatches=32   --total_timesteps=50_000_000 --num-steps=200 --num_eval_steps=200 --gamma=0.99


python ppo_dual_xarm7.py --env_id="MyDualBoxRotation-v0" \
  --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 \
  --num_minibatches=32   --total_timesteps=500_000_000 --num-steps=200 --num_eval_steps=200 --gamma=0.99


python sac_dual_xarm7.py --env_id="MyDualBoxRotation-v0" \
  --num_envs=128 --utd=0.5 --buffer_size=500_000 \
  --total_timesteps=5_000_000 --eval_freq=50_000 --control-mode="pd_joint_delta_pos" \
  --num_eval_envs 4 --num-steps=200 --num_eval_steps=200