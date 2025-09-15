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

# ガチの学習
python ppo_xarm7.py --env_id MyPushCube-v1 --seed 42 --num_envs 1024 --num-steps 8 --update_epochs 8 --num_minibatches 32 --total_timesteps 50_000_000 --num_eval_envs 16 --control-mode pd_joint_delta_pos --exp-name "ppo-MyPushCube-v1-xarm7-42"
