source /opt/ros/humble/setup.bash


# 動画保存のみ
ppo.py（状態ベース、手軽）
python examples/baselines/ppo/ppo.py --env_id PushT-v1 --evaluate --checkpoint runs/<run_name>/final_ckpt.pt --num_eval_envs 8 --num_eval_steps 200
動画保存先: dirname(<ckpt>)/test_videos
スクリプト: mani_skill/trajectory/replay_trajectory.py
コマンド例:
python -m mani_skill.trajectory.replay_trajectory --traj_path path/to/trajectory.h5 --save_video --vis
用途: 評価時に保存された trajectory.h5 から再生・動画化。