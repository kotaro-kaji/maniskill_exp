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
    --total-timesteps 25_000_000 --eval-freq 5 --gamma 0.99

# cardboard cabinet just-return PPO: tasks/single/task_single_cardboard_cabinet_just_return.py を明示的に先に登録して学習
uv run python -c 'import runpy, sys; import tasks.single.task_single_cardboard_cabinet_just_return; sys.argv = ["ppo_dual_xarm7.py"] + sys.argv[1:]; runpy.run_path("ppo_dual_xarm7.py", run_name="__main__")' \
    --exp-name cardboard_just_return_armonly_nodelta_ppo_seed1 \
    --env-id MyDualCardboardCabinet-v1 \
    --control-mode pd_joint_delta_pos --robot-init-noise-scale 0.0 \
    --num-envs 1024 --num-steps 100 --num-eval-steps 100 \
    --num-eval-envs 64 --num-eval-video-envs 4 \
    --update-epochs 8 --num-minibatches 32 \
    --total-timesteps 25_000_000 --eval-freq 5 --gamma 0.99

# cardboard cabinet just-return PPO rollout CSV:
# ckpt_41 が出たら学習を止めて、このコマンドで state/action CSV を列展開形式で保存する。
# RunPod では `uv run python` の代わりに `.venv/bin/python` を使う。
mkdir -p tmp_note/rollout_csv
rm -f rollout_dual_log.csv rollout_dual_info.jsonl
uv run python -c 'import runpy, sys; import tasks.single.task_single_cardboard_cabinet_just_return; sys.argv = ["ppo_rollout_single_xarm7.py"] + sys.argv[1:]; runpy.run_path("ppo_rollout_single_xarm7.py", run_name="__main__")' \
    --checkpoint runs/cardboard_just_return_armonly_nodelta_ppo_seed1/ckpt_41.pt \
    --env-id MyDualCardboardCabinet-v1 \
    --control-mode pd_joint_delta_pos --robot-init-noise-scale 0.0 \
    --num-eval-envs 1 --num-eval-steps 100 \
    --no-capture-video
mv rollout_dual_log.csv tmp_note/rollout_csv/cardboard_just_return_armonly_nodelta_ckpt41_state_action.csv
mv rollout_dual_info.jsonl tmp_note/rollout_csv/cardboard_just_return_armonly_nodelta_ckpt41_info.jsonl

# single-arm checkpoint policy rollout: state と action の時系列を rollout_dual_log.csv に保存
uv run python ppo_rollout_single_xarm7.py \
    --checkpoint runs/cardboard_drawer_soft_pd_small_return_jump_ppo_seed1/ckpt_241.pt \
    --control-mode pd_joint_delta_pos \
    --num-eval-envs 1 --num-eval-steps 100 \
    --no-capture-video --sim-backend cpu

# cardboard cabinet drawer SAC: joint delta controller での比較
uv run python sac.py \
    --env-id MyDualCardboardCabinet-v1 \
    --control-mode pd_joint_delta_pos --robot-init-noise-scale 0.0 \
    --num-envs 256 --training-freq 256 --utd 0.5 --buffer-size 1_000_000 \
    --total-timesteps 25_000_000 --eval-freq 25_000 \
    --num-eval-envs 64 --num-eval-video-envs 4 --num-steps 100 --num-eval-steps 100 \
    --gamma 0.99

# cardboard cabinet drawer SAC: EEF pose controller での比較
uv run python sac.py \
    --env-id MyDualCardboardCabinet-v1 \
    --control-mode pd_ee_delta_pose --robot-init-noise-scale 0.0 \
    --num-envs 128 --training-freq 128 --utd 0.5 --buffer-size 1_000_000 \
    --total-timesteps 25_000_000 --eval-freq 25_000 \
    --num-eval-envs 64 --num-eval-video-envs 4 --num-steps 100 --num-eval-steps 100 \
    --gamma 0.99
