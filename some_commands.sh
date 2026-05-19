source /opt/ros/humble/setup.bash


# 動画保存のみ
ppo.py（状態ベース、手軽）
python examples/baselines/ppo/ppo.py --env_id PushT-v1 --evaluate --checkpoint runs/<run_name>/final_ckpt.pt --num_eval_envs 8 --num_eval_steps 120
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

python ppo_xarm7_mjcf.py --env_id MyPushCube-v1 --seed 0 --num_envs 64 --num-steps 4 --update_epochs 2 --num_minibatches 8 --total_timesteps 256 --exp-name "smoke-ppo-xarm7-pushcube"

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

python ppo_xarm7.py --env_id="MyEEAlignMarker-v0"   --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 --num_minibatches=32   --total_timesteps=50_000_000 --num-steps=120 --num_eval_steps=120 --gamma=0.99

python ppo_xarm7.py --env_id="MyEEAlignMarker-v0"   --control-mode official_pd_joint_delta_pos --num_envs=1024 --update_epochs=8 --num_minibatches=32   --total_timesteps=100_000_000 --num-steps=100 --num_eval_steps=100 --gamma=0.99 --print_eval_actions -checkpoint runs/MyEEAlignMarker-v0__ppo_xarm7__1__1760590273/ckpt_76.pt

python ppo_xarm7.py --env_id="MyEEAlignMarker-v0"   --control-mode local_pd_joint_delta_pos --num_envs=1024 --update_epochs=8 --num_minibatches=32   --total_timesteps=100_000_000 --num-steps=100 --num_eval_steps=100 --gamma=0.99 --print_eval_actions



python ppo_xarm7.py --env_id="MyEEAlignMarker-v0" \
  --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 \
  --num_minibatches=32   --total_timesteps=500_000_000 --num-steps=120 --num_eval_steps=120 --gamma=0.99


python ppo_dual_xarm7.py --env_id="MyDualSimple-v0" \
  --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 \
  --num_minibatches=32   --total_timesteps=50_000_000 --num-steps=120 --num_eval_steps=120 --gamma=0.99


python ppo_dual_xarm7.py --env_id="MyDualBoxRotation-v0" \
  --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 \
  --num_minibatches=32   --total_timesteps=500_000_000 --num-steps=120 --num_eval_steps=120 --gamma=0.99

python ppo_dual_xarm7.py --env_id="MyDualBoxRotationAblated-v0" \
  --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 \
  --num_minibatches=32   --total_timesteps=500_000_000 --num-steps=120 --num_eval_steps=120 \
  --gamma=0.99

python ppo_dual_xarm7.py --env_id="MyDualBoxRotationRegrasp-v0" \
  --control-mode pd_joint_delta_pos --num_envs=3072 --update_epochs=8 \
  --num_minibatches=32   --total_timesteps=500_000_000 --num-steps=120 --num_eval_steps=120 \
  --gamma=0.99

# サンドイッチ版（Y面を挟むpushpoint）
python ppo_dual_xarm7.py --env_id="MyDualBoxRotationSandwitch-v0" \
  --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 \
  --num_minibatches=32   --total_timesteps=500_000_000 --num-steps=120 --num_eval_steps=120 \
  --gamma=0.99

python ppo_rollout_dual_xarm7.py \
  --checkpoint ckpts/ckpt_101.pt \
  --env-id MyDualBoxRotationAblated-v0 --num_eval_steps 120
s
python ppo_rollout_dual_xarm7.py \
--checkpoint runs/MyDualSimple-v0__ppo_dual_xarm7__1__1763541530/ckpt_51.pt --env-id MyDualSimple-v0



python ppo_rollout_virtual_dual_xarm7.py \
runs/MyDualSimple-v0__ppo_dual_xarm7__1__1763541530/ckpt_26.pt


python sac_dual_xarm7.py --env_id="MyDualBoxRotationAblated-v0" \
  --num_envs=128 --utd=0.5 --buffer_size=500_000 \
  --total_timesteps=5_000_000 --eval_freq=50_000 --control-mode="pd_joint_delta_pos" \
  --num_eval_envs 4 --num-steps=120 --num_eval_steps=120

python3 visualize_urdf.py --urdf xarm7_rs_g2_ft.urdf

#Allegro touch or Allegro_right_hand_
python tests/ppo.py --env_id="RotateSingleObjectInHandLevel0-v1"   --num_envs=128 --update_epochs=8   --num_minibatches=32   --total_timesteps=50_000_000 --num-steps=120 --num_eval_steps=120 --gamma=0.99 --no-partial-reset

python sac.py --env_id="MyDualSimple-v0" \
  --num_envs=32 --utd=0.5 --buffer_size=500_000 \
  --total_timesteps=500_000 --eval_freq=50_000 --control-mode="pd_joint_delta_pos" 

python sac.py --env_id="MyDualBoxRotationAblated-v0"   \
  --num_envs=64 --utd=0.5 --buffer_size=500_000   --total_timesteps=25_000_000 \
  --eval_freq=50_000 --control-mode="pd_joint_delta_pos" --num_eval_envs=4 --num_steps 120 --num_eval_steps 120


python ppo_dual_xarm7.py --env_id="MyDualBoxRotationRegrasp-v0" \
  --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 \
  --num_minibatches=32   --total_timesteps=500_000_000 --num-steps=120 --num_eval_steps=120 \
  --gamma=0.99


# SAC
python sac.py --env_id="MyDualBoxRotation-v0"   \
  --num_envs=256 --training_freq 256 --utd=0.5 --buffer_size=1_000_000 \
  --total_timesteps=25_000_000 --eval_freq=100_000 --control-mode="pd_joint_delta_pos" \
  --num_eval_envs=64 --num_eval_video_envs=4 --num_steps 120 --num_eval_steps 120 \
  --track --wandb-project-name MyDualBoxRotation-v0_fast --robot_init_noise_scale=1.0 --seed 1

# SAC rgbd test (Grasp Cube)
python sac_rgbd_official.py --env_id="SO100GraspCube-v1" --obs_mode="rgb+segmentation" \
  --num_envs=16 --training_freq 32 --utd=0.5 --buffer_size=2_000 \
  --control-mode="pd_joint_delta_pos" --camera_width=128 --camera_height=128 \
  --total_timesteps=1_000_000 --eval_freq=10_000

# SAC rgbd example (Pick Cube)
python sac_rgbd_official.py --env_id="PickCube-v1" --obs_mode="rgb" \
  --num_envs=32 --training_freq 64 --utd=0.5 --buffer_size=300_000 \
  --control-mode="pd_ee_delta_pos" --camera_width=128 --camera_height=128 \
  --total_timesteps=1_000_000 --eval_freq=10_000 --num_eval_envs=4


#SAC rgb test(Box Rotation)
python sac_rgbd.py --env_id="MyDualBoxRotation-v0" --obs_mode="rgb" \
  --num_envs=64 --training_freq 64 --utd=0.5 --buffer_size=2_000 \
  --num_eval_envs=64 --num_eval_video_envs=8 --num_steps 120 --num_eval_steps 120 \
  --control-mode="pd_joint_delta_pos" --camera_width=256 --camera_height=192 \
  --total_timesteps=10_000_000 --eval_freq=5_000 \
  --robot_init_noise_scale=1.0

#SAC rgb on abci (Box Rotation) 
python sac_rgbd.py --env_id="MyDualBoxRotation-v0" --obs_mode="rgb" \
  --num_envs=128 --training_freq 128 --utd=0.05 --buffer_size=300_000 \
  --num_eval_envs=128 --num_eval_video_envs=8 --num_steps 120 --num_eval_steps 120 \
  --control-mode="pd_joint_delta_pos" --camera_width=256 --camera_height=192 \
  --total_timesteps=15_000_000 --eval_freq=100_000 \
  --robot_init_noise_scale=1.0


#hdf5の確認######################################################
ipython 
from robo_manip_baselines.common import RmbData, DataKey

path = "dataset/20260219_114618_ckpt_16900096/episode_004_x+0.06_y+0.00_t+0.0.rmb"

rmb = RmbData(path)
rmb.open()

print(list(rmb.keys()))
print(rmb[DataKey.MEASURED_JOINT_POS].shape)
###############################################################


#rmbデータの収集
python sac_collect_rmb_data.py --checkpoint runs/ckpt/ckpt_11700224.pt
=======
python sac.py --env_id="MyDualBoxRotationRegrasp-v0"   \
  --num_envs=64 --utd=0.5 --buffer_size=500_000   --total_timesteps=25_000_000 \
  --eval_freq=50_000 --control-mode="pd_joint_delta_pos" --num_eval_envs=4 --num_steps 200 --num_eval_steps 200

uv run python sac.py --env_id="MyDualSimple-v0" \
    --num_envs=32 --num_eval_envs=1 \
    --utd=0.5 --buffer_size=500_000 \
    --total_timesteps=500_000 --eval_freq=50_000 \
    --control-mode="pd_joint_delta_pos" \
    --no-capture-video
