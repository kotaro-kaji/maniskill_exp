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
uv run python src/bin/teleop_spacemouse_dual_xarm7.py --env-id MyDualCardboardCabinet-v0
