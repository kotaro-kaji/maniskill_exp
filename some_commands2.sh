python ppo_xarm7.py --env_id="PickCube-v1"  --control-mode pd_joint_delta_pos \
    --num_envs=1024 --update_epochs=8 --num_minibatches=32   --total_timesteps=25_000_000 \
    --num-steps=50 --num_eval_steps=50 --gamma=0.99 

python ppo_xarm7.py --env_id="MyXarm7PickCube-v1"   --robot_uid panda \
    --control-mode pd_joint_delta_pos --num_envs=1024 --update_epochs=8 \
    --num_minibatches=32   --total_timesteps=25_000_000 --gamma=0.99 

