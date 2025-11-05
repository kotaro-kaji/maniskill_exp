import gymnasium as gym
import mani_skill.envs


import task_pushcube_beatiful
import tasks.du.task_dual_simple

#env_id = "MyPushCube-v1"
env_id = "MyDualSimple-v0"
#env_id = "TwoRobotPickCube-v1"

env = gym.make(
    env_id,
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    render_mode="human"
)

print(f"Environment ID: {env_id}")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")


obs, _ = env.reset(seed=0)
done = False
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    #print(f"Step: {env.num_steps}")
    #print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    done = terminated or truncated
    env.render()  # GUIウィンドウ表示
print(f"Episode finished: {info}")
env.close()
