import gymnasium as gym
import mani_skill.envs

from collections import OrderedDict
import torch
import task_pushcube_beatiful
import tasks.dual.task_dual_simple
import tasks.dual.task_dual_box_rotation
import numpy as np

#env_id = "MyPushCube-v1"
env_id = "MyDualSimple-v0"
env_id = "MyDualBoxRotation-v0"
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
    #print(action)

    zero_action = OrderedDict((k, np.zeros_like(v)) for k, v in action.items())
    obs, reward, terminated, truncated, info = env.step(zero_action)
    contact = info.get("contact/force", None)

    #print("\n\n\n")
    #print(contact)
    #print("\n\n\n")

    done = terminated or truncated
    env.render()  # GUIウィンドウ表示
print(f"Episode finished: {info}")
env.close()
