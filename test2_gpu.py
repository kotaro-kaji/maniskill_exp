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

    if contact is not None:
        forces, names = contact
        forces = torch.as_tensor(forces)
        if forces.ndim == 1:
            forces = forces.unsqueeze(0)
        # 特定ペアは常に表示
        for target in (
            "L_link_tcp_stick|box",
            "R_link_tcp_stick|box",
            "L_xarm_gripper_base_link|box",
            "R_xarm_gripper_base_link|box",
        ):
            if target in names:
                idx = names.index(target)
                vec = forces[:, idx]
                print(f"{target}: {vec}")
        # マスク：どこかの環境でしきい値を超える接触があるペアだけ表示
        mask = (forces > 0).any(dim=0)
        if mask.any():
            print("\nContact pairs with nonzero force:")
            for name, force_vec in zip(names, forces.T):
                if force_vec.max() <= 0:
                    continue
                # 環境ごとに表示（0ベースの env idx）
                for env_idx, val in enumerate(force_vec):
                    if val <= 0:
                        continue
                    print(f"  env {env_idx}: {name} -> {val.item():.6f}")
            # ついでに接触ペナルティも表示
            if "reward_contact_penalty" in info:
                pen = info["reward_contact_penalty"]
                if not isinstance(pen, torch.Tensor):
                    pen = torch.as_tensor(pen)
                print("  reward_contact_penalty:", pen.detach().cpu().numpy())
    done = terminated or truncated
    env.render()  # GUIウィンドウ表示
print(f"Episode finished: {info}")
env.close()
