import gymnasium as gym
import mani_skill.envs
import numpy as np
import torch

import tasks.dual.task_dual_box_rotation  # noqa: F401


def main():
    env_id = "MyDualBoxRotation-v0"
    env = gym.make(
        env_id,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="human",
    )
    print(f"Environment ID: {env_id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, _ = env.reset(seed=0)
    step = 0
    while True:
        action = env.action_space.sample()
        zero_action = {k: np.zeros_like(v) for k, v in action.items()}
        obs, reward, terminated, truncated, info = env.step(zero_action)
        step += 1

        # Penalty pairs (contact/force) and ball-box, gripper-box logs
        contact_force = info.get("contact/force")
        if contact_force is not None:
            forces, names = contact_force
            forces = torch.as_tensor(forces)
            if forces.ndim == 1:
                forces = forces.unsqueeze(0)
            mask = (forces > 0).any(dim=0)
            if mask.any():
                print(f"\n[step {step}] contact/force (nonzero):")
                for name, vec in zip(names, forces.T):
                    if vec.max() <= 0:
                        continue
                    for env_idx, val in enumerate(vec):
                        if val <= 0:
                            continue
                        print(f"  env {env_idx}: {name} -> {val.item():.6f}")
        if "contact/penalty" in info:
            pen = torch.as_tensor(info["contact/penalty"])
            if pen.max() > 0:
                print(f"[step {step}] penalty applied: {pen.cpu().numpy()}")

        for key in ("contact/ball_box_force", "contact/gripper_box_force"):
            special = info.get(key)
            if special is None:
                continue
            forces, names = special
            forces = torch.as_tensor(forces)
            if forces.ndim == 1:
                forces = forces.unsqueeze(0)
            print(f"[step {step}] {key}:")
            for name, vec in zip(names, forces.T):
                for env_idx, val in enumerate(vec):
                    print(f"  env {env_idx}: {name} -> {val.item():.6f}")

        if terminated or truncated:
            print(f"Episode finished at step {step}")
            break

        env.render()

    env.close()


if __name__ == "__main__":
    main()
