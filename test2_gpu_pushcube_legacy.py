import gymnasium as gym
import mani_skill.envs  # registers built-in envs
import numpy as np
import robotagents.my_xarm7  # registers updated custom robot UID
import robotagents.my_xarm7_mjcf  # keeps MJCF variant registered if needed
import task_pushcube_beatiful  # registers MyPushCube-v1 from your file


def main():
    # Prefer the modern PhysX CUDA backend, but gracefully fall back to legacy alias.
    backend_pairs = [
        ("physx_cuda", "cuda"),
        ("gpu", "gpu"),
    ]
    last_exc = None
    for sim_backend, render_backend in backend_pairs:
        try:
            env = gym.make(
                "MyPushCube-v1",
                obs_mode="state",
                control_mode="pd_joint_delta_pos",
                robot_uids="my_xarm7",
                render_mode="human",
                sim_backend=sim_backend,
                render_backend=render_backend,
            )
            break
        except Exception as exc:  # pragma: no cover - best effort fallback
            last_exc = exc
            env = None
    if env is None:
        raise RuntimeError(
            "Failed to create MyPushCube-v1 with the available GPU backends"
        ) from last_exc

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    obs, _ = env.reset(seed=0)
    done = False

    # Keep arm still (zeros) and smoothly open/close the gripper with small deltas.
    # Action space is normalized [-1, 1] for pd_joint_delta_pos, so choose a small step.
    while not done:
        # Read current gripper driver joint position to flip direction at bounds
        if hasattr(env, "single_action_space"):
            action_space = env.single_action_space
        else:
            action_space = env.action_space

        action = np.zeros(action_space.shape, dtype=np.float32)
        action = np.clip(action, action_space.low, action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
    env.close()


if __name__ == "__main__":
    main()
