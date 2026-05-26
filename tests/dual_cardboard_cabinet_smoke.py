import sys
from pathlib import Path

import gymnasium as gym
import mani_skill.envs  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasks.dual.task_dual_cardboard_cabinet import (  # noqa: F401
    MyDualCardboardCabinetEnv,
)


def main():
    env = gym.make(
        "MyDualCardboardCabinet-v1",
        num_envs=1,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
    )
    obs, info = env.reset(seed=0)
    print("reset_ok", type(obs).__name__, type(info).__name__)
    print("action_space", env.action_space)
    for step in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print(
            "step",
            step,
            "reward",
            reward,
            "terminated",
            terminated,
            "truncated",
            truncated,
        )
    env.close()


if __name__ == "__main__":
    main()
