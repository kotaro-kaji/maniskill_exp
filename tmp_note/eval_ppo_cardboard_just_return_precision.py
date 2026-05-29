import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import tasks.single.task_single_cardboard_cabinet_just_return  # noqa: F401


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        action_dim = int(np.prod(envs.single_action_space.shape))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=100)
    args = parser.parse_args()

    raw_env = gym.make(
        "MyDualCardboardCabinet-v1",
        num_envs=1,
        obs_mode="state",
        reward_mode="normalized_dense",
        control_mode="pd_joint_delta_pos",
        sim_backend="physx_cuda",
        robot_init_noise_scale=0.0,
        render_mode=None,
        reconfiguration_freq=1,
    )
    if isinstance(raw_env.action_space, gym.spaces.Dict):
        raw_env = FlattenActionSpaceWrapper(raw_env)
    env = ManiSkillVectorEnv(raw_env, 1, ignore_terminations=True, record_metrics=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    target = raw_env.unwrapped.RETURN_TARGET_QPOS.to(device=device)
    obs, _ = env.reset(seed=args.seed)

    best_arm_err = float("inf")
    best_step = -1
    best_qpos = None
    final_arm_err = None
    final_qpos = None

    print("checkpoint", args.checkpoint)
    print("seed", args.seed)
    print("target_arm", target.detach().cpu().numpy().round(6).tolist())
    print("step return arm_mean_abs_err arm_max_abs_err qpos_arm")
    for step in range(args.num_steps):
        qpos = raw_env.unwrapped.agent.robot.get_qpos()[0]
        arm_abs_err = torch.abs(qpos[:7] - target)
        arm_mean_err = float(arm_abs_err.mean())
        arm_max_err = float(arm_abs_err.max())
        return_reward = float(raw_env.unwrapped.evaluate()["return_to_target_qpos_reward"][0])

        if arm_mean_err < best_arm_err:
            best_arm_err = arm_mean_err
            best_step = step
            best_qpos = qpos.detach().clone()

        if step in [0, 1, 5, 10, 20, 30, 40, 50, 75, 99]:
            print(
                f"{step:03d}",
                f"{return_reward:.6f}",
                f"{arm_mean_err:.6f}",
                f"{arm_max_err:.6f}",
                qpos[:7].detach().cpu().numpy().round(6).tolist(),
            )

        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        obs, _, _, _, _ = env.step(torch.clamp(action, -1.0, 1.0))

        final_arm_err = arm_mean_err
        final_qpos = qpos.detach().clone()

    print("best_step", best_step)
    print("best_arm_mean_abs_err", best_arm_err)
    print("best_arm_max_abs_err", float(torch.abs(best_qpos[:7] - target).max()))
    print("best_arm_qpos", best_qpos[:7].detach().cpu().numpy().round(6).tolist())
    print("final_arm_mean_abs_err", final_arm_err)
    print("final_arm_qpos", final_qpos[:7].detach().cpu().numpy().round(6).tolist())
    env.close()


if __name__ == "__main__":
    main()
