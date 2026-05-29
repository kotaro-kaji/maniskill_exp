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
from mani_skill.utils.wrappers.record import RecordEpisode
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    env = gym.make(
        "MyDualCardboardCabinet-v1",
        num_envs=1,
        obs_mode="state",
        reward_mode="normalized_dense",
        render_mode="rgb_array",
        sim_backend="gpu",
        control_mode="pd_joint_delta_pos",
        robot_init_noise_scale=0.0,
        reconfiguration_freq=1,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    env = RecordEpisode(
        env,
        output_dir=args.output_dir,
        save_trajectory=False,
        trajectory_name="rollout",
        max_steps_per_video=args.num_steps,
        video_fps=20,
    )
    env = ManiSkillVectorEnv(env, 1, ignore_terminations=True, record_metrics=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=device))
    agent.eval()

    obs, _ = env.reset(seed=args.seed)
    for _ in range(args.num_steps):
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=True)
        obs, _, _, _, _ = env.step(torch.clamp(action, -1.0, 1.0))
    env.close()
    print("output_dir", args.output_dir)


if __name__ == "__main__":
    main()
