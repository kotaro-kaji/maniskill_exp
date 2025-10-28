from collections import defaultdict
from dataclasses import dataclass
import os
import random
import time
from typing import Optional

import tqdm

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro

import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.structs.types import SimConfig
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from task_joint_hold import MyJointHoldEnv  # noqa: F401  # registers the task
from task_marker_align_official import MyEEAlignMarkerEnv  # noqa: F401  # registers the task
import task_simple  # noqa: F401  # registers additional tasks
from task_pushcube_beatiful import MyPushCubeEnv  # noqa: F401  # registers the task


SIM_FREQUENCY_HZ = 250
CONTROL_FREQUENCY_HZ = 50


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "SAC"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    log_freq: int = 1_000
    """logging frequency in terms of environment steps"""

    env_id: str = "MyPushCube-v1"
    """the id of the environment"""
    sim_backend: str = "physx_cuda"
    """the physics backend to use (cpu or physx_cuda)"""
    num_envs: int = 16
    """the number of parallel environments"""
    num_eval_envs: int = 16
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 25
    """evaluation frequency in terms of environment steps"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""

    total_timesteps: int = 1_000_000
    """total timesteps of the experiments"""
    buffer_size: int = 1_000_000
    """the replay memory buffer size"""
    buffer_device: str = "cuda"
    """where the replay buffer is stored. Can be 'cpu' or 'cuda' for GPU"""
    gamma: float = 0.8
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""
    learning_starts: int = 4_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network optimizer"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    utd: float = 0.5
    """update to data ratio"""
    bootstrap_at_done: str = "always"
    """the bootstrap method to use when a done signal is received. Can be 'always', 'never', or 'truncated'"""

    # to be filled in runtime
    training_freq: int = 0
    """the total number of environment steps collected between updates"""
    grad_steps_per_iteration: int = 0
    """the number of gradient updates per iteration"""
    steps_per_env: int = 0
    """the number of steps each parallel env takes per iteration"""


@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, env, num_envs: int, buffer_size: int, storage_device: torch.device, sample_device: torch.device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        self.obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.next_obs = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_observation_space.shape).to(storage_device)
        self.actions = torch.zeros((self.per_env_buffer_size, self.num_envs) + env.single_action_space.shape).to(storage_device)
        self.rewards = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)
        self.dones = torch.zeros((self.per_env_buffer_size, self.num_envs)).to(storage_device)

    def add(self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor):
        if self.storage_device == torch.device("cpu"):
            obs = obs.cpu()
            next_obs = next_obs.cpu()
            action = action.cpu()
            reward = reward.cpu()
            done = done.cpu()

        self.obs[self.pos] = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size,))
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size,))
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,))
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
        )


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=-1))


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.log_std = nn.Linear(256, np.prod(env.single_action_space.shape))

    def forward(self, x):
        hidden = self.net(x)
        mean = self.mean(hidden)
        log_std = torch.clamp(self.log_std(hidden), -20, 2)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean

    def get_eval_action(self, x):
        mean, _ = self.forward(x)
        return torch.tanh(mean)

    def to(self, device):
        self.device = device
        return super().to(device)


class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            import wandb

            wandb.log({tag: scalar_value}, step=step)
        if self.writer is not None:
            self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        if self.writer is not None:
            self.writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.training_freq = args.num_envs * args.num_steps
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.num_steps
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        sim_config=SimConfig(sim_freq=SIM_FREQUENCY_HZ, control_freq=CONTROL_FREQUENCY_HZ),
    )
    control_mode = args.control_mode
    if control_mode is None:
        control_mode = "pd_joint_delta_pos"
    if args.env_id == "MyEEAlignMarker-v0" and control_mode == "pd_joint_delta_pos":
        control_mode = "official_pd_joint_delta_pos"
    env_kwargs["control_mode"] = control_mode
    envs = gym.make(
        args.env_id,
        num_envs=args.num_envs if not args.evaluate else 1,
        reconfiguration_freq=args.reconfiguration_freq,
        **env_kwargs,
    )
    eval_envs = gym.make(
        args.env_id,
        num_envs=args.num_eval_envs,
        reconfiguration_freq=args.eval_reconfiguration_freq,
        human_render_camera_configs=dict(shader_pack="default"),
        **env_kwargs,
    )
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(
                envs,
                output_dir=f"runs/{run_name}/train_videos",
                save_trajectory=False,
                save_video_trigger=save_video_trigger,
                max_steps_per_video=args.num_steps,
                video_fps=CONTROL_FREQUENCY_HZ,
            )
        eval_envs = RecordEpisode(
            eval_envs,
            output_dir=eval_output_dir,
            save_trajectory=args.save_trajectory,
            save_video=args.capture_video,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=CONTROL_FREQUENCY_HZ,
        )
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(
        eval_envs,
        args.num_eval_envs,
        ignore_terminations=not args.eval_partial_reset,
        record_metrics=True,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb

            config = vars(args).copy()
            config["env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=args.partial_reset,
            )
            config["eval_env_cfg"] = dict(
                **env_kwargs,
                num_envs=args.num_eval_envs,
                env_id=args.env_id,
                reward_mode="normalized_dense",
                env_horizon=max_episode_steps,
                partial_reset=False,
            )
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["sac", "walltime_efficient"],
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt["actor"])
        qf1.load_state_dict(ckpt["qf1"])
        qf2.load_state_dict(ckpt["qf2"])
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
        log_alpha = torch.tensor(np.log(alpha), device=device)
        a_optimizer = None

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        env=envs,
        num_envs=args.num_envs,
        buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device),
        sample_device=device,
    )

    obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    global_step = 0
    global_update = 0
    learning_has_started = False

    global_steps_per_iteration = args.num_envs * args.steps_per_env
    cumulative_times = defaultdict(float)
    pbar = tqdm.tqdm(total=args.total_timesteps, dynamic_ncols=True)

    while global_step < args.total_timesteps:
        iteration_start_step = global_step
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            actor.eval()
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(actor.get_eval_action(eval_obs))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
            if eval_metrics_mean:
                description_parts = []
                if "success_once" in eval_metrics_mean:
                    description_parts.append(f"success_once: {eval_metrics_mean['success_once']:.2f}")
                if "return" in eval_metrics_mean:
                    description_parts.append(f"return: {eval_metrics_mean['return']:.2f}")
                if description_parts:
                    pbar.set_description(", ".join(description_parts))
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
            actor.train()

            if args.save_model:
                model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
                torch.save(
                    {
                        "actor": actor.state_dict(),
                        "qf1": qf1_target.state_dict(),
                        "qf2": qf2_target.state_dict(),
                        "log_alpha": log_alpha,
                    },
                    model_path,
                )
                print(f"model saved to {model_path}")

        rollout_time = time.perf_counter()
        for _ in range(args.steps_per_env):
            global_step += args.num_envs

            if not learning_has_started:
                actions = torch.empty(size=envs.action_space.shape, dtype=torch.float32, device=device).uniform_(-max_action, max_action)
            else:
                actions, _, _ = actor.get_action(obs)
                actions = actions.detach()

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            real_next_obs = next_obs.clone()
            if args.bootstrap_at_done == "never":
                need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
                stop_bootstrap = truncations | terminations
            elif args.bootstrap_at_done == "truncated":
                need_final_obs = truncations & (~terminations)
                stop_bootstrap = terminations
            else:
                need_final_obs = truncations | terminations
                stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool)

            if "final_observation" in infos:
                final_observation = infos["final_observation"]
                real_next_obs[need_final_obs] = final_observation[need_final_obs]

            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)

            obs = next_obs
            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos.get("_final_info")
                if isinstance(final_info, dict) and "episode" in final_info:
                    episode_info = final_info["episode"]
                    if logger is not None:
                        if done_mask is None:
                            done_mask = torch.ones_like(terminations, dtype=torch.bool)
                        for k, v in episode_info.items():
                            if isinstance(v, torch.Tensor):
                                value = v[done_mask].float().mean()
                                logger.add_scalar(f"train/{k}", value.item(), global_step)
                            else:
                                logger.add_scalar(f"train/{k}", float(v), global_step)
                else:
                    if done_mask is None:
                        done_mask_iter = [True] * len(final_info)
                    else:
                        done_mask_iter = done_mask
                    for info, mask in zip(final_info, done_mask_iter):
                        if not mask or info is None:
                            continue
                        if logger is not None and "episode" in info:
                            for k, v in info["episode"].items():
                                if isinstance(v, torch.Tensor):
                                    scalar = v.float().mean().item()
                                else:
                                    scalar = float(v)
                                logger.add_scalar(f"train/{k}", scalar, global_step)

            if global_step >= args.learning_starts:
                learning_has_started = True

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        steps_this_iteration = global_step - iteration_start_step
        remaining = max(args.total_timesteps - pbar.n, 0)
        if steps_this_iteration > 0 and remaining > 0:
            pbar.update(min(steps_this_iteration, remaining))

        if global_step >= args.learning_starts:
            update_time = time.perf_counter()
            actor_loss = torch.tensor(0.0, device=device)
            alpha_loss = torch.tensor(0.0, device=device)
            for _ in range(args.grad_steps_per_iteration):
                global_update += 1
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_obs)
                    qf1_next_target = qf1_target(data.next_obs, next_state_actions)
                    qf2_next_target = qf2_target(data.next_obs, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(data.obs, data.actions).view(-1)
                qf2_a_values = qf2(data.obs, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_update % args.policy_frequency == 0:
                    pi, log_pi, _ = actor.get_action(data.obs)
                    qf1_pi = qf1(data.obs, pi)
                    qf2_pi = qf2(data.obs, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.obs)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()
                    else:
                        alpha_loss = torch.tensor(0.0, device=device)

                if global_update % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
            update_time = time.perf_counter() - update_time
            cumulative_times["update_time"] += update_time

            if logger is not None and (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
                logger.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                logger.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                logger.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                logger.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                logger.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                logger.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                logger.add_scalar("losses/alpha", alpha, global_step)
                logger.add_scalar("time/update_time", update_time, global_step)
                logger.add_scalar("time/rollout_time", rollout_time, global_step)
                logger.add_scalar("time/rollout_fps", global_steps_per_iteration / max(rollout_time, 1e-6), global_step)
                for k, v in cumulative_times.items():
                    logger.add_scalar(f"time/total_{k}", v, global_step)
                logger.add_scalar(
                    "time/total_rollout+update_time",
                    cumulative_times.get("rollout_time", 0.0) + cumulative_times.get("update_time", 0.0),
                    global_step,
                )
                if args.autotune:
                    logger.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    if not args.evaluate and args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(
            {
                "actor": actor.state_dict(),
                "qf1": qf1_target.state_dict(),
                "qf2": qf2_target.state_dict(),
                "log_alpha": log_alpha,
            },
            model_path,
        )
        print(f"model saved to {model_path}")
    if logger is not None:
        logger.close()
    pbar.close()
    envs.close()
    eval_envs.close()
