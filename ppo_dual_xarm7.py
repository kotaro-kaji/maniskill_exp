import csv
from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from task_joint_hold import MyJointHoldEnv
from task_marker_align_official import MyEEAlignMarkerEnv
import task_simple
from task_pushcube_beatiful import MyPushCubeEnv
from tasks.dual.task_dual_simple import MyDualSimpleEnv
from tasks.dual.task_dual_cardboard_cabinet import MyDualCardboardCabinetEnv
from tasks.dual.task_dual_box_rotation import MyDualBoxRotationEnv
from tasks.dual.task_dual_box_rotation_regrasp import MyDualBoxRotationRegraspEnv
from tasks.dual.task_dual_box_rotation_sandwitch import MyDualBoxRotationSandwitchEnv


#デフォルトはSIM_FREQUENCY_HZ=100, CONTROL_FREQUENCY_HZ=20
SIM_FREQUENCY_HZ = 100
CONTROL_FREQUENCY_HZ = 20


class InfoDirectoryLogger:
    """Utility to dump per-key info streams into CSV files."""

    def __init__(self, root_dir: str, num_envs: int):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.num_envs = num_envs
        self._column_names = {}

    def register_field(self, key: str, column_names):
        """Register human-readable column names for a specific key."""
        if column_names is None:
            self._column_names.pop(key, None)
            return
        self._column_names[key] = list(column_names)

    def log(self, infos: dict, step_idx: int):
        for key, value in infos.items():
            if key.startswith("_") or key in ("final_info", "final_observation"):
                continue
            column_names = None
            if isinstance(value, tuple) and len(value) == 2:
                value, column_names = value
                if column_names is not None:
                    self.register_field(key, column_names)
            array = self._to_numpy(value)
            if array is None:
                continue
            if array.ndim == 0:
                array = np.repeat(array[None], self.num_envs, axis=0)
            if array.shape[0] != self.num_envs:
                continue
            array = array.reshape(self.num_envs, -1)
            for env_id in range(self.num_envs):
                self._write_row(key, env_id, step_idx, array[env_id])

    def close(self):
        pass

    def _write_row(self, key: str, env_id: int, step_idx: int, row):
        directory = self.root_dir / key
        directory.mkdir(parents=True, exist_ok=True)
        row_array = np.asarray(row).reshape(-1)
        column_headers = self._column_names.get(key)
        if column_headers is None or len(column_headers) != len(row_array):
            column_headers = [f"{key}_{i}" for i in range(len(row_array))]
        file_path = directory / f"env_{env_id:03d}.csv"
        write_header = not file_path.exists()
        with open(file_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                headers = ["step"] + column_headers
                writer.writerow(headers)
            row_values = row_array.tolist()
            writer.writerow([step_idx] + row_values)

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return None


def _extract_returns(final_info: dict, done_mask) -> list:
    episode_info = final_info.get("episode", {})
    if not episode_info:
        return []
    returns = None
    for key in ("reward", "r"):
        if key in episode_info:
            returns = episode_info[key]
            break
    if returns is None:
        return []
    returns = torch.as_tensor(returns)
    mask = torch.as_tensor(done_mask, dtype=torch.bool, device=returns.device)
    return returns[mask].detach().cpu().view(-1).tolist()


def _save_return_plot(data_points, out_dir: Path):
    if not data_points:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data_points, dtype=float)
    csv_path = out_dir / "returns.csv"
    np.savetxt(csv_path, arr, delimiter=",", header="timestep,return", comments="")
    plt.figure()
    plt.plot(arr[:, 0], arr[:, 1], marker=".", linewidth=1)
    plt.xlabel("timestep")
    plt.ylabel("return")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_dir / "returns.png")
    plt.close()


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
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    anchor_checkpoint: Optional[str] = None
    """path to a fixed policy checkpoint used to regularize PPO fine-tuning"""
    anchor_coef: float = 0.0
    """MSE weight for keeping deterministic actions close to the anchor policy"""
    print_eval_actions: bool = False
    """if toggled, prints the actions issued during evaluation"""

    # Algorithm specific arguments
    env_id: str = "MyDualSimple-v0"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
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
    control_mode: Optional[str] = None
    """the control mode to use for the environment (defaults per task)"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False
    # simulation backend: "cpu" or "physx_cuda"
    sim_backend: str = "physx_cuda"
    robot_init_noise_scale: float = 1.0
    """Scale factor for robot initial joint randomization (1.0 = default training noise, 0.0 = fixed)."""


    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5)

    def get_value(self, x):
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # Bump PhysX contact buffers to avoid overflow when many envs are run in parallel.
    gpu_mem_cfg = GPUMemoryConfig(
        max_rigid_contact_count=2**22,
        max_rigid_patch_count=2**20,
        temp_buffer_capacity=2**24,
        heap_capacity=2**22,
    )

    env_kwargs = dict(
        obs_mode="state",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        sim_config=SimConfig(
            sim_freq=SIM_FREQUENCY_HZ,
            control_freq=CONTROL_FREQUENCY_HZ,
            gpu_memory_config=gpu_mem_cfg,
        ),
        robot_init_noise_scale=args.robot_init_noise_scale,
    )

    if args.control_mode is not None:
        control_mode = args.control_mode
    else:
        control_mode = "pd_joint_delta_pos"
    env_kwargs["control_mode"] = control_mode
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    info_output_root = None
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
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
            save_trajectory=args.evaluate,
            trajectory_name="trajectory",
            max_steps_per_video=args.num_eval_steps,
            video_fps=CONTROL_FREQUENCY_HZ,
        )
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    eval_return_trace = []
    return_plot_dir = Path(f"runs/{run_name}/info")
    return_plot_dir.mkdir(parents=True, exist_ok=True)
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group="PPO",
                tags=["ppo", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    action_space_low, action_space_high = torch.from_numpy(envs.single_action_space.low).to(device), torch.from_numpy(envs.single_action_space.high).to(device)
    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))
    anchor_agent = None
    if args.anchor_checkpoint is not None:
        assert args.anchor_coef > 0
        anchor_agent = Agent(envs).to(device)
        anchor_agent.load_state_dict(torch.load(args.anchor_checkpoint))
        anchor_agent.eval()
        for param in anchor_agent.parameters():
            param.requires_grad_(False)

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            eval_info_logger = None
            if info_output_root is not None:
                iter_info_dir = os.path.join(info_output_root, f"iter_{iteration:04d}")
                eval_info_logger = InfoDirectoryLogger(iter_info_dir, args.num_eval_envs)
            num_episodes = 0
            for eval_step in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_action = agent.get_action(eval_obs, deterministic=True)
                    if args.print_eval_actions:
                        print(f"[eval] step={eval_step} actions={eval_action.detach().cpu().numpy()}")
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(eval_action)
                    if eval_info_logger is not None:
                        eval_info_logger.log({"actions": eval_action}, eval_step)
                        eval_info_logger.log(
                            {
                                "rewards": eval_rew,
                                "terminations": eval_terminations,
                                "truncations": eval_truncations,
                                "next_obs": eval_obs,
                            },
                            eval_step,
                        )
                        eval_info_logger.log(eval_infos, eval_step)
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            if eval_info_logger is not None:
                eval_info_logger.close()
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            eval_metrics_mean = {}
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                eval_metrics_mean[k] = mean
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            # Log eval return to info folder (plot+csv)
            mean_return = None
            for key in ("reward", "r"):
                if key in eval_metrics_mean:
                    mean_return = float(eval_metrics_mean[key].detach().cpu().item())
                    break
            if mean_return is not None and return_plot_dir is not None:
                eval_return_trace.append((global_step, mean_return))
                _save_return_plot(eval_return_trace, return_plot_dir)
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_time = time.time()
        for step in range(0, args.num_steps):
            attempt = 0
            while True:
                attempt += 1
                # sanitize observations in case a simulator glitch produced NaNs/Infs
                if not torch.isfinite(next_obs).all():
                    print(f"[warn] invalid observation detected at global_step={global_step}; resetting environments")
                    next_obs, _ = envs.reset()
                    next_done = torch.zeros(args.num_envs, device=device)

                try:
                    with torch.no_grad():
                        action, logprob, _, value = agent.get_action_and_value(next_obs)
                except ValueError as exc:
                    # Rarely, action_mean may contain NaN/Inf and torch distributions will raise.
                    if "Expected parameter" in str(exc):
                        print(f"[warn] invalid action distribution at global_step={global_step}; resetting environments")
                        next_obs, _ = envs.reset()
                        next_done = torch.zeros(args.num_envs, device=device)
                        if attempt >= 5:
                            raise RuntimeError("Unable to recover from invalid action distribution") from exc
                        continue
                    raise

                if not (torch.isfinite(action).all() and torch.isfinite(logprob).all() and torch.isfinite(value).all()):
                    print(f"[warn] non-finite rollout tensors at global_step={global_step}; resetting environments")
                    next_obs, _ = envs.reset()
                    next_done = torch.zeros(args.num_envs, device=device)
                    if attempt >= 5:
                        raise RuntimeError("Unable to recover from non-finite rollout tensors")
                    continue
                break

            obs[step] = next_obs
            dones[step] = next_done
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            candidate_next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            global_step += args.num_envs

            # Validate rollout tensors before committing them to buffers.
            invalid_obs = ~torch.isfinite(candidate_next_obs).view(args.num_envs, -1).all(dim=1)
            invalid_reward = ~torch.isfinite(reward.view(args.num_envs, -1)).all(dim=1)
            if invalid_obs.any() or invalid_reward.any():
                print(f"[warn] rollout step produced invalid tensors at global_step={global_step}; resetting environments")
                candidate_next_obs, _ = envs.reset()
                reward = torch.zeros_like(reward)
                terminations = torch.zeros_like(terminations)
                truncations = torch.zeros_like(truncations)
                infos = {}
                next_done = torch.zeros(args.num_envs, device=device)
            else:
                next_done = torch.logical_or(terminations, truncations).to(torch.float32)

            next_obs = candidate_next_obs
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"][done_mask]).view(-1)
        rollout_time = time.time() - rollout_time
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Final sanity check before optimization to avoid silently propagating NaNs/Infs.
        if not torch.isfinite(b_obs).all():
            raise RuntimeError("non-finite observations detected in rollout buffer")
        if not torch.isfinite(b_actions).all():
            raise RuntimeError("non-finite actions detected in rollout buffer")
        if not torch.isfinite(b_returns).all() or not torch.isfinite(b_advantages).all():
            raise RuntimeError("non-finite returns/advantages detected in rollout buffer")

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                if anchor_agent is not None:
                    with torch.no_grad():
                        anchor_action = anchor_agent.get_action(
                            b_obs[mb_inds],
                            deterministic=True,
                        )
                    current_action = agent.get_action(
                        b_obs[mb_inds],
                        deterministic=True,
                    )
                    anchor_loss = ((current_action - anchor_action) ** 2).mean()
                    loss = loss + args.anchor_coef * anchor_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
    if not args.evaluate:
        if args.save_model:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        logger.close()
    envs.close()
    eval_envs.close()
