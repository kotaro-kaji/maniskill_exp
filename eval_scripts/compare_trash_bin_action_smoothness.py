from __future__ import annotations

import argparse
import csv
import html
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from ppo_dual_xarm7 import (
    Agent,
    CONTROL_FREQUENCY_HZ,
    SIM_FREQUENCY_HZ,
)
from tasks.dual.task_dual_trash_bin_rolling import (  # noqa: F401
    MyDualTrashBinRollingStage3Env,
)


ARM_ACTION_INDICES = tuple(range(7)) + tuple(range(8, 15))
ACTION_CHANGE_NO_PENALTY_THRESHOLD = 0.01
ACTION_CHANGE_REFERENCE_DELTA = 0.2
ACTION_CHANGE_REFERENCE_PENALTY = 0.25


@dataclass
class RolloutResult:
    label: str
    policy_mode: str
    arm_actions: np.ndarray
    step_max_change: np.ndarray
    metrics: dict[str, float]


def build_env(args) -> ManiSkillVectorEnv:
    gpu_memory_config = GPUMemoryConfig(
        max_rigid_contact_count=2**22,
        max_rigid_patch_count=2**20,
        temp_buffer_capacity=2**24,
        heap_capacity=2**22,
    )
    env = gym.make(
        "MyDualTrashBinRollingStage3-v0",
        num_envs=args.num_envs,
        obs_mode="state",
        reward_mode="normalized_dense",
        render_mode=None,
        sim_backend=args.sim_backend,
        sim_config=SimConfig(
            sim_freq=SIM_FREQUENCY_HZ,
            control_freq=CONTROL_FREQUENCY_HZ,
            gpu_memory_config=gpu_memory_config,
        ),
        control_mode="pd_joint_delta_pos",
        robot_init_noise_scale=args.robot_init_noise_scale,
        reconfiguration_freq=1,
    )
    assert isinstance(env.action_space, gym.spaces.Dict), env.action_space
    env = FlattenActionSpaceWrapper(env)
    assert env.single_action_space.shape == (16,), env.single_action_space
    return ManiSkillVectorEnv(
        env,
        args.num_envs,
        ignore_terminations=True,
        record_metrics=True,
    )


def load_agent(
    agent: Agent,
    checkpoint: Path,
    device: torch.device,
):
    state_dict = torch.load(
        checkpoint,
        map_location=device,
        weights_only=True,
    )
    agent.load_state_dict(state_dict)
    agent.eval()


def action_change_penalty(step_max_change: np.ndarray) -> np.ndarray:
    penalty = np.maximum(
        step_max_change - ACTION_CHANGE_NO_PENALTY_THRESHOLD,
        0.0,
    )
    penalty *= ACTION_CHANGE_REFERENCE_PENALTY / (
        ACTION_CHANGE_REFERENCE_DELTA
        - ACTION_CHANGE_NO_PENALTY_THRESHOLD
    )
    return penalty


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def final_success(info: dict) -> float:
    assert "final_info" in info, info.keys()
    episode = info["final_info"]["episode"]
    for key in ("success_at_end", "success_once", "success"):
        if key in episode:
            value = episode[key]
            if isinstance(value, torch.Tensor):
                value = value.float().mean().item()
            else:
                value = np.asarray(value, dtype=np.float32).mean()
            return float(value)
    raise AssertionError(episode.keys())


def collect_rollout(
    env: ManiSkillVectorEnv,
    agent: Agent,
    checkpoint: Path,
    label: str,
    deterministic: bool,
    args,
    device: torch.device,
) -> RolloutResult:
    load_agent(agent, checkpoint, device)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    obs, _ = env.reset(seed=args.seed)
    obs = obs.to(device)

    # Use the same policy-sampling noise for baseline and candidate.
    torch.manual_seed(args.policy_noise_seed)
    action_low = torch.as_tensor(
        env.single_action_space.low,
        dtype=torch.float32,
        device=device,
    )
    action_high = torch.as_tensor(
        env.single_action_space.high,
        dtype=torch.float32,
        device=device,
    )

    actions = []
    episode_return = torch.zeros(args.num_envs, device=device)
    info = None
    for _ in range(args.num_steps):
        with torch.no_grad():
            action = agent.get_action(obs, deterministic=deterministic)
        action = torch.clamp(action, action_low, action_high)
        actions.append(action.detach().cpu().numpy())
        obs, reward, _, _, info = env.step(action)
        obs = obs.to(device)
        episode_return += reward.to(device)

    assert info is not None
    all_actions = np.stack(actions)
    arm_actions = all_actions[:, :, ARM_ACTION_INDICES]
    component_change = np.abs(np.diff(arm_actions, axis=0))
    step_max_change = component_change.max(axis=-1)
    penalty = action_change_penalty(step_max_change)

    arm_std = (
        agent.actor_logstd.exp()[0, list(ARM_ACTION_INDICES)]
        .detach()
        .cpu()
        .numpy()
    )
    metrics = {
        "return_mean": float(episode_return.mean().item()),
        "success_at_end": final_success(info),
        "component_change_mean": float(component_change.mean()),
        "component_change_p95": percentile(component_change, 95),
        "step_max_mean": float(step_max_change.mean()),
        "step_max_p50": percentile(step_max_change, 50),
        "step_max_p95": percentile(step_max_change, 95),
        "step_max_p99": percentile(step_max_change, 99),
        "step_max_max": float(step_max_change.max()),
        "step_max_over_0.2_fraction": float(
            np.mean(step_max_change > 0.2)
        ),
        "step_max_over_0.5_fraction": float(
            np.mean(step_max_change > 0.5)
        ),
        "step_max_over_1.0_fraction": float(
            np.mean(step_max_change > 1.0)
        ),
        "penalty_mean": float(penalty.mean()),
        "penalty_p95": percentile(penalty, 95),
        "penalty_max": float(penalty.max()),
        "actor_arm_std_mean": float(arm_std.mean()),
        "actor_arm_std_max": float(arm_std.max()),
    }
    policy_mode = "deterministic" if deterministic else "stochastic"
    return RolloutResult(
        label=label,
        policy_mode=policy_mode,
        arm_actions=arm_actions,
        step_max_change=step_max_change,
        metrics=metrics,
    )


def write_metrics_csv(results: list[RolloutResult], output_path: Path):
    rows = []
    for result in results:
        row = {
            "policy": result.label,
            "policy_mode": result.policy_mode,
        }
        row.update(result.metrics)
        rows.append(row)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def reduction_percent(baseline: float, candidate: float) -> float:
    assert baseline > 0.0, baseline
    return 100.0 * (baseline - candidate) / baseline


def comparison_rows(
    results: dict[tuple[str, str], RolloutResult],
) -> list[dict[str, float | str]]:
    rows = []
    smoothness_metrics = (
        "component_change_mean",
        "step_max_mean",
        "step_max_p95",
        "step_max_p99",
        "penalty_mean",
        "penalty_p95",
        "actor_arm_std_mean",
    )
    for policy_mode in ("deterministic", "stochastic"):
        baseline = results[("baseline", policy_mode)]
        candidate = results[("candidate", policy_mode)]
        for metric in smoothness_metrics:
            baseline_value = baseline.metrics[metric]
            candidate_value = candidate.metrics[metric]
            rows.append(
                {
                    "policy_mode": policy_mode,
                    "metric": metric,
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "reduction_percent": reduction_percent(
                        baseline_value,
                        candidate_value,
                    ),
                }
            )
    return rows


def write_comparison_csv(
    rows: list[dict[str, float | str]],
    output_path: Path,
):
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_representative_trace_csv(
    results: list[RolloutResult],
    output_path: Path,
):
    fieldnames = ["policy", "policy_mode", "step"]
    fieldnames += [f"arm_action_{index + 1}" for index in range(14)]
    fieldnames += [f"arm_delta_{index + 1}" for index in range(14)]
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for result in results:
            actions = result.arm_actions[:, 0]
            delta = np.zeros_like(actions)
            delta[1:] = np.diff(actions, axis=0)
            for step in range(actions.shape[0]):
                row = {
                    "policy": result.label,
                    "policy_mode": result.policy_mode,
                    "step": step,
                }
                row.update(
                    {
                        f"arm_action_{index + 1}": actions[step, index]
                        for index in range(14)
                    }
                )
                row.update(
                    {
                        f"arm_delta_{index + 1}": delta[step, index]
                        for index in range(14)
                    }
                )
                writer.writerow(row)


def plot_action_traces(
    baseline: RolloutResult,
    candidate: RolloutResult,
    output_path: Path,
):
    datasets = (
        ("baseline", baseline.arm_actions[:, 0]),
        ("candidate", candidate.arm_actions[:, 0]),
    )
    fig, axes = plt.subplots(4, 2, figsize=(18, 12), sharex="col")
    colors = plt.get_cmap("tab10").colors
    max_signed_change = max(
        float(np.abs(np.diff(actions, axis=0)).max())
        for _, actions in datasets
    )
    change_limit = max(
        ACTION_CHANGE_REFERENCE_DELTA * 1.1,
        max_signed_change * 1.05,
    )

    for column, (label, actions) in enumerate(datasets):
        steps = np.arange(actions.shape[0])
        delta_steps = steps[1:]
        signed_change = np.diff(actions, axis=0)
        for joint_index in range(7):
            color = colors[joint_index]
            axes[0, column].plot(
                steps,
                actions[:, joint_index],
                color=color,
                label=f"joint {joint_index + 1}",
            )
            axes[1, column].plot(
                delta_steps,
                signed_change[:, joint_index],
                color=color,
            )
            axes[2, column].plot(
                steps,
                actions[:, 7 + joint_index],
                color=color,
                label=f"joint {joint_index + 1}",
            )
            axes[3, column].plot(
                delta_steps,
                signed_change[:, 7 + joint_index],
                color=color,
            )

        axes[0, column].set_title(label)
        axes[0, column].set_ylabel("left arm action")
        axes[1, column].set_ylabel("left Δaction")
        axes[2, column].set_ylabel("right arm action")
        axes[3, column].set_ylabel("right Δaction")
        axes[3, column].set_xlabel("step")
        axes[0, column].set_ylim(-1.05, 1.05)
        axes[2, column].set_ylim(-1.05, 1.05)
        axes[1, column].set_ylim(-change_limit, change_limit)
        axes[3, column].set_ylim(-change_limit, change_limit)

        for row in (1, 3):
            axes[row, column].axhline(
                ACTION_CHANGE_REFERENCE_DELTA,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label="reference Δ ±0.2",
            )
            axes[row, column].axhline(
                -ACTION_CHANGE_REFERENCE_DELTA,
                color="black",
                linestyle="--",
                linewidth=1.0,
            )
        for row in range(4):
            axes[row, column].grid(True, alpha=0.25)

    axes[0, 1].legend(loc="upper right", ncol=2)
    axes[1, 1].legend(loc="upper right")
    fig.suptitle(
        "TrashBin baseline vs candidate arm actions: "
        f"{baseline.policy_mode}, representative env 0",
        fontsize=16,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values.reshape(-1))
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def plot_change_cdf(
    results: dict[tuple[str, str], RolloutResult],
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, policy_mode in zip(
        axes,
        ("deterministic", "stochastic"),
    ):
        for policy, color in (
            ("baseline", "tab:blue"),
            ("candidate", "tab:orange"),
        ):
            result = results[(policy, policy_mode)]
            x, y = empirical_cdf(result.step_max_change)
            axis.plot(x, y, label=policy, color=color)
        axis.axvline(
            ACTION_CHANGE_REFERENCE_DELTA,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="reference Δ=0.2",
        )
        axis.set_title(policy_mode)
        axis.set_xlabel("max |Δaction| across 14 arm joints")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="lower right")
    axes[0].set_ylabel("empirical cumulative probability")
    fig.suptitle("TrashBin per-step maximum action-change distribution")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metric_bars(
    results: dict[tuple[str, str], RolloutResult],
    output_path: Path,
):
    metrics = (
        ("step_max_mean", "mean step max |Δaction|"),
        ("step_max_p95", "P95 step max |Δaction|"),
        ("penalty_mean", "mean action-change penalty"),
        ("actor_arm_std_mean", "mean actor arm std"),
    )
    modes = ("deterministic", "stochastic")
    x = np.arange(len(modes))
    width = 0.35
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, (metric, title) in zip(axes.flat, metrics):
        baseline_values = [
            results[("baseline", mode)].metrics[metric] for mode in modes
        ]
        candidate_values = [
            results[("candidate", mode)].metrics[metric] for mode in modes
        ]
        baseline_bars = axis.bar(
            x - width / 2,
            baseline_values,
            width,
            label="baseline",
        )
        candidate_bars = axis.bar(
            x + width / 2,
            candidate_values,
            width,
            label="candidate",
        )
        axis.bar_label(baseline_bars, fmt="%.3f", padding=3)
        axis.bar_label(candidate_bars, fmt="%.3f", padding=3)
        axis.set_xticks(x, modes)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend()
    fig.suptitle("TrashBin action-smoothness metrics")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_html_report(
    results: dict[tuple[str, str], RolloutResult],
    comparisons: list[dict[str, float | str]],
    args,
    output_path: Path,
):
    metric_rows = []
    for policy_mode in ("deterministic", "stochastic"):
        for policy in ("baseline", "candidate"):
            result = results[(policy, policy_mode)]
            metric_rows.append(
                "<tr>"
                f"<td>{html.escape(policy)}</td>"
                f"<td>{html.escape(policy_mode)}</td>"
                f"<td>{result.metrics['success_at_end']:.3f}</td>"
                f"<td>{result.metrics['return_mean']:.3f}</td>"
                f"<td>{result.metrics['step_max_mean']:.4f}</td>"
                f"<td>{result.metrics['step_max_p95']:.4f}</td>"
                f"<td>{result.metrics['penalty_mean']:.4f}</td>"
                f"<td>{result.metrics['actor_arm_std_mean']:.4f}</td>"
                "</tr>"
            )

    comparison_table_rows = []
    for row in comparisons:
        comparison_table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['policy_mode']))}</td>"
            f"<td>{html.escape(str(row['metric']))}</td>"
            f"<td>{float(row['baseline']):.5f}</td>"
            f"<td>{float(row['candidate']):.5f}</td>"
            f"<td>{float(row['reduction_percent']):+.2f}%</td>"
            "</tr>"
        )

    content = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>TrashBin action smoothness comparison</title>
<style>
body {{ font-family: sans-serif; max-width: 1200px; margin: 32px auto; line-height: 1.6; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
th, td {{ border: 1px solid #ccc; padding: 6px 9px; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
img {{ width: 100%; margin: 12px 0 28px; }}
code {{ background: #f2f2f2; padding: 2px 4px; }}
</style>
</head>
<body>
<h1>TrashBin action滑らかさ比較</h1>
<p>baseline: <code>{html.escape(str(args.baseline_checkpoint))}</code><br>
candidate: <code>{html.escape(str(args.candidate_checkpoint))}</code><br>
backend: <code>{html.escape(args.sim_backend)}</code>,
num_envs: {args.num_envs}, seed: {args.seed}, horizon: {args.num_steps}</p>
<p>各stepで左右arm 14関節のうち最大の
<code>|Δaction|</code>を滑らかさ指標として使用しています。
reductionが正ならcandidateの方が小さく、滑らかです。</p>

<h2>主要指標</h2>
<table>
<thead><tr><th>policy</th><th>mode</th><th>success</th><th>return</th>
<th>mean max Δ</th><th>P95 max Δ</th><th>mean penalty</th><th>actor std</th></tr></thead>
<tbody>{''.join(metric_rows)}</tbody>
</table>

<h2>減少率</h2>
<table>
<thead><tr><th>mode</th><th>metric</th><th>baseline</th>
<th>candidate</th><th>reduction</th></tr></thead>
<tbody>{''.join(comparison_table_rows)}</tbody>
</table>

<h2>定量比較</h2>
<img src="action_smoothness_metrics.png" alt="metric bars">
<img src="action_change_cdf.png" alt="action change CDF">

<h2>代表環境のaction trace</h2>
<img src="action_trace_deterministic.png" alt="deterministic action trace">
<img src="action_trace_stochastic.png" alt="stochastic action trace">
</body>
</html>
"""
    output_path.write_text(content, encoding="utf-8")


def print_summary(
    results: dict[tuple[str, str], RolloutResult],
    comparisons: list[dict[str, float | str]],
):
    for policy_mode in ("deterministic", "stochastic"):
        print(f"{policy_mode}:")
        for policy in ("baseline", "candidate"):
            metrics = results[(policy, policy_mode)].metrics
            print(
                f"  {policy}: success={metrics['success_at_end']:.3f} "
                f"return={metrics['return_mean']:.3f} "
                f"step_max_mean={metrics['step_max_mean']:.4f} "
                f"step_max_p95={metrics['step_max_p95']:.4f} "
                f"penalty_mean={metrics['penalty_mean']:.4f} "
                f"actor_std={metrics['actor_arm_std_mean']:.4f}"
            )
        for row in comparisons:
            if (
                row["policy_mode"] == policy_mode
                and row["metric"] in ("step_max_mean", "step_max_p95")
            ):
                print(
                    f"  {row['metric']}_reduction="
                    f"{float(row['reduction_percent']):+.2f}%"
                )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=Path("good_ckpts/TrashBinRolling/best_ckpt.pt"),
    )
    parser.add_argument(
        "--candidate-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--policy-noise-seed", type=int, default=10001)
    parser.add_argument("--sim-backend", default="physx_cuda")
    parser.add_argument("--robot-init-noise-scale", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    assert args.baseline_checkpoint.is_file(), args.baseline_checkpoint
    assert args.candidate_checkpoint.is_file(), args.candidate_checkpoint
    assert args.num_envs > 0, args.num_envs
    assert args.num_steps == 120, args.num_steps
    if args.sim_backend == "physx_cuda":
        assert torch.cuda.is_available()
    else:
        assert args.sim_backend == "cpu", args.sim_backend
        assert args.num_envs == 1, args.num_envs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    env = build_env(args)
    agent = Agent(env).to(device)
    env.reset(seed=args.seed)

    ordered_results = []
    results = {}
    for deterministic in (True, False):
        policy_mode = "deterministic" if deterministic else "stochastic"
        for policy, checkpoint in (
            ("baseline", args.baseline_checkpoint),
            ("candidate", args.candidate_checkpoint),
        ):
            result = collect_rollout(
                env,
                agent,
                checkpoint,
                policy,
                deterministic,
                args,
                device,
            )
            ordered_results.append(result)
            results[(policy, policy_mode)] = result
    env.close()

    comparisons = comparison_rows(results)
    write_metrics_csv(
        ordered_results,
        args.output_dir / "action_smoothness_metrics.csv",
    )
    write_comparison_csv(
        comparisons,
        args.output_dir / "action_smoothness_reduction.csv",
    )
    write_representative_trace_csv(
        ordered_results,
        args.output_dir / "representative_action_trace.csv",
    )
    plot_action_traces(
        results[("baseline", "deterministic")],
        results[("candidate", "deterministic")],
        args.output_dir / "action_trace_deterministic.png",
    )
    plot_action_traces(
        results[("baseline", "stochastic")],
        results[("candidate", "stochastic")],
        args.output_dir / "action_trace_stochastic.png",
    )
    plot_change_cdf(
        results,
        args.output_dir / "action_change_cdf.png",
    )
    plot_metric_bars(
        results,
        args.output_dir / "action_smoothness_metrics.png",
    )
    write_html_report(
        results,
        comparisons,
        args,
        args.output_dir / "index.html",
    )
    print_summary(results, comparisons)
    print(f"saved report: {(args.output_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
