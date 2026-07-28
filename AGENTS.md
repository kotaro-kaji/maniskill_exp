# Repository Guidelines

## Project Structure & Module Organization
This repository is a ManiSkill-based robotics experimentation workspace centered on xArm7 manipulation tasks. Core environment code lives in `tasks/` and `tasks/dual/`, robot definitions and controller variants live in `robotagents/`, and reusable scene setup code lives in `scenebuilders/`. Training and rollout entry points are top-level scripts such as `ppo_dual_xarm7.py`, `sac_dual_xarm7.py`, and `ppo_rollout_virtual_dual_xarm7.py`.

Static assets and robot descriptions are stored in `assets/`, `xarm_description/`, and top-level `*.urdf` / `*.stl` files. Use `tests/` for validation scripts, rollout checks, and plotting utilities. Put reusable policy/evaluation utilities in `eval_scripts/`; these are not throwaway notes and should be kept readable and generally useful. Treat `dont_touch_this_folder/` as archival or fragile experiment areas unless a task explicitly requires it.

Use `outputs_to_user/` for polished artifacts that the user is expected to read directly, especially HTML summaries, reports, and compact experiment briefs. Files in this directory should be self-contained, readable in a browser, and written with the assumption that the user will use them as durable context. Do not put scratch logs, raw CSV dumps, temporary probe outputs, or bulky run artifacts there; keep those in `tmp_note/`, `runs/`, or an explicit artifact directory instead.

When sharing videos with the user, prefer embedding compact MP4 files directly in an `outputs_to_user/` HTML report with normal `<video controls>` elements. Keep only the videos that are actually referenced by the report in `outputs_to_user/`; store extra or exploratory recordings under `tmp_note/`.

## Build, Test, and Development Commands
Use `uv` for environment management because the repo includes `pyproject.toml` and `uv.lock`. When running repository scripts locally, always prefix the command with `uv run` so the locked environment and interpreter are used consistently. Do not invoke repository Python entry points with plain `python ...`; use `uv run python ...` instead.

- `uv sync`: install the pinned Python dependencies.
- `uv run python ppo_dual_xarm7.py --help`: inspect training options for the main PPO entry point.
- `uv run python ppo_dual_xarm7.py --env-id MyDualBoxRotation-v0`: start a training run.
- `uv run python ppo_rollout_dual_xarm7.py --checkpoint <path>`: evaluate or replay a saved policy.
- `uv run python tests/ppo.py` or `uv run python tests/rollout_record.py`: run the lightweight validation scripts in `tests/`.

## Remote GPU / RunPod Notes
The user often provides RunPod-style GPU servers for long ManiSkill experiments.
Before setting up or debugging a new server, read `knowledge/runpod_maniskill_gpu.md`
and `knowledge/remote_experiment_workflow.md`. Reuse those notes when the server
looks similar, but still verify the actual GPU, Python environment, CUDA, Vulkan,
and ManiSkill smoke tests on the current server.

### GPU evaluation and sandbox false negatives

A failed `nvidia-smi`, CUDA, or Vulkan probe inside the sandbox is not evidence
that the machine has no usable GPU. The sandbox may hide the host GPU or deny
driver access. Before reporting that GPU execution is unavailable, rerun the
relevant checks with host/escalated execution and verify `nvidia-smi`,
`torch.cuda.is_available()`, and a small ManiSkill CUDA smoke test as needed.
If a required GPU command fails only because of sandbox restrictions, request
escalation instead of silently changing the evaluation method.

When an experiment or reference command uses GPU-vectorized environments, never
replace it with CPU-sequential rollouts for success-rate measurement or
reproduction. CPU and GPU simulation backends, vectorization, reset behavior,
and random-number consumption are not equivalent, and ManiSkill CPU execution
may support only one environment. Preserve the requested/reference simulation
backend, number of parallel environments, environment ID, controller, seed,
noise settings, and rollout horizon. A CPU run may be used only as an explicitly
labeled diagnostic; it must not be presented as a comparable evaluation result.
Always report the actual simulation backend and parallel environment count with
evaluation metrics.

Keep server-specific runtime files such as `server_env.sh` untracked unless the
user explicitly asks to commit them. Store durable operational findings in
`knowledge/`, and keep throwaway scripts/logs in `tmp_note/`.

For cardboard insertion experiments, read
`knowledge/cardboard_insertion_policy.md` before reporting results. Do not use
fixed evaluation-time action offsets; results should reflect the policy action
itself, with only normal action clipping.

When an experiment finds a strong checkpoint or otherwise produces a good
behavioral result, always generate an evaluation video and show the video path to
the user together with the scalar metrics. Do not report only metrics for good
robotics results unless video generation is genuinely blocked; if blocked,
explain the blocker and keep the checkpoint path clear.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for functions and modules, PascalCase for classes, and explicit constant names such as `SIM_FREQUENCY_HZ`. Keep task registrations and environment IDs descriptive, for example `MyDualBoxRotation-v0`. Prefer small, focused changes; many scripts are experiment entry points rather than reusable packages.

There is no enforced formatter config in the repo, so match surrounding code and keep imports, typing, and docstrings consistent with the edited file. In docs, PR notes, and shell snippets, show Python entry points as `uv run python ...`, not plain `python ...`.

When editing command-list files such as `some_commands.sh` or `some_commands2.sh`, keep new examples as simple as the surrounding examples. Do not add separate smoke-test commands unless explicitly requested. Do not add `--exp-name` unless the user asks for a named run. Do not spell out arguments that already have acceptable defaults in the target script. If the user asks to edit a specific line range, change that existing range rather than inserting a new duplicated block elsewhere.

## Testing Guidelines
There is no single pytest suite yet; verification is script-driven. Add targeted checks under `tests/` and name files after the behavior being exercised, such as `rollout_record.py` or `search_camera_transform.py`. Run validation commands via `uv run python ...`, and for task or controller changes, run at least one relevant training or rollout script and capture the exact command in your PR notes.

## Commit & Pull Request Guidelines
Recent history mixes short sync commits with conventional prefixes like `feat:` and `refactor:`. Prefer `feat:`, `fix:`, or `refactor:` followed by a concise imperative summary. Keep each commit scoped to one task, controller, or asset update.

Pull requests should explain the experiment goal, list changed entry points or environments, note any asset or URDF updates, and include evidence of validation. Attach plots or screenshots when a change affects rollouts, camera views, or robot geometry.
