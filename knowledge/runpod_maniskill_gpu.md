# RunPod ManiSkill GPU Server Notes

These notes capture what worked on a RunPod-style GPU server for this repo.
The exact GPU or image may change, so treat this as a checklist and verify each
step on the new server.

## SSH

Prefer adding a host alias in `~/.ssh/config`:

```sshconfig
Host runpod-gpu
    HostName <host>
    Port <port>
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

Then use:

```bash
ssh runpod-gpu
scp <local-path> runpod-gpu:<remote-path>
```

## Repository Setup

Typical remote layout:

```bash
mkdir -p /root/work
cd /root/work
git clone <public-repo-url> maniskill_exp
cd maniskill_exp
git checkout sync
```

Use `uv` for Python:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --python 3.12
```

Sanity check:

```bash
uv run python - <<'PY'
import torch, mani_skill
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
print(mani_skill.__version__)
PY
```

## Vulkan / Rendering Problem

On the previous RunPod-like server, CUDA worked but ManiSkill GPU rendering
failed until NVIDIA GL/Vulkan runtime libraries were made available. The working
pattern was:

1. Identify that `torch.cuda.is_available()` is true but Vulkan tools or
   ManiSkill render env creation fails.
2. Install or extract matching NVIDIA GL/Vulkan libraries.
3. Point `LD_LIBRARY_PATH` and `VK_ICD_FILENAMES` at that runtime.

The working server used an extracted NVIDIA 580.126.09 runtime under:

```bash
/opt/nvidia-gl-580-126
```

and this repo-local `server_env.sh`:

```bash
#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
export NVIDIA_GL=/opt/nvidia-gl-580-126/usr
export LD_LIBRARY_PATH="$NVIDIA_GL/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/opt/nvidia-gl-580-126/nvidia_icd_egl.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```

Always source it before running repo scripts on that server:

```bash
cd /root/work/maniskill_exp
source ./server_env.sh
uv run python ...
```

Verify Vulkan when available:

```bash
source ./server_env.sh
vulkaninfo --summary
```

If `vulkaninfo` is missing, use a ManiSkill env smoke test instead.

## ManiSkill Smoke Test

Use this after setup or after changing driver/runtime variables:

```bash
source ./server_env.sh
uv run python - <<'PY'
import gymnasium as gym
import mani_skill.envs
import tasks.single_arm.pick_cube  # registers MyXarm7PickCube-v1

env = gym.make(
    "MyXarm7PickCube-v1",
    num_envs=2,
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    sim_backend="physx_cuda",
    render_mode=None,
)
obs, info = env.reset(seed=0)
print(env.action_space)
env.close()
PY
```

For video/render checks, use `render_mode="rgb_array"` after the state smoke
test succeeds.

## Known Pitfalls

- `rg` may not be installed on the remote image. Fall back to `grep -R` or
  install tools only if needed.
- `apt-get` may fail on some RunPod images because NVIDIA files are bind-mounted
  or partially managed by the base image. Prefer non-invasive runtime extraction
  when driver packages conflict.
- On GPU simulation, manually changing actor poses may require:

```python
scene._gpu_apply_all()
scene.px.gpu_update_articulation_kinematics()
scene._gpu_fetch_all()
```

Use this when doing scripted diagnostic checks.

- `RecordEpisode` may fail at close if env kwargs contain non-JSON-serializable
  objects such as `SimConfig`. The mp4 may still be written. Check the output
  directory before rerunning.
- For eval videos, if the main PPO script crashes only during JSON trajectory
  flush, copy the generated `0.mp4` first.

## Useful Verification Commands

GPU:

```bash
nvidia-smi
uv run python - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

Repo and branch:

```bash
git status --short
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
```

Long-running process:

```bash
ps -eo pid,pcpu,pmem,cmd | grep -E "ppo_xarm7|sac.py|python" | grep -v grep
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
```

