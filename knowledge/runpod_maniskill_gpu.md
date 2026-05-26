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

Current aliases used in this project:

```sshconfig
Host runpod-gpu
    HostName 69.30.85.182
    Port 22172
    User root
    IdentityFile ~/.ssh/id_ed25519

Host runpod-gpu-2
    HostName 157.157.221.29
    Port 30906
    User root
    IdentityFile ~/.ssh/id_ed25519
```
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

The A5000 server used an extracted NVIDIA 580.126.09 runtime under:

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

The RTX 4090 server had driver `580.126.20` and initially had CUDA working but
Vulkan failing with `vk::createInstanceUnique: ErrorIncompatibleDriver`. The
working fix was to install the loader/tools and extract matching NVIDIA GL
userspace packages without installing the full driver:

```bash
apt-get update -y
apt-get install -y libvulkan1 vulkan-tools
mkdir -p /tmp/nvidia-gl-debs /opt/nvidia-gl-580-126-20
cd /tmp/nvidia-gl-debs
apt-get download \
  libnvidia-gl-580=580.126.20-1ubuntu1 \
  libnvidia-common-580=580.126.20-1ubuntu1 \
  libnvidia-gpucomp-580=580.126.20-1ubuntu1
for deb in *.deb; do dpkg-deb -x "$deb" /opt/nvidia-gl-580-126-20; done
cat > /opt/nvidia-gl-580-126-20/nvidia_icd_egl.json <<'EOF'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "/opt/nvidia-gl-580-126-20/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.580.126.20",
    "api_version": "1.3.280"
  }
}
EOF
```

Then use this repo-local `server_env.sh`:

```bash
#!/usr/bin/env bash
export PATH="$HOME/.local/bin:$PATH"
export NVIDIA_GL=/opt/nvidia-gl-580-126-20/usr
export LD_LIBRARY_PATH="$NVIDIA_GL/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES=/opt/nvidia-gl-580-126-20/nvidia_icd_egl.json
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```

For this server, the relative-path ICD at `/etc/vulkan/icd.d/nvidia_icd.json`
continued to fail. Use the absolute `libEGL_nvidia` ICD file above.

## RunPod Cost Notes

Observed RunPod prices:

- RTX A5000: about `$0.29/hour`
- RTX 4090: about `$0.69/hour`

Use the cheaper A5000 for overnight runs when waiting is acceptable, especially
from around midnight to 7am. Use the RTX 4090 during the day when fast feedback
matters or when running short ablations that should finish quickly.

Measured on the cardboard SAC run with `num_envs=256`, `training_freq=256`,
and `utd=0.5`:

- RTX A5000: roughly `65-75 env steps/s`, 100k steps in about 24 minutes.
- RTX 4090: roughly `100-150 env steps/s`, 100k steps in about 15 minutes.

The 4090 is faster, but the SAC loop is not purely GPU-compute bound, so expect
about `1.5-2x` rather than a raw TFLOPS-speedup.

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
- Some RunPod images may not have `rsync` installed. Use `git archive | ssh tar`
  for a clean tracked-file copy, or `scp` individual files when updating a small
  set of scripts.
- On the A6000 image with NVIDIA driver `550.127.05` / CUDA `12.4`, the default
  `uv sync` resolved PyTorch `2.11.0+cu130`, which cannot use the older driver.
  Install a CUDA 12.4 PyTorch build instead and run scripts with
  `.venv/bin/python` directly so `uv run` does not resync the environment:

```bash
uv pip uninstall torch triton nvidia-cublas nvidia-cuda-cupti \
  nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft \
  nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse \
  nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink \
  nvidia-nvshmem-cu13 nvidia-nvtx
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
.venv/bin/python -m pip install --force-reinstall --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu124 \
  nvidia-cudnn-cu12==9.1.0.70 nvidia-nccl-cu12==2.21.5 \
  nvidia-cublas-cu12==12.4.5.8
```

  Add the `.venv` NVIDIA library directories to `LD_LIBRARY_PATH` in
  `server_env.sh`.
- `apt-get` may fail on some RunPod images because NVIDIA files are bind-mounted
  or partially managed by the base image. Prefer non-invasive runtime extraction
  when driver packages conflict.
- The root filesystem can fill quickly because each copied workspace may have its
  own `.venv`, and `uv sync` leaves a large cache under `/root/.cache/uv`.
  On a 20GB pod, two workspaces with `.venv` plus the uv cache can consume more
  than 16GB. Keep only the active workspace `.venv`, and remove the uv cache
  after the environment is installed:

```bash
rm -rf /root/.cache/uv /root/.cache/pip /tmp/*
rm -rf /root/work/maniskill_exp/.venv  # only if using another active workspace
```

- Do not leave training-time eval videos or per-env `info/` CSV dumps enabled for
  long runs. Record videos only for selected checkpoints after a run shows a good
  distance metric. If disk usage rises during training, check:

```bash
df -h /root/work
du -xh -d 2 /root /root/work 2>/dev/null | sort -h | tail -60
find /root/work -maxdepth 4 -type d -name info -exec du -sh {} \;
```

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
