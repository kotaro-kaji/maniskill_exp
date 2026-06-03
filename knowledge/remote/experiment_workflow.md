# Remote Experiment Workflow

Use this when running long ManiSkill training or evaluation on a remote GPU
server, especially RunPod.

## Basic Pattern

```bash
ssh runpod-gpu 'cd /root/work/maniskill_exp && source ./server_env.sh && uv run python ppo_xarm7.py ...'
```

## Keep Rental GPUs Busy

For RunPod-style rental GPUs, after diagnosing a training issue and applying a
plausible fix, restart the relevant learning run by default instead of leaving
the GPU idle. The user can often learn from the next eval video even when scalar
metrics are inconclusive, so it is usually better to keep a PPO/SAC run moving
while reporting the change and log path.

If the pod has an older driver and `uv run` would reinstall an incompatible
PyTorch build, run through the existing virtualenv interpreter instead:

```bash
cd /workspace/maniskill_exp
source ./server_env.sh
.venv/bin/python ppo_dual_xarm7.py ...
```

When starting a long run, use `nohup` and write logs under `tmp_note/logs/`.
Record the PID, log path, and exact command in the response.

For long batches, prefer `nohup` with logs:

```bash
ssh runpod-gpu 'cd /root/work/maniskill_exp && source ./server_env.sh && \
mkdir -p tmp_note/logs && \
nohup bash -lc "uv run python ppo_xarm7.py ..." > tmp_note/logs/run.out 2>&1 & echo $!'
```

Then monitor:

```bash
ssh runpod-gpu 'cd /root/work/maniskill_exp && tail -120 tmp_note/logs/run.out'
ssh runpod-gpu 'ps -p <pid> -o pid,etime,cmd'
```

## Artifact Copying

Copy checkpoints or videos back to the local workspace:

```bash
mkdir -p runs/<run-name>/test_videos
scp runpod-gpu:/root/work/maniskill_exp/runs/<run-name>/test_videos/0.mp4 \
    runs/<run-name>/test_videos/0.mp4
```

When a video is intended to be shown to the user, or the user asks to see a
video, copy or rename it to a Japanese filename by default, such as
`現在のベスト結果_近接視点.mp4` or `こちらをご参照ください.mp4`. This keeps
shared artifacts recognizable in chat and file browsers. Internal batch outputs
can still use ASCII names when scripts benefit from that.

## TensorBoard Scalar Extraction

For quick comparison without opening TensorBoard:

```bash
uv run python - <<'PY'
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

for run in sorted(Path("runs").glob("tmp_*")):
    if not list(run.glob("events.out.tfevents*")):
        continue
    ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    print(run.name)
    for key in ["eval/success_once", "eval/success_at_end", "eval/return"]:
        if key not in tags:
            continue
        vals = ea.Scalars(key)
        best = max(vals, key=lambda v: v.value)
        print(f"  {key}: best={best.value:.4f}@{best.step} last={vals[-1].value:.4f}")
PY
```

## Notes Policy

- Put throwaway investigation scripts under `tmp_note/`.
- Put reusable operational lessons under `knowledge/`.
- Do not commit remote-only files such as `server_env.sh` unless explicitly
  requested; each server may need a different runtime path.
