# Remote Experiment Workflow

Use this when running long ManiSkill training or evaluation on a remote GPU
server, especially RunPod.

## Basic Pattern

```bash
ssh runpod-gpu 'cd /root/work/maniskill_exp && source ./server_env.sh && uv run python ppo_xarm7.py ...'
```

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

