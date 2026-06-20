import argparse
import html
import re
import subprocess
import sys
from pathlib import Path


ENV_INDICES = [
    0,
    5,
    9,
    15,
    22,
    25,
    29,
    38,
    51,
    58,
    69,
    78,
    89,
    95,
    100,
    115,
    136,
    146,
    155,
    156,
]
ENV_INDICES_TEXT = ",".join(str(env_idx) for env_idx in ENV_INDICES)
ENV_ID = "MySingleCardboardCabinetRandomized-v2"
NUM_ENVS = "160"
NUM_STEPS = "100"
SEED = "1"
ROBOT_INIT_NOISE_SCALE = "0.05"
RENDER_WIDTH = "640"
RENDER_HEIGHT = "640"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def parse_env_metrics(output: str) -> dict[int, dict[str, float]]:
    metrics = {}
    pattern = re.compile(
        r"^env (?P<env_idx>\d+) "
        r"return=(?P<return>[-+0-9.eE]+) "
        r"max_open=(?P<max_open>[-+0-9.eE]+)"
    )
    for line in output.splitlines():
        match = pattern.match(line.strip())
        if match is None:
            continue
        env_idx = int(match.group("env_idx"))
        metrics[env_idx] = {
            "return": float(match.group("return")),
            "max_open": float(match.group("max_open")),
        }
    return metrics


def render_html(
    *,
    checkpoint: Path,
    robot_uid: str,
    video_dir: Path,
    html_path: Path,
    metrics: dict[int, dict[str, float]],
):
    title = f"{checkpoint.name} {robot_uid} Randomized v2 Samples"
    video_prefix = video_dir.relative_to(html_path.parent)
    sections = []
    for env_idx in ENV_INDICES:
        item = metrics[env_idx]
        sections.append(
            f"""
      <section>
        <h2>env {env_idx}</h2>
        <dl>
          <dt>return</dt><dd>{item["return"]:.6f}</dd>
          <dt>max_open</dt><dd>{item["max_open"]:.6f}</dd>
        </dl>
        <video controls preload="metadata" src="{video_prefix}/env_{env_idx}.mp4"></video>
      </section>"""
        )

    html_path.write_text(
        f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f6f7f9;
      color: #1f2933;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      font-weight: 700;
    }}
    .meta {{
      margin: 0 0 18px;
      color: #5b6675;
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    section {{
      background: #ffffff;
      border: 1px solid #d9dee6;
      border-radius: 8px;
      padding: 12px;
    }}
    h2 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    dl {{
      display: grid;
      grid-template-columns: max-content 1fr;
      gap: 3px 10px;
      margin: 0 0 10px;
      font-size: 13px;
    }}
    dt {{
      color: #5b6675;
    }}
    dd {{
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }}
    video {{
      display: block;
      width: 100%;
      background: #000000;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(title)}</h1>
    <p class="meta">160 env eval, base seed 1, {html.escape(ENV_ID)}, robot_init_noise_scale 0.05</p>
    <div class="grid">
{''.join(sections)}
    </div>
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--robot-uid", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    assert checkpoint.exists(), checkpoint

    run_name = f"{safe_name(checkpoint.stem)}_{safe_name(args.robot_uid)}"
    video_dir = Path("outputs_to_user") / "videos" / f"{run_name}_20env_samples"
    html_path = Path("outputs_to_user") / f"{run_name}_20env_samples.html"
    video_dir.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "eval_scripts/record_cardboard_policy_video.py",
        "--checkpoint",
        str(checkpoint),
        "--env-id",
        ENV_ID,
        "--robot-uid",
        args.robot_uid,
        "--num-envs",
        NUM_ENVS,
        "--num-steps",
        NUM_STEPS,
        "--seed",
        SEED,
        "--robot-init-noise-scale",
        ROBOT_INIT_NOISE_SCALE,
        "--record-env-indices",
        ENV_INDICES_TEXT,
        "--render-width",
        RENDER_WIDTH,
        "--render-height",
        RENDER_HEIGHT,
        "--output-dir",
        str(video_dir),
    ]
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(result.stdout, end="")

    metrics = parse_env_metrics(result.stdout)
    missing_metrics = [env_idx for env_idx in ENV_INDICES if env_idx not in metrics]
    assert not missing_metrics, missing_metrics
    missing_videos = [
        env_idx for env_idx in ENV_INDICES if not (video_dir / f"env_{env_idx}.mp4").exists()
    ]
    assert not missing_videos, missing_videos

    render_html(
        checkpoint=checkpoint,
        robot_uid=args.robot_uid,
        video_dir=video_dir,
        html_path=html_path,
        metrics=metrics,
    )
    print(f"html {html_path}")
    print(f"video_dir {video_dir}")


if __name__ == "__main__":
    main()
