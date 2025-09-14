# rollout_record_pusht.py
import os, torch, gymnasium as gym, mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
ENV_ID = "PushT-v1"
OUT_DIR = "videos/pusht_show"
os.makedirs(OUT_DIR, exist_ok=True)
# まずは単環境で “見せる用” を撮る（安定）
env = gym.make(ENV_ID, num_envs=1, obs_mode="state", render_mode="rgb_array")
env = RecordEpisode(
    env,
    output_dir=OUT_DIR,
    save_video=True, video_fps=30,
    save_trajectory=True, trajectory_name="pusht_demo"
)
obs, _ = env.reset(seed=0)
# ---- 方策：ここをあなたの学習済み ckpt に置換 ----
# 例としてランダム行動で1エピソードだけ撮る
done = False
steps = 0
while not done and steps < 500:
    action = env.action_space.sample()
    obs, rew, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    steps += 1
print(f"[OK] saved video/trajectory to {OUT_DIR}")