# quick_test_pusht_gpu.py
import os, time, torch, gymnasium as gym, mani_skill.envs
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
# ---- 実験設定（短時間テスト）----
ENV_ID = "PushT-v1"
NUM_ENVS = 16            # GPUが軽いなら 32〜64 に
TOTAL_STEPS = 500        # 動作確認だけ
SEED = 0
CKPT_DIR = "runs/pusht_min/ckpts"
os.makedirs(CKPT_DIR, exist_ok=True)
# ---- 環境（GPU）----
env = gym.make(ENV_ID, num_envs=NUM_ENVS, obs_mode="state", reconfiguration_freq=1)
env = ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=True, record_metrics=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
obs, _ = env.reset(seed=SEED)
step = 0
# ---- ダミーモデル（あとで置換）----
# 例：観測次元→行動次元の線形。まず形だけ（本当の学習器に差し替え予定）
obs_dim = env.single_observation_space.shape[0]
act_dim = env.single_action_space.shape[0]
policy = torch.nn.Sequential(torch.nn.Linear(obs_dim, act_dim)).to(device)
optim = torch.optim.Adam(policy.parameters(), lr=3e-4)
# ---- ループ（ランダム行動→あとで方策に）----
while step < TOTAL_STEPS:
    with torch.no_grad():
        action = env.action_space.sample()  # ← 後で policy(obs) に差し替え
    obs, rew, terminated, truncated, info = env.step(action)
    step += 1
# ---- ckpt 保存（テンプレ）----
ckpt = {
    "model": policy.state_dict(),
    "optim": optim.state_dict(),
    "step": step,
    "meta": {"env_id": ENV_ID, "num_envs": NUM_ENVS, "seed": SEED}
}
torch.save(ckpt, os.path.join(CKPT_DIR, f"step_{step}.pt"))
print(f"[OK] quick test done. saved ckpt at step {step}.")