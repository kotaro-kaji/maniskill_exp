# Cardboard SAC Results

## Best checkpoint so far

Remote host: `runpod-gpu-2`  
Run: `runs/cardboard_insert_sac_obs9_from200k_lr1e4_utd01_eval50k_1m_seed1`  
Checkpoint: `ckpt_350208.pt`

Distance eval command:

```bash
cd /root/work/maniskill_exp
source ./server_env.sh
uv run python tmp_note/eval_sac_cardboard_insert_distance.py \
  --checkpoint runs/cardboard_insert_sac_obs9_from200k_lr1e4_utd01_eval50k_1m_seed1/ckpt_350208.pt \
  --num-envs 128 --num-steps 200
```

Result:

```text
finite 128 / 128
mean 0.004242143128067255
median 0.002466617850586772
min 0.00027683991356752813
max 0.022634804248809814
under_0.005 107 / 128
under_0.010 116 / 128
under_0.020 119 / 128
under_0.030 128 / 128
```

## Important SAC findings

- The cardboard env originally exposed only 16-dimensional robot qpos state to SAC.
  Adding marker/target geometry to state obs raised obs dim to 25 and made SAC learn.
- `sac.py` saved `log_alpha` but did not reload it from checkpoint. At the useful
  100k checkpoint, `alpha` was about `0.0009`; resume had been resetting it to
  `1.0`, which destroyed the policy quickly.
- Continuing with full `utd=0.5` after the first improvement over-trains and
  collapses. Best behavior came from:
  1. train from scratch to 100k with `utd=0.5`;
  2. resume with alpha restored, `utd=0.5` to another 100k;
  3. resume the best checkpoint with `utd=0.1`, `policy_lr=q_lr=1e-4`,
     and evaluate every 50k.
- Later checkpoints after `ckpt_350208.pt` started to degrade, so the run was stopped.

## Cardboard insertion PPO diagnostics

- Correcting the eval scripts to accept `--env-id` was necessary. Earlier variant
  checkpoint distance checks were accidentally evaluated in `MyDualCardboardCabinet-v0`.
- Geometric variants such as `SlotCenter` and `HighTarget` were removed because
  they changed the task geometry. They are not valid baselines for the cabinet
  insertion task. Use `MyDualCardboardCabinet-v0` as the single source of
  geometric truth, and keep variants limited to reward shaping only.
- The removed high-target PPO result was useful only as a diagnostic that PPO and
  the controller can solve a nearby point-matching task. It should not be reported
  as task success.

## Base geometry runs started on 2026-05-24

Both runs use the same target geometry as `MyDualCardboardCabinet-v0`.

- A5000 / `runpod-gpu`: PPO with reward-only variant
  `MyDualCardboardCabinetYzShaped-v0`.
  Log: `/root/work/maniskill_exp_basegeom/tmp_note/logs/basegeom_ppo_yz_10m_seed1.out`
- RTX 4090 / `runpod-gpu-2`: SAC with base env
  `MyDualCardboardCabinet-v0`.
  Log: `/root/work/maniskill_exp_basegeom/tmp_note/logs/basegeom_sac_base_5m_seed1.out`
