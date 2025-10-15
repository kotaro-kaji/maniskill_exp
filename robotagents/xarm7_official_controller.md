# xArm7 Official Controller Overview

このドキュメントは `robotagents/my_xarm7_official.py` で採用した
レートリミット付きジョイントコントローラの構成と調整方法をまとめる。
目的は、ManiSkill 上の xArm7 エージェントで実機（Mode 6）の
速度・加速度制約に整合する動作を再現すること。

- [概要](#概要)
- [主要コンポーネント](#主要コンポーネント)
- [ハードウェア仕様と必要な実測値](#ハードウェア仕様と必要な実測値)
- [挙動チューニングのヒント](#挙動チューニングのヒント)
- [参考資料](#参考資料)

## 概要

- コントローラ本体は ManiSkill の
  [`RateLimitedJointPosController`](../ManiSkill/mani_skill/agents/controllers/pd_joint_pos.py)
  を使用し、速度 (`max_velocity`)・加速度 (`max_acceleration`)・（将来的に）ジャーク
  制限を適用しながら `PDJointPos` ドライブに目標角を渡す。
- 腕はデルタ角度インターフェース (`use_delta=True`) を維持しつつ
  `use_target=True` で内部ターゲットを滑らかに積算する。
- グリッパはこれまで通り `PDJointPosMimicController` を利用し、
  Mimic マップによる 1-DOF 制御を保持する。
- 既存の PD コントローラ (`pd_joint_pos`, `pd_joint_delta_pos`) も後方互換のために
  残してあり、コントローラ UID `rate_limited_pd_joint_delta_pos` を追加した。

## 主要コンポーネント

### `ManiSkill/mani_skill/agents/controllers/utils/rate_limit.py`

- `apply_rate_limits` 関数が速度・加速度（必要ならジャーク）制限を一括で処理。
- すべてバッチ演算対応で、`(num_envs, dof)` 形状を前提。
- 速度制限 → 加速度制限 → ジャーク制限 → 目標位置計算の順で適用。

### `ManiSkill/mani_skill/agents/controllers/pd_joint_pos.py`

- `RateLimitedJointPosController` を追加。
- `set_action` では最新コマンドを保存するだけで、実際の補間は
  毎ステップの `before_simulation_step` 内で実施。
- `get_state` / `set_state` は目標位置・速度・加速度を保存／復元し、
  `use_target=True` の際も滑らかにリジュームできる。

### `robotagents/my_xarm7_official.py`

- 腕用の `RateLimitedJointPosControllerConfig` を構築し、
  Mode 6 のデフォルト値（20°/s, 500°/s² 相当）を初期値として設定。
- 実機の PD ゲイン・力制限はベースエージェント (`my_xarm7`) の値をそのまま使用。
  シミュレーションを実機に合わせる場合は、実測値で上書きすること。
- グリッパ設定は既存 Mimic コントローラを流用。

## ハードウェア仕様と必要な実測値

実機挙動を正確に再現するには、以下の値をユーザー自身で取得／設定する必要がある。

| 項目 | 適用先 | 入手先メモ |
| ---- | ------ | ----------- |
| 各関節の速度上限 (`rad/s`) | `max_velocity` | `set_servo_angle` 実行時の `speed` 応答、または SDK パラメータダンプ |
| 各関節の加速度上限 (`rad/s²`) | `max_acceleration` | SDK の `mvacc` パラメータ／実機ログ |
| （任意）ジャーク上限 (`rad/s³`) | `max_jerk` | 実機挙動を微分して推定。SDK からは提供されない |
| 関節別 PD ゲイン (P/D) | `stiffness`, `damping` | ROS `xarm7_controllers.yaml` または `set_servo_angle` 実験で同定 |
| 力制限 (`Nm`) | `force_limit` | `xarm7.urdf.xacro` の `effort` 値 |

SDK から直接取得できない項目（特に PD ゲイン、ジャーク上限）は、
ユーザーによる実機計測が必須である。

## 挙動チューニングのヒント

1. **速度リミットを下回る場合**
   - `control_dt` より小さなシミュレーション刻み (`px.timestep`) で補間しているため、
     コマンド差分が小さいと速度上限には達しない。PD ゲイン調整で補う。

2. **振動する／応答が遅すぎる場合**
   - `stiffness` を上げる、`damping` を適度に調整する。
   - 加速度上限が低すぎるとレスポンスが鈍るので、実機値に合わせて再設定する。

3. **デルタ指令が累積してしまう場合**
   - `use_target=True` 前提で設計している。もし累積したくない場合は、
     環境側でリセット時に `set_state` で `desired_qpos` を初期化する。

4. **ジャーク制限を導入したい場合**
   - `RateLimitedJointPosControllerConfig.max_jerk` に値を渡せば有効化できる。
   - その際は `apply_rate_limits` に過去ステップの加速度を渡す必要があるが、
     コントローラが自動で状態管理するため、値をセットするだけで良い。

## 参考資料

- UFactory xArm SDK
  - [`xarm/x3/xarm.py`](../xArm-Python-SDK/xarm/x3/xarm.py)
  - [`xarm/x3/base.py`](../xArm-Python-SDK/xarm/x3/base.py)
  - [`xarm/core/wrapper/uxbus_cmd.py`](../xArm-Python-SDK/xarm/core/wrapper/uxbus_cmd.py)
- ROS パッケージ
  - `xarm_ros/xarm_description/urdf/xarm7/xarm7.urdf.xacro`
  - `xarm_ros/xarm_controller/config/xarm7/xarm7_controllers.yaml`
- コード参照
  - `ManiSkill/mani_skill/agents/controllers/pd_joint_pos.py`
  - `robotagents/my_xarm7_official.py`
