# XArm7 Official Delta Controller Design

このドキュメントでは、`robotagents/my_xarm7_official.py` に実装した
xArm7 用デルタ角度コントローラの設計思想と根拠を体系的にまとめる。
対象読者は ManiSkill / 物理シミュレーション環境で
ハードウェア準拠の挙動を再現したい開発者である。

- [概要](#概要)
- [実機 SDK に基づく要件整理](#実機-sdk-に基づく要件整理)
- [ManiSkill 側の制約と既存実装](#maniskill-側の制約と既存実装)
- [実装構成](#実装構成)
- [ハードウェア仕様の反映ポイント](#ハードウェア仕様の反映ポイント)
- [挙動チューニングと既知の注意点](#挙動チューニングと既知の注意点)
- [参考資料](#参考資料)

## 概要

- xArm 公式 Python SDK の `move_servoj` / `set_servo_angle_j` を参考に、
  ManiSkill のデルタ角度制御 (`pd_joint_delta_pos`) と互換性を維持したまま、
  実機の速度・加速度制限と PID 由来のスムージングを近似する。
- 実機の SDK では関節コマンドが「現在姿勢 + デルタ」で与えられるため、
  `use_delta=True` で現在角度からの増分を直接目標とする（累積ターゲットにしない）。
- ManiSkill の PD ドライブが最終的なトルク生成を担うため、
  SDK の速度／加速度制限は「内部ターゲット更新」の段階で反映する。

## 実機 SDK に基づく要件整理

### 参照ソース

- リポジトリ: `xArm-Python-SDK`
  - `xarm/x3/xarm.py`
    - `set_servo_angle_j`: `move_servoj` コマンドを組み立てる主要関数。
    - `__get_joint_motion_params`: デフォルト速度（rad/s）、加速度（rad/s²）を求める。
    - `_min_joint_speed` / `_max_joint_speed` 等の内部定数（`xarm/x3/base.py`）。
  - `xarm/core/wrapper/uxbus_cmd.py`
    - `move_servoj`: 実際に 7 関節の位置・速度・加速度をバスへ送信。
- これらから得られる知見:
  1. SDK は 7 関節をまとめて同じ速度・加速度制限の範囲で補間する。
  2. 速度上限は `min(max(speed, min_joint_speed), π)` でクリップされる。
  3. 加速度上限は 20 rad/s²（`_min_joint_acc`, `_max_joint_acc`）。
  4. 実機サーボ側はさらに固有の PID を持つが、外部 API では速度・加速度を指定するのみ。

### ハードウェア固有パラメータ

- 公式 URDF (`xarm_ros/xarm_description/urdf/xarm7/xarm7.urdf.xacro`) を参照し、
  関節角度範囲と努力限界（joint effort limits）を確認した。
- 努力限界:
  - 関節1-2: 50 Nm, 関節3-5: 30 Nm, 関節6-7: 20 Nm。
  - シミュレーションでは PD ドライブの `force_limit` に反映する。

### ROS1/ROS2 用 PID 設定

- `xarm_ros/xarm_controller/config/xarm7/xarm7_controllers.yaml`
  に Gazebo 用 PID (P/I/D) がある。
- 実機ではサーボ内部の PID が主に働くが、シミュレーションでは
  これらの値を目安に「どの程度の応答性/減衰が必要か」を推定できる。

## ManiSkill 側の制約と既存実装

- ManiSkill の PD コントローラ (`PDJointPosController`)
  - アクションは `target_qpos` に変換され、PhysX の PD ドライブが角速度・トルクを生成する。
  - `use_delta=True` かつ `use_target=False` の場合、
    「現在角度 + デルタ」がそのままターゲットになる。
- 既存 `my_xarm7` (`robotagents/my_xarm7.py`)
  - デルタ角度を直接ターゲットに加算する簡易な PD 制御。
  - 速度・加速度制限は設けず、PD ドライブ任せになる。
- `my_xarm7_official` はこの挙動と互換性を保ちつつ、
  内部で SDK に近い速度/加速度制限を適用するのが目的。

## 実装構成

### 1. `XArmSDKJointDeltaControllerConfig`

- `max_joint_speed`, `max_joint_acc` を rad/s, rad/s² で指定。
- `positional_gain`, `velocity_damping` は SDK PID を参考に縮約。
  - 例: `hardware_p / 150`, `hardware_d / 5` など、
    Gazebo PID を ManiSkill の時間スケールに合わせてダウンサンプリング。
- `force_limit` は URDF の effort 値をそのまま使用。

### 2. `XArmSDKJointDeltaController`

制御周期 (`control_dt`) ごとに以下を実施する。

1. **デルタ→目標角変換**
   - `use_target=False` により、行動は常に現在角 (`self.qpos`) からの相対変位として解釈。
   - 関節角の上下限 (`self._joint_lower`, `_joint_upper`) でクリップ。

2. **速度・加速度の制限**
   - 誤差 `error = commanded - servo_target` を計算。
   - `desired_acc = pos_gain * error - vel_damp * servo_velocity`
   - `desired_acc` を ±`max_acc` でクリップ。
   - `servo_velocity` に積分し、速度も ±`max_speed` でクリップ。

3. **ターゲット更新 & オーバーシュート抑制**
   - `new_target = servo_target + servo_velocity * dt`
   - 新目標が指令目標を越えそうな場合（sign チェック）には目標位置を直接指令値に固定し、
     その軸の `servo_velocity` を 0 にリセット。
   - 最終的な `servo_target` を PD ドライブへ送信 (`set_drive_targets`).

4. **`before_simulation_step`**
   - ManiSkill のループで毎シミュレーションステップ呼び出される。
   - 最新の `servo_target` を継続的に適用し、PhysX に反映させる。

### 3. `Xarm7Official` エージェント

- `my_xarm7_official.py` の `_controller_configs` で新コントローラを登録：
  - `sdk_joint_delta_pos`: arm -> 新コントローラ、gripper -> 既存の mimic コントローラ。
  - 既存の `pd_joint_pos` / `pd_joint_delta_pos` も後方互換用に残す。

## ハードウェア仕様の反映ポイント

| 項目 | 実装箇所 | 参考ソース |
| --- | --- | --- |
| 速度上限 π rad/s | `max_joint_speed` | `xarm/x3/base.py` `_max_joint_speed` |
| 加速度上限 20 rad/s² | `max_joint_acc` | `xarm/x3/base.py` `_max_joint_acc` |
| PD 応答 (P/D 比) | `positional_gain`, `velocity_damping` | `xarm_ros/.../xarm7_controllers.yaml` |
| 努力限界 (50/30/20 Nm) | `force_limit` | `xarm_ros/xarm_description/.../xarm7.urdf.xacro` |
| デルタ解釈（現在角 + Δ） | `use_target=False` | ManiSkill 既存 `pd_joint_delta_pos` と SDK の一致性 |

## 挙動チューニングと既知の注意点

1. **デルタ累積による暴走**
   - `use_target=True` のままにすると、指令デルタが「前回の目標」に積み上がり、
     実機 / 学習済みポリシーの想定とズレて大きく旋回する。
   - `use_target=False` に戻すことで `現在角 + Δ` の挙動に揃え済み。

2. **オーバーシュートと振動**
   - `positional_gain` / `velocity_damping` のスケールは調整可能。
   - オーバーシュートが大きい場合は P を下げるか D を上げる。
   - 現状のスケールは Gazebo PID を 150 / 5 で割っているが、
     目的に応じて再調整してよい。

3. **速度ペナルティとの関係**
   - `task_marker_align_official.py` の報酬に速度ペナルティがあり、
     過度の加速を防ぐ役割を持つ。
   - ペナルティを外す場合は P/D ゲインや速度上限を弱くすると良い。

4. **既存 PD との互換性**
   - コントローラ UID を変えず `sdk_joint_delta_pos` を追加したため、
     腕デルタは既存ポリシーでも扱える。
   - グリッパは元の mimic PD (`PDJointPosMimicControllerConfig`) を再利用。

## 参考資料

- UFactory xArm SDK
  - [`xarm/x3/xarm.py`](../xArm-Python-SDK/xarm/x3/xarm.py)
  - [`xarm/x3/base.py`](../xArm-Python-SDK/xarm/x3/base.py)
  - [`xarm/core/wrapper/uxbus_cmd.py`](../xArm-Python-SDK/xarm/core/wrapper/uxbus_cmd.py)
- ROS1/ROS2 設定
  - `xarm_ros/xarm_controller/config/xarm7/xarm7_controllers.yaml`
  - `xarm_ros/xarm_description/urdf/xarm7/xarm7.urdf.xacro`
- ManiSkill
  - `robotagents/my_xarm7_official.py`
  - `mani_skill/agents/controllers/pd_joint_pos.py`
  - `mani_skill/utils/wrappers/record.py`（動画生成関連）
- 社内メモ
  - `maniskill_xarm_force_control.md`（force-control 実験の整理）

