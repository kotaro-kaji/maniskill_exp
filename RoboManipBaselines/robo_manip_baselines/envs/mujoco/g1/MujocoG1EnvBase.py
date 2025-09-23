from os import path

import mujoco
import numpy as np
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import ArmConfig, DataKey, get_se3_from_pose
from robo_manip_baselines.teleop import SpacemouseInputDevice

from ..MujocoEnvBase import MujocoEnvBase


class MujocoG1EnvBase(MujocoEnvBase):
    sim_timestep = 0.002
    frame_skip = 16

    default_camera_config = {
        "azimuth": -180.0,
        "elevation": -30.0,
        "distance": 1.5,
        "lookat": [0.3, 0.0, 0.5],
    }
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64),
        }
    )

    def setup_robot(self, init_qpos):
        # TODO: Robot moves quickly to the initial posture and loses balance
        # self.init_qpos[: len(init_qpos)] = init_qpos
        self.init_qvel[:] = 0.0

        mujoco.mj_kinematics(self.model, self.data)

        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__),
                    "../../assets/common/robots/g1/g1_only_arms.urdf",
                ),
                arm_root_pose=self.get_body_pose("pelvis"),
                ik_eef_joint_id=7,
                arm_joint_idxes=np.arange(7),
                gripper_joint_idxes=np.array([7]),
                gripper_joint_idxes_for_limit=np.array([22]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=init_qpos[0:7],
                init_gripper_joint_pos=np.zeros(1),
                exclude_joint_names=[
                    "right_shoulder_pitch_joint",
                    "right_shoulder_roll_joint",
                    "right_shoulder_yaw_joint",
                    "right_elbow_joint",
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
                get_root_pose_func=lambda env: get_se3_from_pose(
                    env.get_body_pose("pelvis")
                ),
            ),
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__),
                    "../../assets/common/robots/g1/g1_only_arms.urdf",
                ),
                arm_root_pose=self.get_body_pose("pelvis"),
                ik_eef_joint_id=7,
                arm_joint_idxes=np.arange(8, 15),
                gripper_joint_idxes=np.array([15]),
                gripper_joint_idxes_for_limit=np.array([30]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([1]),
                eef_idx=1,
                init_arm_joint_pos=init_qpos[8:15],
                init_gripper_joint_pos=np.zeros(1),
                exclude_joint_names=[
                    "left_shoulder_pitch_joint",
                    "left_shoulder_roll_joint",
                    "left_shoulder_yaw_joint",
                    "left_elbow_joint",
                    "left_wrist_roll_joint",
                    "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint",
                ],
                get_root_pose_func=lambda env: get_se3_from_pose(
                    env.get_body_pose("pelvis")
                ),
            ),
        ]

        self.arm_joint_name_list = [
            "shoulder_pitch_joint",
            "shoulder_roll_joint",
            "shoulder_yaw_joint",
            "elbow_joint",
            "wrist_roll_joint",
            "wrist_pitch_joint",
            "wrist_yaw_joint",
        ]
        self.gripper_joint_name = "hand_middle_0_joint"

    def setup_input_device(self, input_device_name, motion_manager, overwrite_kwargs):
        if input_device_name == "spacemouse":
            InputDeviceClass = SpacemouseInputDevice
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid input device key: {input_device_name}"
            )

        default_kwargs = self.get_input_device_kwargs(input_device_name)

        return [
            InputDeviceClass(
                body_manager,
                **{
                    **default_kwargs.get(device_idx, {}),
                    **overwrite_kwargs.get(device_idx, {}),
                },
            )
            for device_idx, body_manager in enumerate(motion_manager.body_manager_list)
        ]

    def get_input_device_kwargs(self, input_device_name):
        if input_device_name == "spacemouse":
            return {0: {"gripper_scale": 0.025}, 1: {"gripper_scale": 0.025}}
        else:
            return super().get_input_device_kwargs(input_device_name)

    @property
    def measured_keys_to_save(self):
        return [
            DataKey.MEASURED_JOINT_POS,
            DataKey.MEASURED_JOINT_VEL,
            DataKey.MEASURED_GRIPPER_JOINT_POS,
            DataKey.MEASURED_EEF_POSE,
        ]

    def step(self, action_sub):
        action_all = np.zeros(self.model.nu)

        for left_right, joint_idx_offset in zip(["left", "right"], [0, 8]):
            for joint_idx_in_sub, joint_name in enumerate(self.arm_joint_name_list):
                joint_idx_in_all = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    left_right + "_" + joint_name,
                )
                action_all[joint_idx_in_all] = action_sub[
                    joint_idx_offset + joint_idx_in_sub
                ]

        for left_right, joint_idx_in_sub in zip(["left", "right"], [7, 15]):
            joint_idx_in_all = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                left_right + "_" + self.gripper_joint_name,
            )
            action_all[joint_idx_in_all] = action_sub[joint_idx_in_sub]

        return super().step(action_all)

    def _get_obs(self):
        left_obs = self._get_obs_single_arm("left")
        right_obs = self._get_obs_single_arm("right")
        return {
            key: np.concatenate([left_obs[key], right_obs[key]])
            for key in left_obs.keys()
        }

    def _get_obs_single_arm(self, left_right):
        arm_joint_pos = np.array(
            [
                self.data.joint(left_right + "_" + joint_name).qpos[0]
                for joint_name in self.arm_joint_name_list
            ]
        )
        arm_joint_vel = np.array(
            [
                self.data.joint(left_right + "_" + joint_name).qvel[0]
                for joint_name in self.arm_joint_name_list
            ]
        )
        gripper_joint_pos = np.array(
            [self.data.joint(left_right + "_" + self.gripper_joint_name).qpos[0]]
        )
        gripper_joint_vel = np.zeros(1)

        return {
            "joint_pos": np.concatenate(
                (arm_joint_pos, gripper_joint_pos), dtype=np.float64
            ),
            "joint_vel": np.concatenate(
                (arm_joint_vel, gripper_joint_vel), dtype=np.float64
            ),
        }
