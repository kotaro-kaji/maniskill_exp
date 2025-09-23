from os import path

import mujoco
import numpy as np
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import ArmConfig, DataKey
from robo_manip_baselines.teleop import (
    KeyboardInputDevice,
    SpacemouseInputDevice,
)

from ..MujocoEnvBase import MujocoEnvBase


class MujocoVx300sEnvBase(MujocoEnvBase):
    default_camera_config = {
        "azimuth": 0.0,
        "elevation": -20.0,
        "distance": 1.8,
        "lookat": [0.0, 0.0, 0.3],
    }
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
        }
    )

    def setup_robot(self, init_qpos):
        self.init_qpos[: len(init_qpos)] = init_qpos
        self.init_qvel[:] = 0.0

        mujoco.mj_kinematics(self.model, self.data)

        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__),
                    "../../assets/common/robots/vx300s/vx300s.urdf",
                ),
                arm_root_pose=self.get_body_pose("base_link"),
                ik_eef_joint_id=6,
                arm_joint_idxes=np.arange(0, 6),
                gripper_joint_idxes=np.array([6]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[0:6],
                init_gripper_joint_pos=np.array([0.037]),
            )
        ]

    def setup_input_device(self, input_device_name, motion_manager, overwrite_kwargs):
        if input_device_name == "spacemouse":
            InputDeviceClass = SpacemouseInputDevice
        elif input_device_name == "keyboard":
            InputDeviceClass = KeyboardInputDevice
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid input device key: {input_device_name}"
            )

        default_kwargs = self.get_input_device_kwargs(input_device_name)

        return [
            InputDeviceClass(
                motion_manager.body_manager_list[0],
                **{**default_kwargs, **overwrite_kwargs},
            )
        ]

    def get_input_device_kwargs(self, input_device_name):
        if input_device_name == "spacemouse":
            return {"rpy_scale": 2e-2}
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

    def _get_obs(self):
        arm_joint_name_list = [
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
        ]
        gripper_joint_name_list = [
            "left_finger",
            "right_finger",
        ]

        arm_joint_pos = np.array(
            [self.data.joint(joint_name).qpos[0] for joint_name in arm_joint_name_list]
        )
        arm_joint_vel = np.array(
            [self.data.joint(joint_name).qvel[0] for joint_name in arm_joint_name_list]
        )
        gripper_qpos = np.array(
            [
                self.data.joint(joint_name).qpos[0]
                for joint_name in gripper_joint_name_list
            ]
        )
        gripper_qvel = np.array(
            [
                self.data.joint(joint_name).qvel[0]
                for joint_name in gripper_joint_name_list
            ]
        )
        gripper_joint_pos = np.rad2deg(gripper_qpos.mean(keepdims=True))
        gripper_joint_vel = np.rad2deg(gripper_qvel.mean(keepdims=True))

        return {
            "joint_pos": np.concatenate(
                (arm_joint_pos, gripper_joint_pos), dtype=np.float64
            ),
            "joint_vel": np.concatenate(
                (arm_joint_vel, gripper_joint_vel), dtype=np.float64
            ),
        }
