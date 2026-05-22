import runpy
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mani_skill.utils import common
from mani_skill.utils.registration import register_env

from robotagents.my_xarm7 import Xarm7
from tasks.single_arm.pick_cube import PickCubeEnv


def old_is_grasping(self, object, min_force=0.5, max_angle=85):
    l_contact_forces = self.scene.get_pairwise_contact_forces(
        self.finger1_link, object
    )
    r_contact_forces = self.scene.get_pairwise_contact_forces(
        self.finger2_link, object
    )
    lforce = torch.linalg.norm(l_contact_forces, axis=1)
    rforce = torch.linalg.norm(r_contact_forces, axis=1)

    ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
    rdirection = self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
    langle = common.compute_angle_between(ldirection, l_contact_forces)
    rangle = common.compute_angle_between(rdirection, r_contact_forces)
    lflag = torch.logical_and(
        lforce >= min_force, torch.rad2deg(langle) <= max_angle
    )
    rflag = torch.logical_and(
        rforce >= min_force, torch.rad2deg(rangle) <= max_angle
    )
    return torch.logical_and(lflag, rflag)


@register_env("AblatePickCubeH100-v1", max_episode_steps=100)
class AblatePickCubeH100(PickCubeEnv):
    pass


@register_env("AblatePickCubeMinForce05-v1", max_episode_steps=150)
class AblatePickCubeMinForce05(PickCubeEnv):
    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube, min_force=0.5)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }


@register_env("AblatePickCubeLinkTcp-v1", max_episode_steps=150)
class AblatePickCubeLinkTcp(PickCubeEnv):
    def _tcp_pos_for_task(self):
        return self.agent.tcp_pose.p


@register_env("AblatePickCubeNoPadReward-v1", max_episode_steps=150)
class AblatePickCubeNoPadReward(PickCubeEnv):
    def compute_dense_reward(self, obs, action, info):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self._tcp_pos_for_task(), axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        fine_reaching_reward = 1 - torch.tanh(20 * tcp_to_obj_dist)

        grasped = info["is_grasped"].to(torch.float32)
        pre_grasp = 1.0 - grasped
        reward = pre_grasp * (reaching_reward + fine_reaching_reward)

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        obj_to_goal_pos = self.goal_site.pose.p - self.cube.pose.p
        obj_to_goal_xy_dist = torch.linalg.norm(obj_to_goal_pos[..., :2], axis=1)
        obj_to_goal_z_dist = torch.abs(obj_to_goal_pos[..., 2])
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        place_xy_reward = 1 - torch.tanh(5 * obj_to_goal_xy_dist)
        place_z_reward = 1 - torch.tanh(5 * obj_to_goal_z_dist)
        lift_reward = torch.clamp(
            (self.cube.pose.p[..., 2] - self.cube_half_size) / 0.12,
            min=0.0,
            max=1.0,
        )
        reward += grasped * (
            2.0
            + lift_reward
            + 4.0 * place_reward
            + 2.0 * place_xy_reward
            + 4.0 * place_z_reward
        )

        qvel = self.agent.robot.get_qvel()
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]
        reward += info["is_obj_placed"] * (3.0 + 2.0 * static_reward)

        qpos = self.agent.robot.get_qpos()
        gripper_closed = torch.clamp(qpos[..., 7], min=0.0, max=0.85) / 0.85
        reward += pre_grasp * reaching_reward * gripper_closed

        reward[info["success"]] = 10
        return reward


@register_env("AblatePickCubeNoPlaceShaping-v1", max_episode_steps=150)
class AblatePickCubeNoPlaceShaping(PickCubeEnv):
    def compute_dense_reward(self, obs, action, info):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self._tcp_pos_for_task(), axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        fine_reaching_reward = 1 - torch.tanh(20 * tcp_to_obj_dist)

        grasped = info["is_grasped"].to(torch.float32)
        pre_grasp = 1.0 - grasped
        reward = pre_grasp * (reaching_reward + fine_reaching_reward)

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 3.0 * (1 - torch.tanh(5 * obj_to_goal_dist))
        reward += grasped * (0.5 + place_reward)

        qvel = self.agent.robot.get_qvel()
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        qpos = self.agent.robot.get_qpos()
        gripper_closed = torch.clamp(qpos[..., 7], min=0.0, max=0.85) / 0.85
        left_pad_pos, right_pad_pos = self._finger_pad_pos_for_task()
        pad_mid_pos = (left_pad_pos + right_pad_pos) / 2
        pad_to_obj_dist = torch.linalg.norm(self.cube.pose.p - pad_mid_pos, axis=1)
        pad_center_reward = 1 - torch.tanh(20 * pad_to_obj_dist)
        pad_width = torch.linalg.norm(left_pad_pos - right_pad_pos, axis=1)
        target_width = self.cube_half_size * 2
        pad_width_reward = 1 - torch.tanh(60 * torch.abs(pad_width - target_width))
        reward += pre_grasp * 3.0 * pad_center_reward * pad_width_reward
        reward += pre_grasp * 3.0 * pad_center_reward * gripper_closed

        reward[info["success"]] = 10
        return reward


@register_env("AblatePickCubeNoPadWidth-v1", max_episode_steps=150)
class AblatePickCubeNoPadWidth(PickCubeEnv):
    def compute_dense_reward(self, obs, action, info):
        reward = super().compute_dense_reward(obs, action, info)
        grasped = info["is_grasped"].to(torch.float32)
        pre_grasp = 1.0 - grasped
        left_pad_pos, right_pad_pos = self._finger_pad_pos_for_task()
        pad_mid_pos = (left_pad_pos + right_pad_pos) / 2
        pad_to_obj_dist = torch.linalg.norm(self.cube.pose.p - pad_mid_pos, axis=1)
        pad_center_reward = 1 - torch.tanh(20 * pad_to_obj_dist)
        pad_width = torch.linalg.norm(left_pad_pos - right_pad_pos, axis=1)
        target_width = self.cube_half_size * 2
        pad_width_reward = 1 - torch.tanh(60 * torch.abs(pad_width - target_width))
        return reward - pre_grasp * 3.0 * pad_center_reward * pad_width_reward


@register_env("AblatePickCubeNoPadClosed-v1", max_episode_steps=150)
class AblatePickCubeNoPadClosed(PickCubeEnv):
    def compute_dense_reward(self, obs, action, info):
        reward = super().compute_dense_reward(obs, action, info)
        grasped = info["is_grasped"].to(torch.float32)
        pre_grasp = 1.0 - grasped
        qpos = self.agent.robot.get_qpos()
        gripper_closed = torch.clamp(qpos[..., 7], min=0.0, max=0.85) / 0.85
        left_pad_pos, right_pad_pos = self._finger_pad_pos_for_task()
        pad_mid_pos = (left_pad_pos + right_pad_pos) / 2
        pad_to_obj_dist = torch.linalg.norm(self.cube.pose.p - pad_mid_pos, axis=1)
        pad_center_reward = 1 - torch.tanh(20 * pad_to_obj_dist)
        return reward - pre_grasp * 3.0 * pad_center_reward * gripper_closed


if "--old-grasp-direction" in sys.argv:
    sys.argv.remove("--old-grasp-direction")
    Xarm7.is_grasping = old_is_grasping

runpy.run_path("ppo_xarm7.py", run_name="__main__")
