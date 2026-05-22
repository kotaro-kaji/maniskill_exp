from typing import Any, Union

import numpy as np
import sapien
import torch


import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import SO100, Fetch, Panda, WidowXAI, XArm6Robotiq
from robotagents.my_xarm7 import Xarm7


from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose
from scenebuilders.xarm7_table_scene_builder import (
    Xarm7TableSceneBuilder,
    PEDESTAL_HEIGHT,
    ROBOT_BASE_X_OFFSET,
)

PICK_CUBE_DOC_STRING = """**Task Description:**
A simple task where the objective is to grasp a red cube with the {robot_id} robot and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

**Randomizations:**
- the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
- the cube's z-axis rotation is randomized to a random angle
- the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

**Success Conditions:**
- the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
- the robot is static (q velocity < 0.2)
"""


@register_env("MyXarm7PickCube-v1", max_episode_steps=150)
class PickCubeEnv(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
        "my_xarm7",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100, WidowXAI, Xarm7]
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)

    def __init__(
        self,
        *args,
        robot_uids="my_xarm7",
        robot_init_qpos_noise=0.02,
        robot_init_noise_scale=1.0,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise * robot_init_noise_scale
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = Xarm7TableSceneBuilder(self)
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            xyz[:, 0] += self.cube_spawn_center[0] - 0.22
            xyz[:, 1] += self.cube_spawn_center[1]

            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0] -0.20
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _tcp_pos_for_task(self):
        if self.robot_uids != "my_xarm7":
            return self.agent.tcp_pose.p

        b = self.agent.tcp_pose.p.shape[0]
        grasp_offset = torch.tensor([[0.0, 0.0, 0.1564]], device=self.device).repeat(
            b, 1
        )
        gripper_base = self.agent.robot.links_map["xarm_gripper_base_link"]
        return (gripper_base.pose * Pose.create_from_pq(grasp_offset)).p

    def _finger_pad_pos_for_task(self):
        assert self.robot_uids == "my_xarm7"
        b = self.agent.tcp_pose.p.shape[0]
        left_pad_offset = torch.tensor(
            [[0.0, -0.02110, 0.04295]], device=self.device
        ).repeat(b, 1)
        right_pad_offset = torch.tensor(
            [[0.0, 0.02110, 0.04295]], device=self.device
        ).repeat(b, 1)
        left_pad_pos = (
            self.agent.finger1_link.pose * Pose.create_from_pq(left_pad_offset)
        ).p
        right_pad_pos = (
            self.agent.finger2_link.pose * Pose.create_from_pq(right_pad_offset)
        ).p
        return left_pad_pos, right_pad_pos

    def _get_obs_extra(self, info: dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        tcp_pos = self._tcp_pos_for_task()
        tcp_pose = self.agent.tcp_pose.raw_pose.clone()
        tcp_pose[..., :3] = tcp_pos
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=tcp_pose,
            goal_pos=self.goal_site.pose.p,
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                tcp_to_obj_pos=self.cube.pose.p - tcp_pos,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
            <= self.goal_thresh
        )
        grasp_min_force = 0.1 if self.robot_uids == "my_xarm7" else 0.5
        is_grasped = self.agent.is_grasping(self.cube, min_force=grasp_min_force)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.cube.pose.p - self._tcp_pos_for_task(), axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        fine_reaching_reward = 1 - torch.tanh(20 * tcp_to_obj_dist)

        is_grasped = info["is_grasped"]
        grasped = is_grasped.to(torch.float32)
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
        if self.robot_uids in ["panda", "widowxai"]:
            qvel = qvel[..., :-2]
        elif self.robot_uids == "so100":
            qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]
        reward += info["is_obj_placed"] * (3.0 + 2.0 * static_reward)

        qpos = self.agent.robot.get_qpos()
        gripper_closed = torch.clamp(qpos[..., 7], min=0.0, max=0.85) / 0.85
        if self.robot_uids == "my_xarm7":
            left_pad_pos, right_pad_pos = self._finger_pad_pos_for_task()
            pad_mid_pos = (left_pad_pos + right_pad_pos) / 2
            pad_to_obj_dist = torch.linalg.norm(self.cube.pose.p - pad_mid_pos, axis=1)
            pad_center_reward = 1 - torch.tanh(20 * pad_to_obj_dist)

            pad_width = torch.linalg.norm(left_pad_pos - right_pad_pos, axis=1)
            target_width = self.cube_half_size * 2
            pad_width_reward = 1 - torch.tanh(60 * torch.abs(pad_width - target_width))
            reward += pre_grasp * 3.0 * pad_center_reward * pad_width_reward
            reward += pre_grasp * 3.0 * pad_center_reward * gripper_closed
        else:
            reward += pre_grasp * reaching_reward * gripper_closed


        reward[info["success"]] = 10
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10


PickCubeEnv.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="Panda")


@register_env("PickCubeSO100-v1", max_episode_steps=50)
class PickCubeSO100Env(PickCubeEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCubeSO100-v1_rt.mp4"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="so100", **kwargs)


PickCubeSO100Env.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="SO100")


@register_env("PickCubeWidowXAI-v1", max_episode_steps=50)
class PickCubeWidowXAIEnv(PickCubeEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCubeWidowXAI-v1_rt.mp4"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, robot_uids="widowxai", **kwargs)


PickCubeWidowXAIEnv.__doc__ = PICK_CUBE_DOC_STRING.format(robot_id="WidowXAI")
