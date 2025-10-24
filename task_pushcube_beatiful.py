import sapien
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv

####the part which i made#################################
from scenebuilders.xarm7_table_scene_builder import (
    Xarm7TableSceneBuilder,
    ROBOT_BASE_X_OFFSET,
)
###############################################################


from mani_skill.utils.registration import register_env

from mani_skill.utils.structs.pose import Pose
import torch

from mani_skill.utils.building import actors
from mani_skill.sensors.camera import CameraConfig

from typing import Union, Dict, Any
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from transforms3d.euler import euler2quat
import numpy as np


# ★ 自作ロボットを import（これで登録の副作用が走る）
from robotagents.my_xarm7 import Xarm7  # ← your file/module path に合わせて
# my_xarm7_mjcf も登録の副作用が必要なので import
import robotagents.my_xarm7_mjcf  # registers Xarm7MJCF (uid: "my_xarm7_mjcf")

# （必要なら他ロボも残す）
# from mani_skill.agents.robots import Fetch, Panda


@register_env("MyPushCube-v1", max_episode_steps=50)
class MyPushCubeEnv(BaseEnv):

    goal_radius = 0.1
    cube_half_extent_x = 0.072
    cube_half_extent_y = 0.0409
    cube_half_extent_z = 0.0254


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    # ★ サポートロボに自作UIDを追加（自作だけにするなら ["my_xarm7"] だけでOK）
    # my_xarm7 に加えて my_xarm7_mjcf も選択可能に
    SUPPORTED_ROBOTS = ["my_xarm7", "my_xarm7_mjcf"]  # , "panda", "fetch"]

    # ★ 型ヒントも自作に
    agent: Xarm7  # Union[Xarm7, Xarm7MJCF] などでもOK

    def __init__(self, *args, robot_uids="my_xarm7", **kwargs):
        # "panda" や "fetch" も許すなら、タプル/リストで受けられるのは元のまま
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        # ベースポーズは環境に合わせて。床置きなら z=0 付近でOK
        base_pose = sapien.Pose(p=[0.0, 0.0, 0.0])
        super()._load_agent(options, base_pose)

    #@property
    #def _default_sensor_configs(self):
        ## トレーニング用の小さめカメラ（高速）
        #pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        #return [
            #CameraConfig(
                #"base_camera",
                #pose=pose,
                #width=128,
                #height=128,
                #fov=np.pi / 2,
                #near=0.01,
                #far=100,
            #)
        #]

    @property
    def _default_human_render_camera_configs(self):
        # 動画保存・レンダリング用の高解像度カメラ
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            # for boxes we specify half length of each side
            half_size=[
                self.cube_half_extent_x,
                self.cube_half_extent_y,
                self.cube_half_extent_z,
            ],
        )
        builder.add_box_visual(
            half_size=[
                self.cube_half_extent_x,
                self.cube_half_extent_y,
                self.cube_half_extent_z,
            ],
            material=sapien.render.RenderMaterial(
                # RGBA values, set to white for the rectangular prism
                base_color=[1, 1, 1, 1],
            ),
        )

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )
        # strongly recommended to set initial poses for objects, even if you plan to modify them later
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.02], q=[1, 0, 0, 0])
        self.obj = builder.build(name="cube")
        # PushCube has some other code after this removed for brevity that 
        # spawns a goal object (a red/white target) stored at self.goal_region

        self.table_scene = Xarm7TableSceneBuilder(
            env=self,
        )
        self.table_scene.build()

    # BaseEnv.close() -> self._clear() 経由の終了時例外対策
    # インタプリタシャットダウン時に module 参照が None 化し gc.collect が呼べず
    # TypeError になるケースがあるため、ここで安全に wrap した _clear を提供します。
    def _clear(self):
        # BaseEnv._clear と同等の後片付け。gc.collect だけ try で囲う。
        self._close_viewer()
        self.agent = None
        self._sensors = dict()
        self._human_render_cameras = dict()
        self.scene = None
        self._hidden_objects = []
        try:
            import gc as _gc
            _gc.collect()
        except Exception:
            pass

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # using torch.device context manager to auto create tensors 
        # on CPU/CUDA depending on self.device, the device the env runs on
        with torch.device(self.device):
            b = len(env_idx)
            # use the TableSceneBuilder to init all objects in that scene builder
            self.table_scene.initialize(env_idx)

            # Randomize cube position closer to the robot side of the table
            target_x_base = torch.rand((b,), device=self.device) * 0.15 + 0.32
            target_y_base = torch.rand((b,), device=self.device) * 0.42 - 0.30
            cube_x_base = target_x_base - 0.1
            cube_y_base = target_y_base

            cube_positions_world = torch.stack(
                (
                    cube_x_base + ROBOT_BASE_X_OFFSET,
                    cube_y_base,
                    torch.full(
                        (b,),
                        self.cube_half_extent_z,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                ),
                dim=-1,
            )
            cube_orientation = torch.zeros(
                (b, 4), device=self.device, dtype=torch.float32
            )
            cube_orientation[..., 0] = 1.0
            obj_pose = Pose.create_from_pq(p=cube_positions_world, q=cube_orientation)
            self.obj.set_pose(obj_pose)

            # place the visual goal region slightly in front of the cube on the table
            target_positions_world = torch.stack(
                (
                    target_x_base + ROBOT_BASE_X_OFFSET,
                    target_y_base,
                    torch.full(
                        (b,),
                        1e-3,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                ),
                dim=-1,
            )
            goal_orientation = torch.tensor(
                euler2quat(0, np.pi / 2, 0), device=self.device, dtype=torch.float32
            ).repeat(b, 1)
            if b == 0:
                goal_orientation = goal_orientation.view(0, 4)
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_positions_world,
                    q=goal_orientation,
                )
            )

    def evaluate(self):
        # success: cube xy within goal radius of target and cube is on table
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        ) & (self.obj.pose.p[..., 2] < self.cube_half_extent_z + 5e-3)

        return {
            "success": is_obj_placed,
        }
    
    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            # if the observation mode requests to use state, we provide ground truth information about where the cube is.
            # for visual observation modes one should rely on the sensed visual data to determine where the cube is
            obs.update(
                goal_pos=self.goal_region.pose.p,
                obj_pose=self.obj.pose.raw_pose,
            )
        return obs

    def compute_staged_dense_reward(self, obs: Any, action: Array, info: Dict):
        # We also create a pose marking where the robot should push the cube from that is easiest (pushing from behind the cube)
        tcp_push_pose = Pose.create_from_pq(
            p=self.obj.pose.p
            + torch.tensor(
                [-self.cube_half_extent_x - 0.005, 0, 0], device=self.device
            )
        )
        tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
        tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
        reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
        reward = reaching_reward

        # compute a placement reward to encourage robot to move the cube to the center of the goal region
        # we further multiply the place_reward by a mask reached so we only add the place reward if the robot has reached the desired push pose
        # This reward design helps train RL agents faster by staging the reward out.
        reached = tcp_to_push_pose_dist < 0.01
        obj_to_goal_dist = torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * reached
        
        # Compute a z reward to encourage the robot to keep the cube on the table
        desired_obj_z = self.cube_half_extent_z
        current_obj_z = self.obj.pose.p[..., 2]
        z_deviation = torch.abs(current_obj_z - desired_obj_z)
        z_reward = 1 - torch.tanh(5 * z_deviation)
        # We multiply the z reward by the place_reward and reached mask so that 
        #   we only add the z reward if the robot has reached the desired push pose
        #   and the z reward becomes more important as the robot gets closer to the goal.
        reward += place_reward * z_reward * reached

        # Encourage gentle closing of the gripper; 0 rad=open, 0.85 rad=closed.
        drive_joint = self.agent.robot.joints_map.get("drive_joint")
        if drive_joint is not None and drive_joint.active_index is not None:
            drive_qpos = drive_joint.qpos
            grip_closure = torch.clamp(drive_qpos / 0.85, 0.0, 1.0)
            reward += 0.5 * grip_closure

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 4
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 4.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward

    def compute_simple_place_reward(self, obs: Any, action: Array, info: Dict):
        """
        Minimal alternative reward that only encourages the cube to reach the goal.
        No waypoint/reaching term and no gating – always evaluates the placement distance.
        """
        obj_to_goal_dist = torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)

        reward = place_reward.clone()
        reward[info["success"]] = 4
        return reward

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        """Default dense reward encouraging the TCP to align with the ideal push point."""
        return self._compute_pushpoint_distance_reward(obs=obs, action=action, info=info)

    def _compute_pushpoint_distance_reward(
        self, obs: Any, action: Array, info: Dict
    ) -> torch.Tensor:
        """
        Dense reward that reduces the distance between the TCP and the optimal push point
        on the cube surface, and penalizes the TCP for hovering too far above the cube.
        """
        tcp_pose = self.agent.tcp.pose
        tcp_xy = tcp_pose.p[..., :2]
        cube_pose = self.obj.pose
        cube_xy = cube_pose.p[..., :2]
        goal_xy = self.goal_region.pose.p[..., :2]

        push_vec = goal_xy - cube_xy
        eps = 1e-6
        push_norm = torch.linalg.norm(push_vec, dim=1, keepdim=True)
        default_dir = torch.tensor([1.0, 0.0], device=self.device).view(1, 2)
        safe_dir = torch.where(
            (push_norm < eps).expand(-1, 2),
            default_dir.expand_as(push_vec),
            push_vec / torch.clamp(push_norm, min=eps),
        )

        half_extents = torch.tensor(
            [self.cube_half_extent_x, self.cube_half_extent_y],
            device=self.device,
            dtype=torch.float32,
        )
        denom = torch.clamp(torch.abs(safe_dir), min=eps)
        t = torch.min(half_extents / denom, dim=1, keepdim=True).values
        contact_xy = cube_xy - safe_dir * t

        xy_dist = torch.linalg.norm(tcp_xy - contact_xy, dim=1)

        cube_top_z = cube_pose.p[..., 2] + self.cube_half_extent_z
        tcp_z = tcp_pose.p[..., 2]
        z_offset = torch.clamp(tcp_z - cube_top_z, min=0.0)

        total_dist = torch.sqrt(xy_dist**2 + z_offset**2)
        reward = 1 - torch.tanh(5 * total_dist)
        reward[info["success"]] = 4
        return reward
