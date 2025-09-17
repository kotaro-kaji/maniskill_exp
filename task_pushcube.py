import sapien
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from xarm7_table_scene_builder import Xarm7TableSceneBuilder
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
from my_xarm7 import Xarm7  # ← your file/module path に合わせて
# my_xarm7_mjcf も登録の副作用が必要なので import
import my_xarm7_mjcf  # registers Xarm7MJCF (uid: "my_xarm7_mjcf")

# （必要なら他ロボも残す）
# from mani_skill.agents.robots import Fetch, Panda


@register_env("MyPushCube-v1", max_episode_steps=50)
class MyPushCubeEnv(BaseEnv):

    goal_radius = 0.1
    cube_half_size = 0.02


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
            half_size=[0.02] * 3,
        )
        builder.add_box_visual(
            half_size=[0.02] * 3,
            material=sapien.render.RenderMaterial(
                # RGBA values, this is a red cube
                base_color=[1, 0, 0, 1],
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

            # here is randomization code that randomizes the x, y position 
            # of the cube we are pushing in the range [-0.1, -0.1] to [0.1, 0.1]
            p = torch.zeros((b, 3))
            p[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            p[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=p, q=q)
            self.obj.set_pose(obj_pose)

            # place the visual goal region slightly in front of the cube on the table
            target_region_xyz = p + torch.tensor([0.1 + self.goal_radius, 0, 0])
            target_region_xyz[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=target_region_xyz,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )

    def evaluate(self):
        # success: cube xy within goal radius of target and cube is on table
        is_obj_placed = (
            torch.linalg.norm(
                self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        ) & (self.obj.pose.p[..., 2] < self.cube_half_size + 5e-3)

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

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # We also create a pose marking where the robot should push the cube from that is easiest (pushing from behind the cube)
        tcp_push_pose = Pose.create_from_pq(
            p=self.obj.pose.p
            + torch.tensor([-self.cube_half_size - 0.005, 0, 0], device=self.device)
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
        desired_obj_z = self.cube_half_size
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
            drive_idx = int(drive_joint.active_index.squeeze().item())
            drive_qpos = self.agent.robot.get_qpos()[..., drive_idx]
            grip_closure = torch.clamp(drive_qpos / 0.85, 0.0, 1.0)
            reward += 0.05 * grip_closure

        # assign rewards to parallel environments that achieved success to the maximum of 3.
        reward[info["success"]] = 4
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = 4.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
