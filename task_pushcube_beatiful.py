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

from typing import Union, Dict, Any, Tuple
import sapien

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig
from transforms3d.euler import euler2quat
import numpy as np


# ★ 自作ロボットを import（これで登録の副作用が走る）
from robotagents.my_xarm7 import Xarm7  # ← your file/module path に合わせて
from robotagents.my_xarm7_official import Xarm7Official
# my_xarm7_mjcf も登録の副作用が必要なので import
import robotagents.my_xarm7_mjcf  # registers Xarm7MJCF (uid: "my_xarm7_mjcf")

# （必要なら他ロボも残す）
# from mani_skill.agents.robots import Fetch, Panda


@register_env("MyPushCube-v1", max_episode_steps=200)
class MyPushCubeEnv(BaseEnv):

    goal_radius = 0.1
    box_half_extent_x = 0.072
    box_half_extent_y = 0.0409
    box_half_extent_z = 0.0254
    push_waypoint_threshold = 0.05
    max_dense_reward = 3.5


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    SUPPORTED_ROBOTS = ["my_xarm7", "my_xarm7", "panda"]  # , "panda", "fetch"]

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
        # 動画保存・レンダリング用の一般的な解像度カメラ
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=1080, height=1080, fov=np.pi / 3, near=0.01, far=100
        )

    def _load_scene(self, options: dict):
        builder = self.scene.create_actor_builder()
        box_half_size = [
            self.box_half_extent_x,
            self.box_half_extent_y,
            self.box_half_extent_z,
        ]
        builder.add_box_collision(
            half_size=box_half_size,
            density=500.0,
        )
        builder.add_box_visual(
            half_size=box_half_size,
            material=sapien.render.RenderMaterial(
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
        self.obj = builder.build_dynamic(name="box")
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

            # Randomize box position closer to the robot side of the table
            box_x_base = torch.rand((b,), device=self.device) * 0.15 + 0.26
            box_y_base = torch.rand((b,), device=self.device) * 0.42 - 0.30

            # Randomize a positive x-offset for the goal so it sits ahead of the box.
            goal_x_offset = torch.rand((b,), device=self.device) * 0.05 + 0.08
            goal_x_base = box_x_base + goal_x_offset
            goal_y_base = torch.rand((b,), device=self.device) * 0.42 - 0.30

            box_positions_world = torch.stack(
                (
                    box_x_base + ROBOT_BASE_X_OFFSET,
                    box_y_base,
                    torch.full(
                        (b,),
                        self.box_half_extent_z,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                ),
                dim=-1,
            )
            box_orientation = torch.zeros(
                (b, 4), device=self.device, dtype=torch.float32
            )
            box_orientation[..., 0] = 1.0
            obj_pose = Pose.create_from_pq(p=box_positions_world, q=box_orientation)
            self.obj.set_pose(obj_pose)

            # place the visual goal region slightly in front of the box on the table
            target_positions_world = torch.stack(
                (
                    goal_x_base + ROBOT_BASE_X_OFFSET,
                    goal_y_base,
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
        delta_xy = self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2]
        goal_distance = torch.linalg.norm(delta_xy, axis=1)
        on_table = self.obj.pose.p[..., 2] < self.box_half_extent_z + 5e-3
        return {
            "goal_distance": goal_distance,
            "on_table": on_table,
        }
    
    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        tcp_pose = self.agent.tcp.pose
        tcp_matrix = tcp_pose.to_transformation_matrix()[..., :3, :3]
        tcp_pose_6d = torch.cat([tcp_pose.p, tcp_matrix[..., :, 0], tcp_matrix[..., :, 1]], dim=-1)
        obs = dict(tcp_pose_6d=tcp_pose_6d)
        if self.obs_mode_struct.use_state:
            box_pose = self.obj.pose
            box_matrix = box_pose.to_transformation_matrix()[..., :3, :3]
            box_pose_6d = torch.cat([box_pose.p, box_matrix[..., :, 0], box_matrix[..., :, 1]], dim=-1)

            goal_pose = self.goal_region.pose
            goal_matrix = goal_pose.to_transformation_matrix()[..., :3, :3]
            goal_pose_6d = torch.cat([goal_pose.p, goal_matrix[..., :, 0], goal_matrix[..., :, 1]], dim=-1)

            obs.update(
                goal_pose_6d=goal_pose_6d,
                obj_pose_6d=box_pose_6d,
            )
        return obs

    def compute_staged_dense_reward(self, obs: Any, action: Array, info: Dict):
        """
        Dense reward that first guides the TCP to the pushing waypoint and then,
        once the waypoint is reached, rewards goal alignment and height stability.
        """
        push_reward, push_reached = self._push_waypoint_metrics()
        reached_weight = push_reached.to(push_reward.dtype)

        reward = push_reward
        reward += self._goal_alignment_reward() * reached_weight
        reward += self._height_stability_reward() * reached_weight
        reward += self._gripper_closure_reward()

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        # this should be equal to compute_dense_reward / max possible reward
        max_reward = self.max_dense_reward
        dense = torch.clamp(
            self.compute_staged_dense_reward(obs=obs, action=action, info=info),
            max=max_reward,
        )
        return dense / max_reward


    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        """
        Dense reward that mirrors the staged formulation without normalization.
        """
        reward = self.compute_staged_dense_reward(obs=obs, action=action, info=info)
        return torch.clamp(reward, max=self.max_dense_reward)

    def _push_waypoint_metrics(self) -> Tuple[torch.Tensor, torch.Tensor]:
        tcp_pose = self.agent.tcp.pose
        box_pose = self.obj.pose
        goal_xy = self.goal_region.pose.p[..., :2]
        box_xy = box_pose.p[..., :2]

        push_vec = goal_xy - box_xy
        eps = 1e-6
        push_norm = torch.linalg.norm(push_vec, dim=1, keepdim=True)
        default_dir = push_vec.new_tensor([1.0, 0.0]).view(1, 2)
        safe_dir = torch.where(
            (push_norm < eps).expand(-1, 2),
            default_dir.expand_as(push_vec),
            push_vec / torch.clamp(push_norm, min=eps),
        )

        contact_half_extents = safe_dir.new_tensor(
            [self.box_half_extent_x, self.box_half_extent_y]
        ).view(1, 2)
        contact_half_extents = contact_half_extents * 1.0  #ちゃんとBoxと同じ高さにする。
        contact_half_extents = contact_half_extents.expand_as(safe_dir)
        denom = torch.clamp(torch.abs(safe_dir), min=eps)
        t = torch.min(contact_half_extents / denom, dim=1, keepdim=True).values
        contact_xy = box_xy - safe_dir * t

        tcp_xy = tcp_pose.p[..., :2]
        xy_dist = torch.linalg.norm(tcp_xy - contact_xy, dim=1)

        box_top_z = box_pose.p[..., 2] + self.box_half_extent_z
        tcp_z = tcp_pose.p[..., 2]
        z_offset = torch.clamp(tcp_z - box_top_z, min=0.0)

        total_dist = torch.sqrt(xy_dist**2 + z_offset**2)
        base_reward = 1 - torch.tanh(5 * total_dist)
        reward = torch.where(
            total_dist < self.push_waypoint_threshold,
            torch.ones_like(base_reward),
            base_reward,
        )
        reached = total_dist < self.push_waypoint_threshold
        return reward, reached

    def _goal_alignment_reward(self) -> torch.Tensor:
        delta_xy = self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2]
        distance = torch.linalg.norm(delta_xy, dim=1)
        return 1 - torch.tanh(5 * distance)

    def _height_stability_reward(self) -> torch.Tensor:
        current_obj_z = self.obj.pose.p[..., 2]
        desired_obj_z = self.box_half_extent_z
        z_deviation = torch.abs(current_obj_z - desired_obj_z)
        return 1 - torch.tanh(5 * z_deviation)

    def _gripper_closure_reward(self) -> torch.Tensor:
        drive_joint = self.agent.robot.joints_map.get("drive_joint")
        base = torch.zeros_like(self.obj.pose.p[..., 0])
        if drive_joint is None or drive_joint.active_index is None:
            return base
        drive_qpos = drive_joint.qpos
        grip_closure = torch.clamp(drive_qpos / 0.85, 0.0, 1.0)
        return 0.5 * grip_closure
