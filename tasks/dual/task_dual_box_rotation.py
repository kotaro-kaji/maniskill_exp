import math
import os
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
import sapien

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.utils import get_active_joint_indices

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.rotation_conversions import (
    matrix_to_quaternion,
    quaternion_to_matrix,
)
from mani_skill.utils.structs import Actor, Link
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.common import flatten_dict_keys, flatten_state_dict, to_tensor
from robotagents.xarm_ball_ee import Xarm7BallEE
from robotagents.xarm_ball_ee_kinematics import Xarm7BallEEKinematicsLeft
from robotagents.xarm_ball_ee_wo_force_sensor import Xarm7BallEEWoForceSensor


from scenebuilders.dual_xarm7_table_scene_builder import (
    DualXarm7TableSceneBuilder,
    LEFT_ARM_Y_OFFSET,
    RIGHT_ARM_Y_OFFSET,
)
from scenebuilders.xarm7_initial_randomization_scene_builder import (
    Xarm7InitialRandomizationSceneBuilder,
)
from scenebuilders.xarm7_table_scene_builder import (
    PEDESTAL_HEIGHT,
    ROBOT_BASE_X_OFFSET,
)
from .get_obs_extra import get_obs_extra_full, get_obs_extra_ablation



def smoothstep(x: torch.Tensor) -> torch.Tensor:
    """
    Smootherstep function (0 < x < 1)
    f(x) = 6x^5 - 15x^4 + 10x^3
    Properties:
      f(0)=0, f(1)=1
      f'(0)=f'(1)=0
      f''(x) changes sign at x=0.5
      smoother near endpoints than smoothstep
    """
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    x = torch.clamp(x, 0.0, 1.0)
    return 6 * x**5 - 15 * x**4 + 10 * x**3


@register_env("MyDualBoxRotation-v0", max_episode_steps=200)
class MyDualBoxRotationEnv(BaseEnv):
    SUPPORTED_ROBOTS = [
        ("xarm7_ball_ee", "xarm7_ball_ee"),
        ("xarm7_ball_ee_kinematics_left", "xarm7_ball_ee_kinematics_left"),
        ("xarm7_ball_ee_wo_force_sensor", "xarm7_ball_ee_wo_force_sensor"),
    ]
    agent: MultiAgent[
        Tuple[
            Xarm7BallEE | Xarm7BallEEKinematicsLeft | Xarm7BallEEWoForceSensor,
            Xarm7BallEE | Xarm7BallEEKinematicsLeft | Xarm7BallEEWoForceSensor,
        ]
    ]
    _obs_extra_fn = staticmethod(get_obs_extra_full)

    CONTACT_FORCE_THRESHOLD = 0.001  # N
    CONTACT_PENALTY_WEIGHT = 2.0
    
    BOX_HALF_SIZE = np.array([0.2172*0.5, 0.2845*0.5, 0.1140*0.5])
    BOX_DENSITY = 200.0
    BOX_X_OFFSET_FROM_BASE = 0.342
    BOX_CENTER_MIN_X = 0.25
    BOX_Y_JITTER = 0.08
    BOX_X_JITTER = 0.04
    BOX_ROTATION_MEAN_DEG = 0.0
    BOX_ROTATION_JITTER_DEG = 20.0
    BOX_ROTATION_JITTER_RAD = math.radians(BOX_ROTATION_JITTER_DEG)
    DISTANCE_SCALE = 4.0
    PUSHPOINT_DISTANCE_SCALE = 5.0
    PUSHPOINT_DISTANCE_THRESHOLD = 0.035
    BOX_CENTER_MAX_OFFSET = 0.15
    BOX_TRANSLATION_PENALTY_MAX = 0.5
    MAX_ROTATION_PACE = 540.0 / 10.0  # degrees per second for full reward
    BOX_GOAL_YAW_DEG = 90.0 
    BOX_OVERSHOOT_PENALTY_START_DEG = 90.5
    BOX_OVERSHOOT_PENALTY_END_DEG = 110.0
    BOX_OVERSHOOT_PENALTY_MAX = 5.0
    TCP_LEAD_SATURATION = 0.020
    TCP_LEAD_MAX = 0.35
    TCP_HEIGHT_MARGIN = 0.005
    TCP_HEIGHT_PENALTY = 0.5
    TCP_MIN_X_THRESHOLD = 0.095
    TCP_MIN_X_PENALTY = 2.0
    OBS_ABS_MAX = 1e15
    BOX_OBS_TRANSLATION_OFFSET_M = 0.005
    BOX_OBS_YAW_OFFSET_RAD = 0.02
    BOX_OBS_TRANSLATION_JITTER_M = 0.005
    BOX_OBS_YAW_JITTER_RAD = 0.02

    def __init__(
        self,
        *args,
        robot_uids=("xarm7_ball_ee", "xarm7_ball_ee"),
        robot_init_noise_scale=1.0,
        collect_rmb_data: bool = False,
        **kwargs,
    ):
        self.robot_init_noise_scale = robot_init_noise_scale
        self.collect_rmb_data = bool(collect_rmb_data)
        self._box_obs_translation_offset: Optional[torch.Tensor] = None
        self._box_obs_yaw_offset: Optional[torch.Tensor] = None
        self._flat_obs_column_names: Optional[List[str]] = None
        self._agent_obs_joint_indices: Dict[str, torch.Tensor] = dict()
        self._last_good_obs: Optional[Dict[str, Any]] = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: Dict[str, Any]):
        super()._load_agent(options, [sapien.Pose(p=[0,1,0]), sapien.Pose(p=[0,-1,0])])
        self._configure_observed_joint_indices()
        #super()._load_agent(options, sapien.Pose[p=])


# 0.00539962 -0.97814688 0.20784496 0.11480299
# -0.99990853 -0.00270374 0.01325251 -0.00192624
# -0.01240094 -0.20789751 -0.97807200 0.95438088
# 0.00000000 0.00000000 0.00000000 1.00000000


    @property
    def _default_sensor_configs(self):
        bimanual_center_pose = self._compute_bimanual_center_pose()
        bimanual_center_T_front_camera = np.array(
            # [
            #     [0.00539962, -0.97814688, 0.20784496, 0.11480299],
            #     [-0.99990853, -0.00270374, 0.01325251, -0.00192624],
            #     [-0.01240094, -0.20789751, -0.97807200, 0.95438088],
            #     [0.00000000, 0.00000000, 0.00000000, 1.00000000],
            # ],

            [

                [0.00604744, -0.97743607, 0.21114487, 0.11216216],
                [-0.99990406, -0.00327925, 0.01345807, -0.00125407],
                [-0.01246200, -0.21120600, -0.97736212, 0.95348328],
                [0.0, 0.0, 0.0, 1.0]
            ],

            dtype=np.float32,
        )
        rotation_np = bimanual_center_T_front_camera[:3, :3]
        # 実機カメラ座標とManiSkill上のカメラ座標の定義差を吸収する変換。
        # これは「カメラ座標系だけ」を入れ替える解析的な固定回転で、
        # bimanual_center座標系には手を入れない。
        # 軸対応: x_new = -y_old, y_new = -z_old, z_new = x_old
        camera_axis_map = np.array(
            [
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        rotation_np = rotation_np @ camera_axis_map
        rotation = torch.tensor(rotation_np, dtype=torch.float32)
        translation = torch.tensor(
            bimanual_center_T_front_camera[:3, 3], dtype=torch.float32
        )
        front_camera_pose_in_bimanual_center = Pose.create_from_pq(
            p=translation,
            q=matrix_to_quaternion(rotation),
        )
        # print(f"translation: {translation}")
        # print(f"rotation: {rotation}")
        # print(f"q:{matrix_to_quaternion(rotation)}")
        pose = Pose.create(bimanual_center_pose) * front_camera_pose_in_bimanual_center
        scale = 1.0 if self.collect_rmb_data else 0.4
        base_intrinsic = np.array(
            [
                [606.135498046875, 0.0, 330.1974182128906],
                [0.0, 605.4591674804688, 244.23666381835938],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        intrinsic = base_intrinsic.copy()
        intrinsic[0, 0] *= scale
        intrinsic[1, 1] *= scale
        intrinsic[0, 2] *= scale
        intrinsic[1, 2] *= scale
        return [
            CameraConfig(
                "top",
                pose=pose,
                width=int(640 * scale),
                height=int(480 * scale),
                intrinsic=intrinsic,
                fov=None,

                near=0.01,
                far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.2, 1.0, 1.0], [0.0, 0.0, 0.4])
        return CameraConfig(
            "render_camera", pose=pose, width=800, height=800, fov=1.0, near=0.01, far=5.0
        )

    def _load_lighting(self, options: dict):
        obs_mode = str(getattr(self, "obs_mode", "")).lower()
        if "rgb" in obs_mode or self.collect_rmb_data:
            for scene in self.scene.sub_scenes:
                scene.render_system.ambient_light = np.random.uniform(
                    0.2, 0.5, size=(3,)
                ).astype(np.float32)
        else:
            self.scene.set_ambient_light([0.3, 0.3, 0.3])
        self.scene.add_directional_light(
            [1, 1, -1], [1, 1, 1], shadow=False, shadow_scale=5, shadow_map_size=2048
        )
        self.scene.add_directional_light([0, 0, -1], [1, 1, 1])


    def _load_scene(self, options: Dict[str, Any]):
        self.table_scene = DualXarm7TableSceneBuilder(env=self)
        self.table_scene.build()
        self.bimanual_center_pose = self._compute_bimanual_center_pose()
        builder = self.scene.create_actor_builder()
        #builder.add_box_collision(
            #half_size=self.BOX_HALF_SIZE,
            #density=self.BOX_DENSITY,
        #)

        random_box_objects = []
        box_material = sapien.render.RenderMaterial()
        box_material.set_base_color([1.0, 1.0, 1.0, 1.0])

        # Initial pose will be overwritten during episode init; place safely above table for now.
        builder.initial_pose = sapien.Pose(
                p=self._bimanual_center_point_to_world(
                    [
                        self.BOX_X_OFFSET_FROM_BASE,
                        0.0,
                        self.BOX_HALF_SIZE[2] + 1e-2,
                    ]
                )
            )
        
        builder.add_box_collision(
            half_size=self.BOX_HALF_SIZE,
            density=self.BOX_DENSITY,
        )

        visual_file = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "pose_estimation",
                "box_data",
                "cardboard3.obj",
            )
        )
        builder.add_visual_from_file(
            filename=visual_file,

            scale=[0.01, 0.01, 0.01],
            # Local-axis rotation: +90° about X, then +90° about Y
            pose=sapien.Pose(q=[0.5, 0.5, 0.5, 0.5]),

        )
            
        
        
        #for i in range(self.num_envs):
            #builder = self.scene.create_actor_builder()
            #ここで箱オブジェクトに関するあらゆるプロパティをランダマイズすることが可能です。
            #builder = self.scene.create_actor_builder()
            
            #randomized_box_half_size = self.BOX_HALF_SIZE + np.random.uniform(-0.1, 0.1, size=self.BOX_HALF_SIZE.shape)

            #builder.add_box_collision(half_size=randomized_box_half_size)
            #builder.add_box_visual(half_size=randomized_box_half_size,material=box_material)
            
            #obj = builder.build(name = f"box_object_{i}")
            #self.remove_from_state_dict_registry(obj)
            #random_box_objects.append(obj)
        #self.box_objects = Actor.merge(random_box_objects, name = "object")
        #self.add_to_state_dict_registry(self.box_objects)

       
        self._enable_pushpoint_debug = True #Now it is completely True(No overwrite)
        
        

        self.box = builder.build(name="box_between_arms")
        if self._enable_pushpoint_debug:
            # pushpoint_radius = 0.02
            # self.pushpoint_left_site = actors.build_sphere(
            #     self.scene,
            #     radius=pushpoint_radius,
            #     color=(1.0, 0.0, 0.0, 1.0),
            #     name="pushpoint_left_site",
            #     body_type="kinematic",
            #     add_collision=False,
            #     initial_pose=sapien.Pose(
            #         p=self._bimanual_center_point_to_world(
            #             [0.0, 0.0, self.BOX_HALF_SIZE[2]]
            #         )
            #     ),
            # )
            # self.pushpoint_right_site = actors.build_sphere(
            #     self.scene,
            #     radius=pushpoint_radius,
            #     color=(0.3, 0.3, 0.9, 0.8),
            #     name="pushpoint_right_site",
            #     body_type="kinematic",
            #     add_collision=False,
            #     initial_pose=sapien.Pose(
            #         p=self._bimanual_center_point_to_world(
            #             [0.0, 0.0, self.BOX_HALF_SIZE[2]]
            #         )
            #     ),
            # )
            self.pushpoint_left_site = None
            self.pushpoint_right_site = None
        else:
            self.pushpoint_left_site = None
            self.pushpoint_right_site = None


        #Visualize bimanual center as a small orange sphere for debugging
        # self.bimanual_center_site = actors.build_sphere(
        #     self.scene,
        #     radius=0.02,
        #     color=(1.0, 0.0, 0.0, 0.8),
        #     name="bimanual_center_site",
        #     body_type="kinematic",
        #     add_collision=False,
        #     initial_pose=sapien.Pose(
        #         p=[
        #             self.bimanual_center_pose.p[0],
        #             self.bimanual_center_pose.p[1],
        #             self.bimanual_center_pose.p[2]
        #         ]
        #     ),
        # )
        self.bimanual_center_site = None
        
    def _configure_observed_joint_indices(self):
        self._agent_obs_joint_indices = dict()
        if not isinstance(self.agent, MultiAgent):
            return
        for idx, sub_agent in enumerate(self.agent.agents):
            key = f"{sub_agent.uid}-{idx}"
            indices = self._build_agent_joint_indices(sub_agent)
            if indices is None:
                if hasattr(sub_agent, "_obs_joint_indices"):
                    delattr(sub_agent, "_obs_joint_indices")
                continue
            self._agent_obs_joint_indices[key] = indices
            sub_agent._obs_joint_indices = indices

    def _build_agent_joint_indices(self, sub_agent) -> Optional[torch.Tensor]:
        observed_joint_names: List[str] = []
        arm_joint_names = getattr(sub_agent, "arm_joint_names", None)
        if arm_joint_names is not None:
            observed_joint_names.extend(list(arm_joint_names))
        gripper_joint_names = getattr(sub_agent, "gripper_joint_names", None)
        if gripper_joint_names:
            drive_joint = gripper_joint_names[0]  # mimicチェーンの代表として0番目を観測へ残す
            if drive_joint not in observed_joint_names:
                observed_joint_names.append(drive_joint)
        if not observed_joint_names:
            return None
        return get_active_joint_indices(
            sub_agent.robot, observed_joint_names
        ).long()

    def _clear(self):
        super()._clear()
        self._agent_obs_joint_indices = dict()

    def _ensure_pushpoint_buffers(self):
        num_envs = getattr(self, "num_envs", 1)
        needs_init = not hasattr(self, "pushpoint_local_right")
        if not needs_init:
            needs_init = self.pushpoint_local_right.shape[0] != num_envs
        if not needs_init:
            return
        zeros = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)
        self.pushpoint_local_right = zeros.clone()
        self.pushpoint_local_left = zeros.clone()

    def _ensure_rotation_buffers(self) -> bool:
        num_envs = getattr(self, "num_envs", 1)
        needs_init = not hasattr(self, "_prev_box_theta")
        if not needs_init:
            needs_init = self._prev_box_theta.shape[0] != num_envs
        if not needs_init:
            return False
        zeros = torch.zeros((num_envs,), device=self.device, dtype=torch.float32)
        self._prev_box_theta = zeros.clone()
        self._positive_yaw_progress = zeros.clone()
        return True

    def _reset_box_obs_offset_buffers(self, env_idx: torch.Tensor):
        assert env_idx.ndim == 1
        env_idx_long = env_idx.long()
        if self._box_obs_translation_offset is None:
            assert self.num_envs is not None
            self._box_obs_translation_offset = torch.zeros(
                (self.num_envs, 3), device=self.device, dtype=torch.float32
            )
            self._box_obs_yaw_offset = torch.zeros(
                (self.num_envs,), device=self.device, dtype=torch.float32
            )
        else:
            assert self._box_obs_translation_offset.shape == (self.num_envs, 3)
            assert self._box_obs_yaw_offset.shape == (self.num_envs,)
        translation, yaw = self._sample_box_obs_noise(
            env_idx_long.numel(),
            self.BOX_OBS_TRANSLATION_OFFSET_M,
            self.BOX_OBS_YAW_OFFSET_RAD,
        )
        self._box_obs_translation_offset[env_idx_long] = translation
        self._box_obs_yaw_offset[env_idx_long] = yaw

    def _sample_box_obs_noise(
        self, batch_size: int, max_translation: float, max_yaw: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        translation = (
            (torch.rand((batch_size, 3), device=self.device) - 0.5)
            * 2.0
            * float(max_translation)
        )
        yaw = (
            (torch.rand((batch_size,), device=self.device) - 0.5)
            * 2.0
            * float(max_yaw)
        )
        return translation, yaw

    def _yaw_rotation_matrix(self, yaw: torch.Tensor) -> torch.Tensor:
        cos = torch.cos(yaw)
        sin = torch.sin(yaw)
        zeros = torch.zeros_like(cos)
        ones = torch.ones_like(cos)
        return torch.stack(
            [
                torch.stack([cos, -sin, zeros], dim=-1),
                torch.stack([sin, cos, zeros], dim=-1),
                torch.stack([zeros, zeros, ones], dim=-1),
            ],
            dim=-2,
        )

    def _apply_box_obs_noise(
        self,
        position: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        yaw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert position.ndim == 2
        assert rotation.ndim == 3
        assert translation.ndim == 2
        assert yaw.ndim == 1
        assert position.shape[0] == rotation.shape[0]
        assert position.shape[0] == translation.shape[0]
        assert position.shape[0] == yaw.shape[0]
        position = position + translation
        rot_yaw = self._yaw_rotation_matrix(yaw)
        rotation = torch.matmul(rot_yaw, rotation)
        return position, rotation

    def _apply_box_obs_offset(
        self, position: torch.Tensor, rotation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._box_obs_translation_offset is not None
        assert self._box_obs_yaw_offset is not None
        batch = position.shape[0]
        translation = self._box_obs_translation_offset[:batch]
        yaw = self._box_obs_yaw_offset[:batch]
        return self._apply_box_obs_noise(position, rotation, translation, yaw)

    def _apply_box_obs_step_jitter(
        self, position: torch.Tensor, rotation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        translation, yaw = self._sample_box_obs_noise(
            position.shape[0],
            self.BOX_OBS_TRANSLATION_JITTER_M,
            self.BOX_OBS_YAW_JITTER_RAD,
        )
        return self._apply_box_obs_noise(position, rotation, translation, yaw)

    def _reset_rotation_buffers(
        self, env_idx: torch.Tensor, theta_deg: torch.Tensor
    ):
        self._ensure_rotation_buffers()
        if env_idx.numel() == 0:
            return
        env_idx_long = env_idx.long()
        # Use actual box yaw at reset to avoid spurious large deltas on first step.
        current_theta = self._get_box_theta_deg().to(
            device=self.device, dtype=torch.float32
        )
        if current_theta.ndim == 0:
            current_theta = current_theta.unsqueeze(0)
        self._prev_box_theta[env_idx_long] = current_theta[env_idx_long]
        self._positive_yaw_progress[env_idx_long] = 0.0

    def _theta_to_quaternion(self, theta_deg: torch.Tensor) -> torch.Tensor:
        """Convert planar angle in degrees (around z) to quaternion."""
        if not isinstance(theta_deg, torch.Tensor):
            theta_deg = torch.tensor(theta_deg, device=self.device, dtype=torch.float32)
        else:
            theta_deg = theta_deg.to(device=self.device, dtype=torch.float32)
        theta_rad = theta_deg * (torch.pi / 180.0)
        half = 0.5 * theta_rad
        zeros = torch.zeros_like(theta_rad)
        w = torch.cos(half)
        z = torch.sin(half)
        return torch.stack((w, zeros, zeros, z), dim=-1)

    def _get_box_theta_deg(self) -> torch.Tensor:
        """Return box yaw in degrees within [0, 360)."""
        box_pose = Pose.create(self.box.pose, device=self.device)
        rotation = quaternion_to_matrix(box_pose.q)
        yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
        theta = torch.rad2deg(yaw)
        theta = torch.remainder(theta, 360.0)
        theta = torch.remainder(-theta, 360.0)
        return theta

    def _update_initial_pushpoints(
        self, env_idx: torch.Tensor, positions: torch.Tensor, orientations: torch.Tensor
    ):
        batch_size = positions.shape[0]
        if batch_size == 0:
            return
        rotations = quaternion_to_matrix(orientations)
        hx = float(self.BOX_HALF_SIZE[0])
        hy = float(self.BOX_HALF_SIZE[1])

        long_side_offsets_local = torch.tensor(
            [[hx, 0.0, 0.0], [-hx, 0.0, 0.0]],
            device=self.device,
            dtype=torch.float32,
        )
        long_side_offsets_local = long_side_offsets_local.transpose(0, 1)  # (3, 2)
        long_side_offsets_world = torch.matmul(
            rotations, long_side_offsets_local.unsqueeze(0).expand(batch_size, -1, -1)
        )  # (batch, 3, 2)
        long_side_offsets_world = long_side_offsets_world.transpose(1, 2)  # (batch, 2, 3)
        long_side_centers_world = positions.unsqueeze(1) + long_side_offsets_world
        # Treat +X (local +hx) as the forward side deterministically.
        initial_forward_side_center = long_side_centers_world[:, 0, :]

        push_offset_local = torch.tensor(
            [0.0, -0.8 * hy, 0.0],
            device=self.device,
            dtype=torch.float32,
        )
        push_offset_world = torch.einsum("bij,j->bi", rotations, push_offset_local)
        initial_pushpoint_by_right = initial_forward_side_center + push_offset_world
        initial_pushpoint_by_left = (
            2.0 * positions - initial_pushpoint_by_right
        )
        env_idx_long = env_idx.long()
        rotations_T = rotations.transpose(1, 2)
        local_right = torch.einsum(
            "bij,bj->bi", rotations_T, initial_pushpoint_by_right - positions
        )
        local_left = torch.einsum(
            "bij,bj->bi", rotations_T, initial_pushpoint_by_left - positions
        )
        self.pushpoint_local_right[env_idx_long] = local_right
        self.pushpoint_local_left[env_idx_long] = local_left
        if getattr(self, "_enable_pushpoint_debug", False):
            if self.pushpoint_left_site is not None:
                self.pushpoint_left_site.set_pose(
                    Pose.create_from_pq(p=initial_pushpoint_by_left)
                )
            if self.pushpoint_right_site is not None:
                self.pushpoint_right_site.set_pose(
                    Pose.create_from_pq(p=initial_pushpoint_by_right)
                )

    def _initialize_episode(self, env_idx: torch.Tensor, options: Dict[str, Any]):
        with torch.device(self.device):
            noise_scale = self.robot_init_noise_scale
            if options is not None:
                noise_scale = float(options.get("robot_init_noise_scale", noise_scale))
            if hasattr(self.table_scene, "set_noise_scale"):
                self.table_scene.set_noise_scale(noise_scale)
            self.table_scene.initialize(env_idx)
            batch_size = len(env_idx)
            if batch_size == 0:
                return

            self._ensure_pushpoint_buffers()
            positions, theta = self._sample_initial_box_pose(batch_size, options)
            orientations = self._theta_to_quaternion(theta)
            #self.box_objects[env_idx].set_pose(Pose.create_from_pq(positions, orientations))
            self.box.set_pose(Pose.create_from_pq(positions, orientations))
            self._update_initial_pushpoints(env_idx, positions, orientations)
            self._reset_rotation_buffers(env_idx, theta)
            self._reset_box_obs_offset_buffers(env_idx)

    def _sample_initial_box_pose(
        self, batch_size: int, options: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_center = None
        y_center = None
        theta = None
        if options is not None:
            fixed_xy = options.get("box_jitter_xy", None)
            if fixed_xy is not None:
                fixed_xy = torch.tensor(
                    fixed_xy, device=self.device, dtype=torch.float32
                )
                if fixed_xy.ndim == 1:
                    assert batch_size == 1
                    fixed_xy = fixed_xy.unsqueeze(0)
                assert fixed_xy.shape == (batch_size, 2)
                x_center = self.BOX_X_OFFSET_FROM_BASE + fixed_xy[:, 0]
                y_center = fixed_xy[:, 1]

            fixed_theta = options.get("box_jitter_theta_deg", None)
            if fixed_theta is not None:
                fixed_theta = torch.tensor(
                    fixed_theta, device=self.device, dtype=torch.float32
                )
                if fixed_theta.ndim == 0:
                    fixed_theta = fixed_theta.unsqueeze(0)
                if fixed_theta.ndim == 1 and fixed_theta.shape[0] == 1:
                    assert batch_size == 1
                assert fixed_theta.shape == (batch_size,)
                theta = (
                    torch.full(
                        (batch_size,),
                        self.BOX_ROTATION_MEAN_DEG,
                        device=self.device,
                        dtype=torch.float32,
                    )
                    + fixed_theta
                )

        if x_center is None:
            x_center = (
                self.BOX_X_OFFSET_FROM_BASE
                + (torch.rand(batch_size, device=self.device) - 0.5)
                * 2
                * self.BOX_X_JITTER
            )
        if y_center is None:
            y_center = (
                (torch.rand(batch_size, device=self.device) - 0.5)
                * 2
                * self.BOX_Y_JITTER
            )
        z_center = torch.full(
            (batch_size,),
            self.BOX_HALF_SIZE[2] - PEDESTAL_HEIGHT,
            device=self.device,
            dtype=torch.float32,
        )
        positions = torch.stack((x_center, y_center, z_center), dim=-1)
        positions = self._bimanual_center_tensor_to_world(positions)
        if theta is None:
            theta = torch.full(
                (batch_size,),
                self.BOX_ROTATION_MEAN_DEG,
                device=self.device,
                dtype=torch.float32,
            )
            if self.BOX_ROTATION_JITTER_DEG > 0.0:
                # jitter is specified in degrees; keep theta in degrees before quaternion conversion
                theta = theta + (
                    (torch.rand(batch_size, device=self.device) - 0.5)
                    * 2.0
                    * float(self.BOX_ROTATION_JITTER_DEG)
                )
        return positions, theta

    def _pose_to_6d(self, pose: Pose, *, center_frame: bool = False) -> torch.Tensor:
        matrix = pose.to_transformation_matrix()[..., :3, :3]
        position = pose.p
        if position.ndim == 1:
            position = position.unsqueeze(0)
            matrix = matrix.unsqueeze(0)
        if center_frame:
            center = torch.tensor(
                self.bimanual_center_pose.p,
                dtype=position.dtype,
                device=position.device,
            )
            position = position - center
        return torch.cat([position, matrix[..., :, 0], matrix[..., :, 1]], dim=-1)

    def _compute_contact_penalty(self, info: Dict[str, Any]) -> torch.Tensor:
        """Contact penalty using explicit pair list only."""
        device = self.device
        threshold = self.CONTACT_FORCE_THRESHOLD
        weight = self.CONTACT_PENALTY_WEIGHT

        def _ensure_batch(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0) if t.ndim == 1 else t

        penalty_forces = []
        penalty_names = []

        ball_box_forces = []
        ball_box_names = []
        ball_box_above_top_flags = []

        # Prepare environment actors
        table_actor = getattr(self.table_scene, "table", None)
        pedestal_actor = getattr(self.table_scene, "robot_pedestal", None)
        if table_actor is None:
            raise ValueError("Expected environment actor 'table' to exist, got None")
        if self.box is None:
            raise ValueError("Expected environment actor 'box' to exist, got None")
        if pedestal_actor is None:
            raise ValueError("Expected environment actor 'robot_pedestal' to exist, got None")

        box_center = _ensure_batch(Pose.create(self.box.pose, device=device).p)
        box_top_z = box_center[..., 2] + float(self.BOX_HALF_SIZE[2])
        tcp_positions = [
            _ensure_batch(Pose.create(self.agent.agents[0].tcp_pose, device=device).p),
            _ensure_batch(Pose.create(self.agent.agents[1].tcp_pose, device=device).p),
        ]

        # Environment-object penalty: box hitting pedestal
        f_box_pedestal = self.scene.get_pairwise_contact_forces(self.box, pedestal_actor).to(device)
        penalty_forces.append(torch.linalg.norm(f_box_pedestal, dim=-1))
        penalty_names.append("box|robot_pedestal")

        for idx, sub_agent in enumerate(self.agent.agents):
            prefix = "L" if idx == 0 else "R"
            links = list(sub_agent.robot.get_links())
            name_map = {l.name: l for l in links}
            ball_link = name_map.get("link_tcp_ball")
            stick_link = name_map.get("link_tcp_stick")
            base_link = name_map.get("xarm_gripper_base_link")
            link2 = name_map.get("link2")
            link6 = name_map.get("link6")
            link5 = name_map.get("link5")
            link1 = name_map.get("link1")
            if (
                ball_link is None
                or stick_link is None
                or base_link is None
                or link2 is None
                or link6 is None
                or link5 is None
                or link1 is None
            ):
                raise ValueError("Missing expected link on agent")

            # Penalty pairs
            for pair_name, a, b in (
                (f"{prefix}_xarm_gripper_base_link|box", base_link, self.box),
                (f"{prefix}_link_tcp_stick|box", stick_link, self.box),
                (f"{prefix}_link_tcp_ball|table", ball_link, table_actor),
                (f"{prefix}_link2|{prefix}_link6", link2, link6),
                (f"{prefix}_link6|{prefix}_link1", link6, link1),
                (f"{prefix}_link5|{prefix}_link1", link5, link1),
            ):
                f = self.scene.get_pairwise_contact_forces(a, b).to(device)
                penalty_forces.append(torch.linalg.norm(f, dim=-1))
                penalty_names.append(pair_name)

            # Ball-box contact is penalized only when the TCP is above the box top surface.
            tcp_pos = tcp_positions[idx]
            f_ball_box = self.scene.get_pairwise_contact_forces(ball_link, self.box).to(device)
            ball_box_force = torch.linalg.norm(f_ball_box, dim=-1)
            tcp_above_top = tcp_pos[..., 2] > box_top_z
            assert ball_box_force.shape == tcp_above_top.shape
            ball_box_penalty_force = ball_box_force * tcp_above_top.float()

            penalty_forces.append(ball_box_penalty_force)
            penalty_names.append(f"{prefix}_link_tcp_ball|box_above_top")

            ball_box_forces.append(ball_box_force)
            ball_box_names.append(f"{prefix}_link_tcp_ball|box")
            ball_box_above_top_flags.append(tcp_above_top)



        if penalty_forces:
            stacked = torch.stack(penalty_forces, dim=-1)  # [B, num_pairs]
            has_contact = (stacked > threshold).any(dim=-1)
            penalty = has_contact.float() * weight
            info["contact/force"] = (stacked.detach().cpu(), penalty_names)
            info["contact/penalty"] = penalty.detach().cpu()
        else:
            penalty = torch.zeros((self.num_envs,), device=device)

        if ball_box_forces:
            ball_box_stacked = torch.stack(ball_box_forces, dim=-1)
            info["contact/ball_box_force"] = (
                ball_box_stacked.detach().cpu(),
                ball_box_names,
            )
        if ball_box_above_top_flags:
            above_top_stacked = torch.stack(ball_box_above_top_flags, dim=-1)
            info["contact/ball_box_tcp_above_top"] = (
                above_top_stacked.detach().cpu(),
                ball_box_names,
            )

        return penalty

    @dataclass
    class RewardContext:
        current_box_center: torch.Tensor
        target_pushpoint_left: torch.Tensor
        target_pushpoint_right: torch.Tensor
        left_tcp_pos: torch.Tensor
        right_tcp_pos: torch.Tensor
        box_theta_deg: torch.Tensor
        box_rotation_matrix: torch.Tensor

    def _gather_reward_context(
        self, info: Dict[str, Any]
    ) -> "MyDualBoxRotationEnv.RewardContext":
        theta = self._get_box_theta_deg()
        if theta.ndim == 0:
            theta = theta.unsqueeze(0)
        theta = theta.to(device=self.device, dtype=torch.float32)
        self._ensure_pushpoint_buffers()
        box_pose = Pose.create(self.box.pose, device=self.device)
        current_box_center = box_pose.p
        if current_box_center.ndim == 1:
            current_box_center = current_box_center.unsqueeze(0)
        rotation_current = quaternion_to_matrix(box_pose.q)
        (
            target_pushpoint_left,
            target_pushpoint_right,
            inversion_conditions,
            world_pushpoint_left,
            world_pushpoint_right,
        ) = self._compute_pushpoints_from_pose(
            current_box_center, rotation_current, theta
        )

        left_tcp_pos = Pose.create(
            self.agent.agents[0].tcp_pose, device=self.device
        ).p
        left_tcp_pos = (
            left_tcp_pos.squeeze(0)
            if left_tcp_pos.ndim == 2 and left_tcp_pos.shape[0] == 1
            else left_tcp_pos
        )
        right_tcp_pos = Pose.create(
            self.agent.agents[1].tcp_pose, device=self.device
        ).p
        right_tcp_pos = (
            right_tcp_pos.squeeze(0)
            if right_tcp_pos.ndim == 2 and right_tcp_pos.shape[0] == 1
            else right_tcp_pos
        )

        info["box_theta_deg"] = theta.detach().cpu()
        #info["pushpoint_inversion_conditions"] = inversion_conditions.detach().cpu()
        center = torch.tensor(
            self.bimanual_center_pose.p,
            device=self.device,
            dtype=target_pushpoint_left.dtype,
        )
        info["current_pushpoint_by_right"] = target_pushpoint_right.detach().cpu()
        info["current_pushpoint_by_left"] = target_pushpoint_left.detach().cpu()
        info["current_pushpoint_by_right_from_bimanual_center"] = (
            target_pushpoint_right - center
        ).detach().cpu()
        info["current_pushpoint_by_left_from_bimanual_center"] = (
            target_pushpoint_left - center
        ).detach().cpu()

        if getattr(self, "_enable_pushpoint_debug", False):
            if self.pushpoint_left_site is not None:
                self.pushpoint_left_site.set_pose(
                    Pose.create_from_pq(p=target_pushpoint_left)
                )
            if self.pushpoint_right_site is not None:
                self.pushpoint_right_site.set_pose(
                    Pose.create_from_pq(p=target_pushpoint_right)
                )

        return self.RewardContext(
            current_box_center=current_box_center,
            target_pushpoint_left=target_pushpoint_left,
            target_pushpoint_right=target_pushpoint_right,
            left_tcp_pos=left_tcp_pos,
            right_tcp_pos=right_tcp_pos,
            box_theta_deg=theta,
            box_rotation_matrix=rotation_current,
        )

    def compute_closeness_score_to_reset_state(self) -> torch.Tensor:
        """
        Return the joint-position return reward for both arms.

        Each arm joint contributes linearly:
        - 0.25 reward at the exact baseline reset pose
        - 0.0 reward at pi radians or farther from the baseline

        With two xArm7 arms, the maximum return reward is 14 * 0.25 = 3.5.

        This score is intentionally independent of any initialization noise parameters
        (e.g., ``robot_init_noise_scale`` / ``noise_scale`` / ``_OFFSET_LOW/_OFFSET_HIGH``).
        """
        baseline = Xarm7InitialRandomizationSceneBuilder._JOINT_ONLY_RESET_STATE_OF_ROBOMANIPBASELINES_.to(
            device=self.device, dtype=torch.float32
        )
        if baseline.numel() != 7:
            raise ValueError(
                "Expected _JOINT_ONLY_RESET_STATE_OF_ROBOMANIPBASELINES_ to have 7 elements, "
                f"got {baseline.numel()}"
            )

        scores: List[torch.Tensor] = []
        for sub_agent in getattr(self.agent, "agents", []):
            qpos = sub_agent.robot.get_qpos()
            if qpos.ndim == 1:
                qpos = qpos.unsqueeze(0)

            dof = int(qpos.shape[-1])
            if dof < 7:
                raise ValueError(
                    f"Expected each sub-agent to have at least 7 DoF qpos, got {dof}"
                )
            qpos_joint_only = qpos[..., :7].to(device=self.device, dtype=torch.float32)

            delta = (qpos_joint_only - baseline.unsqueeze(0)).abs()
            normalized_delta = torch.clamp(delta / torch.pi, min=0.0, max=1.0)
            joint_rewards = 0.25 * (1.0 - normalized_delta)
            scores.append(joint_rewards.sum(dim=-1))

        if not scores:
            return torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        return torch.stack(scores, dim=0).sum(dim=0)

    def _compute_pushpoints_from_pose(
        self,
        current_box_center: torch.Tensor,
        rotation_current: torch.Tensor,
        theta: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute world-frame pushpoints from the box pose.
        Separated for easy customization of how pushpoints are derived (e.g., adding extra rotation).
        """
        world_pushpoint_right = current_box_center + torch.einsum(
            "bij,bj->bi", rotation_current, self.pushpoint_local_right
        )
        world_pushpoint_left = current_box_center + torch.einsum(
            "bij,bj->bi", rotation_current, self.pushpoint_local_left
        )
        normal_conditions = (theta >= 0.0) & (theta < 125.0) | (theta >= 305.0)
        inversion_conditions = ~normal_conditions
        inversion_conditions_mask = inversion_conditions.unsqueeze(-1)
        target_pushpoint_by_right = torch.where(
            inversion_conditions_mask, world_pushpoint_left, world_pushpoint_right
        )
        target_pushpoint_by_left = torch.where(
            inversion_conditions_mask, world_pushpoint_right, world_pushpoint_left
        )
        return (
            target_pushpoint_by_left.clone(),
            target_pushpoint_by_right.clone(),
            inversion_conditions,
            world_pushpoint_left,
            world_pushpoint_right,
        )

    def _log_observation_to_info(self, info: Dict[str, Any]):
        """Copy structured observation entries into info so they can be logged with names."""
        if info is None:
            return
        try:
            obs_state = self._get_obs_state_dict(info)
        except Exception:
            return
        flat_tensor = flatten_state_dict(obs_state, use_torch=True, device=self.device)
        column_names = self._get_obs_column_names(obs_state)
        info["obs_flat"] = (flat_tensor.detach().cpu(), column_names)
        flat_obs = flatten_dict_keys(obs_state)
        for key, value in flat_obs.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                tensor = value
            else:
                try:
                    tensor = torch.as_tensor(value, device=self.device)
                except Exception:
                    continue
            info[f"obs/{key}"] = tensor.detach().cpu()

    def _get_obs_column_names(self, obs_state: Dict[str, Any]) -> List[str]:
        if self._flat_obs_column_names is None:
            columns: List[str] = []
            self._collect_obs_columns(obs_state, prefix="", out=columns)
            self._flat_obs_column_names = columns
        return self._flat_obs_column_names

    def _collect_obs_columns(self, value: Any, prefix: str, out: List[str]):
        if isinstance(value, dict):
            for key, subvalue in value.items():
                new_prefix = f"{prefix}{key}" if not prefix else f"{prefix}/{key}"
                self._collect_obs_columns(subvalue, new_prefix, out)
            return
        tensor = to_tensor(value, device=self.device)
        if tensor.numel() == 0:
            return
        if tensor.ndim <= 1:
            out.append(prefix)
            return
        feature_dim = tensor.shape[-1]
        if feature_dim == 1:
            out.append(prefix)
            return
        for i in range(feature_dim):
            out.append(f"{prefix}_{i}")

    def _get_obs_agent(self):
        obs = super()._get_obs_agent()
        if not isinstance(obs, dict):
            return obs
        index_map = getattr(self, "_agent_obs_joint_indices", None)
        if not index_map:
            return obs
        filtered = {}
        for key, value in obs.items():
            indices = index_map.get(key)
            if indices is None:
                filtered[key] = value
                continue
            filtered[key] = self._filter_agent_obs_entry(value, indices)
        return filtered

    def _filter_agent_obs_entry(
        self, obs_entry: Any, indices: torch.Tensor
    ) -> Any:
        if not isinstance(obs_entry, dict):
            return obs_entry
        filtered_entry = dict(obs_entry)
        if "qpos" in obs_entry and obs_entry["qpos"] is not None:
            filtered_entry["qpos"] = self._index_select_joint_tensor(
                obs_entry["qpos"], indices
            )
        if "qvel" in obs_entry and obs_entry["qvel"] is not None:
            filtered_entry["qvel"] = self._index_select_joint_tensor(
                obs_entry["qvel"], indices
            )
        return filtered_entry

    def _index_select_joint_tensor(
        self, tensor: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        if tensor is None:
            return tensor
        idx = indices.to(device=tensor.device, dtype=torch.long)
        dim = tensor.dim() - 1
        return tensor.index_select(dim, idx)

    def _get_obs_extra(self, info: Dict[str, Any]):
        extra = self._obs_extra_fn(self, info)
        if info is not None and self.collect_rmb_data:
            rgb_images = self._capture_rgb_images()
            if rgb_images:
                info["rgb_images"] = rgb_images
        return extra

    def _capture_rgb_images(self) -> Dict[str, Any]:
        """Capture RGB images from all sensors (assumes rgb is always available)."""
        sensor_obs = self._get_obs_sensor_data()
        rgb_images: Dict[str, Any] = {}
        for name, data in sensor_obs.items():
            rgb = data["rgb"]
            rgb_images[name] = rgb.detach().cpu() if torch.is_tensor(rgb) else rgb
        return rgb_images

    def _get_obs_state_dict(self, info: Dict[str, Any]):
        obs = super()._get_obs_state_dict(info)
        bad_mask, bad_keys = self._find_bad_obs(obs)
        if bad_mask is not None and bad_mask.any().item():
            bad_envs = bad_mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
            print(
                "[warn] non-finite/huge obs detected; "
                f"elapsed_steps={info.get('elapsed_steps')}; "
                f"bad_env_idx={bad_envs}; keys={bad_keys}"
            )
            if self._last_good_obs is not None:
                obs = self._replace_obs_with_last(obs, self._last_good_obs, bad_mask)
            else:
                obs = self._replace_obs_with_zeros(obs, bad_mask)
            existing_fail = info.get("fail")
            if existing_fail is None:
                info["fail"] = bad_mask
            else:
                info["fail"] = existing_fail | bad_mask
            return obs
        self._last_good_obs = self._clone_obs(obs)
        return obs

    def _find_bad_obs(self, obs: Any) -> Tuple[Optional[torch.Tensor], List[str]]:
        bad_keys: List[str] = []
        bad_mask: Optional[torch.Tensor] = None

        def walk(value: Any, prefix: str):
            nonlocal bad_mask
            if isinstance(value, dict):
                for key, subvalue in value.items():
                    new_prefix = f"{prefix}{key}" if not prefix else f"{prefix}/{key}"
                    walk(subvalue, new_prefix)
                return
            try:
                tensor = to_tensor(value, device=self.device)
            except Exception:
                return
            if tensor.numel() == 0:
                return
            mask = self._bad_obs_mask(tensor)
            if mask is None:
                return
            if mask.any().item():
                bad_keys.append(prefix)
            if bad_mask is None:
                bad_mask = mask
            else:
                bad_mask = bad_mask | mask

        walk(obs, "")
        return bad_mask, bad_keys

    def _bad_obs_mask(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.ndim == 0:
            tensor = tensor.view(1)
        if tensor.ndim == 1:
            finite = torch.isfinite(tensor)
            abs_max = tensor.abs()
            bad = (~finite) | (~torch.isfinite(abs_max)) | (abs_max > self.OBS_ABS_MAX)
            return bad
        dims = tuple(range(1, tensor.ndim))
        finite = torch.isfinite(tensor).all(dim=dims)
        abs_max = tensor.abs().amax(dim=dims)
        bad = (~finite) | (~torch.isfinite(abs_max)) | (abs_max > self.OBS_ABS_MAX)
        std = tensor.float().std(dim=dims)
        bad = bad | (~torch.isfinite(std))
        return bad

    def _replace_obs_with_last(
        self, obs: Any, last_obs: Any, bad_mask: torch.Tensor
    ) -> Any:
        if isinstance(obs, dict) and isinstance(last_obs, dict):
            return {
                key: self._replace_obs_with_last(
                    obs.get(key), last_obs.get(key), bad_mask
                )
                for key in obs.keys()
            }
        if obs is None or last_obs is None:
            return obs
        tensor = to_tensor(obs, device=self.device)
        last_tensor = to_tensor(last_obs, device=self.device)
        if tensor.ndim == 0 or tensor.shape[0] != bad_mask.shape[0]:
            return tensor
        out = tensor.clone()
        out[bad_mask] = last_tensor[bad_mask]
        return out

    def _replace_obs_with_zeros(self, obs: Any, bad_mask: torch.Tensor) -> Any:
        if isinstance(obs, dict):
            return {key: self._replace_obs_with_zeros(value, bad_mask) for key, value in obs.items()}
        if obs is None:
            return obs
        tensor = to_tensor(obs, device=self.device)
        if tensor.ndim == 0 or tensor.shape[0] != bad_mask.shape[0]:
            return tensor
        out = tensor.clone()
        out[bad_mask] = 0
        return out

    def _clone_obs(self, obs: Any) -> Any:
        if isinstance(obs, dict):
            return {key: self._clone_obs(value) for key, value in obs.items()}
        if obs is None:
            return obs
        tensor = to_tensor(obs, device=self.device)
        return tensor.clone()

    def _compute_bimanual_center_pose(self) -> sapien.Pose:
        base_left = torch.tensor(
            [ROBOT_BASE_X_OFFSET, LEFT_ARM_Y_OFFSET, PEDESTAL_HEIGHT],
            dtype=torch.float32,
        )
        base_right = torch.tensor(
            [ROBOT_BASE_X_OFFSET, RIGHT_ARM_Y_OFFSET, PEDESTAL_HEIGHT],
            dtype=torch.float32,
        )
        center = 0.5 * (base_right + base_left)
        return sapien.Pose(p=center.tolist())

    def _bimanual_center_point_to_world(self, point):
        center = self.bimanual_center_pose.p
        return [
            center[0] + point[0],
            center[1] + point[1],
            center[2] + point[2],
        ]

    def _bimanual_center_tensor_to_world(self, tensor: torch.Tensor) -> torch.Tensor:
        center = torch.tensor(
            self.bimanual_center_pose.p, dtype=tensor.dtype, device=tensor.device
        )
        if tensor.ndim == 1:
            return tensor + center
        if tensor.ndim == 2:
            return tensor + center.unsqueeze(0)
        raise ValueError("Expected tensor to have rank 1 or 2.")

    def _pushpoint_tracking_reward(
        self,
        left_tcp_pos: torch.Tensor,
        right_tcp_pos: torch.Tensor,
        target_pushpoint_left: torch.Tensor,
        target_pushpoint_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        left_tcp_pos = left_tcp_pos.to(device=self.device, dtype=torch.float32)
        right_tcp_pos = right_tcp_pos.to(device=self.device, dtype=torch.float32)
        target_pushpoint_left = target_pushpoint_left.to(
            device=self.device, dtype=torch.float32
        )
        target_pushpoint_right = target_pushpoint_right.to(
            device=self.device, dtype=torch.float32
        )
        if left_tcp_pos.ndim == 1:
            left_tcp_pos = left_tcp_pos.unsqueeze(0)
        if right_tcp_pos.ndim == 1:
            right_tcp_pos = right_tcp_pos.unsqueeze(0)
        distance_left = torch.linalg.norm(
            left_tcp_pos - target_pushpoint_left, dim=-1
        )
        distance_right = torch.linalg.norm(
            right_tcp_pos - target_pushpoint_right, dim=-1
        )
        reward_left = 1 - torch.tanh(
            self.PUSHPOINT_DISTANCE_SCALE * distance_left
        )
        reward_right = 1 - torch.tanh(
            self.PUSHPOINT_DISTANCE_SCALE * distance_right
        )
        reward = 0.5 * (reward_left + reward_right)
        reached_left = distance_left < self.PUSHPOINT_DISTANCE_THRESHOLD
        reached_right = distance_right < self.PUSHPOINT_DISTANCE_THRESHOLD
        pushpoint_stage_complete = (reached_left & reached_right).float()
        info = {
            "distance_left": distance_left,
            "distance_right": distance_right,
            "reached_left": reached_left,
            "reached_right": reached_right,
            "pushpoint_stage_complete": pushpoint_stage_complete,
        }
        return reward, info, pushpoint_stage_complete

    def _box_translation_penalty(
        self, current_box_center: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Deviation from the nominal spawn center in XYZ to discourage lifting/carrying.
        nominal_center_world = torch.tensor(
            self._bimanual_center_point_to_world(
                [
                    self.BOX_X_OFFSET_FROM_BASE,
                    0.0,
                    self.BOX_HALF_SIZE[2] - PEDESTAL_HEIGHT,
                ]
            ),
            device=self.device,
            dtype=torch.float32,
        )
        translation_vector = current_box_center - nominal_center_world
        translation_distance = torch.linalg.norm(translation_vector, dim=-1)
        normalized_distance = torch.clamp(
            translation_distance / self.BOX_CENTER_MAX_OFFSET, 0.0, 1.0
        )
        normalized_penalty = smoothstep(normalized_distance)
        translation_penalty = self.BOX_TRANSLATION_PENALTY_MAX * normalized_penalty
        info = {
            "translation_distance": translation_distance,
            "translation_reference_world": nominal_center_world,
            "normalized_translation_distance": normalized_distance,
            "normalized_translation_penalty": normalized_penalty,
            "translation_penalty": translation_penalty,
        }
        return translation_penalty, info

    def _box_min_x_penalty(
        self, current_box_center: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Linear penalty based on box center x.
        Zero at BOX_X_OFFSET_FROM_BASE, 1.0 at BOX_CENTER_MIN_X, clamped to [0, 1].
        """
        center = torch.tensor(
            self.bimanual_center_pose.p, device=self.device, dtype=torch.float32
        )
        x_center_frame = current_box_center[..., 0] - center[0]
        ref_x = float(self.BOX_X_OFFSET_FROM_BASE - self.BOX_X_JITTER)
        min_x = float(self.BOX_CENTER_MIN_X)
        span = max(ref_x - min_x, 1e-6)
        normalized = torch.clamp((ref_x - x_center_frame) / span, 0.0, 1.0)
        penalty = normalized  # already in [0,1]
        info = {
            "box_min_x_penalty": penalty,
            "box_min_x_normalized": normalized,
            "box_min_x_reference": torch.tensor(ref_x, device=self.device),
            "box_min_x_span": torch.tensor(span, device=self.device),
        }
        return penalty, info

    def _box_yaw_rotation(
        self, theta_deg: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        buffers_reset = self._ensure_rotation_buffers()
        theta_deg = theta_deg.to(device=self.device, dtype=torch.float32)
        if buffers_reset:
            self._prev_box_theta = theta_deg.clone()
            delta = torch.zeros_like(theta_deg)
        else:
            prev_theta = self._prev_box_theta
            delta = torch.remainder(theta_deg - prev_theta + 180.0, 360.0) - 180.0
        signed_delta = delta
        self._prev_box_theta = theta_deg
        self._positive_yaw_progress = torch.clamp(
            self._positive_yaw_progress + signed_delta, min=0.0
        )
        target_delta_value = max(self.MAX_ROTATION_PACE * self.control_timestep, 1e-6)
        target_delta = torch.tensor(
            target_delta_value, device=self.device, dtype=torch.float32
        )
        step_reward = signed_delta / target_delta
        step_reward = torch.clamp(step_reward, min=-1.0, max=1.0)
        delta_rotation_score = 2.0 * step_reward  # range [-2, 2]

        goal_theta = torch.tensor(
            self.BOX_GOAL_YAW_DEG, device=self.device, dtype=torch.float32
        )
        angle_diff_rad = torch.deg2rad(goal_theta - theta_deg)
        rotation_score = (1.0 + torch.cos(angle_diff_rad))
        info = {
            "yaw_delta_deg": signed_delta,
            "yaw_step_reward": delta_rotation_score,
            "yaw_rotation_progress_deg": self._positive_yaw_progress,
            "yaw_rotation_target_delta_deg": signed_delta.new_full(
                signed_delta.shape, target_delta_value
            ),
        }
        return rotation_score, delta_rotation_score, info

    def _tcp_leading_reward(
        self,
        current_box_center: torch.Tensor,
        rotation_matrix: torch.Tensor,
        left_tcp_pos: torch.Tensor,
        right_tcp_pos: torch.Tensor,
        target_pushpoint_left: torch.Tensor,
        target_pushpoint_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Reward when TCP passes the (possibly inverted) pushpoints along +/-X on the box surface.

        Output is squashed by tanh then shifted into [0, TCP_LEAD_MAX], so small負も段階的に残しつつ上限だけクリップ。
        """

        def _ensure_batch(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0) if t.ndim == 1 else t

        current_box_center = _ensure_batch(current_box_center)
        left_tcp_pos = _ensure_batch(left_tcp_pos)
        right_tcp_pos = _ensure_batch(right_tcp_pos)
        target_pushpoint_left = _ensure_batch(target_pushpoint_left)
        target_pushpoint_right = _ensure_batch(target_pushpoint_right)
        if rotation_matrix.ndim == 2:
            rotation_matrix = rotation_matrix.unsqueeze(0)

        rot_T = rotation_matrix.transpose(1, 2)
        left_local = torch.einsum(
            "bij,bj->bi", rot_T, left_tcp_pos - current_box_center
        )
        right_local = torch.einsum(
            "bij,bj->bi", rot_T, right_tcp_pos - current_box_center
        )
        push_left_local = torch.einsum(
            "bij,bj->bi", rot_T, target_pushpoint_left - current_box_center
        )
        push_right_local = torch.einsum(
            "bij,bj->bi", rot_T, target_pushpoint_right - current_box_center
        )

        def _lead(
            push_local: torch.Tensor, tcp_local: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            push_x = push_local[..., 0]
            tcp_x = tcp_local[..., 0]
            direction = torch.sign(push_x)
            progress = direction * (tcp_x - push_x)
            progress = progress * direction.abs()
            progress = torch.clamp(progress, max=self.TCP_LEAD_SATURATION)
            saturation = self.TCP_LEAD_SATURATION + 1e-8
            # Squash with tanh then shift to [0,1], preserving magnitude differences for negative progress.
            scaled_progress = torch.tanh(progress / saturation)  # [-1, 1]
            normalized = 0.5 * (scaled_progress + 1.0)  # [0, 1]
            return normalized * self.TCP_LEAD_MAX, progress

        lead_right, lead_right_progress = _lead(push_right_local, right_local)
        lead_left, lead_left_progress = _lead(push_left_local, left_local)
        reward = 0.5 * (lead_right + lead_left)
        lead_stage_complete = (lead_right_progress >= 0.0) & (lead_left_progress >= 0.0)
        info = {
            "tcp_lead_reward": reward,
            "tcp_lead_right_component": lead_right,
            "tcp_lead_left_component": lead_left,
            "tcp_lead_stage_complete": lead_stage_complete,
        }
        return reward, info

    def _tcp_height_penalty(
        self,
        current_box_center: torch.Tensor,
        left_tcp_pos: torch.Tensor,
        right_tcp_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Penalize TCP dipping below box center z minus tolerance."""

        def _ensure_batch(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0) if t.ndim == 1 else t

        current_box_center = _ensure_batch(current_box_center)
        left_tcp_pos = _ensure_batch(left_tcp_pos)
        right_tcp_pos = _ensure_batch(right_tcp_pos)

        box_center_z = current_box_center[..., 2]
        threshold_z = box_center_z - self.TCP_HEIGHT_MARGIN
        violation_left = left_tcp_pos[..., 2] < threshold_z
        violation_right = right_tcp_pos[..., 2] < threshold_z
        violation = violation_left | violation_right
        penalty = violation.float() * self.TCP_HEIGHT_PENALTY

        info = {
            "tcp_height_penalty": penalty,
            "tcp_height_violation_left": violation_left,
            "tcp_height_violation_right": violation_right,
        }
        return penalty, info

    def _tcp_min_x_penalty(
        self,
        left_tcp_pos: torch.Tensor,
        right_tcp_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Penalize when either TCP x (in bimanual-center frame) is below threshold."""

        def _ensure_batch(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0) if t.ndim == 1 else t

        left_tcp_pos = _ensure_batch(left_tcp_pos)
        right_tcp_pos = _ensure_batch(right_tcp_pos)

        center_x = float(self.bimanual_center_pose.p[0])
        left_tcp_x_center = left_tcp_pos[..., 0] - center_x
        right_tcp_x_center = right_tcp_pos[..., 0] - center_x

        violation_left = left_tcp_x_center < float(self.TCP_MIN_X_THRESHOLD)
        violation_right = right_tcp_x_center < float(self.TCP_MIN_X_THRESHOLD)
        violation = violation_left | violation_right
        penalty = violation.float() * float(self.TCP_MIN_X_PENALTY)

        info = {
            "tcp_min_x_penalty": penalty,
            "tcp_min_x_violation_left": violation_left,
            "tcp_min_x_violation_right": violation_right,
            "tcp_x_center_left": left_tcp_x_center,
            "tcp_x_center_right": right_tcp_x_center,
        }
        return penalty, info

    def _box_yaw_overshoot_penalty(
        self, theta_deg: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Yaw overshoot penalty. Linear only within [start, end], zero outside."""
        theta = theta_deg.to(device=self.device, dtype=torch.float32)
        overshoot_start = float(self.BOX_OVERSHOOT_PENALTY_START_DEG)
        overshoot_end = float(self.BOX_OVERSHOOT_PENALTY_END_DEG)
        overshoot_max = float(self.BOX_OVERSHOOT_PENALTY_MAX)
        overshoot_span = max(overshoot_end - overshoot_start, 1e-6)

        overshoot_active_mask = (theta >= overshoot_start) & (theta <= overshoot_end)
        overshoot_saturated_mask = theta > overshoot_end
        overshoot_linear = ((theta - overshoot_start) / overshoot_span) * overshoot_max

        overshoot_value = torch.where(
            overshoot_active_mask,
            overshoot_linear,
            torch.zeros_like(theta),
        )
        penalty = overshoot_value
        info = {
            "overshoot_active_mask": overshoot_active_mask,
            "overshoot_saturated_mask": overshoot_saturated_mask,
            "overshoot_penalty_value": penalty,
        }
        return penalty, info



    def compute_normalized_dense_reward(self, obs, action, info):

        if not isinstance(self.agent, MultiAgent):
            return torch.zeros(self.num_envs, device=self.device)

        context = self._gather_reward_context(info)
        self._log_observation_to_info(info)


        pushpoint_score, push_info, pushpoint_stage_complete = self._pushpoint_tracking_reward(
            context.left_tcp_pos,
            context.right_tcp_pos,
            context.target_pushpoint_left,
            context.target_pushpoint_right,
        )
        translation_penalty_score, translation_info = self._box_translation_penalty(
            context.current_box_center
        )
        box_min_x_penalty_score, box_min_x_info = self._box_min_x_penalty(
            context.current_box_center
        )
        rotation_score, rotation_step_score, rotation_info = self._box_yaw_rotation(context.box_theta_deg)
        tcp_lead_score, tcp_lead_info = self._tcp_leading_reward(
            context.current_box_center,
            context.box_rotation_matrix,
            context.left_tcp_pos,
            context.right_tcp_pos,
            context.target_pushpoint_left,
            context.target_pushpoint_right,
        )
        tcp_height_penalty_score, tcp_height_info = self._tcp_height_penalty(
            context.current_box_center,
            context.left_tcp_pos,
            context.right_tcp_pos,
        )
        tcp_min_x_penalty_score, tcp_min_x_info = self._tcp_min_x_penalty(
            context.left_tcp_pos,
            context.right_tcp_pos,
        )
        rotation_overshoot_penalty_score, rotation_overshoot_info = self._box_yaw_overshoot_penalty(
            context.box_theta_deg
        )

        contact_penalty_score = self._compute_contact_penalty(info) #non-negative penalty

        reward_pushpoint = pushpoint_score #min = 0.0, max = 1.0
        tcp_lead_stage_complete = tcp_lead_info["tcp_lead_stage_complete"].bool()
        stage_1_mask = tcp_lead_stage_complete
        stage_2_mask = pushpoint_stage_complete.bool() & tcp_lead_stage_complete
        
        reward_rotation = torch.where(
            stage_2_mask, rotation_score, torch.zeros_like(rotation_score)
        )  # min = -3.0, max = 3.0 (only after pushpoint stage)

        positive_rotation = rotation_step_score > 0.0
        
        reward_closeness_to_reset_state = self.compute_closeness_score_to_reset_state() 
        # min = 0.0, max = 3.5

        penalty_inverse_rotation = torch.where(
            positive_rotation, torch.zeros_like(rotation_step_score), rotation_step_score
        ) #min = -2.0, max = 0.0
        penalty_forward_rotation_stage1 = torch.where(
            stage_2_mask, torch.zeros_like(rotation_step_score), -torch.clamp(rotation_step_score, min=0.0)
        ) #min = -2.0, max = 0.0 (only before pushpoints reached)
        reward_tcp_lead = tcp_lead_score #min = -TCP_LEAD_MAX, max = TCP_LEAD_MAX

        penalty_translation = -translation_penalty_score #min = -0.5, max = 0.0
        penalty_contact = -contact_penalty_score #min = -1.0, max = 0.0
        penalty_tcp_height = -tcp_height_penalty_score #min = -TCP_HEIGHT_PENALTY, max = 0.0
        penalty_tcp_min_x = -tcp_min_x_penalty_score #min = -TCP_MIN_X_PENALTY, max = 0.0
        penalty_box_min_x = -box_min_x_penalty_score #min = -1.0, max = 0.0
        penalty_rotation_overshoot = -rotation_overshoot_penalty_score

        

        #腕を振り回すというだけの報酬も用意してみました。
        #reward = (torch.linalg.norm(self.agent.agents[0].tcp.get_linear_velocity(), dim = -1) + torch.linalg.norm(self.agent.agents[1].tcp.get_linear_velocity(), dim = -1))/20.0 
        

        #Stage 0: move tcp to lead target pushpoint line

        #reward = reward_pushpoint + reward_tcp_lead #min = -TCP_LEAD_MAX, max = TCP_LEAD_MAX

        #reward = reward_tcp_lead #min = -TCP_LEAD_MAX, max = TCP_LEAD_MAX


        #Stage 1: after tcp lead is achieved, move tcp to pushpoint
        #stage_1_reward = self.TCP_LEAD_MAX + reward_pushpoint
        #reward[stage_1_mask] = stage_1_reward[stage_1_mask]

        reward = reward_tcp_lead + reward_pushpoint #min = -TCP_LEAD_MAX, max = TCP_LEAD_MAX


        #Stage 2: rotate box to pushpoint
        stage_2_reward = 1.0 + self.TCP_LEAD_MAX + reward_rotation + 1.5 +penalty_translation + penalty_box_min_x
        reward[stage_2_mask] = stage_2_reward[stage_2_mask] #approx range: (1 + TCP_LEAD_MAX - 2.0 + 1.5 - 1.5, 1 + TCP_LEAD_MAX + 2.0 + 1.5 +0.0)

        #Stage 3: goes bask to its initial pose.
        stage_3_reward = 4.75 + reward_closeness_to_reset_state # min = 4.75(at least bigger than stage2) max = 8.25
        stage_3_mask = (context.box_theta_deg >= float(self.BOX_GOAL_YAW_DEG)) & \
                       (context.box_theta_deg <= (float(self.BOX_GOAL_YAW_DEG) + 10.0))
        reward[stage_3_mask] = stage_3_reward[stage_3_mask]
        

        # Add a constant penalty that is independent of the stage.
        reward = reward + penalty_contact + penalty_inverse_rotation + penalty_forward_rotation_stage1 + penalty_tcp_height + penalty_tcp_min_x + penalty_rotation_overshoot
        


        info["reward_pushpoint"] = reward_pushpoint.detach().cpu()
        info["reward_rotation"] = reward_rotation.detach().cpu()
        info["penalty_translation"] = penalty_translation.detach().cpu()
        info["penalty_contact"] = penalty_contact.detach().cpu()
        info["penalty_inverse_rotation"] = penalty_inverse_rotation.detach().cpu()
        info["penalty_forward_rotation_stage1"] = penalty_forward_rotation_stage1.detach().cpu()
        info["penalty_tcp_height"] = penalty_tcp_height.detach().cpu()
        info["penalty_tcp_min_x"] = penalty_tcp_min_x.detach().cpu()
        info["penalty_box_min_x"] = penalty_box_min_x.detach().cpu()
        info["penalty_rotation_overshoot"] = penalty_rotation_overshoot.detach().cpu()
        info["overshoot_active_mask"] = rotation_overshoot_info[
            "overshoot_active_mask"
        ].detach().cpu()
        info["overshoot_saturated_mask"] = rotation_overshoot_info[
            "overshoot_saturated_mask"
        ].detach().cpu()
        info["box_min_x_penalty_norm"] = box_min_x_info[
            "box_min_x_penalty"
        ].detach().cpu()
        info["box_min_x_reference"] = box_min_x_info[
            "box_min_x_reference"
        ].detach().cpu()
        info["box_min_x_span"] = box_min_x_info[
            "box_min_x_span"
        ].detach().cpu()
        info["reward_tcp_lead"] = reward_tcp_lead.detach().cpu()
        info["tcp_lead_stage_complete"] = tcp_lead_stage_complete.detach().cpu()
        info["rewards_t"] = reward.detach().cpu()
        info["reward_closeness_to_reset_state"] = reward_closeness_to_reset_state.detach().cpu()
        info["success_once"] = stage_3_mask
        

        info["pushpoint_left_distance"] = push_info["distance_left"].detach().cpu()
        info["pushpoint_left_reached"] = push_info["reached_left"].detach().cpu()
        info["pushpoint_right_distance"] = push_info["distance_right"].detach().cpu()
        info["pushpoint_right_reached"] = push_info["reached_right"].detach().cpu()
        info["pushpoint_stage_complete"] = push_info[
            "pushpoint_stage_complete"
        ].detach().cpu()
        info["pushpoint_left_world"] = context.target_pushpoint_left.detach().cpu()
        info["pushpoint_right_world"] = context.target_pushpoint_right.detach().cpu()
        info["box_translation_distance"] = translation_info[
            "translation_distance"
        ].detach().cpu()
        info["box_translation_penalty"] = translation_info[
            "translation_penalty"
        ].detach().cpu()
        info["box_rotation_delta"] = rotation_info["yaw_delta_deg"].detach().cpu()
        info["box_rotation_score"] = rotation_score.detach().cpu()
        info["box_rotation_step_reward"] = rotation_info[
            "yaw_step_reward"
        ].detach().cpu()
        info["box_rotation_progress"] = rotation_info[
            "yaw_rotation_progress_deg"
        ].detach().cpu()
        info["box_rotation_target_delta"] = rotation_info[
            "yaw_rotation_target_delta_deg"
        ].detach().cpu()
        info["tcp_lead_reward"] = tcp_lead_info["tcp_lead_reward"].detach().cpu()
        info["tcp_lead_right_component"] = tcp_lead_info[
            "tcp_lead_right_component"
        ].detach().cpu()
        info["tcp_lead_left_component"] = tcp_lead_info[
            "tcp_lead_left_component"
        ].detach().cpu()
        info["tcp_height_violation_left"] = tcp_height_info[
            "tcp_height_violation_left"
        ].detach().cpu()
        info["tcp_height_violation_right"] = tcp_height_info[
            "tcp_height_violation_right"
        ].detach().cpu()
        info["tcp_min_x_violation_left"] = tcp_min_x_info[
            "tcp_min_x_violation_left"
        ].detach().cpu()
        info["tcp_min_x_violation_right"] = tcp_min_x_info[
            "tcp_min_x_violation_right"
        ].detach().cpu()
        info["tcp_x_center_left"] = tcp_min_x_info["tcp_x_center_left"].detach().cpu()
        info["tcp_x_center_right"] = tcp_min_x_info["tcp_x_center_right"].detach().cpu()
        # measured joint positions (left then right) flattened (drop mimic joints via obs indices)
        try:
            left_qpos = self.agent.agents[0].robot.get_qpos()
            right_qpos = self.agent.agents[1].robot.get_qpos()
            # use the same active joint subset as observations to drop mimic joints
            left_idx = self._agent_obs_joint_indices.get(f"{self.agent.agents[0].uid}-0")
            right_idx = self._agent_obs_joint_indices.get(f"{self.agent.agents[1].uid}-1")
            if left_idx is not None:
                left_qpos = self._index_select_joint_tensor(left_qpos, left_idx)
            if right_idx is not None:
                right_qpos = self._index_select_joint_tensor(right_qpos, right_idx)
            info["mesured_q"] = torch.cat([left_qpos, right_qpos], dim=-1).detach().cpu()
        except Exception:
            pass
        if reward.ndim == 0:
            reward = reward.unsqueeze(0)
        return reward.to(self.device)


@register_env("MyDualBoxRotationJustReturn-v0", max_episode_steps=120)
class MyDualBoxRotationJustReturnEnv(MyDualBoxRotationEnv):
    """Only train both arms to return to the fixed reset joint pose."""

    RETURN_SUCCESS_SCORE = 3.45

    def compute_normalized_dense_reward(self, obs, action, info):
        if not isinstance(self.agent, MultiAgent):
            return torch.zeros(self.num_envs, device=self.device)

        self._log_observation_to_info(info)

        return_qpos_score = self.compute_closeness_score_to_reset_state()
        reward = 4.75 + return_qpos_score
        success = return_qpos_score >= self.RETURN_SUCCESS_SCORE

        info["reward_closeness_to_reset_state"] = return_qpos_score.detach().cpu()
        info["return_qpos_score"] = return_qpos_score.detach().cpu()
        info["success_once"] = success.detach().cpu()
        info["rewards_t"] = reward.detach().cpu()

        try:
            left_qpos = self.agent.agents[0].robot.get_qpos()
            right_qpos = self.agent.agents[1].robot.get_qpos()
            left_idx = self._agent_obs_joint_indices.get(f"{self.agent.agents[0].uid}-0")
            right_idx = self._agent_obs_joint_indices.get(f"{self.agent.agents[1].uid}-1")
            if left_idx is not None:
                left_qpos = self._index_select_joint_tensor(left_qpos, left_idx)
            if right_idx is not None:
                right_qpos = self._index_select_joint_tensor(right_qpos, right_idx)
            info["mesured_q"] = torch.cat([left_qpos, right_qpos], dim=-1).detach().cpu()
        except Exception:
            pass

        if reward.ndim == 0:
            reward = reward.unsqueeze(0)
        return reward.to(self.device)


@register_env("MyDualBoxRotationAblated-v0", max_episode_steps=200)
class MyDualBoxRotationAblatedEnv(MyDualBoxRotationEnv):
    """Variant with reduced observation (no TCP or pushpoint features)."""

    _obs_extra_fn = staticmethod(get_obs_extra_ablation)
