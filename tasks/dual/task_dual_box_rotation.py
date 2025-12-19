import math
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
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.structs import Actor, Link
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.common import flatten_dict_keys, flatten_state_dict, to_tensor
from robotagents.xarm_ball_ee import Xarm7BallEE


from scenebuilders.dual_xarm7_table_scene_builder import (
    DualXarm7TableSceneBuilder,
    LEFT_ARM_Y_OFFSET,
    RIGHT_ARM_Y_OFFSET,
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
    SUPPORTED_ROBOTS = [("xarm7_ball_ee", "xarm7_ball_ee")]
    agent: MultiAgent[Tuple[Xarm7BallEE, Xarm7BallEE]]
    _obs_extra_fn = staticmethod(get_obs_extra_full)

    CONTACT_FORCE_THRESHOLD = 0.000001  # N
    CONTACT_PENALTY_WEIGHT = 1.0
    
    BOX_HALF_SIZE = np.array([0.2170*0.5, 0.2845*0.5, 0.1140*0.5])
    BOX_DENSITY = 200.0
    BOX_X_OFFSET_FROM_BASE = 0.337
    BOX_CENTER_MIN_X = 0.25
    BOX_Y_JITTER = 0.02
    BOX_X_JITTER = 0.015
    BOX_ROTATION_JITTER_DEG = 180
    BOX_ROTATION_JITTER_RAD = math.radians(BOX_ROTATION_JITTER_DEG)
    DISTANCE_SCALE = 4.0
    PUSHPOINT_DISTANCE_SCALE = 5.0
    PUSHPOINT_DISTANCE_THRESHOLD = 0.050
    BOX_CENTER_MAX_OFFSET = 0.15
    BOX_CENTER_PENALTY_WEIGHT = 0.5
    MAX_ROTATION_PACE = 540.0 / 10.0  # degrees per second for full reward
    BOX_GOAL_YAW_DEG = 145.0  # raw yawでは -145 度、内部では正値で扱う
    TCP_LEAD_SATURATION = 0.010
    TCP_LEAD_MAX = 0.25
    BOX_TRANSLATION_PENALTY_SCALE = 0.25
    TCP_HEIGHT_MARGIN = 0.005
    TCP_HEIGHT_PENALTY = 0.5

    def __init__(self, *args, robot_uids=("xarm7_ball_ee", "xarm7_ball_ee"), robot_init_noise_scale=1.0,**kwargs):
        self.robot_init_noise_scale = robot_init_noise_scale
        self._flat_obs_column_names: Optional[List[str]] = None
        self._agent_obs_joint_indices: Dict[str, torch.Tensor] = dict()
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: Dict[str, Any]):
        super()._load_agent(options, [sapien.Pose(p=[0,1,0]), sapien.Pose(p=[0,-1,0])])
        self._configure_observed_joint_indices()
        #super()._load_agent(options, sapien.Pose[p=])

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.2, 1.0, 1.0], [0.0, 0.0, 0.4])
        return CameraConfig(
            "render_camera", pose=pose, width=800, height=800, fov=1.0, near=0.01, far=5.0
        )

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

        builder.add_box_visual(half_size=self.BOX_HALF_SIZE ,material=box_material)
            
        
        
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
            pushpoint_radius = 0.02
            self.pushpoint_left_site = actors.build_sphere(
                self.scene,
                radius=pushpoint_radius,
                color=(1.0, 0.0, 0.0, 1.0),
                name="pushpoint_left_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(
                    p=self._bimanual_center_point_to_world(
                        [0.0, 0.0, self.BOX_HALF_SIZE[2]]
                    )
                ),
            )
            self.pushpoint_right_site = actors.build_sphere(
                self.scene,
                radius=pushpoint_radius,
                color=(0.3, 0.3, 0.9, 0.8),
                name="pushpoint_right_site",
                body_type="kinematic",
                add_collision=False,
                initial_pose=sapien.Pose(
                    p=self._bimanual_center_point_to_world(
                        [0.0, 0.0, self.BOX_HALF_SIZE[2]]
                    )
                ),
            )
        else:
            self.pushpoint_left_site = None
            self.pushpoint_right_site = None


        #Visualize bimanual center as a small orange sphere for debugging
        self.bimanual_center_site = actors.build_sphere(
            self.scene,
            radius=0.02,
            color=(1.0, 0.0, 0.0, 0.8),
            name="bimanual_center_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(
                p=[
                    self.bimanual_center_pose.p[0],
                    self.bimanual_center_pose.p[1],
                    self.bimanual_center_pose.p[2] 
                ]
            ),
        )
        
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
        needs_init = not hasattr(self, "initial_forward_side_center")
        if not needs_init:
            needs_init = self.initial_forward_side_center.shape[0] != num_envs
        if not needs_init:
            return
        zeros = torch.zeros((num_envs, 3), device=self.device, dtype=torch.float32)
        self.initial_forward_side_center = zeros.clone()
        self.initial_pushpoint_by_right = zeros.clone()
        self.initial_pushpoint_by_left = zeros.clone()
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
        self.initial_forward_side_center[env_idx_long] = initial_forward_side_center
        self.initial_pushpoint_by_right[env_idx_long] = initial_pushpoint_by_right
        self.initial_pushpoint_by_left[env_idx_long] = initial_pushpoint_by_left
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
                    Pose.create_from_pq(p=self.initial_pushpoint_by_left)
                )
            if self.pushpoint_right_site is not None:
                self.pushpoint_right_site.set_pose(
                    Pose.create_from_pq(p=self.initial_pushpoint_by_right)
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
            x_center = (
                self.BOX_X_OFFSET_FROM_BASE
                + (torch.rand(batch_size, device=self.device) - 0.5)
                * 2
                * self.BOX_X_JITTER
            )
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
            theta = torch.zeros(
                batch_size, device=self.device, dtype=torch.float32
            )
            if self.BOX_ROTATION_JITTER_DEG > 0.0:
                # jitter is specified in degrees; keep theta in degrees before quaternion conversion
                theta = theta + (
                    (torch.rand(batch_size, device=self.device) - 0.5)
                    * 2.0
                    * float(self.BOX_ROTATION_JITTER_DEG)
                )
            orientations = self._theta_to_quaternion(theta)
            #self.box_objects[env_idx].set_pose(Pose.create_from_pq(positions, orientations))
            self.box.set_pose(Pose.create_from_pq(positions, orientations))
            self._update_initial_pushpoints(env_idx, positions, orientations)
            self._reset_rotation_buffers(env_idx, theta)

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

        penalty_forces = []
        penalty_names = []

        ball_box_forces = []
        ball_box_names = []

        # Prepare environment actors
        table_actor = getattr(self.table_scene, "table", None)
        pedestal_actor = getattr(self.table_scene, "robot_pedestal", None)
        if table_actor is None:
            raise ValueError("Expected environment actor 'table' to exist, got None")
        if self.box is None:
            raise ValueError("Expected environment actor 'box' to exist, got None")
        if pedestal_actor is None:
            raise ValueError("Expected environment actor 'robot_pedestal' to exist, got None")

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

            # Non-penalty logging: ball vs box
            f_ball_box = self.scene.get_pairwise_contact_forces(ball_link, self.box).to(device)
            ball_box_forces.append(torch.linalg.norm(f_ball_box, dim=-1))
            ball_box_names.append(f"{prefix}_link_tcp_ball|box")



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
            self.agent.agents[0].tcp.pose, device=self.device
        ).p
        left_tcp_pos = (
            left_tcp_pos.squeeze(0)
            if left_tcp_pos.ndim == 2 and left_tcp_pos.shape[0] == 1
            else left_tcp_pos
        )
        right_tcp_pos = Pose.create(
            self.agent.agents[1].tcp.pose, device=self.device
        ).p
        right_tcp_pos = (
            right_tcp_pos.squeeze(0)
            if right_tcp_pos.ndim == 2 and right_tcp_pos.shape[0] == 1
            else right_tcp_pos
        )

        info["box_theta_deg"] = theta.detach().cpu()
        info["pushpoint_inversion_conditions"] = inversion_conditions.detach().cpu()
        info["initial_forward_side_center"] = (
            self.initial_forward_side_center.detach().cpu()
        )
        info["initial_pushpoint_by_right"] = (
            self.initial_pushpoint_by_right.detach().cpu()
        )
        info["initial_pushpoint_by_left"] = (
            self.initial_pushpoint_by_left.detach().cpu()
        )
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
        normal_conditions = (theta >= 0.0) & (theta < 179.0) | (theta >= 359.0)
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
        return self._obs_extra_fn(self, info)

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
        translation_penalty = (
            self.BOX_CENTER_PENALTY_WEIGHT
            * normalized_penalty
            * self.BOX_TRANSLATION_PENALTY_SCALE
        )
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
        delta_reward = 2.0 * step_reward  # range [-2, 2]

        goal_theta = torch.tensor(
            self.BOX_GOAL_YAW_DEG, device=self.device, dtype=torch.float32
        )
        angle_diff_rad = torch.deg2rad(goal_theta - theta_deg)
        alignment_reward = 2.0 * (1.0 + torch.cos(angle_diff_rad))
        info = {
            "yaw_delta_deg": signed_delta,
            "yaw_rotation_reward": alignment_reward,
            "yaw_step_reward": delta_reward,
            "yaw_rotation_progress_deg": self._positive_yaw_progress,
            "yaw_rotation_target_delta_deg": signed_delta.new_full(
                signed_delta.shape, target_delta_value
            ),
        }
        return delta_reward, info

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

        def _lead(push_local: torch.Tensor, tcp_local: torch.Tensor) -> torch.Tensor:
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
            return normalized * self.TCP_LEAD_MAX

        lead_right = _lead(push_right_local, right_local)
        lead_left = _lead(push_left_local, left_local)
        reward = 0.5 * (lead_right + lead_left)
        info = {
            "tcp_lead_reward": reward,
            "tcp_lead_right_component": lead_right,
            "tcp_lead_left_component": lead_left,
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
        rotation_step_score, rotation_info = self._box_yaw_rotation(context.box_theta_deg)
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

        contact_penalty_score = self._compute_contact_penalty(info) #non-negative penalty

        reward_pushpoint = pushpoint_score #min = 0.0, max = 1.0

        mask = pushpoint_stage_complete.bool()
        rotation_alignment_reward = rotation_info["yaw_rotation_reward"]
        reward_rotation = torch.where(
            mask, rotation_alignment_reward, torch.zeros_like(rotation_alignment_reward)
        )  # min = -3.0, max = 3.0 (only after pushpoint stage)

        positive_rotation = rotation_step_score > 0.0
        penalty_inverse_rotation = torch.where(
            positive_rotation, torch.zeros_like(rotation_step_score), rotation_step_score
        ) #min = -2.0, max = 0.0
        penalty_forward_rotation_stage1 = torch.where(
            mask, torch.zeros_like(rotation_step_score), -torch.clamp(rotation_step_score, min=0.0)
        ) #min = -2.0, max = 0.0 (only before pushpoints reached)
        reward_tcp_lead = tcp_lead_score #min = -TCP_LEAD_MAX, max = TCP_LEAD_MAX

        penalty_translation = -translation_penalty_score #min = -0.1 (scale=0.25), max = 0.0
        penalty_contact = -contact_penalty_score #min = -1.0, max = 0.0
        penalty_tcp_height = -tcp_height_penalty_score #min = -TCP_HEIGHT_PENALTY, max = 0.0
        penalty_box_min_x = -box_min_x_penalty_score #min = -1.0, max = 0.0
        

        #腕を振り回すというだけの報酬も用意してみました。
        #reward = (torch.linalg.norm(self.agent.agents[0].tcp.get_linear_velocity(), dim = -1) + torch.linalg.norm(self.agent.agents[1].tcp.get_linear_velocity(), dim = -1))/20.0 
        

        #Stage 1: move tcp to pushpoint
        reward = reward_pushpoint + reward_tcp_lead #min = -1.0, max = 1 + TCP_LEAD_MAX

        #Stage 2: rotate box to pushpoint
        stage_2_reward = reward_pushpoint + self.TCP_LEAD_MAX +1e-2 + reward_rotation
        reward[mask] = stage_2_reward[mask] #approx range: (1 + TCP_LEAD_MAX - 3.0, 1 + TCP_LEAD_MAX + 3.0)

        #reward = reward_rotation + reward_pushpoint
        # Add a constant penalty that is independent of the stage.
        reward = reward + penalty_contact + penalty_translation + penalty_inverse_rotation + penalty_forward_rotation_stage1 + penalty_tcp_height + penalty_box_min_x 
        


        info["reward_pushpoint"] = reward_pushpoint.detach().cpu()
        info["reward_rotation"] = reward_rotation.detach().cpu()
        info["penalty_translation"] = penalty_translation.detach().cpu()
        info["penalty_contact"] = penalty_contact.detach().cpu()
        info["penalty_inverse_rotation"] = penalty_inverse_rotation.detach().cpu()
        info["penalty_forward_rotation_stage1"] = penalty_forward_rotation_stage1.detach().cpu()
        info["penalty_tcp_height"] = penalty_tcp_height.detach().cpu()
        info["penalty_box_min_x"] = penalty_box_min_x.detach().cpu()
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
        info["rewards_t"] = reward.detach().cpu()


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
        info["box_rotation_reward"] = rotation_info[
            "yaw_rotation_reward"
        ].detach().cpu()
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


@register_env("MyDualBoxRotationAblated-v0", max_episode_steps=200)
class MyDualBoxRotationAblatedEnv(MyDualBoxRotationEnv):
    """Variant with reduced observation (no TCP or pushpoint features)."""

    _obs_extra_fn = staticmethod(get_obs_extra_ablation)
