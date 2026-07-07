import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sapien
import torch

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.utils import get_active_joint_indices
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from robotagents.xarm_ball_ee_wo_force_sensor import Xarm7BallEEWoForceSensor
from scenebuilders.dual_xarm7_table_scene_builder import (
    DualXarm7TableSceneBuilder,
    LEFT_ARM_Y_OFFSET,
    RIGHT_ARM_Y_OFFSET,
)
from scenebuilders.trash_bin_mesh import ensure_trash_bin_frustum_mesh
from scenebuilders.xarm7_table_scene_builder import (
    PEDESTAL_HEIGHT,
    ROBOT_BASE_X_OFFSET,
)


@register_env("MyDualTrashBinRolling-v0", max_episode_steps=120)
class MyDualTrashBinRollingEnv(BaseEnv):
    SUPPORTED_ROBOTS = [
        ("xarm7_ball_ee_wo_force_sensor", "xarm7_ball_ee_wo_force_sensor"),
    ]
    agent: MultiAgent[Tuple[Xarm7BallEEWoForceSensor, Xarm7BallEEWoForceSensor]]

    BIN_HEIGHT = 0.2727
    BIN_BOTTOM_RADIUS = 0.184 / 2.0
    BIN_TOP_RADIUS = 0.218 / 2.0
    BIN_RIM_RADIUS = 0.224 / 2.0
    BIN_RIM_THICKNESS = 0.003
    BIN_DENSITY = 120.0
    BIN_X_OFFSET_FROM_BASE = 0.34

    def __init__(
        self,
        *args,
        robot_uids=("xarm7_ball_ee_wo_force_sensor", "xarm7_ball_ee_wo_force_sensor"),
        robot_init_noise_scale: float = 1.0,
        **kwargs,
    ):
        self.robot_init_noise_scale = robot_init_noise_scale
        self._agent_obs_joint_indices: Dict[str, torch.Tensor] = dict()
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: Dict[str, Any]):
        super()._load_agent(options, [sapien.Pose(p=[0, 1, 0]), sapien.Pose(p=[0, -1, 0])])
        self._configure_observed_joint_indices()

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.1, 1.0, 0.8], [0.0, 0.0, 0.18])
        return CameraConfig(
            "render_camera", pose=pose, width=800, height=800, fov=1.0, near=0.01, far=5.0
        )

    def _load_scene(self, options: Dict[str, Any]):
        self.table_scene = DualXarm7TableSceneBuilder(env=self)
        self.table_scene.build()
        self.bimanual_center_pose = self._compute_bimanual_center_pose()

        mesh_path = ensure_trash_bin_frustum_mesh(
            height=self.BIN_HEIGHT,
            bottom_radius=self.BIN_BOTTOM_RADIUS,
            top_radius=self.BIN_TOP_RADIUS,
        )
        material = sapien.render.RenderMaterial(
            base_color=[0.85, 0.85, 0.88, 1.0],
            roughness=0.45,
            metallic=0.0,
        )
        rim_material = sapien.render.RenderMaterial(
            base_color=[0.70, 0.70, 0.74, 1.0],
            roughness=0.45,
            metallic=0.0,
        )
        physical_material = sapien.physx.PhysxMaterial(
            static_friction=0.9,
            dynamic_friction=0.75,
            restitution=0.05,
        )

        builder = self.scene.create_actor_builder()
        builder.add_convex_collision_from_file(
            filename=str(mesh_path),
            material=physical_material,
            density=self.BIN_DENSITY,
        )
        builder.add_visual_from_file(filename=str(mesh_path), material=material)
        rim_pose = sapien.Pose(
            p=[0.0, 0.0, -self.BIN_HEIGHT / 2.0 - self.BIN_RIM_THICKNESS / 2.0],
            q=[math.sqrt(0.5), 0.0, -math.sqrt(0.5), 0.0],
        )
        builder.add_cylinder_collision(
            pose=rim_pose,
            radius=self.BIN_RIM_RADIUS,
            half_length=self.BIN_RIM_THICKNESS / 2.0,
            material=physical_material,
            density=self.BIN_DENSITY,
        )
        builder.add_cylinder_visual(
            pose=rim_pose,
            radius=self.BIN_RIM_RADIUS,
            half_length=self.BIN_RIM_THICKNESS / 2.0,
            material=rim_material,
        )
        builder.initial_pose = self._initial_bin_pose()
        self.trash_bin = builder.build(name="trash_bin")

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

            pose = self._initial_bin_pose()
            positions = torch.as_tensor(
                np.array([pose.p], dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)
            orientations = torch.as_tensor(
                np.array([pose.q], dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)
            self.trash_bin.set_pose(Pose.create_from_pq(positions, orientations))

    def _initial_bin_pose(self) -> sapien.Pose:
        center_z = self.BIN_TOP_RADIUS
        p = self._bimanual_center_point_to_world(
            [self.BIN_X_OFFSET_FROM_BASE, 0.0, center_z]
        )
        return sapien.Pose(p=p, q=[math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0])

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

    def _clear(self):
        super()._clear()
        self._agent_obs_joint_indices = dict()

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
            drive_joint = gripper_joint_names[0]
            if drive_joint not in observed_joint_names:
                observed_joint_names.append(drive_joint)
        if not observed_joint_names:
            return None
        return get_active_joint_indices(sub_agent.robot, observed_joint_names).long()

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
            else:
                filtered[key] = self._filter_agent_obs_entry(value, indices)
        return filtered

    def _filter_agent_obs_entry(self, obs_entry: Any, indices: torch.Tensor) -> Any:
        if not isinstance(obs_entry, dict):
            return obs_entry
        filtered_entry = dict(obs_entry)
        if "qpos" in obs_entry and obs_entry["qpos"] is not None:
            idx = indices.to(device=obs_entry["qpos"].device, dtype=torch.long)
            filtered_entry["qpos"] = obs_entry["qpos"].index_select(obs_entry["qpos"].dim() - 1, idx)
        filtered_entry.pop("qvel", None)
        return filtered_entry

    def evaluate(self):
        return {"success": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)}

    def compute_dense_reward(self, obs, action, info):
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info)
