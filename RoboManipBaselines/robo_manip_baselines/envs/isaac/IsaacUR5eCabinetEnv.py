from os import path

import numpy as np
from isaacgym import (
    gymapi,  # noqa: F401
    gymtorch,  # noqa: F401
    gymutil,  # noqa: F401
)

from .IsaacUR5eEnvBase import IsaacUR5eEnvBase


class IsaacUR5eCabinetEnv(IsaacUR5eEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        IsaacUR5eEnvBase.__init__(
            self,
            init_qpos=np.array(
                [
                    np.pi,
                    -np.pi / 2,
                    -0.75 * np.pi,
                    -0.75 * np.pi,
                    -0.5 * np.pi,
                    np.pi / 2,
                    0.0,
                ]
            ),
            **kwargs,
        )

        self.original_cabinet_pos = self.get_link_pose("cabinet", "base_link")[0:3]
        self.cabinet_pos_offsets = np.array(
            [
                [0.0, -0.15, 0.0],
                [0.0, -0.09, 0.0],
                [0.0, -0.03, 0.0],
                [0.0, 0.03, 0.0],
                [0.0, 0.09, 0.0],
                [0.0, 0.15, 0.0],
            ]
        )  # [m]

    def get_sim_params(self):
        sim_params = super().get_sim_params()
        sim_params.physx.rest_offset = 0.0001
        sim_params.physx.contact_offset = 0.002
        return sim_params

    def setup_task_specific_variables(self):
        self.cabinet_handle_list = []

    def setup_task_specific_assets(self):
        # Setup cabinet asset
        cabinet_asset_root = path.join(
            path.dirname(__file__), "../assets/isaac/objects/cabinet"
        )
        cabinet_asset_file = "cabinet.urdf"
        cabinet_asset_options = gymapi.AssetOptions()
        cabinet_asset_options.armature = 0.01
        cabinet_asset_options.density = 10.0
        cabinet_asset_options.override_com = True
        cabinet_asset_options.override_inertia = True
        cabinet_asset_options.fix_base_link = True
        cabinet_asset_options.flip_visual_attachments = False
        self.cabinet_asset = self.gym.load_asset(
            self.sim, cabinet_asset_root, cabinet_asset_file, cabinet_asset_options
        )

    def setup_task_specific_actors(self, env_idx):
        env = self.env_list[env_idx]

        # Setup cabinet actor
        cabinet_pose = gymapi.Transform(
            p=gymapi.Vec3(0.67, 0.0, 0.0),
            r=gymapi.Quat.from_euler_zyx(0, 0, -np.pi / 2),
        )
        cabinet_handle = self.gym.create_actor(
            env, self.cabinet_asset, cabinet_pose, "cabinet", env_idx, 0
        )
        self.cabinet_handle_list.append(cabinet_handle)

        # Setup cabinet joint control mode
        cabinet_dof_props = self.gym.get_asset_dof_properties(self.cabinet_asset)
        cabinet_dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        cabinet_dof_props["stiffness"].fill(0.025)
        cabinet_dof_props["damping"].fill(0.1)
        self.gym.set_actor_dof_properties(env, cabinet_handle, cabinet_dof_props)

        # Setup cabinet joint control command
        if env_idx == 0:
            cabinet_num_dofs = self.gym.get_asset_dof_count(self.cabinet_asset)
            self.init_cabinet_dof_state = np.zeros(
                cabinet_num_dofs, gymapi.DofState.dtype
            )
        self.gym.set_actor_dof_states(
            env, cabinet_handle, self.init_cabinet_dof_state, gymapi.STATE_ALL
        )
        self.gym.set_actor_dof_position_targets(
            env, cabinet_handle, self.init_cabinet_dof_state["pos"]
        )

    def setup_task_specific_cameras(self, env_idx):
        env = self.env_list[env_idx]
        camera_handles = self.camera_handles_list[env_idx]
        camera_properties = self.camera_properties_list[env_idx]

        single_camera_properties = gymapi.CameraProperties()
        single_camera_properties.width = 640
        single_camera_properties.height = 480

        camera_handle = self.gym.create_camera_sensor(env, single_camera_properties)
        camera_origin_pos = gymapi.Vec3(-0.1, 0.5, 0.4)
        camera_lookat_pos = gymapi.Vec3(0.3, 0.0, 0.3)
        self.gym.set_camera_location(
            camera_handle, env, camera_origin_pos, camera_lookat_pos
        )
        camera_handles["front"] = camera_handle
        camera_properties["front"] = single_camera_properties

        camera_handle = self.gym.create_camera_sensor(env, single_camera_properties)
        camera_origin_pos = gymapi.Vec3(0.4, -0.8, 0.3)
        camera_lookat_pos = gymapi.Vec3(0.4, 0.8, 0.3)
        self.gym.set_camera_location(
            camera_handle, env, camera_origin_pos, camera_lookat_pos
        )
        camera_handles["side"] = camera_handle
        camera_properties["side"] = single_camera_properties

    def reset_task_specific_actors(self, env_idx):
        env = self.env_list[env_idx]
        cabinet_handle = self.cabinet_handle_list[env_idx]
        self.gym.set_actor_dof_states(
            env, cabinet_handle, self.init_cabinet_dof_state, gymapi.STATE_ALL
        )
        self.gym.set_actor_dof_position_targets(
            env, cabinet_handle, self.init_cabinet_dof_state["pos"]
        )

    def _get_success_list(self):
        success_list = []
        for env, cabinet_handle in zip(self.env_list, self.cabinet_handle_list):
            door_angle = self.gym.get_actor_dof_states(
                env, cabinet_handle, gymapi.STATE_POS
            )["pos"][0]
            success = door_angle < np.deg2rad(-75)
            success_list.append(success)
        return success_list

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.cabinet_pos_offsets)

        cabinet_pos_offset = self.cabinet_pos_offsets[world_idx]
        cabinet_num_bodies = self.gym.get_asset_rigid_body_count(self.cabinet_asset)

        for env, cabinet_handle in zip(self.env_list, self.cabinet_handle_list):
            for body_idx in range(cabinet_num_bodies):
                link_idx = self.gym.get_actor_rigid_body_index(
                    env, cabinet_handle, body_idx, gymapi.DOMAIN_SIM
                )
                new_link_pos = self.init_state[link_idx]["pose"]["p"]
                original_link_pos = self.original_init_state[link_idx]["pose"]["p"]
                new_link_pos["x"] = original_link_pos["x"] + cabinet_pos_offset[0]
                new_link_pos["y"] = original_link_pos["y"] + cabinet_pos_offset[1]
                new_link_pos["z"] = original_link_pos["z"] + cabinet_pos_offset[2]

        return world_idx
