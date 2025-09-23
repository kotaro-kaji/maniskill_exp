from os import path

import numpy as np
from isaacgym import (
    gymapi,  # noqa: F401
    gymtorch,  # noqa: F401
    gymutil,  # noqa: F401
)

from .IsaacUR5eEnvBase import IsaacUR5eEnvBase


class IsaacUR5eChainEnv(IsaacUR5eEnvBase):
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
                    -0.7 * np.pi,
                    -0.3 * np.pi,
                    np.pi / 2,
                    np.pi / 2,
                    0.0,
                ]
            ),
            **kwargs,
        )

        self.original_fook_pos = self.get_link_pose("fook", "box")[0:3]
        self.fook_pos_offsets = np.array(
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
        sim_params.substeps = 4
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.contact_offset = 0.0005
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        return sim_params

    def setup_task_specific_variables(self):
        self.chain_handles_list = []
        self.fook_handle_list = []

    def setup_task_specific_assets(self):
        # Setup chain asset
        chain_asset_root = path.join(
            path.dirname(__file__), "../assets/isaac/objects/chain"
        )
        chain_asset_options = gymapi.AssetOptions()
        chain_asset_options.override_com = True
        chain_asset_options.override_inertia = True
        chain_asset_options.fix_base_link = False
        chain_asset_options.flip_visual_attachments = False
        chain_asset_options.vhacd_enabled = True
        chain_ring_asset_file = "chain_ring.urdf"
        self.chain_ring_asset = self.gym.load_asset(
            self.sim, chain_asset_root, chain_ring_asset_file, chain_asset_options
        )
        chain_start_asset_file = "chain_start.urdf"
        self.chain_start_asset = self.gym.load_asset(
            self.sim, chain_asset_root, chain_start_asset_file, chain_asset_options
        )
        chain_end_asset_file = "chain_end.urdf"
        self.chain_end_asset = self.gym.load_asset(
            self.sim, chain_asset_root, chain_end_asset_file, chain_asset_options
        )

        # Setup fook asset
        fook_asset_options = gymapi.AssetOptions()
        fook_asset_options.density = 10.0
        fook_asset_options.fix_base_link = True
        fook_asset_options.flip_visual_attachments = False
        self.fook_asset = self.gym.create_box(
            self.sim, 0.15, 0.04, 0.02, fook_asset_options
        )

    def setup_task_specific_actors(self, env_idx):
        env = self.env_list[env_idx]

        # Setup chain actor
        chain_handles = []
        num_ring = 9
        for ring_idx in range(num_ring):
            ring_pos = gymapi.Vec3(0.4, 0.04 * (ring_idx - 0.5 * num_ring), 0.1)
            if ring_idx % 2 == 0:
                ring_rot = gymapi.Quat.from_euler_zyx(0, 0, np.pi / 2)
                ring_color = gymapi.Vec3(0.1, 0.8, 0.1)
            else:
                ring_rot = gymapi.Quat.from_euler_zyx(np.pi / 2, 0, np.pi / 2)
                ring_color = gymapi.Vec3(0.8, 0.8, 0.2)
            ring_pose = gymapi.Transform(p=ring_pos, r=ring_rot)
            if ring_idx == 0:
                chain_ring_asset = self.chain_start_asset
                chain_name = "chain_start"
            elif ring_idx == num_ring - 1:
                chain_ring_asset = self.chain_end_asset
                chain_name = "chain_end"
            else:
                chain_ring_asset = self.chain_ring_asset
                chain_name = f"chain_ring{ring_idx}"
            chain_handle = self.gym.create_actor(
                env, chain_ring_asset, ring_pose, chain_name, env_idx, 0
            )
            self.gym.set_rigid_body_color(
                env, chain_handle, 0, gymapi.MESH_VISUAL, ring_color
            )
            if ring_idx in (0, num_ring - 1):
                box_color = gymapi.Vec3(0.1, 0.5, 0.8)
                self.gym.set_rigid_body_color(
                    env, chain_handle, 1, gymapi.MESH_VISUAL, box_color
                )
            chain_handles.append(chain_handle)
        self.chain_handles_list.append(chain_handles)

        # Setup fook actor
        fook_pose = gymapi.Transform(p=gymapi.Vec3(0.55, 0.0, 0.3))
        fook_handle = self.gym.create_actor(
            env, self.fook_asset, fook_pose, "fook", env_idx, 0
        )
        self.gym.set_rigid_body_color(
            env, fook_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.1, 0.5)
        )
        self.fook_handle_list.append(fook_handle)

    def setup_task_specific_cameras(self, env_idx):
        env = self.env_list[env_idx]
        camera_handles = self.camera_handles_list[env_idx]
        camera_properties = self.camera_properties_list[env_idx]

        single_camera_properties = gymapi.CameraProperties()
        single_camera_properties.width = 640
        single_camera_properties.height = 480

        camera_handle = self.gym.create_camera_sensor(env, single_camera_properties)
        camera_origin_pos = gymapi.Vec3(0.9, 0.0, 0.45)
        camera_lookat_pos = gymapi.Vec3(-1.0, 0.0, -0.4)
        self.gym.set_camera_location(
            camera_handle, env, camera_origin_pos, camera_lookat_pos
        )
        camera_handles["front"] = camera_handle
        camera_properties["front"] = single_camera_properties

        camera_handle = self.gym.create_camera_sensor(env, single_camera_properties)
        camera_origin_pos = gymapi.Vec3(0.3, -0.8, 0.3)
        camera_lookat_pos = gymapi.Vec3(0.3, 0.8, 0.3)
        self.gym.set_camera_location(
            camera_handle, env, camera_origin_pos, camera_lookat_pos
        )
        camera_handles["side"] = camera_handle
        camera_properties["side"] = single_camera_properties

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.fook_pos_offsets)

        modified_fook_pos = self.original_fook_pos + self.fook_pos_offsets[world_idx]

        for env, fook_handle in zip(self.env_list, self.fook_handle_list):
            link_idx = self.gym.find_actor_rigid_body_index(
                env, fook_handle, "box", gymapi.DOMAIN_SIM
            )
            self.init_state[link_idx]["pose"]["p"]["x"] = modified_fook_pos[0]
            self.init_state[link_idx]["pose"]["p"]["y"] = modified_fook_pos[1]
            self.init_state[link_idx]["pose"]["p"]["z"] = modified_fook_pos[2]

        return world_idx
