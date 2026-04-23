import math
from dataclasses import dataclass

import sapien
import sapien.physx
import sapien.render
import torch

from mani_skill.utils import sapien_utils


@dataclass(frozen=True)
class CardboardCabinetSpec:
    outer_length_x: float = 0.335
    outer_width_y: float = 0.2122
    outer_height_z: float = 0.1295
    wall_thickness: float = 0.006
    density: float = 250.0
    color_hex: str = "#B0916E"
    yaw_deg: float = -90.0
    name: str = "cardboard_cabinet"


DEFAULT_CARDBOARD_CABINET_SPEC = CardboardCabinetSpec()


def build_cardboard_cabinet_actor(
    scene,
    *,
    initial_pose: sapien.Pose,
    spec: CardboardCabinetSpec = DEFAULT_CARDBOARD_CABINET_SPEC,
):
    builder = scene.create_actor_builder()
    material = sapien.physx.PhysxMaterial(
        static_friction=0.9,
        dynamic_friction=0.75,
        restitution=0.05,
    )
    render_material = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba(spec.color_hex),
        roughness=0.6,
    )

    for panel_pose, half_size in cardboard_cabinet_panel_specs(spec):
        builder.add_box_collision(
            pose=panel_pose,
            half_size=half_size,
            density=spec.density,
            material=material,
        )
        builder.add_box_visual(
            pose=panel_pose,
            half_size=half_size,
            material=render_material,
        )

    builder.initial_pose = initial_pose
    return builder.build(name=spec.name)


def cardboard_cabinet_panel_specs(spec: CardboardCabinetSpec):
    length = spec.outer_length_x
    width = spec.outer_width_y
    height = spec.outer_height_z
    thickness = spec.wall_thickness
    return [
        (
            sapien.Pose(p=[0.0, 0.0, -height / 2 + thickness / 2]),
            [length / 2, width / 2, thickness / 2],
        ),
        (
            sapien.Pose(p=[0.0, 0.0, height / 2 - thickness / 2]),
            [length / 2, width / 2, thickness / 2],
        ),
        (
            sapien.Pose(p=[0.0, width / 2 - thickness / 2, 0.0]),
            [length / 2, thickness / 2, height / 2],
        ),
        (
            sapien.Pose(p=[0.0, -width / 2 + thickness / 2, 0.0]),
            [length / 2, thickness / 2, height / 2],
        ),
        (
            sapien.Pose(p=[length / 2 - thickness / 2, 0.0, 0.0]),
            [thickness / 2, width / 2, height / 2],
        ),
    ]


def cardboard_cabinet_grasp_targets_local(
    spec: CardboardCabinetSpec = DEFAULT_CARDBOARD_CABINET_SPEC,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    return torch.tensor(
        [
            [
                -spec.outer_length_x / 2 + spec.wall_thickness,
                spec.outer_width_y / 2 - spec.wall_thickness / 2,
                0.0,
            ],
            [
                -spec.outer_length_x / 2 + spec.wall_thickness,
                -spec.outer_width_y / 2 + spec.wall_thickness / 2,
                0.0,
            ],
        ],
        dtype=dtype,
        device=device,
    )


def cardboard_cabinet_quaternion(
    spec: CardboardCabinetSpec = DEFAULT_CARDBOARD_CABINET_SPEC,
    *,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    yaw_rad = math.radians(spec.yaw_deg)
    half_yaw = 0.5 * yaw_rad
    return torch.tensor(
        [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
        dtype=dtype,
        device=device,
    )
