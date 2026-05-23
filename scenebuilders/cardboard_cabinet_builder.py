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


@dataclass(frozen=True)
class CardboardInnerBoxSpec:
    outer_length_x: float
    outer_width_y: float
    outer_height_z: float
    wall_thickness: float = 0.006
    density: float = 250.0
    color_hex: str = "#B0916E"
    notch_width_x: float = 0.06
    notch_height_z: float = 0.018
    name: str = "cardboard_inner_box"


def make_cardboard_inner_box_spec(
    cabinet_spec: CardboardCabinetSpec = DEFAULT_CARDBOARD_CABINET_SPEC,
    *,
    length_delta: float = 0.007,
    width_delta: float = 0.015,
    height_delta: float = 0.015,
    notch_width_x: float = 0.06,
    notch_height_z: float = 0.018,
) -> CardboardInnerBoxSpec:
    return CardboardInnerBoxSpec(
        outer_length_x=cabinet_spec.outer_length_x - length_delta,
        outer_width_y=cabinet_spec.outer_width_y - width_delta,
        outer_height_z=cabinet_spec.outer_height_z - height_delta,
        wall_thickness=cabinet_spec.wall_thickness,
        density=cabinet_spec.density,
        color_hex=cabinet_spec.color_hex,
        notch_width_x=notch_width_x,
        notch_height_z=notch_height_z,
    )


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


def build_cardboard_inner_box_actor(
    scene,
    *,
    initial_pose: sapien.Pose,
    spec: CardboardInnerBoxSpec,
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

    for panel_pose, half_size in cardboard_inner_box_panel_specs(spec):
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


def cardboard_inner_box_panel_specs(spec: CardboardInnerBoxSpec):
    length = spec.outer_length_x
    width = spec.outer_width_y
    height = spec.outer_height_z
    thickness = spec.wall_thickness
    rear_thickness = thickness * 2
    notch_width = spec.notch_width_x
    notch_height = spec.notch_height_z
    side_width = max(0.0, (width - notch_width) / 2.0)
    center_panel_height = max(0.0, height - notch_height)

    return [
        (
            sapien.Pose(p=[0.0, 0.0, -height / 2 + thickness / 2]),
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
            [rear_thickness / 2, width / 2, height / 2],
        ),
        (
            sapien.Pose(
                p=[-length / 2 + thickness / 2, -(notch_width / 2 + side_width / 2), 0.0]
            ),
            [thickness / 2, side_width / 2, height / 2],
        ),
        (
            sapien.Pose(
                p=[-length / 2 + thickness / 2, notch_width / 2 + side_width / 2, 0.0]
            ),
            [thickness / 2, side_width / 2, height / 2],
        ),
        (
            sapien.Pose(
                p=[-length / 2 + thickness / 2, 0.0, -notch_height / 2]
            ),
            [thickness / 2, notch_width / 2, center_panel_height / 2],
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
