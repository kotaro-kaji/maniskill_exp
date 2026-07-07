import math
from pathlib import Path


TRASH_BIN_MESH_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "trash_bin"
    / "trash_bin_frustum.obj"
)


def ensure_trash_bin_frustum_mesh(
    height: float,
    bottom_radius: float,
    top_radius: float,
    segments: int = 48,
) -> Path:
    mesh_path = TRASH_BIN_MESH_PATH
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    half_h = height / 2.0
    vertices = []
    # The initial pose lays local +Z toward the front side of the workspace.
    # Put the smaller physical bottom face on local +Z so it faces front.
    for z, radius in ((-half_h, top_radius), (half_h, bottom_radius)):
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            vertices.append((radius * math.cos(theta), radius * math.sin(theta), z))
    vertices.append((0.0, 0.0, -half_h))
    vertices.append((0.0, 0.0, half_h))
    bottom_center = 2 * segments + 1
    top_center = 2 * segments + 2

    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        bottom_i = i + 1
        bottom_j = j + 1
        top_i = segments + i + 1
        top_j = segments + j + 1
        faces.append((bottom_i, bottom_j, top_j))
        faces.append((bottom_i, top_j, top_i))
        faces.append((bottom_center, bottom_j, bottom_i))
        faces.append((top_center, top_i, top_j))

    with mesh_path.open("w", encoding="ascii") as f:
        f.write("# generated truncated cone for MyDualTrashBinRolling-v0\n")
        for v in vertices:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    return mesh_path
