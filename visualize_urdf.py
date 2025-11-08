import time
from pathlib import Path
import collections
import collections.abc
import fractions
import math
import warnings
import numpy as np

# urdfpy currently depends on networkx 2.2, which still imports these aliases
# from collections instead of collections.abc. Patch them before importing urdfpy.
for _name in ("Mapping", "MutableMapping", "Sequence", "Set", "Iterable"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(collections.abc, _name))

if not hasattr(fractions, "gcd"):
    fractions.gcd = math.gcd

np.int = int  # type: ignore[attr-defined]
np.float = float  # type: ignore[attr-defined]
np.bool = np.bool_  # type: ignore[attr-defined]
np.complex = complex  # type: ignore[attr-defined]

warnings.filterwarnings(
    "ignore",
    message="A NumPy version >=",
    category=UserWarning,
    module="scipy",
)
warnings.filterwarnings(
    "ignore",
    message=".*`np.bool`.*",
    category=FutureWarning,
    module="numpy",
)

import meshcat
import meshcat.geometry as g
from urdfpy import URDF, utils as urdf_utils, urdf as urdf_module
import argparse


def _patch_get_filename(package_map):
    original_get_filename = urdf_utils.get_filename

    def _patched(base_path, file_path, makedirs=False):
        if file_path.startswith("package://"):
            package_payload = file_path[len("package://") :]
            if "/" not in package_payload:
                raise ValueError(f"Malformed package URI: {file_path}")
            package_name, relative_path = package_payload.split("/", 1)
            package_root = package_map.get(package_name)
            if package_root is None:
                raise ValueError(f"Unknown package '{package_name}' in {file_path}")
            resolved = Path(package_root) / relative_path
            if makedirs:
                resolved.parent.mkdir(parents=True, exist_ok=True)
            return str(resolved)
        return original_get_filename(base_path, file_path, makedirs)

    urdf_utils.get_filename = _patched
    urdf_module.get_filename = _patched


def main():
    parser = argparse.ArgumentParser(description="Visualize a URDF with Meshcat.")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path(__file__).resolve().parent / "xarm7.urdf",
        help="Path to the URDF file to visualize.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    urdf_path = args.urdf if args.urdf.is_absolute() else (base_dir / args.urdf)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    package_map = {"xarm_description": str(base_dir / "xarm_description")}
    _patch_get_filename(package_map)

    robot = URDF.load(str(urdf_path))
    vis = meshcat.Visualizer().open()

    fk = robot.visual_trimesh_fk()
    for idx, (mesh, pose) in enumerate(fk.items()):
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32)
        geom = g.TriangularMeshGeometry(vertices, faces)
        material = g.MeshLambertMaterial(color=0x999999, reflectivity=0.6)
        node = vis[f"link_{idx}"]
        node.set_object(geom, material)
        node.set_transform(np.asarray(pose, dtype=np.float64))

    print(f"Meshcat is serving the model at: {vis.url()}")
    print("Press Ctrl+C to stop the viewer.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Viewer closed.")


if __name__ == "__main__":
    main()
