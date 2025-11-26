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
import trimesh
from urdfpy import URDF, utils as urdf_utils, urdf as urdf_module
import argparse


def _parse_joint_overrides(parser, robot, override_args):
    overrides = {}
    if not override_args:
        return overrides

    valid_joints = {joint.name for joint in robot.actuated_joints}
    for entry in override_args:
        if "=" not in entry:
            parser.error(
                f"Joint override '{entry}' is malformed. Use the NAME=VALUE format."
            )
        name, value_str = entry.split("=", 1)
        name = name.strip()
        value_str = value_str.strip()
        if not name:
            parser.error(f"Joint override '{entry}' must include a joint name.")
        if name not in valid_joints:
            parser.error(
                f"Unknown joint '{name}'. Available joints: {sorted(valid_joints)}"
            )
        try:
            overrides[name] = float(value_str)
        except ValueError:
            parser.error(
                f"Could not parse value '{value_str}' for joint '{name}'."
            )
    return overrides


def _build_joint_configuration(robot, overrides):
    """Start with zeros, clamp by limits, then apply overrides."""
    cfg = {}
    for joint in robot.actuated_joints:
        value = overrides.get(joint.name, 0.0)
        limit = joint.limit
        if limit is not None:
            lower = limit.lower if limit.lower is not None else -np.inf
            upper = limit.upper if limit.upper is not None else np.inf
            clamped_value = float(np.clip(value, lower, upper))
            if clamped_value != value:
                print(
                    f"Joint '{joint.name}' clamped from {value:.4f} "
                    f"to {clamped_value:.4f} to respect limits."
                )
            value = clamped_value
        cfg[joint.name] = value
    return cfg


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


def _patch_cylinder_mesh_generation():
    """Work around urdfpy<=0.0.22 bug where Cylinder.meshes is broken."""

    cylinder_cls = getattr(urdf_module, "Cylinder", None)
    if cylinder_cls is None:
        return

    def _fixed_meshes(self):
        if self._meshes is None or len(self._meshes) == 0:
            self._meshes = [
                trimesh.creation.cylinder(radius=self.radius, height=self.length)
            ]
        return self._meshes

    cylinder_cls.meshes = property(_fixed_meshes)


def main():
    parser = argparse.ArgumentParser(description="Visualize a URDF with Meshcat.")
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path(__file__).resolve().parent / "xarm7.urdf",
        help="Path to the URDF file to visualize.",
    )
    parser.add_argument(
        "--joint",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Override a joint angle (radians). Repeat for multiple joints, "
            "e.g. --joint drive_joint=0.72"
        ),
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    urdf_path = args.urdf if args.urdf.is_absolute() else (base_dir / args.urdf)
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    package_map = {"xarm_description": str(base_dir / "xarm_description")}
    _patch_get_filename(package_map)
    _patch_cylinder_mesh_generation()

    robot = URDF.load(str(urdf_path))
    joint_overrides = _parse_joint_overrides(parser, robot, args.joint)
    joint_cfg = _build_joint_configuration(robot, joint_overrides)
    vis = meshcat.Visualizer().open()

    fk = robot.visual_trimesh_fk(cfg=joint_cfg)
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
