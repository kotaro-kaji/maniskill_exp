"""xArm7 agent with a force controller that mirrors the official xArm Python SDK API."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Optional, Type

import numpy as np
import torch
from gymnasium import spaces

from mani_skill import format_path
from mani_skill.agents.controllers.base_controller import BaseController, ControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.utils.geometry.rotation_conversions import quaternion_to_matrix
from mani_skill.utils.structs.pose import Pose

from mani_skill.utils.common import deepcopy_dict

from .my_xarm7 import Xarm7


_PID_RANGES = dict(kp=(0.0, 0.05), ki=(0.0, 0.0005), kd=(0.0, 0.05), xe_limit=(0.0, 200.0))
_FORCE_REF_LIMIT = np.array([150.0, 150.0, 200.0, 4.0, 4.0, 4.0], dtype=np.float32)
_DEFAULT_XE_LIMIT = np.array([200.0, 200.0, 200.0, 0.35, 0.35, 0.35], dtype=np.float32)
_MM_TO_M = 1.0 / 1000.0
_JOINT_DELTA_LIMIT = 0.1


@dataclass
class XArmForceControllerConfig(ControllerConfig):
    """Configuration for the xArm force controller."""

    ee_link: str
    urdf_path: str
    tcp_offset: Sequence[float] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    default_coord: int = 1
    default_c_axis: Sequence[int] = (0, 0, 1, 0, 0, 0)
    default_force_ref: Sequence[float] = (0.0, 0.0, 5.0, 0.0, 0.0, 0.0)
    default_kp: Sequence[float] = (0.005, 0.005, 0.005, 0.05, 0.05, 0.05)
    default_ki: Sequence[float] = (0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005)
    default_kd: Sequence[float] = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    default_xe_limit: Sequence[float] = tuple(_DEFAULT_XE_LIMIT.tolist())
    stiffness: float = 800.0
    damping: float = 40.0
    force_limit: float = 120.0
    action_updates_reference: bool = False
    controller_cls: Optional[Type[BaseController]] = None


class XArmForceController(BaseController):
    """Task-space force controller that follows the official xArm force-control semantics."""

    config: XArmForceControllerConfig
    sets_target_qpos = True
    sets_target_qvel = False

    def __init__(self, config, articulation, control_freq, sim_freq=None, scene=None):
        self._sensor_enabled = False
        self._mode = 0
        self._bias_tool: Optional[torch.Tensor] = None
        self._limits = torch.zeros(6)
        self._qlimits = None
        self._tcp_offset: Optional[Pose] = None
        self._pk_chain = None
        self._pk_joint_names: list[str] = []
        self._pk_articulation_indices: list[Optional[int]] = []
        super().__init__(config, articulation, control_freq, sim_freq=sim_freq, scene=scene)
        self._setup_tcp()
        self._setup_kinematics()
        self._setup_limits()
        self.reset()

    # ------------------------------------------------------------------
    # Controller initialization helpers
    # ------------------------------------------------------------------
    def _setup_tcp(self):
        self.ee_link = self.articulation.links_map[self.config.ee_link]
        offset = torch.tensor(self.config.tcp_offset, dtype=torch.float32, device=self.device)
        p = offset[:3].unsqueeze(0)
        q = offset[3:].unsqueeze(0)
        self._tcp_offset = Pose.create_from_pq(p=p, q=q)

    def _setup_kinematics(self):
        try:
            import pytorch_kinematics as pk
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "pytorch_kinematics_ms is required for the xArm force controller"
            ) from exc
        urdf_path = format_path(self.config.urdf_path)
        with open(urdf_path, "rb") as fh:
            urdf_bytes = fh.read()
        self._pk_chain = pk.build_serial_chain_from_urdf(
            urdf_bytes, end_link_name=self.config.ee_link
        ).to(dtype=torch.float32, device=self.device)
        self._pk_joint_names = list(self._pk_chain.get_joint_parameter_names())
        self._pk_articulation_indices = []
        for name in self._pk_joint_names:
            joint = self.articulation.joints_map.get(name)
            if joint is None or joint.active_index is None:
                self._pk_articulation_indices.append(None)
            else:
                self._pk_articulation_indices.append(int(joint.active_index[0]))
        controlled_joint_set = set(self.config.joint_names)
        missing = controlled_joint_set.difference(self._pk_joint_names)
        if missing:
            raise ValueError(
                f"URDF kinematic chain is missing joints required for control: {sorted(missing)}"
            )
        self._pk_control_indices = torch.tensor(
            [self._pk_joint_names.index(name) for name in self.config.joint_names],
            dtype=torch.long,
            device=self.device,
        )

    def _setup_limits(self):
        limits = self.articulation.get_qlimits()[0, self.active_joint_indices]
        self._qlimits = limits.to(self.device)

    # ------------------------------------------------------------------
    # BaseController overrides
    # ------------------------------------------------------------------
    def _initialize_action_space(self):
        limit = np.asarray(_FORCE_REF_LIMIT, dtype=np.float32)
        self.single_action_space = spaces.Box(-limit, limit, dtype=np.float32)

    def set_drive_property(self):
        stiffness = np.broadcast_to(self.config.stiffness, len(self.joints))
        damping = np.broadcast_to(self.config.damping, len(self.joints))
        force_limit = np.broadcast_to(self.config.force_limit, len(self.joints))
        for i, joint in enumerate(self.joints):
            joint.set_drive_properties(
                stiffness[i], damping[i], force_limit=force_limit[i], mode="force"
            )
            joint.set_friction(0.0)

    def reset(self):
        super().reset()
        num_envs = self.scene.num_envs
        device = self.device
        def _tensor(values):
            return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)

        self._kp = _tensor(self.config.default_kp)
        self._ki = _tensor(self.config.default_ki)
        self._kd = _tensor(self.config.default_kd)
        self._xe_limit = _tensor(self.config.default_xe_limit)
        self._force_ref = _tensor(self.config.default_force_ref)
        self._axis_mask = _tensor(self.config.default_c_axis)
        self._coord = int(self.config.default_coord)
        self._integral = torch.zeros(num_envs, 6, device=device)
        self._prev_error = torch.zeros(num_envs, 6, device=device)
        self._bias_tool = torch.zeros(num_envs, 6, device=device)
        self._limits = torch.zeros(6, device=device)
        self._force_ref = self._force_ref.repeat(num_envs, 1)
        self._kp = self._kp.repeat(num_envs, 1)
        self._ki = self._ki.repeat(num_envs, 1)
        self._kd = self._kd.repeat(num_envs, 1)
        self._xe_limit = self._xe_limit.repeat(num_envs, 1)
        self._axis_mask = self._axis_mask.repeat(num_envs, 1)

    def set_action(self, action):
        if not self.config.action_updates_reference:
            return
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        limit = torch.tensor(_FORCE_REF_LIMIT, dtype=torch.float32, device=self.device)
        action = torch.clamp(action, -limit, limit)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        self._force_ref = action * self._axis_mask

    # ------------------------------------------------------------------
    # Public API mirroring xArm SDK
    # ------------------------------------------------------------------
    def set_force_parameters(self, *args, params_limit: bool = True):
        if len(args) == 4 and isinstance(args[0], Iterable) and not isinstance(args[0], (int, float)):
            kp, ki, kd, xe = args
            self._apply_pid(kp, ki, kd, xe, params_limit=params_limit)
            return 0
        if len(args) == 4 and isinstance(args[0], int):
            coord, c_axis, f_ref, limits = args
            self._apply_reference(coord, c_axis, f_ref, limits, params_limit=params_limit)
            return 0
        if len(args) == 8 and isinstance(args[0], Iterable):
            kp, ki, kd, xe, coord, c_axis, f_ref, limits = args
            self._apply_pid(kp, ki, kd, xe, params_limit=params_limit)
            self._apply_reference(coord, c_axis, f_ref, limits, params_limit=params_limit)
            return 0
        raise ValueError("Unsupported argument combination for set_force_parameters")

    def set_sensor_enable(self, enable: int) -> int:
        self._sensor_enabled = bool(enable)
        if not self._sensor_enabled:
            self._integral.zero_()
            self._prev_error.zero_()
        return 0

    def set_sensor_mode(self, mode: int) -> int:
        if mode not in (0, 2):
            raise ValueError("Only modes 0 (disabled) and 2 (force) are supported")
        self._mode = mode
        return 0

    def set_sensor_zero(self) -> int:
        measurement = self._read_wrench(frame="tool", subtract_bias=False)
        self._bias_tool = measurement.detach()
        self._integral.zero_()
        self._prev_error.zero_()
        return 0

    def get_sensor_data(self, frame: str = "tool") -> tuple[int, np.ndarray]:
        measurement = self._read_wrench(frame=frame, subtract_bias=True)
        data = measurement.detach().cpu().numpy()
        return 0, data

    # ------------------------------------------------------------------
    # Internal state updates
    # ------------------------------------------------------------------
    def _apply_pid(self, kp, ki, kd, xe_limit, params_limit: bool = True):
        kp = torch.tensor(list(kp), dtype=torch.float32, device=self.device)
        ki = torch.tensor(list(ki), dtype=torch.float32, device=self.device)
        kd = torch.tensor(list(kd), dtype=torch.float32, device=self.device)
        xe_limit = torch.tensor(list(xe_limit), dtype=torch.float32, device=self.device)
        self._validate_range(kp, _PID_RANGES["kp"], params_limit)
        self._validate_range(ki, _PID_RANGES["ki"], params_limit)
        self._validate_range(kd, _PID_RANGES["kd"], params_limit)
        self._validate_range(xe_limit, _PID_RANGES["xe_limit"], params_limit)
        self._kp = kp.unsqueeze(0).repeat(self.scene.num_envs, 1)
        self._ki = ki.unsqueeze(0).repeat(self.scene.num_envs, 1)
        self._kd = kd.unsqueeze(0).repeat(self.scene.num_envs, 1)
        self._xe_limit = xe_limit.unsqueeze(0).repeat(self.scene.num_envs, 1)

    def _apply_reference(self, coord, c_axis, f_ref, limits, params_limit: bool = True):
        coord = int(coord)
        axis = torch.tensor(list(c_axis), dtype=torch.float32, device=self.device)
        ref = torch.tensor(list(f_ref), dtype=torch.float32, device=self.device)
        self._validate_force_ref(ref, params_limit)
        self._axis_mask = axis.unsqueeze(0).repeat(self.scene.num_envs, 1)
        self._force_ref = (ref.unsqueeze(0) * self._axis_mask).repeat(self.scene.num_envs, 1)
        self._coord = coord
        if limits is not None:
            self._limits = torch.tensor(list(limits), dtype=torch.float32, device=self.device)

    def _validate_range(self, value: torch.Tensor, bounds, params_limit: bool):
        if not params_limit:
            return
        low, high = bounds
        if torch.any(value < low) or torch.any(value > high):
            raise ValueError(f"Controller parameter out of range [{low}, {high}]")

    def _validate_force_ref(self, ref: torch.Tensor, params_limit: bool):
        if not params_limit:
            return
        limit = torch.tensor(_FORCE_REF_LIMIT, device=self.device)
        if torch.any(torch.abs(ref) > limit):
            raise ValueError("force reference exceeds firmware safe bounds")

    # ------------------------------------------------------------------
    # Sensor processing
    # ------------------------------------------------------------------
    def _read_wrench(self, frame: str = "tool", subtract_bias: bool = True) -> torch.Tensor:
        world_wrench = self._compute_world_wrench()
        tool_wrench = self._world_to_tool(world_wrench)
        if subtract_bias:
            tool_wrench = tool_wrench - self._bias_tool
        if frame == "tool":
            return tool_wrench
        if frame == "base":
            return self._tool_to_world(tool_wrench)
        raise ValueError(f"Unknown frame '{frame}'")

    def _compute_world_wrench(self) -> torch.Tensor:
        num_envs = self.scene.num_envs
        dt = float(self.scene.timestep)
        forces = np.zeros((num_envs, 3), dtype=np.float32)
        torques = np.zeros((num_envs, 3), dtype=np.float32)
        if self.scene.gpu_sim_enabled:
            contact_forces = (
                self.articulation.get_net_contact_forces([self.ee_link.name])
                .detach()
                .cpu()
                .numpy()
            )
            forces = contact_forces[:, 0, :]
        else:
            contacts = self.scene.get_contacts()
            if len(contacts) == 0:
                return torch.zeros(num_envs, 6, device=self.device)
            entity_to_env = {
                link.entity: idx for idx, link in enumerate(self.ee_link._objs)
            }
            tcp_pose = self.tcp_pose
            tcp_pos = tcp_pose.p.detach().cpu().numpy()
            for contact in contacts:
                bodies = contact.bodies
                for body_idx in (0, 1):
                    entity = bodies[body_idx].entity
                    env = entity_to_env.get(entity)
                    if env is None:
                        continue
                    sign = 1.0 if body_idx == 0 else -1.0
                    for point in contact.points:
                        impulse = np.array(point.impulse, dtype=np.float32) * sign
                        force = impulse / dt
                        pos = np.array(point.position, dtype=np.float32)
                        forces[env] += force
                        torques[env] += np.cross(pos - tcp_pos[env], force)
        stacked = np.concatenate([forces, torques], axis=1)
        return torch.as_tensor(stacked, dtype=torch.float32, device=self.device)

    @property
    def tcp_pose(self) -> Pose:
        return self.ee_link.pose * self._tcp_offset

    def _world_to_tool(self, wrench_world: torch.Tensor) -> torch.Tensor:
        rotation = quaternion_to_matrix(self.tcp_pose.q)
        f = torch.bmm(rotation.transpose(1, 2), wrench_world[:, :3].unsqueeze(-1)).squeeze(-1)
        tau = torch.bmm(rotation.transpose(1, 2), wrench_world[:, 3:].unsqueeze(-1)).squeeze(-1)
        return torch.cat([f, tau], dim=-1)

    def _tool_to_world(self, wrench_tool: torch.Tensor) -> torch.Tensor:
        rotation = quaternion_to_matrix(self.tcp_pose.q)
        f = torch.bmm(rotation, wrench_tool[:, :3].unsqueeze(-1)).squeeze(-1)
        tau = torch.bmm(rotation, wrench_tool[:, 3:].unsqueeze(-1)).squeeze(-1)
        return torch.cat([f, tau], dim=-1)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------
    def before_simulation_step(self):
        if not self._sensor_enabled or self._mode != 2:
            return
        dt = float(self.scene.timestep)
        if dt <= 0:
            return
        measurement = self._read_wrench(
            frame="tool" if self._coord == 1 else "base", subtract_bias=True
        )
        target = self._force_ref
        error = (target - measurement) * self._axis_mask
        self._integral = self._integral + error * dt
        integral_limit = torch.where(
            self._ki > 0,
            self._xe_limit / torch.clamp(self._ki, min=torch.finfo(torch.float32).eps),
            torch.full_like(self._xe_limit, 1e6),
        )
        self._integral = torch.clamp(self._integral, -integral_limit, integral_limit)
        derivative = (error - self._prev_error) / dt
        self._prev_error = error
        vel = self._kp * error + self._ki * self._integral + self._kd * derivative
        vel = torch.clamp(vel, -self._xe_limit, self._xe_limit) * self._axis_mask
        if self._coord == 1:
            vel_world = self._tool_to_world(vel)
        else:
            vel_world = vel
        vel_world = vel_world.clone()
        vel_world[:, :3] *= _MM_TO_M
        jac = self._compute_spatial_jacobian()
        if jac is None:
            return
        jac_pinv = torch.linalg.pinv(jac)
        qdot = torch.bmm(jac_pinv, vel_world.unsqueeze(-1)).squeeze(-1)
        dq = qdot * dt
        current_qpos = self.qpos
        qpos = current_qpos + dq
        delta = torch.clamp(qpos - current_qpos, -_JOINT_DELTA_LIMIT, _JOINT_DELTA_LIMIT)
        qpos = current_qpos + delta
        lower = self._qlimits[:, 0].unsqueeze(0)
        upper = self._qlimits[:, 1].unsqueeze(0)
        qpos = torch.max(torch.min(qpos, upper), lower)
        self.articulation.set_joint_drive_targets(qpos, self.joints, self.active_joint_indices)

    def _compute_spatial_jacobian(self) -> Optional[torch.Tensor]:
        if self._pk_chain is None:
            return None
        q_full = torch.zeros(
            self.scene.num_envs, len(self._pk_joint_names), device=self.device
        )
        qpos = self.articulation.get_qpos()
        for idx, art_idx in enumerate(self._pk_articulation_indices):
            if art_idx is not None:
                q_full[:, idx] = qpos[:, art_idx]
        jac = self._pk_chain.jacobian(q_full)
        return jac[:, :, self._pk_control_indices]


@register_agent()
class Xarm7Official(Xarm7):
    """xArm7 agent with an SDK-faithful force controller."""

    uid = "my_xarm7_official"

    def _make_force_controller_config(self) -> XArmForceControllerConfig:
        return XArmForceControllerConfig(
            joint_names=self.arm_joint_names,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
            tcp_offset=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            default_force_ref=(0.0, 0.0, 5.0, 0.0, 0.0, 0.0),
            default_c_axis=(0, 0, 1, 0, 0, 0),
            default_coord=1,
            controller_cls=XArmForceController,
        )

    @property
    def _controller_configs(self):
        parent = super()._controller_configs
        force_cfg = dict(
            arm=self._make_force_controller_config(),
            gripper=parent["pd_joint_delta_pos"]["gripper"],
        )
        configs = {"force_control": force_cfg}
        configs.update(parent)
        return deepcopy_dict(configs)

    @property
    def force_controller(self) -> XArmForceController:
        controller = self.controllers.get("force_control")
        if controller is None:
            config = self._make_force_controller_config()
            controller = config.controller_cls(
                config, self.robot, self._control_freq, scene=self.scene
            )
            controller.set_drive_property()
            self.controllers["force_control"] = controller
        return controller

    # ------------------------------------------------------------------
    # SDK-compatible helper functions
    # ------------------------------------------------------------------
    def set_ft_sensor_force_parameters(self, *args, **kwargs):
        return self.force_controller.set_force_parameters(*args, **kwargs)

    def set_ft_sensor_enable(self, enable: int):
        return self.force_controller.set_sensor_enable(enable)

    def set_ft_sensor_zero(self):
        return self.force_controller.set_sensor_zero()

    def set_ft_sensor_mode(self, mode: int):
        return self.force_controller.set_sensor_mode(mode)

    def get_ft_sensor_data(self, frame: str = "tool"):
        return self.force_controller.get_sensor_data(frame=frame)
