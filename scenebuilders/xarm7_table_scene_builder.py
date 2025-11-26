import torch
import sapien
import sapien.render
from typing import List

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder


PEDESTAL_HEIGHT = 0.0880
PEDESTAL_HALF_EXTENT_X = 0.072
ROBOT_BASE_X_OFFSET = -0.615


class Xarm7TableSceneBuilder(TableSceneBuilder):
    """Table scene builder with Xarm7 defaults.

    Reuses ManiSkill's TableSceneBuilder for geometry/ground, and only
    customizes the robot initialization for the custom agent uid "my_xarm7".
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._primary_pedestal_pose = sapien.Pose(
            p=[ROBOT_BASE_X_OFFSET, 0.0, PEDESTAL_HEIGHT / 2]
        )

    def build(self):
        super().build()

        pedestal_half_size = (
            PEDESTAL_HALF_EXTENT_X,
            self.table_width / 2,
            PEDESTAL_HEIGHT / 2,
        )

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=pedestal_half_size)
        builder.add_box_visual(
            half_size=pedestal_half_size,
            material=sapien.render.RenderMaterial(
                base_color=[0.75, 0.75, 0.8, 1.0],
                metallic=0.9,
                roughness=0.3,
            ),
        )
        builder.initial_pose = self._primary_pedestal_pose
        self.robot_pedestal = builder.build_static(name="robot_pedestal")
        self.scene_objects.append(self.robot_pedestal)

    def initialize(self, env_idx: torch.Tensor):
        # Let the base class place the table and ground consistently
        super().initialize(env_idx)

        agents = self._get_agent_sequence()
        if not agents:
            return

        try:
            qpos_per_agent = self._compute_initial_qpos(agents, env_idx)
            for idx, agent in enumerate(agents):
                qpos = qpos_per_agent[idx]
                if qpos is not None:
                    agent.reset(qpos)
                pose = self._initial_agent_pose(idx)
                if pose is not None:
                    agent.robot.set_pose(pose)
        except Exception:
            # If anything fails (e.g. missing keyframes), fall back to default placement.
            pass

    # --------------------------------------------------------------------- #
    # Helper hooks for subclasses / multi-agent compatibility
    # --------------------------------------------------------------------- #
    def _get_agent_sequence(self) -> List["BaseAgent"]:
        agent = getattr(self.env, "agent", None)
        if agent is None:
            return []
        if isinstance(agent, MultiAgent):
            return list(agent.agents)
        return [agent]

    def _compute_initial_qpos(self, agents: List["BaseAgent"], env_idx: torch.Tensor):
        batch_size = len(env_idx)
        qpos_per_agent = []
        for agent in agents:
            home_qpos = None
            keyframes = getattr(agent, "keyframes", None)
            if keyframes and "home" in keyframes:
                home_qpos = keyframes["home"].qpos
            qpos_per_agent.append(self._expand_initial_qpos(home_qpos, batch_size))
        return qpos_per_agent

    def _expand_initial_qpos(self, qpos, batch_size: int):
        if qpos is None:
            return None
        if isinstance(qpos, torch.Tensor):
            tensor = qpos.to(self.env.device)
        else:
            tensor = torch.as_tensor(qpos, dtype=torch.float32, device=self.env.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 1 and batch_size > 1:
            tensor = tensor.repeat(batch_size, 1)
        return tensor.clone()

    def _initial_agent_pose(self, agent_index: int) -> sapien.Pose:
        return sapien.Pose([ROBOT_BASE_X_OFFSET, 0, PEDESTAL_HEIGHT])
