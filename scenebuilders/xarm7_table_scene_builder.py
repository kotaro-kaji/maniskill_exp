import numpy as np
import torch
import sapien
import sapien.render
from pathlib import Path
from typing import List
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder.table.scene_builder import TableSceneBuilder


PEDESTAL_HEIGHT = 0.0880
PEDESTAL_HALF_EXTENT_X = 0.075
PEDESTAL_HALF_EXTENT_X_NO_RGB = 0.095
PEDESTAL_LENGTH_Y = 0.83645  # xArmの土台の金属部分のみの左右方向の長さ
PEDESTAL_HALF_EXTENT_Y = PEDESTAL_LENGTH_Y / 2.0
ROBOT_BASE_X_OFFSET = -0.615

# !!! 超重要メモ !!!
# このテーブル定義では「Y方向が奥行き」扱いになっている。
# 一般的な机の設計や他の座標系（Xが奥行き等）と感覚がズレるので、要注意
# 見た目の「横幅・奥行き」を入れ替えてしまうと、机の向きが直感と逆になる。
# 以後、机のサイズや配置を調整する時は、
# 「Y=奥行き」「X=横幅」という前提で必ず確認すること。
# 机のX/Yを入れ替えない（世界座標に合わせる）
TABLE_LENGTH_X = PEDESTAL_HALF_EXTENT_X*2.0 + (0.427 + 0.005)
TABLE_WIDTH_Y = PEDESTAL_LENGTH_Y + 0.0259 + 0.0239
TABLE_HEIGHT = 0.9196429


TABLE_X_OFFSET = -PEDESTAL_HALF_EXTENT_X + ROBOT_BASE_X_OFFSET/2.0
TABLE_Y_OFFSET = -0.001

TABLE_YAW = 0.0


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
        self._build_table()
        obs_mode = str(getattr(self.env, "obs_mode", "")).lower()
        if "rgb" in obs_mode or bool(getattr(self.env, "collect_rmb_data", False)):
            pedestal_half_extent_x = PEDESTAL_HALF_EXTENT_X
        else:
            pedestal_half_extent_x = PEDESTAL_HALF_EXTENT_X_NO_RGB

        pedestal_half_size = (
            pedestal_half_extent_x,
            PEDESTAL_HALF_EXTENT_Y,
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

    def _build_table(self):
        builder = self.scene.create_actor_builder()
        table_half_size = (TABLE_LENGTH_X / 2.0, TABLE_WIDTH_Y / 2.0, TABLE_HEIGHT / 2.0)
        table_physical_material = sapien.physx.PhysxMaterial(
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.1,
        )
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, TABLE_HEIGHT / 2.0]),
            half_size=table_half_size,
            material=table_physical_material,
        )
        table_color = sapien_utils.hex2rgba("#334A3B")
        builder.add_box_visual(
            half_size=table_half_size,
            pose=sapien.Pose(p=[0, 0, TABLE_HEIGHT / 2.0]),
            material=sapien.render.RenderMaterial(
                base_color=table_color,
                roughness=0.3,
            ),
        )
        # NOTE: ここは「仮置き」。build時点で一度登場させる位置を定義しているだけ。
        # 実際の運用では、後段のinitializeで再配置される前提のため、ここは最終確定位置ではない。
        builder.initial_pose = sapien.Pose(
            p=[TABLE_X_OFFSET, TABLE_Y_OFFSET, -TABLE_HEIGHT], q=euler2quat(0, 0, TABLE_YAW)
        )
        self.table = builder.build_kinematic(name="table-workspace")
        self.table_length = TABLE_LENGTH_X
        self.table_width = TABLE_WIDTH_Y
        self.table_height = TABLE_HEIGHT
        floor_width = 500 if self.scene.parallel_in_single_scene else 100
        flat_ground_texture = (
            Path(__file__).resolve().parents[1]
            / "tests"
            / "ground_flat_from_grid.png"
        )
        self.ground = build_ground(
            self.scene,
            floor_width=floor_width,
            altitude=-self.table_height,
            texture_file=str(flat_ground_texture),
        )
        self.scene_objects = [self.table, self.ground]

    def initialize(self, env_idx: torch.Tensor):
        # Let the base class place the table and ground consistently
        super().initialize(env_idx)
        if hasattr(self, "table") and self.table is not None:
            self.table.set_pose(
                sapien.Pose(
                    p=[TABLE_X_OFFSET, TABLE_Y_OFFSET, -TABLE_HEIGHT],
                    q=euler2quat(0, 0, TABLE_YAW),
                )
            )

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
