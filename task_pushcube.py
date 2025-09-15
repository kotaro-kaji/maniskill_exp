import sapien
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from xarm7_table_scene_builder import Xarm7TableSceneBuilder
from mani_skill.utils.registration import register_env

from mani_skill.utils.structs.pose import Pose
import torch

from mani_skill.utils.building import actors

from typing import Union, Dict
import sapien

from mani_skill.envs.sapien_env import BaseEnv

# ★ 自作ロボットを import（これで登録の副作用が走る）
from my_xarm7 import Xarm7  # ← your file/module path に合わせて

# （必要なら他ロボも残す）
# from mani_skill.agents.robots import Fetch, Panda


@register_env("MyPushCube-v1", max_episode_steps=50)
class MyPushCubeEnv(BaseEnv):

    goal_radius = 0.1
    cube_half_size = 0.02

    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    # ★ サポートロボに自作UIDを追加（自作だけにするなら ["my_xarm7"] だけでOK）
    SUPPORTED_ROBOTS = ["my_xarm7"]  # , "panda", "fetch"]

    # ★ 型ヒントも自作に
    agent: Xarm7  # Union[Xarm7, Panda, Fetch] みたいにしてもOK

    def __init__(self, *args, robot_uids="my_xarm7", **kwargs):
        # "panda" や "fetch" も許すなら、タプル/リストで受けられるのは元のまま
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        # ベースポーズは環境に合わせて。床置きなら z=0 付近でOK
        base_pose = sapien.Pose(p=[0.0, 0.0, 0.0])
        super()._load_agent(options, base_pose)

    def _load_scene(self, options: dict):
        builder = self.scene.create_actor_builder()
        builder.add_box_collision(
            # for boxes we specify half length of each side
            half_size=[0.02] * 3,
        )
        builder.add_box_visual(
            half_size=[0.02] * 3,
            material=sapien.render.RenderMaterial(
                # RGBA values, this is a red cube
                base_color=[1, 0, 0, 1],
            ),
        )
        # strongly recommended to set initial poses for objects, even if you plan to modify them later
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.02], q=[1, 0, 0, 0])
        self.obj = builder.build(name="cube")
        # PushCube has some other code after this removed for brevity that 
        # spawns a goal object (a red/white target) stored at self.goal_region

        self.table_scene = Xarm7TableSceneBuilder(
            env=self,
        )
        self.table_scene.build()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # using torch.device context manager to auto create tensors 
        # on CPU/CUDA depending on self.device, the device the env runs on
        with torch.device(self.device):
            b = len(env_idx)
            # use the TableSceneBuilder to init all objects in that scene builder
            self.table_scene.initialize(env_idx)

            # here is randomization code that randomizes the x, y position 
            # of the cube we are pushing in the range [-0.1, -0.1] to [0.1, 0.1]
            p = torch.zeros((b, 3))
            p[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            p[..., 2] = self.cube_half_size
            q = [1, 0, 0, 0]
            obj_pose = Pose.create_from_pq(p=p, q=q)
            self.obj.set_pose(obj_pose)
    
    def _get_obs_extra(self, info: Dict):
        # some useful observation info for solving the task includes the pose of the tcp (tool center point) which is the point between the
        # grippers of the robot
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            # if the observation mode requests to use state, we provide ground truth information about where the cube is.
            # for visual observation modes one should rely on the sensed visual data to determine where the cube is
            obs.update(
                goal_pos=self.goal_region.pose.p,
                obj_pose=self.obj.pose.raw_pose,
            )
        return obs
