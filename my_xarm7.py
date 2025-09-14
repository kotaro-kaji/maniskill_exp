import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent

#import gymnasium as gym
#env = gym.make("EmptyEnv-v1", robot_uids="my_xarm7")

@register_agent()
class Xarm7(BaseAgent):
    uid = "my_xarm7"
    urdf_path = f"xarm7.urdf"
    
    


