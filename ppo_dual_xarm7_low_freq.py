import os
import runpy


os.environ["PPO_DUAL_XARM7_SIM_FREQUENCY_HZ"] = "100"
os.environ["PPO_DUAL_XARM7_CONTROL_FREQUENCY_HZ"] = "20"

runpy.run_module("ppo_dual_xarm7", run_name="__main__")
