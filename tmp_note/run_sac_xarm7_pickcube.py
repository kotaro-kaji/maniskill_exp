import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tasks.single_arm.pick_cube  # noqa: F401

runpy.run_path("sac.py", run_name="__main__")
