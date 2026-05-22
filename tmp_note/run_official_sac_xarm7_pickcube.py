import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tasks.single_arm.pick_cube  # noqa: F401

official_sac = Path("/root/work/ManiSkill/examples/baselines/sac/sac.py")
runpy.run_path(str(official_sac), run_name="__main__")
