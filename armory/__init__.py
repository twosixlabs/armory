"""
Adversarial Robustness Evaluation Test Bed
"""

from pathlib import Path

from armory.logs import log
from armory.utils import typedef, version

Config = typedef.Config


SRC_ROOT = Path(__file__).parent


def __getattr__(name):
    if name == "__version__":
        return version.get_version()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


END_SENTINEL = "Scenario has finished running cleanly"
