"""
Adversarial Robustness Evaluation Test Bed
"""

from typing import Dict, Any

from armory.logs import log
from armory.version import get_version


def __getattr__(name):
    if name == "__version__":
        return get_version()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# typedef for a widely used JSON-like configuration specification
Config = Dict[str, Any]

END_SENTINEL = "Scenario has finished running cleanly"
