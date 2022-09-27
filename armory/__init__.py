"""
Adversarial Robustness Evaluation Test Bed
"""

from armory.logs import log
from armory.utils import version, typedef

__all__: set = ("__version__", "Config")
__version__: str = version.get_version()


Config = typedef.Config


END_SENTINEL = "Scenario has finished running cleanly"
