"""Adversarial Robustness Evaluation Test Bed"""

# Set up logging for a library
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Semantic Version
<<<<<<< HEAD
__version__ = "0.14.0"
=======
__version__ = "0.14.2"
>>>>>>> a6b3f05cd557787c83894d97d8e1ca753bb55eb5

# typedef for a widely used JSON-like configuration specification
from typing import Dict, Any

Config = Dict[str, Any]

# Submodule imports
try:
    import coloredlogs

    from armory import art_experimental
    from armory import baseline_models
    from armory import data
    from armory import docker
    from armory import eval
    from armory import paths
    from armory import utils
    from armory import webapi
except ImportError as e:
    module = e.name
    print(f"ERROR: cannot import '{module}' module")
    print("    Please run: $ pip install -r requirements.txt")
    raise

END_SENTINEL = "Scenario has finished running cleanly"
