"""Adversarial Robustness Evaluation Test Bed"""

# Set up logging for a library
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Submodule imports
try:
    import coloredlogs

    from armory import art_experimental
    from armory import baseline_models
    from armory import data
    from armory import docker
    from armory import eval
    from armory import paths
    from armory import scenarios
    from armory import utils
    from armory import webapi
except ImportError as e:
    module = e.name
    print(f"ERROR: cannot import '{module}' module")
    print(f"    Please run: $ pip install -r requirements.txt")
    raise


# Semantic Version
__version__ = "0.6.0"
DEV = "-dev"


def is_dev():
    return __version__.endswith(DEV)
