"""Adversarial Robustness Evaluation Test Bed"""

# Set up logging for a library
import logging
import sys

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Submodule imports
try:
    import coloredlogs

    from armory import art_experimental
    from armory import data
    from armory import baseline_models
    from armory import docker
    from armory import eval
    from armory import eval_scripts
    from armory import utils
    from armory import webapi
    from armory import paths
except ImportError as e:
    module = e.name
    print(f"ERROR: cannot import '{module}' module", file=sys.stderr)
    print(f"    Please run: $ pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)


# Semantic Version
__version__ = "0.4.1"
