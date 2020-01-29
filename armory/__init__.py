"""Adversarial Robustness Evaluation Platform for DARPA GARD"""

# Set up logging for a library
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Submodule imports
from armory import baseline_models
from armory import docker
from armory import eval
from armory import webapi

# Semantic Version
__version__ = "0.1.0"
