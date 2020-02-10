"""
Handles pathnames for armory
"""

import json
import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).parents[1]
CWD = os.getcwd()
USER = os.path.expanduser("~")
USER_ARMORY = os.path.join(USER, ".armory")
USER_KERAS = os.path.join(USER, ".keras")
ARMORY_CONFIG = os.path.join(USER_ARMORY, "config.json")
if os.path.exists(ARMORY_CONFIG):
    try:
        with open(ARMORY_CONFIG) as f:
            config = json.load(f)
    except json.decoder.JSONDecodeError:
        logger.exception(f"Armory config file {ARMORY_CONFIG} could not be decoded")
        raise
    except OSError:
        logger.exception(f"Armory config file {ARMORY_CONFIG} could not be read")
        raise
else:
    os.makedirs(USER_ARMORY, exist_ok=True)
    config = {}
    with open(ARMORY_CONFIG, "w") as f:
        json.dump(config, f)

DATASETS = config.get("datasets") or os.path.join(USER_ARMORY, "datasets")
EXTERNAL_REPOS = os.path.join(DATASETS, "external_repos")
MODELS = config.get("models") or os.path.join(USER_ARMORY, "models")
TMP = config.get("tmp") or os.path.join(USER_ARMORY, "tmp")
OUTPUTS = config.get("outputs") or os.path.join(USER_ARMORY, "outputs")

DOCKER = "/root"
DOCKER_ARMORY = os.path.join(DOCKER, ".armory")
DOCKER_TMP = Path(os.path.join(DOCKER_ARMORY, "tmp")).as_posix()
