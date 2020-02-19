"""
Handles pathnames for armory
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def _parse_global_config(path, armory_dir):
    if os.path.exists(path):
        try:
            with open(path) as f:
                config = json.load(f)
        except json.decoder.JSONDecodeError:
            logger.exception(f"Armory config file {path} could not be decoded")
            raise
        except OSError:
            logger.exception(f"Armory config file {path} could not be read")
            raise
    else:
        os.makedirs(armory_dir, exist_ok=True)
        config = {
            "cached_dataset_dir": os.path.join(armory_dir, "datasets"),
            "output_dir": os.path.join(armory_dir, "outputs"),
        }
        with open(path, "w") as f:
            json.dump(config, f)

    try:
        os.makedirs(config["cached_dataset_dir"], exist_ok=True)
        os.makedirs(config["output_dir"], exist_ok=True)

    # TODO: Recommend to use armory config (#165)
    except KeyError:
        raise KeyError(
            "Error parsing config.json. If you previously ran an older "
            "version of armory you may need to remove the ~/.armory directory since"
            "permission changes."
        )
    return config


class HostPaths:
    def __init__(self):
        self.cwd = os.getcwd()
        self.user_dir = os.path.expanduser("~")
        self.armory_dir = os.path.join(self.user_dir, ".armory")
        self.tmp_dir = os.path.join(self.armory_dir, "tmp")
        self.armory_config = os.path.join(self.armory_dir, "config.json")

        # Parse paths from config
        config = _parse_global_config(self.armory_config, self.armory_dir)
        self.dataset_dir = config.get("cached_dataset_dir")
        self.output_dir = config.get("output_dir")
        self.external_repo_dir = os.path.join(self.dataset_dir, "external_repos")


class DockerPaths:
    def __init__(self):
        self.armory_dir = "/armory"
        self.tmp_dir = "/armory/tmp"
        self.dataset_dir = "/datasets"
        self.output_dir = "/outputs"
