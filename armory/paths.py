"""
Handles pathnames for armory
"""

import functools
import json
import logging
import os

logger = logging.getLogger(__name__)


# Only initialize the host and docker paths once
@functools.lru_cache(maxsize=1)
def default():
    return HostDefault()


@functools.lru_cache(maxsize=1)
def host():
    return HostPaths()


@functools.lru_cache(maxsize=1)
def docker():
    return DockerPaths()


def validate_config(config):
    if not isinstance(config, dict):
        raise TypeError(f"config is a {type(config)}, not a dict")
    keys = ("dataset_dir", "saved_model_dir", "output_dir", "tmp_dir")
    for key in keys:
        if key not in config:
            raise KeyError(f"config is missing key {key}")
    for key, value in config.items():
        if key not in keys:
            raise KeyError(f"config has additional key {key}")
        if not isinstance(value, str):
            raise ValueError(f"{key} value {value} is not a string")


def save_config(config):
    validate_config(config)
    os.makedirs(default().armory_dir, exist_ok=True)
    with open(default().armory_config, "w") as f:
        f.write(json.dumps(config, sort_keys=True, indent=4) + "\n")


def load_config():
    path = default().armory_config
    try:
        with open(path) as f:
            config = json.load(f)
    except json.decoder.JSONDecodeError:
        logger.exception(f"Armory config file {path} could not be decoded")
        raise
    except OSError:
        logger.exception(f"Armory config file {path} could not be read")
        raise

    try:
        validate_config(config)
    except (TypeError, KeyError, ValueError):
        logger.error(
            "Error parsing config.json. Please run `armory configure`.\n"
            "    If you previously ran an older version of armory, you may\n"
            f"    need to remove the {default().armory_dir} directory due to changes"
        )
        raise

    return config


class HostPaths:
    def __init__(self):
        self.cwd = default().cwd
        self.user_dir = default().user_dir
        self.armory_dir = default().armory_dir
        self.armory_config = default().armory_config
        if os.path.isfile(self.armory_config):
            # Parse paths from config
            config = load_config()
            for k in "dataset_dir", "saved_model_dir", "output_dir", "tmp_dir":
                setattr(self, k, config[k])
            self.external_repo_dir = os.path.join(self.tmp_dir, "external")
        else:
            logger.warning(f"No {self.armory_config} file. Using default paths.")
            logger.warning("Please run `armory configure`")
            self.dataset_dir = default().dataset_dir
            self.saved_model_dir = default().saved_model_dir
            self.tmp_dir = default().tmp_dir
            self.output_dir = default().output_dir
            self.external_repo_dir = default().external_repo_dir

        logger.info("Creating armory directories if they do not exist")
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.saved_model_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


def get_external(tmp_dir):
    return os.path.join(tmp_dir, "external")


class HostDefault:
    def __init__(self):
        self.cwd = os.getcwd()
        self.user_dir = os.path.expanduser("~")
        self.armory_dir = os.path.join(self.user_dir, ".armory")
        self.armory_config = os.path.join(self.armory_dir, "config.json")

        self.dataset_dir = os.path.join(self.armory_dir, "datasets")
        self.saved_model_dir = os.path.join(self.armory_dir, "saved_models")
        self.tmp_dir = os.path.join(self.armory_dir, "tmp")
        self.output_dir = os.path.join(self.armory_dir, "outputs")
        self.external_repo_dir = get_external(self.tmp_dir)


class DockerPaths:
    def __init__(self):
        self.cwd = "/workspace"
        armory_dir = "/armory"
        self.dataset_dir = armory_dir + "/datasets"
        self.saved_model_dir = armory_dir + "/saved_models"
        self.tmp_dir = armory_dir + "/tmp"
        self.output_dir = armory_dir + "/outputs"
        self.external_repo_dir = self.tmp_dir + "/external"
