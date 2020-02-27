"""
Handles pathnames for armory
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


# Only initialize the host and docker paths once
_MAP = {}


def default():
    if "default" not in _MAP:
        _MAP["default"] = HostDefault()
    return _MAP["default"]


def host():
    if "host" not in _MAP:
        _MAP["host"] = HostPaths()
    return _MAP["host"]


def docker():
    if "docker" not in _MAP:
        _MAP["docker"] = DockerPaths()
    return _MAP["docker"]


def init():
    """
    Load configurations
    """
    default()
    host()
    docker()


def validate_config(config):
    if not isinstance(config, dict):
        raise TypeError(f"config is a {type(config)}, not a dict")
    keys = ("dataset_dir", "model_dir", "output_dir", "tmp_dir")
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
            for k in "dataset_dir", "model_dir", "output_dir", "tmp_dir":
                setattr(self, k, config[k])
            self.external_repo_dir = os.path.join(self.dataset_dir, "external_repos")
        else:
            logger.warning(f"No {self.armory_config} file. Using default paths.")
            logger.warning("Please run `armory configure`")
            self.dataset_dir = default().dataset_dir
            self.model_dir = default().model_dir
            self.tmp_dir = default().tmp_dir
            self.output_dir = default().output_dir
            self.external_repo_dir = default().external_repo_dir

    def makedirs(self):
        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        # tmp and output dirs should be handled by evaluator


class HostDefault:
    def __init__(self):
        self.cwd = os.getcwd()
        self.user_dir = os.path.expanduser("~")
        self.armory_dir = os.path.join(self.user_dir, ".armory")
        self.armory_config = os.path.join(self.armory_dir, "config.json")

        self.dataset_dir = os.path.join(self.armory_dir, "datasets")
        self.model_dir = os.path.join(self.armory_dir, "models")
        self.tmp_dir = os.path.join(self.armory_dir, "tmp")
        self.output_dir = os.path.join(self.armory_dir, "outputs")
        self.external_repo_dir = os.path.join(self.dataset_dir, "external_repos")


class DockerPaths:
    def __init__(self):
        self.cwd = "/workspace"
        armory_dir = "/armory"
        self.dataset_dir = armory_dir + "/datasets"
        self.model_dir = armory_dir + "/models"
        self.tmp_dir = armory_dir + "/tmp"
        self.output_dir = armory_dir + "/outputs"
        # TODO: external_repo_dir?
