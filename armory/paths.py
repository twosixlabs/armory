"""
Reference objects for armory paths
"""

import os
import warnings  # armory.logs initialization depends on this module, use warnings instead

from armory import configuration

NO_DOCKER = False


def set_mode(mode, set_art_path=True):
    """
    Set path mode to "docker" or "host"
    """
    MODES = ("docker", "host")
    global NO_DOCKER
    if mode == "docker":
        NO_DOCKER = False
    elif mode == "host":
        NO_DOCKER = True
    else:
        raise ValueError(f"mode {mode} is not in {MODES}")

    if set_art_path:
        set_art_data_path()


def set_art_data_path():
    """
    Sets the art data path if art can be imported in the current environment
    """
    try:
        from art import config

        config.set_data_path(os.path.join(runtime_paths().saved_model_dir, "art"))
    except ImportError:
        pass


def runtime_paths():
    """
    Delegates armory evaluation paths to be either Host or Docker paths.
    """
    if NO_DOCKER:
        return HostPaths()
    else:
        return DockerPaths()


class DockerPaths:
    def __init__(self):
        self.cwd = "/workspace"
        armory_dir = "/armory"
        self.dataset_dir = armory_dir + "/datasets"
        self.local_git_dir = armory_dir + "/git"
        self.saved_model_dir = armory_dir + "/saved_models"
        self.pytorch_dir = self.saved_model_dir + "/pytorch"
        self.tmp_dir = armory_dir + "/tmp"
        self.output_dir = armory_dir + "/outputs"
        self.external_repo_dir = self.tmp_dir + "/external"


class HostDefaultPaths:
    def __init__(self):
        self.cwd = os.getcwd()
        self.user_dir = os.path.expanduser("~")
        self.armory_dir = os.path.join(self.user_dir, ".armory")
        self.armory_config = os.path.join(self.armory_dir, "config.json")
        self.dataset_dir = os.path.join(self.armory_dir, "datasets")
        self.local_git_dir = os.path.join(self.armory_dir, "git")
        self.saved_model_dir = os.path.join(self.armory_dir, "saved_models")
        self.pytorch_dir = os.path.join(self.armory_dir, "saved_models", "pytorch")
        self.tmp_dir = os.path.join(self.armory_dir, "tmp")
        self.output_dir = os.path.join(self.armory_dir, "outputs")
        self.external_repo_dir = os.path.join(self.tmp_dir, "external")


class HostPaths(HostDefaultPaths):
    def __init__(self):
        super().__init__()
        if os.path.isfile(self.armory_config):
            # Parse paths from config
            config = configuration.load_global_config(self.armory_config)
            for k in (
                "dataset_dir",
                "local_git_dir",
                "saved_model_dir",
                # pytorch_dir should not be found in the config file
                # because it is not configurable (yet)
                "output_dir",
                "tmp_dir",
            ):
                setattr(self, k, config[k])
        else:
            warnings.warn(f"No {self.armory_config} file. Using default paths.")
            warnings.warn("Please run `armory configure`")

        os.makedirs(self.dataset_dir, exist_ok=True)
        os.makedirs(self.local_git_dir, exist_ok=True)
        os.makedirs(self.saved_model_dir, exist_ok=True)
        os.makedirs(self.pytorch_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
