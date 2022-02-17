import os
import pytest
from armory import paths


@pytest.fixture()
def ensure_armory_dirs(request):
    """
    CI doesn't mount volumes
    """
    docker_paths = paths.DockerPaths()
    saved_model_dir = docker_paths.saved_model_dir
    pytorch_dir = docker_paths.pytorch_dir
    dataset_dir = docker_paths.dataset_dir
    output_dir = docker_paths.output_dir

    os.makedirs(saved_model_dir, exist_ok=True)
    os.makedirs(pytorch_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


@pytest.fixture()
def scenario_configs():
    """Pointer to armory.scenario_configs file"""
    dirname = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "scenario_configs"
    )
    if not os.path.exists(dirname):
        raise Exception(
            "Something is wrong... Scenario Config Dir: {} does not exist".format(
                dirname
            )
        )
    return dirname
