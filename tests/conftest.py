import os

import pytest
import docker
from armory import paths
import logging
from docker.errors import ImageNotFound

logger = logging.getLogger(__name__)

REQUIRED_DOCKER_IMAGES = [
    "twosixarmory/armory-base:dev",
    "twosixarmory/armory-pytorch:dev",
    "twosixarmory/armory-tf2:dev",
]


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


@pytest.fixture(scope="session")
def docker_client():
    try:
        client = docker.from_env()
        logger.info("Docker Client Established...")
    except Exception as e:
        logger.error("Docker Server is not running!!")
        raise e

    for img in REQUIRED_DOCKER_IMAGES:
        try:
            client.images.get(name=img)
        except ImageNotFound:
            logger.error("Could not find Image: {}".format(img))
            raise

    return client
