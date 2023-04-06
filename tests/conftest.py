import logging
import os

import pytest
import requests

from armory import __version__, paths
import docker
from docker.errors import ImageNotFound

logger = logging.getLogger(__name__)

REQUIRED_DOCKER_IMAGES = [
    f"twosixarmory/armory:{__version__}",
    f"twosixarmory/pytorch-deepspeech:{__version__}",
]


armory_test_parameters = (
    (
        ("--armory-mode"),
        {
            "choices": ["native", "docker", "both"],
            "default": "docker",
            "help": "Set Armory Mode [native|docker|both]",
            "required": False,
        },
    ),
    (
        ("--scenario-path"),
        {
            "default": "None",
            "help": "Set path to scenario config file",
            "required": False,
        },
    ),
    (
        ("--github-ci"),
        {
            "action": "store_true",
            "help": "Running in a GitHub CI environment",
        },
    ),
)


# Added this to make run local
paths.set_mode("host")


@pytest.fixture()
def ensure_armory_dirs(request):
    """
    CI doesn't mount volumes
    """
    # Changing this to make more appropriate
    # docker_paths = paths.DockerPaths()
    docker_paths = paths.runtime_paths()
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


@pytest.fixture(scope="session")
@pytest.mark.docker_required
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


def pytest_addoption(parser):
    for args, kwargs in armory_test_parameters:
        args = args if isinstance(args, tuple) else (args,)
        parser.addoption(*args, **kwargs)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_test_if_armory_mode(amory-mode): this mark skips the tests for the given armory mode",
    )


def pytest_runtest_setup(item):
    # Setting up for `--armory-mode`
    parameters = [
        mark.args[0] for mark in item.iter_markers(name="skip_test_if_armory_mode")
    ]
    if parameters:
        if item.config.getoption("--armory-mode") in parameters:
            pytest.skip("Test skipped because armory-mode is {!r}".format(parameters))


@pytest.fixture()
def pass_parameters(request):
    setattr(request.cls, "config", request.config)

    for args, kwargs in armory_test_parameters:
        param = args.replace("--", "").replace("-", "_")
        setattr(request.cls, param, request.config.getoption(param))


@pytest.fixture()
def external_resources():
    try:
        requests.get("https://www.google.com/").status_code
    except Exception as e:
        logger.error("Cannot Reach External Resources")
        raise e


@pytest.fixture
def armory_dataset_dir():
    from armory import paths

    paths.set_mode("host")
    return paths.runtime_paths().dataset_dir


@pytest.fixture
def dataset_generator():
    import tensorflow as tf
    import torch

    from armory.data import datasets
    from armory.data.datasets import ArmoryDataGenerator

    def generator(
        name,
        batch_size,
        num_epochs,
        split,
        framework,
        dataset_dir,
        shuffle_files=False,
        preprocessing_fn=None,
    ):
        # Instance types based on framework
        instance_types = {
            "pytorch": torch.utils.data.DataLoader,
            "tf": (tf.compat.v2.data.Dataset, tf.compat.v1.data.Dataset),
            "numpy": ArmoryDataGenerator,
        }

        if framework not in instance_types:
            raise Exception(
                "Unrecognized Armory Dataset Framework: {}".format(framework)
            )

        ds = getattr(datasets, name)
        dataset = ds(
            split=split,
            batch_size=batch_size,
            epochs=num_epochs,
            dataset_dir=dataset_dir,
            framework=framework,
            shuffle_files=shuffle_files,
            preprocessing_fn=preprocessing_fn,
        )
        print(type(dataset))
        assert isinstance(dataset, instance_types[framework])
        if framework == "numpy":
            assert dataset.batch_size == batch_size
            assert dataset.batches_per_epoch == (
                dataset.size // batch_size + bool(dataset.size % batch_size)
            )
        return dataset

    return generator


# override caplog fixture to receve loguru messages as described in
# https://loguru.readthedocs.io/en/stable/resources/migration.html#making-things-work-with-pytest-and-caplog

from _pytest.logging import LogCaptureFixture
from loguru import logger as loguru_logger

# It is good that this fixture is implemented as a generator function so
# that we don't have to worry about the armory.logs sinks added or removed.
# This fixture adds and removes it's own sink only for the duration of the
# pytest function. If we find that messages disappear after this fixture is
# used, we might have to be more delicate in setting up the other sinks.


@pytest.fixture
def caplog(caplog: LogCaptureFixture):
    handler_id = loguru_logger.add(caplog.handler, format="{message}")
    yield caplog
    loguru_logger.remove(handler_id)
