import pathlib
from glob import glob

import jsonschema
import pytest

from armory.utils.configuration import load_config
from armory import __version__


@pytest.mark.parametrize(
    "filePath, error, match",
    [
        ("not_a_file.json", FileNotFoundError, None),
        (
            str(pathlib.Path("tests/scenarios/broken/missing_scenario.json")),
            jsonschema.ValidationError,
            r"'scenario' is a required property",
        ),
        (
            str(pathlib.Path("tests/scenarios/broken/invalid_module.json")),
            jsonschema.ValidationError,
            r"Failed validating 'pattern' in schema\[0\]\['properties'\]\['module'\]",
        ),
    ],
)
def test_bad_configs(filePath, error, match):
    with pytest.raises(error, match=match):
        load_config(filePath)


@pytest.mark.parametrize("file", glob("scenario_configs/*.json"))
def test_scenario_configs(file):
    # TODO: Why the string cast here?
    config = load_config(str(file))
    assert (
        __version__ in config["sysconfig"]["docker_image"]
    ), "Docker image does not match version in repository"


@pytest.mark.parametrize(
    "file",
    (
        glob("tests/scenarios/tf1/*.json")
        + glob("tests/scenarios/tf2/*.json")
        + glob("tests/scenarios/pytorch/*.json")
    ),
)
def test_all_examples(file):
    config = load_config(str(file))
    assert (
        __version__ in config["sysconfig"]["docker_image"]
    ), "Docker image does not match version in repository"
