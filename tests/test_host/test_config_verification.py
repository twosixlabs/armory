import pathlib
from glob import glob

import jsonschema
import pytest

from armory.utils.configuration import load_config


def test_no_config():
    with pytest.raises(FileNotFoundError):
        load_config("not_a_file.json")


def test_no_scenario():
    with pytest.raises(
        jsonschema.ValidationError, match=r"'scenario' is a required property",
    ):
        load_config(str(pathlib.Path("tests/scenarios/broken/missing_scenario.json")))


def test_invalid_module():
    with pytest.raises(
        jsonschema.ValidationError,
        match=r"Failed validating 'pattern' in schema\[0\]\['properties'\]\['module'\]",
    ):
        load_config(str(pathlib.Path("tests/scenarios/broken/invalid_module.json")))


def test_scenario_configs():
    scenario_jsons = glob("scenario_configs/*.json")

    for json_path in scenario_jsons:
        load_config(str(json_path))


def test_all_examples():
    test_jsons = (
        glob("tests/scenarios/tf1/*.json")
        + glob("tests/scenarios/tf2/*.json")
        + glob("tests/scenarios/pytorch/*.json")
    )
    for json_path in test_jsons:
        load_config(str(json_path))
