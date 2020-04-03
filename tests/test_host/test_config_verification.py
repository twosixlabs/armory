import pathlib

import jsonschema
import pytest

from armory.utils.configuration import load_config


def test_no_config():
    with pytest.raises(FileNotFoundError):
        load_config("not_a_file.txt")


def test_no_evaluation():
    with pytest.raises(
        jsonschema.ValidationError,
        match=r"Failed validating 'type' in schema\['properties'\]\['evaluation'\]",
    ):
        load_config(str(pathlib.Path("tests/configs/broken/missing_eval.json")))


def test_invalid_module():
    with pytest.raises(
        jsonschema.ValidationError,
        match=r"Failed validating 'pattern' in schema\[0\]\['properties'\]\['module'\]",
    ):
        load_config(str(pathlib.Path("tests/configs/broken/invalid_module.json")))


def test_all_examples():
    example_dir = pathlib.Path("tests/configs/")

    for json_path in example_dir.glob("*.json"):
        load_config(str(json_path))
