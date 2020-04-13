import pathlib
from glob import glob

import pytest

from armory.utils.configuration import load_config


def test_no_config():
    with pytest.raises(FileNotFoundError):
        load_config("not_a_file.json")


def test_invalid_config():
    with pytest.raises(KeyError, match="sysconfig"):
        load_config(str(pathlib.Path("tests/scenarios/broken/missing_eval.json")))


def test_all_examples():
    test_jsons = (
        glob("tests/scenarios/tf1/*.json")
        + glob("tests/scenarios/tf2/*.json")
        + glob("tests/scenarios/pytorch/*.json")
    )
    for json_path in test_jsons:
        load_config(str(json_path))
