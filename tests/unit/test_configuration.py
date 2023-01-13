import argparse
from glob import glob
import pathlib

import jsonschema
import pytest

from armory import arguments
from armory.utils.configuration import load_config

# Mark all tests in this file as `unit`
pytestmark = pytest.mark.unit


# TODO Refactor this pattern with Matt's new Config
def test_config_args_merge():
    config = dict(
        sysconfig={
            "output_dir": None,
            "output_filename": "file.out",
            "num_eval_batches": 2,
            "skip_misclassified": True,
        }
    )
    args = argparse.Namespace(
        num_eval_batches=5,
        skip_misclassified=False,
        output_dir="output-dir",
        check=True,
        skip_attack=False,
    )

    (config, args) = arguments.merge_config_and_args(config, args)

    sysconfig = config["sysconfig"]
    assert sysconfig["output_dir"] == "output-dir"
    assert sysconfig["output_filename"] == "file.out"
    assert sysconfig["num_eval_batches"] == 5
    assert sysconfig["skip_misclassified"]
    assert sysconfig["check"]
    assert "skip_attack" not in sysconfig

    assert args.output_dir == "output-dir"
    assert args.output_filename == "file.out"
    assert args.num_eval_batches == 5
    assert args.skip_misclassified
    assert args.check
    assert args.skip_attack is False


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
    # TODO: Fix with refactor of config tin bit
    load_config(str(file))


@pytest.mark.parametrize("file", glob("tests/scenarios/pytorch/*.json"))
def test_all_examples(file):
    load_config(str(file))
