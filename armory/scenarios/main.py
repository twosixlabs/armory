"""
Main script for running scenarios. Users will run a scenario by calling:
    armory run <config_file.json>

The particular scenario and configs will be picked up in the "scenario" field:
    "scenario": {
        "kwargs": {},
        "module": "armory.scenarios.cifar10",
        "name": "Cifar10"
    },

    This is used to instantiate the subclass.
"""

import argparse
import base64
import importlib.resources
import json
import os
import time

import pytest

import armory
from armory import Config, environment, paths, validation
from armory.logs import log, make_logfiles, update_filters
from armory.utils import config_loading, external_repo
from armory.utils.configuration import load_config


def _scenario_setup(config: Config) -> None:
    """
    Creates scenario specific tmp and output directiories.

    Also pulls external repositories ahead of running the scenario in case the scenario
    itself is found in the external repository.
    """

    runtime_paths = paths.runtime_paths()
    if "eval_id" not in config:
        timestamp = time.time()
        log.error(f"eval_id not in config. Inserting current timestamp {timestamp}")
        config["eval_id"] = str(timestamp)

    scenario_output_dir = os.path.join(runtime_paths.output_dir, config["eval_id"])
    scenario_tmp_dir = os.path.join(runtime_paths.tmp_dir, config["eval_id"])
    os.makedirs(scenario_output_dir, exist_ok=True)
    os.makedirs(scenario_tmp_dir, exist_ok=True)

    log.info(f"armory outputs and logs will be written to {scenario_output_dir}")
    make_logfiles(scenario_output_dir)

    # Download any external repositories and add them to the sys path for use
    if config["sysconfig"].get("external_github_repo", None):
        external_repo_dir = os.path.join(scenario_tmp_dir, "external")
        external_repo.download_and_extract_repos(
            config["sysconfig"]["external_github_repo"],
            external_repo_dir=external_repo_dir,
        )
    pythonpaths = config["sysconfig"].get("external_github_repo_pythonpath")
    if isinstance(pythonpaths, str):
        pythonpaths = [pythonpaths]
    elif pythonpaths is None:
        pythonpaths = []
    for pythonpath in pythonpaths:
        external_repo.add_pythonpath(pythonpath, external_repo_dir=external_repo_dir)
    local_paths = config["sysconfig"].get("local_repo_path")
    if isinstance(local_paths, str):
        local_paths = [local_paths]
    elif local_paths is None:
        local_paths = []
    for local_path in local_paths:
        external_repo.add_local_repo(local_path)


def _get_config(config_json, from_file=False) -> Config:
    """
    Reads a config specification from json, dedcodes it, and returns the
    resultant dict.
    """
    if from_file:
        config = load_config(config_json)
    else:
        config_base64_bytes = config_json.encode("utf-8")
        config_b64_bytes = base64.b64decode(config_base64_bytes)
        config_string = config_b64_bytes.decode("utf-8")
        config = json.loads(config_string)
    return config


def run_validation(config_json, from_file=False) -> None:
    """
    Test a configuration spec for jsonschema correctness. Fault on error.
    """
    config = _get_config(config_json, from_file=from_file)
    _scenario_setup(config)
    model_config = json.dumps(config.get("model"))
    test_path_context = importlib.resources.path(validation, "test_config")
    with test_path_context as test_path:
        return_val = pytest.main(["-x", str(test_path), "--model-config", model_config])
    assert return_val == pytest.ExitCode.OK, "Model configuration validation failed"


def get(
    config_json,
    from_file=True,
    check_run=False,
    num_eval_batches=None,
    skip_benign=None,
    skip_attack=None,
    skip_misclassified=None,
):
    """
    Init environment variables and initialize scenario class with config;
    returns a constructed Scenario subclass based on the config specification.
    """
    config = _get_config(config_json, from_file=from_file)
    scenario_config = config.get("scenario")
    if scenario_config is None:
        raise KeyError('"scenario" missing from evaluation config')
    _scenario_setup(config)

    ScenarioClass = config_loading.load_fn(scenario_config)
    kwargs = scenario_config.get("kwargs", {})
    kwargs.update(
        dict(
            check_run=check_run,
            num_eval_batches=num_eval_batches,
            skip_benign=skip_benign,
            skip_attack=skip_attack,
            skip_misclassified=skip_misclassified,
        )
    )
    scenario_config["kwargs"] = kwargs
    scenario = ScenarioClass(config, **kwargs)
    return scenario


def run_config(*args, **kwargs):
    """
    Convenience wrapper around 'load'
    """
    scenario = get(*args, **kwargs)
    log.trace(f"scenario loaded {scenario}")
    scenario.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="scenario", description="run armory scenario")
    parser.add_argument(
        "config",
        metavar="<config json>",
        type=str,
        help="scenario config JSON",
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="synonym for --log-level=armory:debug",
    )
    parser.add_argument(
        "--log-level",
        action="append",
        help="set log level per-module (ex. art:debug) can be used mulitple times",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Whether to use Docker or a local environment with armory run",
    )
    parser.add_argument(
        "--base64",
        dest="from_file",
        action="store_false",
        help="If the config argument is a base64 serialized JSON instead of a filepath",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Whether to quickly check to see if scenario code runs",
    )
    parser.add_argument(
        "--num-eval-batches",
        type=int,
        help="Number of batches to use for evaluation of benign and adversarial examples",
    )
    parser.add_argument(
        "--skip-benign",
        action="store_true",
        help="Skip benign inference and metric calculations",
    )
    parser.add_argument(
        "--skip-attack",
        action="store_true",
        help="Skip attack generation and metric calculations",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate model configuration against several checks",
    )
    parser.add_argument(
        "--skip-misclassified",
        action="store_true",
        help="Skip attack of inputs that are already misclassified",
    )
    args = parser.parse_args()
    update_filters(args.log_level, args.debug)
    log.trace(f"main.py called update_filters({args.log_level} debug: {args.debug})")
    calling_version = os.getenv(environment.ARMORY_VERSION, "UNKNOWN")
    if calling_version != armory.__version__:
        log.warning(
            f"armory calling version {calling_version} != "
            f"armory imported version {armory.__version__}"
        )

    if args.no_docker:
        paths.set_mode("host")

    if args.check and args.num_eval_batches:
        log.warning("--num_eval_batches will be overridden since --check was passed")
        args.num_eval_batches = None

    if args.validate_config:
        run_validation(args.config, args.from_file)
    else:
        run_config(
            args.config,
            from_file=args.from_file,
            check_run=args.check,
            num_eval_batches=args.num_eval_batches,
            skip_benign=args.skip_benign,
            skip_attack=args.skip_attack,
            skip_misclassified=args.skip_misclassified,
        )
    print(
        armory.END_SENTINEL
    )  # indicates to host that the scenario finished w/out error
