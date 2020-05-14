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

import abc
import base64
import argparse
import json
import logging
import os
import time

import coloredlogs

import armory
from armory import paths
from armory.utils import config_loading
from armory.utils import external_repo
from armory.utils.configuration import load_config


logger = logging.getLogger(__name__)


class Scenario(abc.ABC):
    def evaluate(self, config: dict):
        """
        Evaluate a config for robustness against attack.
        """
        results = self._evaluate(config)
        if results is None:
            logger.warning(f"{self._evaluate} returned None, not a dict")
        self.save(config, results)

    @abc.abstractmethod
    def _evaluate(self, config: dict) -> dict:
        """
        Evaluate the config and return a results dict
        """
        raise NotImplementedError

    def save(self, config: dict, results: dict, adv_examples=None):
        """
        Saves a results json-formattable output to file

        adv_examples are (optional) instances of the actual examples used.
            It will be saved in a binary format.
        """
        if adv_examples is not None:
            raise NotImplementedError("saving adversarial examples")

        runtime_paths = paths.runtime_paths()
        scenario_output_dir = os.path.join(runtime_paths.output_dir, config["eval_id"])

        scenario_name = config["scenario"]["name"]
        timestamp = int(time.time())
        filename = f"{scenario_name}_{timestamp}.json"
        logger.info(f"Saving evaluation results saved to <output_dir>/{filename}")
        with open(os.path.join(scenario_output_dir, filename), "w") as f:
            output_dict = {
                "armory_version": armory.__version__,
                "config": config,
                "results": results,
                "timestamp": timestamp,
            }
            f.write(json.dumps(output_dict, sort_keys=True, indent=4) + "\n")


def parse_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config


def _scenario_setup(config: dict):
    """
    Creates scenario specific tmp and output directiories.

    Also pulls external repositories ahead of running the scenario in case the scenario
    itself is found in the external repository.
    """

    runtime_paths = paths.runtime_paths()

    scenario_output_dir = os.path.join(runtime_paths.output_dir, config["eval_id"])
    scenario_tmp_dir = os.path.join(runtime_paths.tmp_dir, config["eval_id"])
    os.makedirs(scenario_output_dir, exist_ok=True)
    os.makedirs(scenario_tmp_dir, exist_ok=True)
    logger.warning(f"Outputs will be written to {scenario_output_dir}")

    # Download any external repositories and add them to the sys path for use
    if config["sysconfig"].get("external_github_repo", None):
        external_repo_dir = os.path.join(scenario_tmp_dir, "external")
        external_repo.download_and_extract_repo(
            config["sysconfig"]["external_github_repo"],
            external_repo_dir=external_repo_dir,
        )


def run_config(config_json, from_file=False):
    if from_file:
        config = load_config(config_json)
    else:
        config_base64_bytes = config_json.encode("utf-8")
        config_b64_bytes = base64.b64decode(config_base64_bytes)
        config_string = config_b64_bytes.decode("utf-8")
        config = json.loads(config_string)
    scenario_config = config.get("scenario")
    if scenario_config is None:
        raise KeyError('"scenario" missing from evaluation config')
    _scenario_setup(config)
    scenario = config_loading.load(scenario_config)
    scenario.evaluate(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="scenario", description="run armory scenario")
    parser.add_argument(
        "config", metavar="<config json>", type=str, help="scenario config JSON",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="log_level",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
        help="Debug output (logging=DEBUG)",
    )
    parser.add_argument(
        "--no-docker",
        dest="no_docker",
        action="store_const",
        const=True,
        default=False,
        help="Whether to use Docker or a local environment with armory run",
    )
    parser.add_argument(
        "--load-config-from-file",
        dest="from_file",
        action="store_const",
        const=True,
        default=False,
        help="If the config argument is a path instead of serialized JSON",
    )
    args = parser.parse_args()
    coloredlogs.install(level=args.log_level)
    paths.NO_DOCKER = args.no_docker
    run_config(args.config, args.from_file)
