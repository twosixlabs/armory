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
from typing import Optional

import coloredlogs
import pymongo
import pymongo.errors

import armory
from armory import paths
from armory.utils import config_loading
from armory.utils import external_repo
from armory.utils.configuration import load_config


logger = logging.getLogger(__name__)


MONGO_PORT = 27017
MONGO_DATABASE = "armory"
MONGO_COLLECTION = "scenario_results"


class Scenario(abc.ABC):
    def __init__(self):
        self.check_run = False

    def evaluate(self, config: dict, mongo_host: Optional[str]):
        """
        Evaluate a config for robustness against attack.
        """
        if self.check_run:
            # Modify dataset entries
            config["dataset"]["check_run"] = True
            if config["model"]["fit"]:
                config["model"]["fit_kwargs"]["nb_epochs"] = 1
            if config.get("attack", {}).get("type") == "preloaded":
                config["attack"]["check_run"] = True

        results = self._evaluate(config)
        if results is None:
            logger.warning(f"{self._evaluate} returned None, not a dict")
        output = self._prepare_results(config, results)
        self._save(output)
        if mongo_host is not None:
            self._send_to_mongo(mongo_host, output)

    def set_check_run(self, check_run):
        """
        Set whether to check_run if the code runs (instead of a full evaluation)
        """
        self.check_run = bool(check_run)

    @abc.abstractmethod
    def _evaluate(self, config: dict) -> dict:
        """
        Evaluate the config and return a results dict
        """
        raise NotImplementedError

    def _prepare_results(self, config: dict, results: dict, adv_examples=None) -> dict:
        """
        Build the JSON results blob for _save() and _send_to_mongo()

        adv_examples are (optional) instances of the actual examples used.
            They will be saved in a binary format.
        """
        if adv_examples is not None:
            raise NotImplementedError("saving adversarial examples")

        timestamp = int(time.time())
        output = {
            "armory_version": armory.__version__,
            "config": config,
            "results": results,
            "timestamp": timestamp,
        }
        return output

    def _save(self, output: dict):
        """
        Save json-formattable output to a file
        """

        runtime_paths = paths.runtime_paths()
        scenario_output_dir = os.path.join(
            runtime_paths.output_dir, output["config"]["eval_id"]
        )

        override_name = output["config"]["sysconfig"].get("output_filename", None)
        scenario_name = (
            override_name if override_name else output["config"]["scenario"]["name"]
        )
        filename = f"{scenario_name}_{output['timestamp']}.json"
        logger.info(f"Saving evaluation results saved to <output_dir>/{filename}")
        with open(os.path.join(scenario_output_dir, filename), "w") as f:
            f.write(json.dumps(output, sort_keys=True, indent=4) + "\n")

    def _send_to_mongo(self, mongo_host: str, output: dict):
        """
        Send results to a Mongo database at mongo_host
        """
        client = pymongo.MongoClient(mongo_host, MONGO_PORT)
        db = client[MONGO_DATABASE]
        col = db[MONGO_COLLECTION]
        logger.info(
            f"Sending evaluation results to MongoDB instance {mongo_host}:{MONGO_PORT}"
        )
        try:
            col.insert_one(output)
        except pymongo.errors.PyMongoError as e:
            logger.error(f"Encountered error {e} sending evaluation results to MongoDB")


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
        external_repo.download_and_extract_repos(
            config["sysconfig"]["external_github_repo"],
            external_repo_dir=external_repo_dir,
        )


def run_config(config_json, from_file=False, check=False, mongo_host=None):
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
    scenario.set_check_run(check)
    scenario.evaluate(config, mongo_host)


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
        action="store_true",
        help="Whether to use Docker or a local environment with armory run",
    )
    parser.add_argument(
        "--load-config-from-file",
        dest="from_file",
        action="store_true",
        help="If the config argument is a path instead of serialized JSON",
    )
    parser.add_argument(
        "--mongo",
        dest="mongo_host",
        default=None,
        help="Send scenario results to a MongoDB instance at the given host (eg 'localhost', '1.2.3.4', 'mongodb://USER:PASS@5.6.7.8')",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Whether to quickly check to see if scenario code runs",
    )
    args = parser.parse_args()
    coloredlogs.install(level=args.log_level)
    if args.no_docker:
        paths.set_mode("host")

    run_config(args.config, args.from_file, args.check, args.mongo_host)
