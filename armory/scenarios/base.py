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
import sys
import pytest
import importlib.resources

import coloredlogs
import pymongo
import pymongo.errors
import re

import armory
from armory import paths
from armory import environment
from armory.utils import config_loading
from armory.utils import external_repo
from armory.utils.configuration import load_config
from armory.scenarios import END_SENTINEL
from armory import validation


logger = logging.getLogger(__name__)


MONGO_PORT = 27017
MONGO_DATABASE = "armory"
MONGO_COLLECTION = "scenario_results"


class Scenario(abc.ABC):
    def __init__(self):
        self.check_run = False
        self.scenario_output_dir = None

    def evaluate(
        self,
        config: dict,
        mongo_host: Optional[str],
        num_eval_batches: Optional[int],
        skip_benign: Optional[bool],
        skip_attack: Optional[bool],
        skip_misclassified: Optional[bool],
    ):
        """
        Evaluate a config for robustness against attack.
        """
        self._set_output_dir(config)
        if self.check_run:
            # Modify dataset entries
            config["dataset"]["check_run"] = True
            if config["model"]["fit"]:
                config["model"]["fit_kwargs"]["nb_epochs"] = 1
            if config.get("attack", {}).get("type") == "preloaded":
                config["attack"]["check_run"] = True
            # For poisoning scenario
            if config.get("adhoc") and config.get("adhoc").get("train_epochs"):
                config["adhoc"]["train_epochs"] = 1

        try:
            self._check_config_and_cli_args(
                config, num_eval_batches, skip_benign, skip_attack, skip_misclassified
            )
            results = self._evaluate(
                config, num_eval_batches, skip_benign, skip_attack, skip_misclassified
            )
        except Exception as e:
            if str(e) == "assignment destination is read-only":
                logger.exception(
                    "Encountered error during scenario evaluation. Be sure "
                    + "that the classifier's predict() isn't directly modifying the "
                    + "input variable itself, as this can cause unexpected behavior in ART."
                )
            else:
                logger.exception("Encountered error during scenario evaluation.")
            sys.exit(1)

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
    def _evaluate(
        self,
        config: dict,
        num_eval_batches: Optional[int],
        skip_benign: Optional[bool],
        skip_attack: Optional[bool],
        skip_misclassified: Optional[bool],
    ) -> dict:
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

        override_name = output["config"]["sysconfig"].get("output_filename", None)
        scenario_name = (
            override_name if override_name else output["config"]["scenario"]["name"]
        )
        filename = f"{scenario_name}_{output['timestamp']}.json"
        logger.info(
            f"Saving evaluation results to {self.scenario_output_dir}/{filename} path "
            f"inside container."
        )
        with open(os.path.join(self.scenario_output_dir, filename), "w") as f:
            f.write(json.dumps(output, sort_keys=True, indent=4) + "\n")

    def _send_to_mongo(self, mongo_host: str, output: dict):
        """
        Send results to a Mongo database at mongo_host
        """
        client = pymongo.MongoClient(mongo_host, MONGO_PORT)
        db = client[MONGO_DATABASE]
        col = db[MONGO_COLLECTION]
        # strip user/pass off of mongodb url for logging
        tail_of_host = re.findall(r"@([^@]*$)", mongo_host)
        if len(tail_of_host) > 0:
            mongo_ip = tail_of_host[0]
        else:
            mongo_ip = mongo_host
        logger.info(
            f"Sending evaluation results to MongoDB instance {mongo_ip}:{MONGO_PORT}"
        )
        try:
            col.insert_one(output)
        except pymongo.errors.PyMongoError as e:
            logger.error(f"Encountered error {e} sending evaluation results to MongoDB")

    def _set_output_dir(self, config):
        runtime_paths = paths.runtime_paths()
        self.scenario_output_dir = os.path.join(
            runtime_paths.output_dir, config["eval_id"]
        )

    def _check_config_and_cli_args(
        self, config, num_eval_batches, skip_benign, skip_attack, skip_misclassified
    ):
        if skip_misclassified:
            if skip_attack or skip_benign:
                raise ValueError(
                    "Cannot pass skip_misclassified if skip_benign or skip_attack is also passed"
                )
            elif "categorical_accuracy" not in config["metric"].get("task"):
                raise ValueError(
                    "Cannot pass skip_misclassified if 'categorical_accuracy' metric isn't enabled"
                )
            elif config["dataset"].get("batch_size") != 1:
                raise ValueError(
                    "To enable skip_misclassified, 'batch_size' must be set to 1"
                )


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


def _get_config(config_json, from_file=False):
    if from_file:
        config = load_config(config_json)
    else:
        config_base64_bytes = config_json.encode("utf-8")
        config_b64_bytes = base64.b64decode(config_base64_bytes)
        config_string = config_b64_bytes.decode("utf-8")
        config = json.loads(config_string)
    return config


def run_validation(
    config_json, from_file=False,
):
    config = _get_config(config_json, from_file=from_file)
    _scenario_setup(config)
    model_config = config.get("model")
    model_config = json.dumps(model_config)
    test_path_context = importlib.resources.path(validation, "test_config")
    with test_path_context as test_path:
        return_val = pytest.main(["-x", str(test_path), "--model-config", model_config])
    assert return_val == pytest.ExitCode.OK, "Model configuration validation failed"


def run_config(
    config_json,
    from_file=False,
    check=False,
    mongo_host=None,
    num_eval_batches=None,
    skip_benign=None,
    skip_attack=None,
    skip_misclassified=None,
):
    config = _get_config(config_json, from_file=from_file)
    scenario_config = config.get("scenario")
    if scenario_config is None:
        raise KeyError('"scenario" missing from evaluation config')
    _scenario_setup(config)
    scenario = config_loading.load(scenario_config)
    scenario.set_check_run(check)
    scenario.evaluate(
        config,
        mongo_host,
        num_eval_batches,
        skip_benign,
        skip_attack,
        skip_misclassified,
    )


def init_interactive(config_json, from_file=True):
    """
    Init environment variables from config to setup environment for interactive use.
    """
    coloredlogs.install(level=logging.INFO)
    config = _get_config(config_json, from_file=from_file)
    _scenario_setup(config)
    return config


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
        help="Send scenario results to a MongoDB instance at the given host (eg mongodb://USER:PASS@5.6.7.8')",
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
    coloredlogs.install(level=args.log_level)
    calling_version = os.getenv(environment.ARMORY_VERSION, "UNKNOWN")
    if calling_version != armory.__version__:
        logger.warning(
            f"armory calling version {calling_version} != "
            f"armory imported version {armory.__version__}"
        )

    if args.no_docker:
        paths.set_mode("host")

    if args.check and args.num_eval_batches:
        logger.warning(
            "--num_eval_batches will be overwritten and set to 1 since --check was passed"
        )

    if args.validate_config:
        run_validation(
            args.config, args.from_file,
        )
    else:
        run_config(
            args.config,
            args.from_file,
            args.check,
            args.mongo_host,
            args.num_eval_batches,
            args.skip_benign,
            args.skip_attack,
            args.skip_misclassified,
        )
    print(END_SENTINEL)  # indicates to host that the scenario finished w/out error
