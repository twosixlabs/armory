import abc
import argparse
import json
import logging
import os
import time

import coloredlogs

import armory
from armory import paths
from armory.utils import config_loading


logger = logging.getLogger(__name__)


class Scenario(abc.ABC):
    def __init__(self):
        pass

    def validate_config(self, config):
        """
        Validate the scenario config
        """

    def evaluate(self, config: dict):
        """
        Evaluate a config for robustness against attack.
        """
        self.validate_config(config)
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

        filename = f"classifier_extended_{int(time.time())}.json"
        logger.info(f"Saving evaluation results saved to <output_dir>/{filename}")
        with open(os.path.join(paths.docker().output_dir, filename), "w") as f:
            output_dict = {
                "armory_version": armory.__version__,
                "config": config,
                "results": results,
            }
            print(output_dict)
            f.write(json.dumps(output_dict, sort_keys=True, indent=4) + "\n")


def parse_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config


def run_config(config_path):
    config = parse_config(config_path)
    scenario_config = config.get("scenario")
    if scenario_config is None:
        raise KeyError('"scenario" missing from evaluation config')
    scenario = config_loading.load(scenario_config)
    # TODO: fix this to work properly:
    #    if not isinstance(scenario, Scenario):
    #        raise TypeError(f"scenario {scenario} is not an instance of {Scenario}")
    scenario.evaluate(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="scenario", description="run armory scenario")
    parser.add_argument(
        "config_path",
        metavar="<config path>",
        type=str,
        help="system filepath to scenario config JSON",
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
    args = parser.parse_args()
    coloredlogs.install(level=args.log_level)
    run_config(args.config_path)
