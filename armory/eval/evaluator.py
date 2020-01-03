"""
Evaluators control launching of ARMORY evaluations.
"""
import os
import json
import requests

from armory.webapi.data import SUPPORTED_DATASETS
from armory.docker.management import ManagementInstance

import logging

logger = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(self, config: dict):
        self.config = config
        self._verify_config()
        self.manager = ManagementInstance()

    def _verify_config(self) -> None:
        assert isinstance(self.config, dict)

        if self.config["data"] not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Configured data {self.config['data']} not found in"
                f" supported datasets: {list(SUPPORTED_DATASETS.keys())}"
            )

    def run_config(self) -> None:
        tmp_dir = "tmp"
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        tmp_config = os.path.join(tmp_dir, "eval-config.json")
        with open(tmp_config, "w") as fp:
            json.dump(self.config, fp)

        try:
            runner = self.manager.start_armory_instance()
        except requests.exceptions.ConnectionError:
            logger.exception("Starting instance failed. Is Docker Daemon running?")
            return

        try:
            logger.info("Running Evaluation...")
            runner.docker_container.exec_run(
                f"python -m armory.eval.classification {tmp_config}",
                stdout=True,
                stderr=True,
            )
        except KeyboardInterrupt:
            logger.warning("Evaluation interrupted by user. Stopping container.")
        finally:
            os.remove(tmp_config)
            self.manager.stop_armory_instance(runner)
