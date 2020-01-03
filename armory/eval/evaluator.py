"""
Evaluators control launching of ARMORY evaluations.
"""
import os
import json
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
        tmp_config_fn = "tmp-eval-config.json"
        with open(tmp_config_fn, "w") as fp:
            json.dump(self.config, fp)

        try:
            runner = self.manager.start_armory_instance()
        except Exception as e:
            logger.exception(e)
            logger.exception("Starting instance failed. Is Docker Daemon running?")
            return

        try:
            logger.info("Running Evaluation...")
            runner.exec_cmd(f"python -m {self.config['eval_type']} {tmp_config_fn}",)
        except KeyboardInterrupt:
            logger.warning("Evaluation interrupted by user. Stopping container.")
        finally:
            os.remove(tmp_config_fn)
            self.manager.stop_armory_instance(runner)
