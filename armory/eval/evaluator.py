"""
Evaluators control launching of ARMORY evaluations.
"""
import os
import json
from armory.webapi.data import SUPPORTED_DATASETS
from armory.docker.management import ManagementInstance

import logging
import coloredlogs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
coloredlogs.install()


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
        tmp_config = "eval-config.json"
        with open(tmp_config, "w") as fp:
            json.dump(self.config, fp)

        try:
            runner = self.manager.start_armory_instance()
        except Exception:
            logger.exception("Starting instance failed. Is Docker Daemon running?")
            return

        try:
            logger.info("Running Evaluation...")
            runner.docker_container.exec_run(
                f"python -m armory.eval.classification {tmp_config}",
                stdout=True,
                stderr=True,
            )
            logger.info(
                "Evaluation Results written to `outputs/evaluation-results.json"
            )
        except KeyboardInterrupt:
            logger.warning("Evaluation interrupted by user. Stopping container.")
        finally:
            os.remove("eval-config.json")
            self.manager.stop_armory_instance(runner)
