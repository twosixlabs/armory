"""
Evaluators control launching of ARMORY evaluations.
"""

import os
import json
import logging
import shutil
from pathlib import Path

import requests

from armory.webapi.common import SUPPORTED_DATASETS
from armory.docker.management import ManagementInstance
from armory.utils.external_repo import download_and_extract


logger = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(self, config: dict):
        self.config = config
        self._verify_config()

        runtime = "runc"
        if "use_gpu" in self.config.keys():
            if self.config["use_gpu"]:
                runtime = "nvidia"

        self.manager = ManagementInstance(runtime=runtime)

        if self.config["external_github_repo"]:
            self._download_external()

    def _verify_config(self) -> None:
        assert isinstance(self.config, dict)

        if self.config["data"] not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Configured data {self.config['data']} not found in"
                f" supported datasets: {list(SUPPORTED_DATASETS.keys())}"
            )

    def _download_external(self):
        download_and_extract(self.config)

    def run_config(self) -> None:
        tmp_dir = "tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_config = os.path.join(tmp_dir, "eval-config.json")
        with open(tmp_config, "w") as fp:
            json.dump(self.config, fp)

        try:
            runner = self.manager.start_armory_instance()
        except requests.exceptions.RequestException:
            logger.exception("Starting instance failed. Is Docker Daemon running?")
            return

        try:
            logger.info("Running Evaluation...")
            unix_config_path = Path(tmp_config).as_posix()
            runner.exec_cmd(f"python -m {self.config['eval_file']} {unix_config_path}")
        except KeyboardInterrupt:
            logger.warning("Evaluation interrupted by user. Stopping container.")
        finally:
            if os.path.exists("external_repos"):
                shutil.rmtree("external_repos")
            os.remove(tmp_config)
            self.manager.stop_armory_instance(runner)
