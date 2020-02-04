"""
Evaluators control launching of ARMORY evaluations.
"""

import os
import json
import logging
import shutil
import time
from pathlib import Path

import docker
import requests

from armory.data.common import SUPPORTED_DATASETS
from armory.docker.management import ManagementInstance
from armory.utils.external_repo import download_and_extract_repo


logger = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(self, config: dict):
        self.extra_env_vars = None
        self.config = config
        self._verify_config()

        runtime = "runc"
        if "use_gpu" in self.config.keys():
            if self.config["use_gpu"]:
                runtime = "nvidia"

        if "external_github_repo" in self.config.keys():
            if self.config["external_github_repo"]:
                self._download_external()

        if "use_armory_private" in self.config.keys():
            if self.config["use_armory_private"]:
                self._download_private()

        self.manager = ManagementInstance(runtime=runtime)

    def _verify_config(self) -> None:
        assert isinstance(self.config, dict)

        if self.config["data"] not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Configured data {self.config['data']} not found in"
                f" supported datasets: {list(SUPPORTED_DATASETS.keys())}"
            )

    def _download_external(self):
        download_and_extract_repo(self.config["external_github_repo"])

    def _download_private(self):
        download_and_extract_repo("twosixlabs/armory-private")
        self.extra_env_vars = {
            "ARMORY_PRIVATE_S3_ID": os.getenv("ARMORY_PRIVATE_S3_ID"),
            "ARMORY_PRIVATE_S3_KEY": os.getenv("ARMORY_PRIVATE_S3_KEY"),
        }

    def run_config(self) -> None:
        tmp_dir = "tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_config = os.path.join(tmp_dir, "eval-config.json")
        with open(tmp_config, "w") as fp:
            json.dump(self.config, fp)

        try:
            runner = self.manager.start_armory_instance(envs=self.extra_env_vars)
        except requests.exceptions.RequestException as e:
            logger.exception("Starting instance failed.")
            if (
                isinstance(e, docker.errors.APIError)
                and str(e)
                == r'400 Client Error: Bad Request ("Unknown runtime specified nvidia")'
                and self.config.get("use_gpu")
            ):
                logger.error('nvidia runtime failed. Set config "use_gpu" to false')
            else:
                logger.error("Is Docker Daemon running?")
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

    def run_interactive(self) -> None:
        tmp_dir = "tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_config = os.path.join(tmp_dir, "eval-config.json")
        with open(tmp_config, "w") as fp:
            json.dump(self.config, fp)

        try:
            runner = self.manager.start_armory_instance(envs=self.extra_env_vars)
        except requests.exceptions.RequestException:
            logger.exception("Starting instance failed. Is Docker Daemon running?")
            return

        try:
            unix_config_path = Path(tmp_config).as_posix()
            logger.info(
                "Container ready for interactive use.\n"
                "*** In a new terminal, run the following to attach to the container:\n"
                f"    docker exec -itu0 {runner.docker_container.short_id} bash\n"
                "*** To run your script in the container:\n"
                f"    python -m {self.config['eval_file']} {unix_config_path}\n"
                "*** To gracefully shut down container, press: Ctrl-C"
            )
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt caught")
        finally:
            logger.warning("Shutting down interactive container")
            if os.path.exists("external_repos"):
                shutil.rmtree("external_repos")
            os.remove(tmp_config)
            self.manager.stop_armory_instance(runner)
