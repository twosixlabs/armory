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
from docker.errors import ImageNotFound

from armory.data.common import SUPPORTED_DATASETS
from armory.docker.management import ManagementInstance
from armory.utils.external_repo import download_and_extract_repo
from armory.utils.printing import bold, red


logger = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(self, config: dict):
        self.extra_env_vars = None
        self.config = config
        self._verify_config()

        kwargs = dict(runtime="runc")
        if self.config.get("use_gpu", None):
            kwargs["runtime"] = "nvidia"

        if self.config.get("external_github_repo", None):
            self._download_external()

        if self.config.get("use_armory_private", None):
            self._download_private()

        image_name = self.config.get("docker_image")
        kwargs["image_name"] = image_name

        # Download docker image on host
        docker_client = docker.from_env()
        try:
            docker_client.images.get(kwargs["image_name"])
        except ImageNotFound:
            logger.info(f"Image {image_name} was not found. Downloading...")
            docker_client.images.pull(image_name)

        self.manager = ManagementInstance(**kwargs)

    def _verify_config(self) -> None:
        assert isinstance(self.config, dict)

        if self.config["data"] not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Configured data {self.config['data']} not found in"
                f" supported datasets: {list(SUPPORTED_DATASETS.keys())}"
            )

        if not self.config.get("docker_image"):
            raise ValueError("Configurations must have a `docker_image` specified.")

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
                "\n".join(
                    [
                        "Container ready for interactive use.",
                        bold(
                            "*** In a new terminal, run the following to attach to the container:"
                        ),
                        bold(
                            red(
                                f"    docker exec -itu0 {runner.docker_container.short_id} bash"
                            )
                        ),
                        bold("*** To run your script in the container:"),
                        bold(
                            red(
                                f"    python -m {self.config['eval_file']} {unix_config_path}"
                            )
                        ),
                        bold("*** To gracefully shut down container, press: Ctrl-C"),
                        "",
                    ]
                )
            )
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt caught")
        finally:
            logger.warning("Shutting down interactive container")
            self.manager.stop_armory_instance(runner)
        if os.path.exists("external_repos"):
            shutil.rmtree("external_repos")
        os.remove(tmp_config)

    def run_jupyter(self, host_port=8888) -> None:
        container_port = 8888
        tmp_dir = "tmp"
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_config = os.path.join(tmp_dir, "eval-config.json")
        with open(tmp_config, "w") as fp:
            json.dump(self.config, fp)

        try:
            runner = self.manager.start_armory_instance(
                envs=self.extra_env_vars, ports={container_port: host_port}
            )
        except requests.exceptions.RequestException as e:
            logger.exception("Starting instance failed.")
            if str(e).endswith(
                f'Bind for 0.0.0.0:{host_port} failed: port is already allocated")'
            ):
                logger.error(
                    f"Port {host_port} already in use. Try a different one with '--port <port>'"
                )
            else:
                logger.error("Is Docker Daemon running?")
            return

        try:
            logger.info(
                "\n".join(
                    [
                        "About to launch jupyter.",
                        bold(
                            "*** To connect to jupyter, please open the following in a browser:"
                        ),
                        bold(red(f"    http://127.0.0.1:{host_port}")),
                        bold(
                            "*** To connect on the command line as well, in a new terminal, run:"
                        ),
                        bold(
                            f"    docker exec -itu0 {runner.docker_container.short_id} bash"
                        ),
                        bold("*** To gracefully shut down container, press: Ctrl-C"),
                        "",
                        "Jupyter notebook log:",
                    ]
                )
            )
            runner.exec_cmd(
                "jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''",
                user="root",
            )
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt caught")
        finally:
            logger.warning("Shutting down jupyter container")
            if os.path.exists("external_repos"):
                shutil.rmtree("external_repos")
            os.remove(tmp_config)
            self.manager.stop_armory_instance(runner)
