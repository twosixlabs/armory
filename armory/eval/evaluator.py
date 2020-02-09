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
from armory.utils import external_repo
from armory.utils.printing import bold, red


logger = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(
        self, config: dict, tmp_dir="tmp", container_config_name="eval-config.json"
    ):
        self.extra_env_vars = None
        self.config = config
        self._verify_config()
        self.tmp_dir = tmp_dir
        self.tmp_config = os.path.join(self.tmp_dir, container_config_name)
        self.unix_config_path = Path(self.tmp_config).as_posix()

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
        external_repo.download_and_extract_repo(self.config["external_github_repo"])

    def _download_private(self):
        external_repo.download_and_extract_repo("twosixlabs/armory-private")
        self.extra_env_vars = {
            "ARMORY_PRIVATE_S3_ID": os.getenv("ARMORY_PRIVATE_S3_ID"),
            "ARMORY_PRIVATE_S3_KEY": os.getenv("ARMORY_PRIVATE_S3_KEY"),
        }

    def _write_tmp(self):
        os.makedirs(self.tmp_dir, exist_ok=True)
        if os.path.exists(self.tmp_config):
            logger.warning(f"Overwriting {self.tmp_config}!")
        with open(self.tmp_config, "w") as f:
            json.dump(self.config, f)

    def _delete_tmp(self):
        if os.path.exists(external_repo.DIRPATH):
            try:
                shutil.rmtree(external_repo.DIRPATH)
            except OSError as e:
                if not isinstance(e, FileNotFoundError):
                    logger.exception(
                        f"Error removing external repo {external_repo.DIRPATH}"
                    )
        try:
            os.remove(self.tmp_config)
        except OSError as e:
            if not isinstance(e, FileNotFoundError):
                logger.exception(f"Error removing tmp config {self.tmp_config}")

    def run(self, interactive=False, jupyter=False, host_port=8888) -> None:
        container_port = 8888
        self._write_tmp()
        ports = {container_port: host_port} if jupyter else None
        try:
            runner = self.manager.start_armory_instance(
                envs=self.extra_env_vars, ports=ports
            )
            try:
                if jupyter:
                    self._run_jupyter(runner, host_port=host_port)
                elif interactive:
                    self._run_interactive_bash(runner)
                else:
                    self._run_config(runner)
            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt caught")
            finally:
                logger.warning("Shutting down container")
                self.manager.stop_armory_instance(runner)
        except requests.exceptions.RequestException as e:
            logger.exception("Starting instance failed.")
            if str(e).endswith(
                f'Bind for 0.0.0.0:{host_port} failed: port is already allocated")'
            ):
                logger.error(
                    f"Port {host_port} already in use. Try a different one with '--port <port>'"
                )
            elif (
                isinstance(e, docker.errors.APIError)
                and str(e)
                == r'400 Client Error: Bad Request ("Unknown runtime specified nvidia")'
                and self.config.get("use_gpu")
            ):
                logger.error('nvidia runtime failed. Set config "use_gpu" to false')
            else:
                logger.error("Is Docker Daemon running?")
        self._delete_tmp()

    def _run_config(self, runner) -> None:
        logger.info(bold(red("Running evaluation script")))
        runner.exec_cmd(f"python -m {self.config['eval_file']} {self.unix_config_path}")

    def _run_interactive_bash(self, runner) -> None:
        lines = [
            "Container ready for interactive use.",
            bold(
                "*** In a new terminal, run the following to attach to the container:"
            ),
            bold(red(f"    docker exec -itu0 {runner.docker_container.short_id} bash")),
            bold("*** To run your script in the container:"),
            bold(
                red(f"    python -m {self.config['eval_file']} {self.unix_config_path}")
            ),
            bold("*** To gracefully shut down container, press: Ctrl-C"),
            "",
        ]
        logger.info("\n".join(lines))
        while True:
            time.sleep(1)

    def _run_jupyter(self, runner, host_port=8888) -> None:
        lines = [
            "About to launch jupyter.",
            bold("*** To connect to jupyter, please open the following in a browser:"),
            bold(red(f"    http://127.0.0.1:{host_port}")),
            bold("*** To connect on the command line as well, in a new terminal, run:"),
            bold(f"    docker exec -itu0 {runner.docker_container.short_id} bash"),
            bold("*** To gracefully shut down container, press: Ctrl-C"),
            "",
            "Jupyter notebook log:",
        ]
        logger.info("\n".join(lines))
        runner.exec_cmd(
            "jupyter notebook --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''",
            user="root",
        )
