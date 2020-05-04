"""
Evaluators control launching of ARMORY evaluations.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Union

import docker
import requests
from docker.errors import ImageNotFound

from armory.docker.management import ManagementInstance
from armory.docker.host_management import HostManagementInstance
from armory.utils.configuration import load_config
from armory.utils.printing import bold, red
from armory.utils import docker_api
from armory import paths

logger = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(
        self,
        config_path: Union[str, dict],
        no_docker: bool = False,
    ):
        if isinstance(config_path, str):
            try:
                self.config = load_config(config_path)
            except json.decoder.JSONDecodeError:
                logger.error(f"Could not decode {config_path} as a json file.")
                if not config_path.lower().endswith(".json"):
                    logger.warning(f"{config_path} is not a '*.json' file")
                    logger.warning("If using `armory run`, use a json config file.")
                raise
        elif isinstance(config_path, dict):
            self.config = config_path
        else:
            raise ValueError(f"config_path {config_path} must be a str or dict")

        self.host_paths = paths.host()

        # Retrieve environment variables that should be used in evaluation
        self.extra_env_vars = dict()
        self._gather_env_variables()

        kwargs = dict(runtime="runc")
        image_name = self.config["sysconfig"].get("docker_image")
        kwargs["image_name"] = image_name
        self.no_docker = not image_name or no_docker

        if self.no_docker:
            self.docker_paths = paths.host()
            self.manager = HostManagementInstance()
            return

        self.docker_paths = paths.docker()

        # Download docker image on host
        docker_client = docker.from_env()
        try:
            docker_client.images.get(kwargs["image_name"])
        except ImageNotFound:
            logger.info(f"Image {image_name} was not found. Downloading...")
            docker_api.pull_verbose(docker_client, image_name)
        except requests.exceptions.ConnectionError:
            logger.error(f"Docker connection refused. Is Docker Daemon running?")
            raise

        self.manager = ManagementInstance(**kwargs)

    def _gather_env_variables(self):
        """
        Update the extra env variable dictionary to pass into container or run on host
        """
        self.extra_env_vars["ARMORY_GITHUB_TOKEN"] = os.getenv(
            "ARMORY_GITHUB_TOKEN", default=""
        )
        self.extra_env_vars["ARMORY_PRIVATE_S3_ID"] = os.getenv(
            "ARMORY_PRIVATE_S3_ID", default=""
        )
        self.extra_env_vars["ARMORY_PRIVATE_S3_KEY"] = os.getenv(
            "ARMORY_PRIVATE_S3_KEY", default=""
        )

        if not self.host_paths.verify_ssl:
            self.extra_env_vars["VERIFY_SSL"] = "false"

        if self.config["sysconfig"].get("use_gpu", None):
            gpus = self.config["sysconfig"].get("gpus")
            if gpus is not None:
                self.extra_env_vars["NVIDIA_VISIBLE_DEVICES"] = gpus

    def _delete_tmp_config(self):
        if os.path.exists(self.tmp_config):
            os.remove(self.tmp_config)

    def run(
        self, interactive=False, jupyter=False, host_port=8888, command=None
    ) -> None:
        if self.no_docker:
            if jupyter or interactive or command:
                raise ValueError(
                    "jupyter, interactive, or bash commands only supported when running Docker containers."
                )
            runner = self.manager.start_armory_instance(envs=self.extra_env_vars)
            try:
                self._run_config(runner)
            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt caught")
            finally:
                logger.warning("Cleaning up...")
            return

        container_port = 8888
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
                elif command:
                    self._run_command(runner, command)
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
                str(e)
                == '400 Client Error: Bad Request ("Unknown runtime specified nvidia")'
            ):
                logger.error(
                    'NVIDIA runtime failed. Either install nvidia-docker or set config "use_gpu" to false'
                )
            else:
                logger.error("Is Docker Daemon running?")
        if interactive:
            self._delete_tmp_config()

    def _escape_config_string(self):
        escaped_config = str(self.config).replace(" ", "").replace("\'", "\\\"")
        return escaped_config.replace("None", "null").replace("True", "true").replace("False", "false")

    def _run_config(self, runner) -> None:
        logger.info(bold(red("Running evaluation script")))

        if self.no_docker:
            docker_option = " --no-docker"
        else:
            docker_option = ""
        escaped_config = self._escape_config_string()
        runner.exec_cmd(f"python -m armory.scenarios.base {escaped_config}{docker_option}")

    def _run_command(self, runner, command) -> None:
        logger.info(bold(red(f"Running bash command: {command}")))
        runner.exec_cmd(command)

    def _run_interactive_bash(self, runner) -> None:
        user_id = os.getuid() if os.name != "nt" else 0
        group_id = os.getgid() if os.name != "nt" else 0
        lines = [
            "Container ready for interactive use.",
            bold(
                "*** In a new terminal, run the following to attach to the container:"
            ),
            bold(
                red(
                    f"    docker exec -it -u {user_id}:{group_id} {runner.docker_container.short_id} bash"
                )
            ),
        ]
        if self.config.get("scenario"):
            self.tmp_config = os.path.join(paths.host().tmp_dir, "interactive-config.json")
            docker_config_path = os.path.join(paths.docker().tmp_dir, "interactive-config.json")
            with open(self.tmp_config, "w") as f:
                f.write(json.dumps(self.config, sort_keys=True, indent=4) + "\n")

            lines.extend(
                [
                    bold("*** To run your scenario in the container:"),
                    bold(
                        red(
                            f"    python -m armory.scenarios.base {docker_config_path} --load-config-from-file"
                        )
                    ),
                    bold("*** To gracefully shut down container, press: Ctrl-C"),
                    "",
                ]
            )
        logger.info("\n".join(lines))
        while True:
            time.sleep(1)

    def _run_jupyter(self, runner, host_port=8888) -> None:
        user_id = os.getuid() if os.name != "nt" else 0
        group_id = os.getgid() if os.name != "nt" else 0
        lines = [
            "About to launch jupyter.",
            bold("*** To connect to jupyter, please open the following in a browser:"),
            bold(red(f"    http://127.0.0.1:{host_port}")),
            bold("*** To connect on the command line as well, in a new terminal, run:"),
            bold(
                f"    docker exec -it -u {user_id}:{group_id} {runner.docker_container.short_id} bash"
            ),
            bold("*** To gracefully shut down container, press: Ctrl-C"),
            "",
            "Jupyter notebook log:",
        ]
        logger.info("\n".join(lines))
        runner.exec_cmd(
            "jupyter lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''",
            user="root",
        )
