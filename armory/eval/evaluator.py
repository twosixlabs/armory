"""
Evaluators control launching of ARMORY evaluations.
"""
import base64
import os
import json
import logging
import shutil
import time
import datetime

import docker
import requests
from docker.errors import ImageNotFound

from armory.configuration import load_global_config
from armory.docker.management import ManagementInstance, ArmoryInstance
from armory.docker.host_management import HostManagementInstance
from armory.utils.printing import bold, red
from armory.utils import docker_api
from armory import paths

logger = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(
        self, config: dict, no_docker: bool = False,
    ):
        if not isinstance(config, dict):
            raise ValueError(f"config {config} must be a dict")
        self.config = config

        self.host_paths = paths.HostPaths()
        if os.path.exists(self.host_paths.armory_config):
            self.armory_global_config = load_global_config(
                self.host_paths.armory_config
            )
        else:
            self.armory_global_config = {"verify_ssl": True}

        date_time = datetime.datetime.utcnow().isoformat().replace(":", "")
        output_dir = self.config["sysconfig"].get("output_dir", None)
        eval_id = f"{output_dir}_{date_time}" if output_dir else date_time

        self.config["eval_id"] = eval_id
        self.output_dir = os.path.join(self.host_paths.output_dir, eval_id)
        self.tmp_dir = os.path.join(self.host_paths.tmp_dir, eval_id)

        if self.config["sysconfig"].get("use_gpu", None):
            kwargs = dict(runtime="nvidia")
        else:
            kwargs = dict(runtime="runc")
        image_name = self.config["sysconfig"].get("docker_image")
        kwargs["image_name"] = image_name
        self.no_docker = not image_name or no_docker

        # Retrieve environment variables that should be used in evaluation
        self.extra_env_vars = dict()
        self._gather_env_variables()

        if self.no_docker:
            self.manager = HostManagementInstance()
            return

        # Download docker image on host
        docker_client = docker.from_env()
        try:
            docker_client.images.get(kwargs["image_name"])
        except ImageNotFound:
            logger.info(f"Image {image_name} was not found. Downloading...")
            if "twosixarmory" in image_name and "-dev" in image_name:
                raise ValueError(
                    (
                        "You are attempting to pull an armory developer "
                        "docker image; however, these are not published. This "
                        "is likely because you're running armory from its "
                        "master branch. If you want a stable release with "
                        "published docker images try pip installing 'armory-testbed' "
                        "or checking out one of the stable branches on the git repository. "
                        "If you'd like to continue working on the developer image please "
                        "build it from source on your machine as described here: "
                        "https://armory.readthedocs.io/en/latest/contributing/#development-docker-containers"
                    )
                )
            docker_api.pull_verbose(docker_client, image_name)
        except requests.exceptions.ConnectionError:
            logger.error("Docker connection refused. Is Docker Daemon running?")
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
        self.extra_env_vars["ARMORY_INCLUDE_SUBMISSION_BUCKETS"] = os.getenv(
            "ARMORY_INCLUDE_SUBMISSION_BUCKETS", default=""
        )

        if not self.armory_global_config["verify_ssl"]:
            self.extra_env_vars["VERIFY_SSL"] = "false"

        if self.config["sysconfig"].get("use_gpu", None):
            gpus = self.config["sysconfig"].get("gpus")
            if gpus is not None:
                self.extra_env_vars["NVIDIA_VISIBLE_DEVICES"] = gpus

    def _cleanup(self):
        logger.info(f"Deleting tmp_dir {self.tmp_dir}")
        try:
            shutil.rmtree(self.tmp_dir)
        except OSError as e:
            if not isinstance(e, FileNotFoundError):
                logger.exception(f"Error removing tmp_dir {self.tmp_dir}")

        logger.info(f"Removing output_dir {self.output_dir} if empty")
        try:
            os.rmdir(self.output_dir)
        except OSError:
            pass

    def run(
        self,
        interactive=False,
        jupyter=False,
        host_port=None,
        command=None,
        check_run=False,
    ) -> None:
        if self.no_docker:
            if jupyter or interactive or command:
                raise ValueError(
                    "jupyter, interactive, or bash commands only supported when running Docker containers."
                )
            runner = self.manager.start_armory_instance(envs=self.extra_env_vars)
            try:
                self._run_config(runner, check_run=check_run)
            except KeyboardInterrupt:
                logger.warning("Keyboard interrupt caught")
            finally:
                logger.warning("Cleaning up...")
            self._cleanup()
            return

        if check_run and (jupyter or interactive or command):
            raise ValueError(
                "check_run incompatible with interactive, jupyter, or command"
            )

        # Handle docker and jupyter ports
        if jupyter or host_port:
            if host_port:
                ports = {host_port: host_port}
            else:
                ports = {8888: 8888}
        else:
            ports = None

        try:
            runner = self.manager.start_armory_instance(
                envs=self.extra_env_vars, ports=ports
            )
            try:
                if jupyter:
                    self._run_jupyter(runner, ports)
                elif interactive:
                    self._run_interactive_bash(runner)
                elif command:
                    self._run_command(runner, command)
                else:
                    self._run_config(runner, check_run=check_run)
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
        self._cleanup()

    def _b64_encode_config(self):
        bytes_config = json.dumps(self.config).encode("utf-8")
        base64_bytes = base64.b64encode(bytes_config)
        return base64_bytes.decode("utf-8")

    def _run_config(self, runner: ArmoryInstance, check_run=False) -> None:
        logger.info(bold(red("Running evaluation script")))

        b64_config = self._b64_encode_config()
        options = ""
        if self.no_docker:
            options += " --no-docker"
        if check_run:
            options += " --check"
        if logger.level == logging.DEBUG:
            options += " --debug"

        runner.exec_cmd(f"python -m armory.scenarios.base {b64_config}{options}")

    def _run_command(self, runner: ArmoryInstance, command: str) -> None:
        logger.info(bold(red(f"Running bash command: {command}")))
        runner.exec_cmd(command)

    def _run_interactive_bash(self, runner: ArmoryInstance) -> None:
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
            tmp_dir = os.path.join(self.host_paths.tmp_dir, self.config["eval_id"])
            os.makedirs(tmp_dir)
            self.tmp_config = os.path.join(tmp_dir, "interactive-config.json")
            docker_config_path = os.path.join(
                paths.runtime_paths().tmp_dir,
                self.config["eval_id"],
                "interactive-config.json",
            )
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

    def _run_jupyter(self, runner: ArmoryInstance, ports: dict) -> None:
        user_id = os.getuid() if os.name != "nt" else 0
        group_id = os.getgid() if os.name != "nt" else 0
        port = list(ports.keys())[0]
        lines = [
            "About to launch jupyter.",
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
            f"jupyter lab --ip=0.0.0.0 --port {port} --no-browser --allow-root",
            user="root",
        )
