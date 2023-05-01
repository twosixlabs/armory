"""
Evaluators control launching of ARMORY evaluations.
"""
import base64
import datetime
import json
import os
import shutil
import sys
import time

import requests

import armory
from armory import environment, paths
from armory.configuration import load_global_config
from armory.docker.host_management import HostManagementInstance
from armory.docker.management import ArmoryInstance, ManagementInstance
from armory.logs import added_filters, is_debug, log
from armory.utils.printing import bold, red


class Evaluator(object):
    def __init__(
        self,
        config: dict,
        no_docker: bool = False,
        root: bool = False,
    ):
        log.info("Constructing Evaluator Object")
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
        self.root = root

        # Retrieve environment variables that should be used in evaluation
        log.info("Retrieving Environment Variables")
        self.extra_env_vars = dict()
        self._gather_env_variables()

        if self.no_docker:
            if self.root:
                raise ValueError("running with --root is incompatible with --no-docker")
            if kwargs["image_name"] is not None:
                log.warning("Running in --no-docker mode. Setting 'image_name' to None")
            kwargs["image_name"] = None
            self.manager = HostManagementInstance()
        else:
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
        if self.config["sysconfig"].get("set_pythonhashseed"):
            self.extra_env_vars["PYTHONHASHSEED"] = "0"

        if not self.no_docker:
            self.extra_env_vars["HOME"] = "/tmp"

        # Because we may want to allow specification of ARMORY_TORCH_HOME
        # this constant path is placed here among the other imports
        if self.no_docker:
            torch_home = paths.HostPaths().pytorch_dir
        else:
            torch_home = paths.DockerPaths().pytorch_dir
        self.extra_env_vars["TORCH_HOME"] = torch_home

        self.extra_env_vars[environment.ARMORY_VERSION] = armory.__version__

    def _cleanup(self):
        log.info(f"deleting tmp_dir {self.tmp_dir}")
        try:
            shutil.rmtree(self.tmp_dir)
        except OSError as e:
            if not isinstance(e, FileNotFoundError):
                log.exception(f"Error removing tmp_dir {self.tmp_dir}")

        try:
            os.rmdir(self.output_dir)
            log.warning(f"removed output_dir {self.output_dir} because it was empty")
        except FileNotFoundError:
            log.warning(f"output_dir {self.output_dir} was deleted or never created")
        except OSError:
            jsons = [x for x in os.listdir(self.output_dir) if x.endswith(".json")]
            if len(jsons) == 1:
                json = jsons[0]
            else:
                json = ""
            output_path = os.path.join(self.output_dir, json)
            log.info(f"results output written to:\n{output_path}")

    def run(
        self,
        interactive=False,
        jupyter=False,
        host_port=None,
        command=None,
        check_run=False,
        num_eval_batches=None,
        skip_benign=None,
        skip_attack=None,
        skip_misclassified=None,
        validate_config=None,
    ) -> int:
        exit_code = 0
        if self.no_docker:
            if jupyter or interactive or command:
                raise ValueError(
                    "jupyter, interactive, or bash commands only supported when running Docker containers."
                )
            runner = self.manager.start_armory_instance(
                envs=self.extra_env_vars,
            )
            try:
                exit_code = self._run_config(
                    runner,
                    check_run=check_run,
                    num_eval_batches=num_eval_batches,
                    skip_benign=skip_benign,
                    skip_attack=skip_attack,
                    skip_misclassified=skip_misclassified,
                    validate_config=validate_config,
                )
            except KeyboardInterrupt:
                log.warning("Keyboard interrupt caught")
            finally:
                log.info("cleaning up...")
            self._cleanup()
            return exit_code

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
                envs=self.extra_env_vars,
                ports=ports,
                user=self.get_id(),
            )
            try:
                if jupyter:
                    self._run_jupyter(
                        runner,
                        ports,
                        check_run=check_run,
                        num_eval_batches=num_eval_batches,
                        skip_benign=skip_benign,
                        skip_attack=skip_attack,
                        skip_misclassified=skip_misclassified,
                    )
                elif interactive:
                    self._run_interactive_bash(
                        runner,
                        check_run=check_run,
                        num_eval_batches=num_eval_batches,
                        skip_benign=skip_benign,
                        skip_attack=skip_attack,
                        skip_misclassified=skip_misclassified,
                        validate_config=validate_config,
                    )
                elif command:
                    exit_code = self._run_command(runner, command)
                else:
                    exit_code = self._run_config(
                        runner,
                        check_run=check_run,
                        num_eval_batches=num_eval_batches,
                        skip_benign=skip_benign,
                        skip_attack=skip_attack,
                        skip_misclassified=skip_misclassified,
                        validate_config=validate_config,
                    )
            except KeyboardInterrupt:
                log.warning("keyboard interrupt caught")
            finally:
                log.trace("Shutting down container {self.manager.instances.keys()}")
                self.manager.stop_armory_instance(runner)
        except requests.exceptions.RequestException as e:
            log.exception("Starting instance failed.")
            if str(e).endswith(
                f'Bind for 0.0.0.0:{host_port} failed: port is already allocated")'
            ):
                log.error(
                    f"Port {host_port} already in use. Try a different one with '--port <port>'"
                )
            elif (
                str(e)
                == '400 Client Error: Bad Request ("Unknown runtime specified nvidia")'
            ):
                log.error(
                    'NVIDIA runtime failed. Either install nvidia-docker or set config "use_gpu" to false'
                )
            else:
                log.error("Is Docker Daemon running?")
        self._cleanup()
        return exit_code

    def _b64_encode_config(self):
        bytes_config = json.dumps(self.config).encode("utf-8")
        base64_bytes = base64.b64encode(bytes_config)
        return base64_bytes.decode("utf-8")

    def _run_config(
        self,
        runner: ArmoryInstance,
        check_run=False,
        num_eval_batches=None,
        skip_benign=None,
        skip_attack=None,
        skip_misclassified=None,
        validate_config=None,
    ) -> int:
        log.info(bold(red("Running evaluation script")))

        b64_config = self._b64_encode_config()
        options = self._build_options(
            check_run=check_run,
            num_eval_batches=num_eval_batches,
            skip_benign=skip_benign,
            skip_attack=skip_attack,
            skip_misclassified=skip_misclassified,
            validate_config=validate_config,
        )
        if self.no_docker:
            kwargs = {}
            python = sys.executable
        else:
            kwargs = {"user": self.get_id()}
            python = "python"

        cmd = f"{python} -m armory.scenarios.main {b64_config}{options} --base64"
        return runner.exec_cmd(cmd, **kwargs)

    def _run_command(self, runner: ArmoryInstance, command: str) -> int:
        log.info(bold(red(f"Running bash command: {command}")))
        return runner.exec_cmd(command, user=self.get_id(), expect_sentinel=False)

    def get_id(self):
        """
        Return uid, gid
        """
        # Windows docker does not require synchronizing file and
        # directory permissions via uid and gid.
        if os.name == "nt" or self.root:
            user_id = 0
            group_id = 0
        else:
            user_id = os.getuid()
            group_id = os.getgid()
        return f"{user_id}:{group_id}"

    def _run_interactive_bash(
        self,
        runner: ArmoryInstance,
        check_run=False,
        num_eval_batches=None,
        skip_benign=None,
        skip_attack=None,
        skip_misclassified=None,
        validate_config=None,
    ) -> None:
        user_group_id = self.get_id()
        lines = [
            "Container ready for interactive use.",
            bold("# In a new terminal, run the following to attach to the container:"),
            bold(
                red(
                    f"docker exec -it -u {user_group_id} {runner.docker_container.short_id} bash"
                )
            ),
            "",
        ]
        if self.config.get("scenario"):
            options = self._build_options(
                check_run=check_run,
                num_eval_batches=num_eval_batches,
                skip_benign=skip_benign,
                skip_attack=skip_attack,
                skip_misclassified=skip_misclassified,
                validate_config=validate_config,
            )
            init_options = self._constructor_options(
                check_run=check_run,
                num_eval_batches=num_eval_batches,
                skip_benign=skip_benign,
                skip_attack=skip_attack,
                skip_misclassified=skip_misclassified,
            )

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
                    bold("# To run your scenario in the container:"),
                    bold(
                        red(
                            f"python -m armory.scenarios.main {docker_config_path}{options}"
                        )
                    ),
                    "",
                    bold("# To run your scenario interactively:"),
                    bold(
                        red(
                            "python\n"
                            "from armory.scenarios.main import get as get_scenario\n"
                            f's = get_scenario("{docker_config_path}"{init_options}).load()\n'
                            "s.evaluate()"
                        )
                    ),
                    "",
                    bold("# To gracefully shut down container, press: Ctrl-C"),
                    "",
                ]
            )
        log.info("\n".join(lines))
        while True:
            time.sleep(1)

    def _run_jupyter(
        self,
        runner: ArmoryInstance,
        ports: dict,
        check_run=False,
        num_eval_batches=None,
        skip_benign=None,
        skip_attack=None,
        skip_misclassified=None,
    ) -> None:
        if not self.root:
            log.warning("Running Jupyter Lab as root inside the container.")

        user_group_id = self.get_id()
        port = list(ports.keys())[0]
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
        init_options = self._constructor_options(
            check_run=check_run,
            num_eval_batches=num_eval_batches,
            skip_benign=skip_benign,
            skip_attack=skip_attack,
            skip_misclassified=skip_misclassified,
        )
        lines = [
            "About to launch jupyter.",
            bold("# To connect on the command line as well, in a new terminal, run:"),
            bold(
                red(
                    f"docker exec -it -u {user_group_id} {runner.docker_container.short_id} bash"
                )
            ),
            "",
        ]
        if "scenario" in self.config:
            # If not, config is not valid to load into scenario
            lines.extend(
                [
                    bold("# To run, inside of a notebook:"),
                    bold(
                        red(
                            "from armory.scenarios.main import get as get_scenario\n"
                            f's = get_scenario("{docker_config_path}"{init_options}).load()\n'
                            "s.evaluate()"
                        )
                    ),
                    "",
                ]
            )
        lines.extend(
            [
                bold("# To gracefully shut down container, press: Ctrl-C"),
                "",
                "Jupyter notebook log:",
            ]
        )
        log.info("\n".join(lines))
        runner.exec_cmd(
            f"jupyter lab --ip=0.0.0.0 --port {port} --no-browser",
            user=user_group_id,
            expect_sentinel=False,
        )

    def _build_options(
        self,
        check_run,
        num_eval_batches,
        skip_benign,
        skip_attack,
        skip_misclassified,
        validate_config,
    ):
        options = ""
        if self.no_docker:
            options += " --no-docker"
        if check_run:
            options += " --check"
        if is_debug():
            options += " --debug"
        if num_eval_batches:
            options += f" --num-eval-batches {num_eval_batches}"
        if skip_benign:
            options += " --skip-benign"
        if skip_attack:
            options += " --skip-attack"
        if skip_misclassified:
            options += " --skip-misclassified"
        if validate_config:
            options += " --validate-config"
        for module, level in added_filters.items():
            options += f" --log-level {module}:{level}"
        return options

    def _constructor_options(
        self,
        check_run=False,
        num_eval_batches=None,
        skip_benign=None,
        skip_attack=None,
        skip_misclassified=None,
    ):
        kwargs = dict(
            check_run=check_run,
            num_eval_batches=num_eval_batches,
            skip_benign=skip_benign,
            skip_attack=skip_attack,
            skip_misclassified=skip_misclassified,
        )
        options = "".join(f", {str(k)}={str(v)}" for k, v in kwargs.items() if v)
        return options
