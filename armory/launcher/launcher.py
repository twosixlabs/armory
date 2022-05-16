"""Armory Launcher Docker Orchestration"""
import os
from subprocess import Popen, PIPE, CalledProcessError
from dataclasses import dataclass
from typing import List
from armory.logs import log
from armory.utils.experiment import ExperimentParameters
from armory.utils.environment import EnvironmentParameters
from datetime import datetime as dt
import time
from string import Template
import armory.logs


@dataclass
class DockerMount:
    type: str
    source: str
    target: str
    readonly: bool

    def __str__(self):
        msg = ",".join(
            [f"{i}={getattr(self, i)}" for i in ["type", "source", "target"]]
        )
        msg += ",readonly" if self.readonly else ""
        return f"--mount {msg}"


@dataclass
class DockerPort:
    host: int
    container: int
    type: str = "tcp"

    def __str__(self):
        return f"-p {self.host}:{self.container}/{self.type}"


def get_mounts_from_environment(environment: EnvironmentParameters):
    """Get list of DockerMount(s) from EnvironmentParameters """
    mounts = []
    mounts.append(
        DockerMount(
            type="bind",
            source=os.path.dirname(environment.profile),
            target="/armory/",
            readonly=False,
        )
    )

    mounts.append(
        DockerMount(
            type="bind",
            source=environment.armory_source_directory,
            target="/armory_src/armory",
            readonly=True,
        )
    )
    return mounts


def execute_cmd(cmd: list, cwd=None, pre_prompt=""):
    log.info(f"Executing Command in subprocess...")
    log.debug(f"\t{' '.join(cmd)}\n")
    log.trace(f"\t Actual List: {cmd}")
    with Popen(cmd, stdout=PIPE, bufsize=1, universal_newlines=True, cwd=cwd) as p:
        for line in p.stdout:
            print(f"{pre_prompt} {line}", end="\r")

    log.debug(f"Subprocess Execution Completed with returncode: {p.returncode}")
    if p.returncode != 0:
        log.error("Command Execution in Subprocess Failed!!")
        raise CalledProcessError(p.returncode, p.args)
    else:
        log.success("Command Execution in Subprocess Succeeded!!")


def execute_docker_cmd(
    image: str,
    cmd: list,
    cwd: str = "/workspace",
    runtime: str = "runc",
    mounts: List[DockerMount] = [],
    ports: List[DockerPort] = [],
    shm_size="16G",
):
    command = ["docker", "run", "-it", "--rm", "-w", f"{cwd}", f"--runtime={runtime}"]
    for mnt in mounts:
        command += f"{mnt}".split(" ")

    for prt in ports:
        command += f"{prt}".split(" ")

    command += [
        f"--shm-size={shm_size}",
        f"{image}",
    ]
    command += cmd
    log.info(f"Executing command:\n\t{' '.join(command)}")
    execute_cmd(command, pre_prompt="<in docker>")



def execute_experiment(
    experiment: ExperimentParameters, environment: EnvironmentParameters
):
    log.info("Executing Armory Experiment")
    log.debug(f"Experiment Parameters: {experiment}")
    log.debug(f"Environment Parameters: {environment}")

    output_directory = os.path.join(
        environment.paths.output_directory,
        dt.utcfromtimestamp(time.time()).strftime("%Y-%m-%dT%H-%M-%SUTC"),
    )
    log.info(f"Creating Output Directory: {output_directory}")
    os.makedirs(output_directory)

    if experiment.execution.mode == "native":
        experiment_filename = os.path.join(output_directory, "experiment.yml")
        experiment.save(experiment_filename)
        log.debug(f"Saved Experiment (for reference) to: {experiment_filename}")
        environment_filename = os.path.join(output_directory, "environment.yml")
        environment.save(environment_filename)
        log.debug(f"Saved Environment (for reference) to: {environment_filename}")

        template_parameters = {
            "armory_sys_path": "",
            "experiment_filename": experiment_filename,
            "environment_filename": environment_filename,
            "output_directory": output_directory,
            "armory_log_filters": [f"{k}:{v}" for k, v in armory.logs.filters.items()],
        }

        with open(os.path.join(os.path.dirname(__file__), "execute_template.py")) as f:
            template = Template(f.read())
            script = template.substitute(template_parameters)
        log.debug(f"Execution script: \n{script}")
        with open(os.path.join(output_directory, "execution_script.py"), "w") as f:
            f.write(script)
        log.info("Launching Armory in `native` mode")
        execute_cmd(["python", "execution_script.py"], cwd=output_directory)
        log.success("Native Execution Complete!!")
    elif experiment.execution.mode == "docker":
        docker_image = experiment.execution.docker_image
        experiment_filename = os.path.join(output_directory, "experiment.yml")
        experiment.execution.mode = "native"
        experiment.execution.docker_image = None
        experiment.save(experiment_filename)
        log.debug(f"Saved Experiment (for reference) to: {experiment_filename}")

        environment_filename = os.path.join(output_directory, "environment.yml")
        environment.paths.change_base("/armory/")
        environment.save(environment_filename)
        log.debug(f"Saved Environment (for reference) to: {environment_filename}")

        template_parameters = {
            "armory_sys_path": "import sys; sys.path.insert(0, '/armory_src/')",
            "experiment_filename": "/armory_output/experiment.yml",
            "environment_filename": "/armory_output/environment.yml",
            "output_directory": "/armory_output/",
            "armory_log_filters": [f"{k}:{v}" for k, v in armory.logs.filters.items()],
        }

        with open(os.path.join(os.path.dirname(__file__), "execute_template.py")) as f:
            template = Template(f.read())
            script = template.substitute(template_parameters)
        log.debug(f"Execution script: \n{script}")
        with open(os.path.join(output_directory, "execution_script.py"), "w") as f:
            f.write(script)

        mounts = get_mounts_from_environment(environment)
        mounts.append(
            DockerMount(
                type="bind",
                source=output_directory,
                target="/armory_output/",
                readonly=False,
            )
        )
        execute_docker_cmd(
            image=docker_image,
            cmd=["python","execution_script.py"],
            cwd="/armory_output/",
            mounts=mounts,
        )

    else:
        raise ValueError(f"Unrecognized Execution Mode: {experiment.execution.mode}")

    log.info(f"Experiment Execution Complete!!  Results can be found at: {output_directory}")


if __name__ == "__main__":
    mount = DockerMount(
        source=os.path.expanduser("~"), target="/my_space", type="bind", readonly=True
    )
    print(mount)
    execute_docker_cmd("alpine", "ls /my_space", mounts=[mount])
