"""Armory Launcher Docker Orchestration"""

import os
import subprocess
from dataclasses import dataclass
from typing import List
from armory.utils.experiment import Experiment
from armory.logs import log

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

def execute_native_cmd(
    cmd: str,
):
    log.info(f"Executing cmd in native environment:\n\t{cmd}")
    result = subprocess.run(f"{cmd}", shell=True, capture_output=True)

    if result.returncode != 0:
        log.error(f"Cmd returned error: {result.returncode}")
    else:
        log.success("Docker CMD Execution Success!!")
        log.debug(result)
    return result

def execute_docker_cmd(
    image: str,
    cmd: str,
    runtime: str = "runc",
    mounts: List[DockerMount] = [],
    ports: List[DockerPort] = [],
    remove=True,
    shm_size="16G",
):
    cmd = " ".join(
        [
            "docker run -it",
            "--rm" if remove else "",
            f"--runtime={runtime}",
            " ".join([f"{mnt}" for mnt in mounts]),
            " ".join([f"{port}" for port in ports]),
            f"--shm-size={shm_size}",
            f"{image}",
            f"{cmd}",
        ]
    )
    log.info(f"Executing cmd:\n\t{cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True)
    log.debug(result)

    if result.returncode != 0:
        log.error(f"Cmd returned error: {result.returncode}")
    else:
        log.success("Docker CMD Execution Success!!")
    return result

def execute_experiment(experiment):
    pass


if __name__ == "__main__":
    mount = DockerMount(
        source=os.path.expanduser("~"), target="/my_space", type="bind", readonly=True
    )
    print(mount)
    execute_docker_cmd("alpine", "ls /my_space", mounts=[mount])
