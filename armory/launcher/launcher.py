"""Armory Launcher Docker Orchestration"""
import importlib
import os
import subprocess
from dataclasses import dataclass
from typing import List
from armory.logs import log
from armory.utils.experiment import ExperimentParameters
from armory.utils.environment import EnvironmentParameters


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


def execute_experiment(
    experiment: ExperimentParameters, environment: EnvironmentParameters
):
    log.info("Executing Armory Experiment")
    log.debug(f"Experiment Parameters: {experiment}")
    log.debug(f"Envioronment Parameters: {environment}")

    config = experiment.as_old_config()
    config["environment_parameters"] = environment.dict()
    from datetime import datetime as dt
    import time

    eval_id = dt.utcfromtimestamp(time.time()).strftime("%Y-%m-%dT%H:%M:%SUTC")
    config["eval_id"] = eval_id

    # Import here to avoid dependency tree in launcher
    log.debug("Loading Scenario Module & fn")
    module = importlib.import_module(experiment.scenario.module_name)
    ScenarioClass = getattr(module, experiment.scenario.function_name)
    log.trace(f"ScenarioClass Loaded: {ScenarioClass}")
    log.debug("Constructing Scenario Class...")
    if experiment.scenario.kwargs is not None:
        pars = experiment.scenario.kwargs.dict()
        pars.pop("__root__")
    else:
        pars = {}

    scenario = ScenarioClass(config, **pars)
    log.trace(f"constructed scenario: {scenario}")
    log.debug("Calling .evaluate()")
    scenario.evaluate()
    log.success("Evaluation Complete!!")
    # # from armory.engine.utils.configuration import load_config
    # log.debug(f"Loading Config: {config}")
    # # config = load_config(config, from_file=True)
    #
    # # scenario_config = config.get("scenario")
    # # if scenario_config is None:
    # #     raise KeyError('"scenario" missing from evaluation config')
    # # _scenario_setup(config)
    #
    # ScenarioClass = config_loading.load_fn(scenario_config)
    # kwargs = scenario_config.get("kwargs", {})
    # kwargs.update(
    #     dict(
    #         check_run=check_run,
    #         num_eval_batches=num_eval_batches,
    #         skip_benign=skip_benign,
    #         skip_attack=skip_attack,
    #         skip_misclassified=skip_misclassified,
    #     )
    # )
    # scenario_config["kwargs"] = kwargs
    # scenario = ScenarioClass(config, **kwargs)
    # log.trace(f"scenario loaded {scenario}")
    # scenario.evaluate()


if __name__ == "__main__":
    mount = DockerMount(
        source=os.path.expanduser("~"), target="/my_space", type="bind", readonly=True
    )
    print(mount)
    execute_docker_cmd("alpine", "ls /my_space", mounts=[mount])
