from pydantic import BaseModel
from armory.logs import log
import os
import yaml
import json
from typing import Union, List, Dict, Optional
from armory.utils import set_overrides
from enum import Enum
from pydoc import locate

class Values(BaseModel):
    value: str
    type: str


class ArbitraryKwargs(BaseModel):
    __root__: Optional[Dict[str, Values]]

    def __getattr__(self, item):  # if you want to use '.'
        print(item)
        return self.__root__[item].value
        # return self.__root__[item]

    def __setattr__(self, key, value):
        print(key, value)
        tp = locate(self.__root__[key].type)
        self.__root__[key] = tp(value)

class ExecutionMode(str, Enum):
    """Armory Execution Mode
    docker  ->  Means that armory will execute
                experiments inside the prescribed
                docker container
    native  ->  Means that armory will execute
                experiments in the python environment
                in which it is called.
    """

    docker = "docker"
    native = "native"


class AttackParameters(BaseModel):
    """Armory Data Class for `Attack` Parameters"""

    name: str
    module: str
    knowledge: str
    kwargs: dict
    type: str = None


class DatasetParameters(BaseModel):
    """Armory Dataclass For `Dataset` Parameters"""

    name: str
    module: str
    framework: str
    batch_size: int


class DefenseParameters(BaseModel):
    """Armory Dataclass for `Defense` Parameters"""

    name: str
    type: str
    module: str
    kwargs: dict


class MetricParameters(BaseModel):
    """Armory Dataclass for Evaluation `Metric` Parameters"""

    means: bool
    perturbation: str
    record_metric_per_sample: bool
    task: list


class ModelParameters(BaseModel):
    """Armory Dataclass for `Model` Parameters"""

    name: str
    module: str
    weights_file: str = None
    wrapper_kwargs: dict
    model_kwargs: dict
    fit_kwargs: ArbitraryKwargs
    fit: bool


class ScenarioParameters(BaseModel):
    """Armory Dataclass for `Scenario` Parameters"""

    function_name: str
    module_name: str
    kwargs: dict


class SystemConfigurationParameters(BaseModel):
    """Armory Dataclass for Environment Configuration Paramters"""

    docker_image: str = None
    gpus: str = None
    external_github_repo: str = None
    output_dir: str = None
    output_filename: str = None
    use_gpu: bool = False


class MetaData(BaseModel):
    name: str
    author: str
    description: str


class PoisonParameters(BaseModel):
    pass


class ExecutionParameters(BaseModel):
    execution_mode: ExecutionMode
    docker_image: str = None


class ExperimentParameters(BaseModel):
    """Armory Dataclass for Experiment Parameters"""

    _meta: MetaData
    poison: PoisonParameters = None
    attack: AttackParameters = None
    dataset: DatasetParameters
    defense: DefenseParameters = None
    metric: MetricParameters = None
    model: ModelParameters
    scenario: ScenarioParameters
    execution: ExecutionParameters = None

    @classmethod
    def load(cls, filename: str, overrides: List = []):
        fname, fext = os.path.splitext(filename)
        log.info(
            f"Attempting to Load Experiment from file: {filename} and applying cli overrides: {overrides}"
        )
        with open(filename, "r") as f:
            data = yaml.safe_load(f.read())

        if "environment" in data:
            log.warning(
                f"Overriding Environment Setting using data from Experiment File: {data['environment']}"
            )
            env_overrides = data["environment"]
            data.pop("environment")
        else:
            env_overrides = []

        log.debug(f"Parsing Class Object from: {data}")
        exp = cls.parse_obj(data)

        log.debug(f"Recieved Exp: {exp.pretty_print()}")
        # import pdb;
        # pdb.set_trace()
        set_overrides(exp, overrides)
        return exp, env_overrides

    def pretty_print(self):
        print(self.dict())
        return json.dumps(self.dict(), indent=2, sort_keys=True)


# class Experiment(object):
#     """Execution Class to `run` armory experiments"""
#
#     def __init__(self, experiment_parameters, environment_parameters):
#         log.info(f"Constructing Experiment using parameters: \n{experiment_parameters}")
#         self.exp_pars = experiment_parameters
#         self.env_pars = environment_parameters
#         log.info(f"Importing Scenario Module: {self.exp_pars.scenario.module_name}")
#         self.scenario_module = import_module(self.exp_pars.scenario.module_name)
#         log.info(f"Loading Scenario Function: {self.exp_pars.scenario.function_name}")
#         self.scenario_fn = getattr(
#             self.scenario_module, self.exp_pars.scenario.function_name
#         )
