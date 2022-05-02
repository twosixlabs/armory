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
    """ Base Model for Arbitrary Dict Type attribute values"""
    value: str
    type: str


class ArbitraryDict(BaseModel):
    """Base Model substitute for Dict

    ArbitraryDict is inteded to be used as a data model
    when the `keys` of the incoming data are not known
    and/or can't be anticipated.  The main example of this
    is in the `kwarg` type options within the armory experiment
    file.  To use this:

    # Say we have data model called `AttackParameters` and we want
    it to have a `kwargs` attribute that can store arbitrary key/value
    pairs, then we would write something like:

    class AttackParameters(BaseModel):
        kwargs: ArbitraryDict
    """
    __root__: Optional[Dict[str, Values]]

    def __getattr__(self, item):  # need this to access using `.`
        return self.__root__[item].value

    def __setattr__(self, key, value):
        # Get class for specified value type
        tp = locate(self.__root__[key].type)

        # return the value typed correctly
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
    kwargs: ArbitraryDict
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
    kwargs: ArbitraryDict


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
    wrapper_kwargs: ArbitraryDict
    model_kwargs: ArbitraryDict
    fit_kwargs: ArbitraryDict
    fit: bool


class ScenarioParameters(BaseModel):
    """Armory Dataclass for `Scenario` Parameters"""

    function_name: str
    module_name: str
    kwargs: ArbitraryDict


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
    mode: ExecutionMode
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

        log.debug(f"Parsed Experiment: {exp.pretty_print()}")

        log.debug(f"Applying experiment overrides: {overrides}")
        set_overrides(exp, overrides)
        log.debug(f"Final Experiment: {exp}")
        return exp, env_overrides

    def pretty_print(self):
        return json.dumps(self.dict(), indent=2, sort_keys=True)

    def as_old_config(self):
        return self.dict()