from pydantic import BaseModel
from armory.logs import log
import os
import yaml
import json
from armory.utils import parse_overrides
from importlib import import_module

# TODO: Update this so that validation occurs at lowest
#  level possible.  I.e. we do NOT validate at edge,
#  and let process throw exception, therefore data model
#  is assumed to be "unsafe" and each function needs to
#  protect its inputs.


class AttackParameters(BaseModel):
    """Armory Data Class for `Attack` Parameters

    """

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
    fit_kwargs: dict
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
    # sysconfig: SystemConfigurationParameters = None

    # def save(self, filename):
    #     with open(filename, "w") as f:
    #         f.write(self.json())

    @classmethod
    def load(cls, filename, overrides=[]):
        overrides = parse_overrides(overrides)
        valid_ext = (".aexp", ".json")
        fname, fext = os.path.splitext(filename)
        log.info(f"Attempting to Load Experiment from file: {filename}")
        if fext == ".json":
            with open(filename, "r") as f:
                data = json.loads(f.read())
        elif fext in (".aexp", ".yml", ".yaml"):
            with open(filename, "r") as f:
                data = yaml.safe_load(f.read())
        else:
            raise ValueError(
                f"Experiment File: {filename} has invalid extension....must be in {valid_ext}"
            )

        log.debug(f"Parsing Class Object from: {data}")
        exp = cls.parse_obj(data)
        # Getting Environment Overrides from File (if available)
        if "environment" in data:
            file_overrides = parse_overrides(data["environment"])
        else:
            file_overrides = []

        return exp, file_overrides


class Experiment(object):
    """Execution Class to `run` armory experiments

    """

    def __init__(self, experiment_parameters, environment_parameters):
        log.info(f"Constructing Experiment using parameters: \n{experiment_parameters}")
        self.exp_pars = experiment_parameters
        self.env_pars = environment_parameters
        log.info(f"Importing Scenario Module: {self.exp_pars.scenario.module_name}")
        self.scenario_module = import_module(self.exp_pars.scenario.module_name)
        log.info(f"Loading Scenario Function: {self.exp_pars.scenario.function_name}")
        self.scenario_fn = getattr(
            self.scenario_module, self.exp_pars.scenario.function_name
        )
