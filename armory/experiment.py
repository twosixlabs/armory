from pydantic import BaseModel
from armory.logs import log
import os

# TODO: Change class names to [thing]Parameters format
#  e.g. Attack-> AttackParameters

# TODO:  Make Experiment class which is like old
#  scneario.main

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

    name: str
    module: str
    kwargs: dict


class SystemConfigurationParameters(BaseModel):
    """Armory Dataclass for Environment Configuration Paramters"""

    docker_image: str = None
    gpus: str = None
    external_github_repo: str = None
    output_dir: str = None
    output_filename: str = None
    use_gpu: bool = False


class ExperimentParameters(BaseModel):
    """Armory Dataclass for Experiment Parameters"""

    name: str = None
    _description: str = None
    adhoc: bool = None  # TODO Figure out what this is for...maybe poison?
    attack: AttackParameters = None
    dataset: DatasetParameters
    defense: DefenseParameters = None
    metric: MetricParameters = None
    model: ModelParameters
    scenario: ScenarioParameters
    sysconfig: SystemConfigurationParameters = None

    def save(self, filename):
        with open(filename, "w") as f:
            f.write(self.json())

    @classmethod
    def load(cls, filename):
        valid_ext = (".aexp", ".json")
        if os.path.splitext(filename)[1] not in valid_ext:
            raise ValueError(
                f"Experiment File: {filename} has invalid extension....must be in {valid_ext}"
            )

        if not os.path.exists(filename):
            raise ValueError(f"Experiment File: {filename} does not exist!")

        try:
            with open(filename, "r") as f:
                exp = cls.parse_raw(f.read())
        except Exception as e:
            log.error(f"Could not parse Experiment from: {filename}")
            raise e
        return exp


class Experiment(object):
    """Execution Class to `run` armory experiments

    """

    def __init__(self):
        pass
