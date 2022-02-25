from pydantic import BaseModel, validator
from importlib import import_module
from armory.logs import log
import os


class Attack(BaseModel):
    """Armory Data Class for `Attack` Parameters

    """

    name: str
    module: str
    knowledge: str
    kwargs: dict
    type: str = None

    @validator("knowledge")
    def validate_knowledge(cls, v):
        if v not in ["white", "black"]:
            raise ValueError(
                f"invalid attack.knowledge: {v}...must be ['white'|'black']"
            )
        return v

    @validator("module")
    def validate_module(cls, module_name):
        print(cls, module_name)
        try:
            import_module(module_name)
        except Exception as e:
            log.error("Invalid Attack.module: {}".format(module_name))
            raise e
        return module_name

    # @validator("name")
    # def validate_name(cls, function_name):
    #     print(cls.module, function_name)
    #     try:
    #         getattr(cls.module, function_name)
    #     except Exception as e:
    #         log.error("Invalid Attack.module name: {}".format(function_name))
    #         raise e
    #     return function_name
    #
    # @validator("type")
    # def validate_type(cls, type):
    #     valid_types = [None, "preloaded", "patch", "sweep"]
    #     if type not in valid_types:
    #         raise ValueError(f"Attack type: {type} is invalid... must be in {valid_types}")


class Dataset(BaseModel):
    """Armory Dataclass For `Dataset` Parameters"""

    name: str
    module: str
    framework: str
    batch_size: int


class Defense(BaseModel):
    """Armory Dataclass for `Defense` Parameters"""

    name: str
    type: str
    module: str
    kwargs: dict


class Metric(BaseModel):
    """Armory Dataclass for Evaluation `Metric` Parameters"""

    means: bool
    perturbation: str
    record_metric_per_sample: bool
    task: list


class Model(BaseModel):
    """Armory Dataclass for `Model` Parameters"""

    name: str
    module: str
    weights_file: str = None
    wrapper_kwargs: dict
    model_kwargs: dict
    fit_kwargs: dict
    fit: bool


class Scenario(BaseModel):
    """Armory Dataclass for `Scenario` Parameters"""

    name: str
    module: str
    kwargs: dict


class SystemConfiguration(BaseModel):
    """Armory Dataclass for Environment Configuration Paramters"""

    docker_image: str = None
    gpus: str = None
    external_github_repo: str = None
    output_dir: str = None
    output_filename: str = None
    use_gpu: bool = False


class Experiment(BaseModel):
    """Armory Dataclass for Experiment Parameters"""

    name: str = None
    _description: str = None
    adhoc: bool = None  # TODO Figure out what this is for...maybe poison?
    attack: Attack = None
    dataset: Dataset
    defense: Defense = None
    metric: Metric = None
    model: Model
    scenario: Scenario
    sysconfig: SystemConfiguration = None

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
