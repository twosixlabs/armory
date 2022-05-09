from pydantic import BaseModel
from armory.logs import log
import os
import yaml
import json
from typing import List, Dict, Any
from armory.utils import set_overrides
from enum import Enum

# from pydoc import locate

#
# class ArbitraryValue(BaseModel):
#     value: str
#     type: str


class ArbitraryDict(BaseModel):
    __root__: Dict[str, Any]

    def __getattr__(self, item):
        if item in self.__root__:
            return self.__root__[item]

    def __setattr__(self, key, value):
        if key in self.__root__:
            self.__root__[key] = value

    # def __getattr__(self, item):  # if you want to use '.'
    #     print(f'Getting Item: {item}')
    #     if item in self.__root__:
    #         print(f"Found item in root: {self.__root__[item]}")
    #         tp = locate(self.__root__[item].Type)
    #         return tp(self.__root__[item].Value)


# class Values(BaseModel):
#     """ Base Model for Arbitrary Dict Type attribute values"""
#     value: str
#     type: str
#     #
#     # def __getattr__(self, item):
#     #     if item == "type"
#     #     tp = locate(self.type)
#
#
# class ArbitraryDict(BaseModel):
#     __root__: Optional[Dict[str, Values]]
#     __values__ = {}
#
#     def __setattr__(self, key, value):
#         log.debug(f"Setting {key} to {value}")
#         if isinstance(value, dict):
#             self.__values__[key] = locate(value['type'])(value['value'])
#         elif isinstance(value, Values):
#             self.__values__[key] = locate(value.type)(value.value)
#         else:
#             self.__values__[key] = value
#
#     def __getattr__(self, item):
#         if item not in self.__values__:
#             msg = f"Arbitrary Dict: {self.__values__} does not have attribute: {item}"
#             log.exception(msg)
#             raise AttributeError(msg)
#         else:
#             return self.__values__[item]


# class ArbitraryDict(BaseModel):
#     """Base Model substitute for Dict
#
#     ArbitraryDict is inteded to be used as a data model
#     when the `keys` of the incoming data are not known
#     and/or can't be anticipated.  The main example of this
#     is in the `kwarg` type options within the armory experiment
#     file.  To use this:
#
#     # Say we have data model called `AttackParameters` and we want
#     it to have a `kwargs` attribute that can store arbitrary key/value
#     pairs, then we would write something like:
#
#     class AttackParameters(BaseModel):
#         kwargs: ArbitraryDict
#     """
#     __root__: Optional[Dict[str, Values]]
#     __values__: Dict
#
#     # def __init__(self, *args, **kwargs):
#     #     super().__init__(*args, **kwargs)
#     #     log.info("Constructing Arbitrary Dict")
#         # for k,v in self.__root__.items():
#         #     print(k,v)
#         #     tp = locate(v.type)
#         #     setattr(self, k, tp(v.value))
#
#     # def __getattr__(self, item):  # need this to access using `.`
#     #     tp = locate(self.__root__[item].type)
#     #     return tp(self.__root__[item].value)
#     #     # return self.__root__[item].value
#
#     def __setattr__(self, key, value):
#         if isinstance(value, dict):
#             tp = locate(value['type'])
#         else
#         # Get class for specified value type
#         tp = locate(self.__root__[key].type)
#
#         # return the value typed correctly
#         self.__root__[key] = tp(value)
#         setattr(self, key, tp(value))
#
#     def as_dict(self):
#         tmp = {}
#         for k,v in self.__root__:
#             tp = locate(v.type)
#             tmp[k] = tp(v.value)
#         return tmp


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
    use_label: bool = True


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
    kwargs: ArbitraryDict = None


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
    # TODO: currently armory.enging does not work if attack
    #  is not specified... so this must be set
    attack: AttackParameters
    dataset: DatasetParameters
    defense: DefenseParameters = None
    metric: MetricParameters  # TODO: Figure out why this can't be none
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

        log.debug(f"Loaded YAML: \n{data}\n")
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
        print(self)

        return self.dict()

    def save(self, filename):
        with open(filename, "w") as f:
            f.write(self.pretty_print())
