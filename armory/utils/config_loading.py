"""
Helper utilies to load things from armory configuration files.
"""

from importlib import import_module
import logging

# import torch before tensorflow to ensure torch.utils.data.DataLoader can utilize
#     all CPU resources when num_workers > 1
try:
    import torch  # noqa: F401
except ImportError:
    pass
from art.attacks import Attack
from art import defences
from art.classifiers import Classifier

from armory.data.datasets import ArmoryDataGenerator, CheckGenerator

logger = logging.getLogger(__name__)


def load(sub_config):
    module = import_module(sub_config["module"])
    fn = getattr(module, sub_config["name"])
    args = sub_config.get("args", [])
    kwargs = sub_config.get("kwargs", {})
    return fn(*args, **kwargs)


def load_fn(sub_config):
    module = import_module(sub_config["module"])
    return getattr(module, sub_config["name"])


def load_dataset(dataset_config, *args, **kwargs):
    """
    Loads a dataset from configuration file
    """
    dataset_module = import_module(dataset_config["module"])
    dataset_fn = getattr(dataset_module, dataset_config["name"])
    batch_size = dataset_config["batch_size"]
    framework = dataset_config.get("framework", "numpy")
    dataset = dataset_fn(batch_size=batch_size, framework=framework, *args, **kwargs)
    if not isinstance(dataset, ArmoryDataGenerator):
        raise ValueError(f"{dataset} is not an instance of {ArmoryDataGenerator}")
    if dataset_config.get("check_run"):
        return CheckGenerator(dataset)
    return dataset


def load_model(model_config):
    """
    Loads a model and preprocessing function from configuration file

    preprocessing_fn can be a tuple of functions or None values
        If so, it applies to training and inference separately
    """
    model_module = import_module(model_config["module"])
    model_fn = getattr(model_module, model_config["name"])
    weights_file = model_config.get("weights_file", None)
    model = model_fn(
        model_config["model_kwargs"], model_config["wrapper_kwargs"], weights_file
    )
    if not isinstance(model, Classifier):
        raise TypeError(f"{model} is not an instance of {Classifier}")
    if not weights_file and not model_config["fit"]:
        logger.warning(
            "You're attempting to evaluate an unfitted model with no "
            "pre-trained weights!"
        )

    preprocessing_fn = getattr(model_module, "preprocessing_fn", None)
    if preprocessing_fn is not None:
        if isinstance(preprocessing_fn, tuple):
            if len(preprocessing_fn) != 2:
                raise ValueError(
                    f"preprocessing tuple length {len(preprocessing_fn)} != 2"
                )
            elif not all([x is None or callable(x) for x in preprocessing_fn]):
                raise TypeError(
                    f"preprocessing_fn tuple elements {preprocessing_fn} must be None or callable"
                )
        elif not callable(preprocessing_fn):
            raise TypeError(
                f"preprocessing_fn {preprocessing_fn} must be None, tuple, or callable"
            )
    return model, preprocessing_fn


def load_attack(attack_config, classifier):
    attack_module = import_module(attack_config["module"])
    attack_fn = getattr(attack_module, attack_config["name"])
    attack = attack_fn(classifier=classifier, **attack_config["kwargs"])
    if not isinstance(attack, Attack):
        raise TypeError(f"attack {attack} is not an instance of {Attack}")
    return attack


def load_adversarial_dataset(config, preprocessing_fn=None, **kwargs):
    if config.get("type") != "preloaded":
        raise ValueError(f"attack type must be 'preloaded', not {config.get('type')}")
    dataset_module = import_module(config["module"])
    dataset_fn = getattr(dataset_module, config["name"])
    dataset_kwargs = config["kwargs"]
    dataset_kwargs.update(kwargs)
    if "description" in dataset_kwargs:
        dataset_kwargs.pop("description")
    dataset = dataset_fn(preprocessing_fn=preprocessing_fn, **dataset_kwargs)
    if not isinstance(dataset, ArmoryDataGenerator):
        raise ValueError(f"{dataset} is not an instance of {ArmoryDataGenerator}")
    if config.get("check_run"):
        return CheckGenerator(dataset)
    return dataset


def _check_defense_api(defense, defense_baseclass):
    if not isinstance(defense, defense_baseclass):
        raise ValueError(
            f"defense {defense} does not extend type {type(defense_baseclass)}"
        )


def load_defense_wrapper(defense_config, classifier):
    defense_type = defense_config["type"]
    if defense_type == "Transformer":
        raise NotImplementedError("Transformer API not yet implemented into scenarios")
    elif defense_type != "Trainer":
        raise ValueError(
            f"Wrapped defenses must be of type Trainer, found {defense_type}"
        )

    defense_module = import_module(defense_config["module"])
    defense_fn = getattr(defense_module, defense_config["name"])
    defense = defense_fn(classifier=classifier, **defense_config["kwargs"])
    _check_defense_api(defense, defences.Trainer)

    return defense


def load_defense_internal(defense_config, classifier):
    defense = load(defense_config)

    defense_type = defense_config["type"]
    if defense_type == "Preprocessor":
        _check_defense_api(defense, defences.Preprocessor)
        if classifier.preprocessing_defences:
            classifier.preprocessing_defences.append(defense)
        else:
            classifier.preprocessing_defences = [defense]
    elif defense_type == "Postprocessor":
        _check_defense_api(defense, defences.Postprocessor)
        if classifier.postprocessing_defences:
            classifier.postprocessing_defences.append(defense)
        else:
            classifier.postprocessing_defences = [defense]
    else:
        raise ValueError(
            f"Internal defenses must be of either type [Preprocessor, Postprocessor], found {defense_type}"
        )

    return classifier
