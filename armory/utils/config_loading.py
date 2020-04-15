"""
Helper utilies to load things from armory configuration files.
"""

from importlib import import_module

from art.attacks import Attack
from art import defences
from art.classifiers import Classifier

from armory.data.datasets import ArmoryDataGenerator


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
    dataset = dataset_fn(batch_size=batch_size, *args, **kwargs)
    if not isinstance(dataset, ArmoryDataGenerator):
        raise ValueError(f"{dataset} is not an instance of {ArmoryDataGenerator}")
    return dataset


def load_model(model_config):
    """
    Loads a model and preprocessing function from configuration file
    """
    model_module = import_module(model_config["module"])
    model_fn = getattr(model_module, model_config["name"])
    weights_file = model_config.get("weights_file", None)
    model = model_fn(
        model_config["model_kwargs"], model_config["wrapper_kwargs"], weights_file
    )
    if not isinstance(model, Classifier):
        raise TypeError(f"{model} is not an instance of {Classifier}")

    preprocessing_fn = getattr(model_module, "preprocessing_fn", None)
    if preprocessing_fn is not None and not callable(preprocessing_fn):
        raise TypeError(f"preprocessing_fn {preprocessing_fn} must be None or callable")
    return model, preprocessing_fn


def load_attack(attack_config, classifier):
    attack_module = import_module(attack_config["module"])
    attack_fn = getattr(attack_module, attack_config["name"])
    attack = attack_fn(classifier=classifier, **attack_config["kwargs"])
    if not isinstance(attack, Attack):
        raise TypeError(f"attack {attack} is not an instance of {Attack}")
    return attack


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
