"""
Helper utilies to load things from armory configuration files.
"""

from importlib import import_module

from art.attacks import Attack
from art.defences import Preprocessor, Postprocessor, Trainer, Transformer
from art.classifiers import Classifier

from armory.data.datasets import ArmoryDataGenerator


DEFENSES = (Preprocessor, Postprocessor, Trainer, Transformer)


def load(sub_config):
    module = import_module(sub_config["module"])
    fn = getattr(module, sub_config["name"])
    args = sub_config.get("args", [])
    kwargs = sub_config.get("kwargs", {})
    return fn(*args, **kwargs)


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


def load_defense(defense_config, classifier):
    defense_module = import_module(defense_config["module"])
    defense_fn = getattr(defense_module, defense_config["name"])
    defense = defense_fn(classifier=classifier, **defense_config["kwargs"])
    if not any(isinstance(defense, x) for x in DEFENSES):
        raise TypeError(f"defense {defense} is not a defense instance: {DEFENSES}")
    return defense
