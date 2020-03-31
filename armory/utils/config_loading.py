"""
Helper utilies to load things from armory configuration files.
"""
from importlib import import_module


def load_dataset(dataset_config, *args, **kwargs):
    """
    Loads a dataset from configuration file
    """
    dataset_module = import_module(dataset_config["module"])
    dataset_fn = getattr(dataset_module, dataset_config["name"])
    batch_size = dataset_config["batch_size"]
    return dataset_fn(batch_size=batch_size, *args, **kwargs)


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

    # If no preprocessing function with model return `None`
    preprocessing_fn = getattr(model_module, "preprocessing_fn", None)
    return model, preprocessing_fn


def load_attack(attack_config):
    raise NotImplementedError


def load_defense(defense_config):
    raise NotImplementedError
