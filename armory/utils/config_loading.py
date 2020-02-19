"""
Helper utilies to load things from armory configuration files.
"""
from importlib import import_module


def load_dataset(dataset_config, *args, **kwargs):
    """
    Return dataset or raise KeyError

    Convenience function, essentially.
    """
    dataset_module = import_module(dataset_config["module"])
    dataset_fn = getattr(dataset_module, dataset_config["name"])
    return dataset_fn(*args, **kwargs)


def load_model(model_config, *args, **kwargs):
    raise NotImplementedError
