import importlib
from typing import Callable

from armory.metrics import compute, perturbation, statistical, task
from armory.metrics.common import get_result_formatter, result_formatter, supported


def _instantiate_validate(function, name, instantiate_if_class=True):
    if isinstance(function, type) and issubclass(function, object):
        if instantiate_if_class:
            function = function()
    if not callable(function):
        raise ValueError(f"function {name} is not callable")
    return function


def is_supported(name):
    """
    Return whether given name is a supported metric
    """
    return name in supported


def get_supported_metric(name, instantiate_if_class=True):
    try:
        function = supported[name]
    except KeyError:
        raise KeyError(f"{name} is not part of armory.metrics")
    return _instantiate_validate(
        function, name, instantiate_if_class=instantiate_if_class
    )


def load(string, instantiate_if_class=True):
    """
    Import load a function from the given '.'-separated identifier string
    """
    tokens = string.split(".")
    if not all(token.isidentifier() for token in tokens):
        raise ValueError(f"{string} is not a valid '.'-separated set of identifiers")
    if len(tokens) < 2:
        raise ValueError(f"{string} not a valid module and function path")

    errors = []
    for i in range(len(tokens) - 1, 0, -1):
        module_name = ".".join(tokens[:i])
        metric_name = ".".join(tokens[i:])
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            errors.append(f"ImportError: import {module_name}")
            continue
        try:
            obj = module
            for name in tokens[i:]:
                obj = getattr(obj, name)
            function = obj
            break
        except AttributeError:
            errors.append(
                f"AttributeError: module {module_name} has no attribute {metric_name}"
            )
    else:
        error_string = "\n    ".join([""] + errors)
        raise ValueError(
            f"Could not import metric {string}. "
            f"The following errors occurred: {error_string}"
        )

    return _instantiate_validate(
        function, string, instantiate_if_class=instantiate_if_class
    )


def get(name, instantiate_if_class=True):
    """
    Get the given metric, first by looking for it in armory, then via import
        instantiate_if_class - if a class is returned, instantiate it when True
    """
    try:
        return get_supported_metric(name, instantiate_if_class=instantiate_if_class)
    except KeyError:
        return load(name, instantiate_if_class=instantiate_if_class)
