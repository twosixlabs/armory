"""
Lookup functions for metrics
"""

from armory.metrics import task, perturbation


def _get_namespace(module, batch=True):
    if batch:
        return module.batch
    else:
        return module.element


def supported(batch=True):
    task_namespace = _get_namespace(task, batch=batch)
    perturbation_namespace = _get_namespace(perturbation, batch=batch)

    supported = []
    supported.extend(vars(task_namespace))
    supported.extend(vars(perturbation_namespace))
    return sorted(set(supported))


def get(name, batch=True):
    task_namespace = _get_namespace(task, batch=batch)
    perturbation_namespace = _get_namespace(perturbation, batch=batch)

    metric = getattr(task_namespace, name, None)
    if metric is None:
        metric = getattr(perturbation_namespace, name, None)
    if metric is None:
        raise KeyError(f"metric {name} is not supported")
    return metric


def register(name):
    raise NotImplementedError
