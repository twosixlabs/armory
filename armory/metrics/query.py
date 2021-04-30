"""
Lookup functions for metrics
"""

from armory.metrics import task, perturbation


def _get_namespace(module, batch=True):
    if batch:
        return module.batch
    else:
        return module.element


def supported_metrics(batch=True):
    task_namespace = _get_namespace(task, batch=batch)
    perturbation_namespace = _get_namespace(perturbation, batch=batch)

    supported = []
    supported.extend(vars(task_namespace))
    supported.extend(vars(perturbation_namespace))
    return sorted(set(supported))


def get_metric(name, batch=True):
    task_namespace = _get_namespace(task, batch=batch)
    perturbation_namespace = _get_namespace(perturbation, batch=batch)

    metric = getattr(task_namespace, name, None)
    if metric is None:
        metric = getattr(perturbation_namespace, name, None)
    return metric
