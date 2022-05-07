"""
Metrics for scenarios
"""

from armory.metrics import perturbation, task, statistical


SUPPORTED_METRICS = {}
for namespace in (
    perturbation.batch,
    task.batch,
    task.aggregate,
    task.population,
    statistical.registered,
):
    assert not any(k in namespace for k in SUPPORTED_METRICS)
    SUPPORTED_METRICS.update(namespace)


def get_supported_metric(name):
    try:
        function = SUPPORTED_METRICS[name]
    except KeyError:
        raise KeyError(f"{name} is not part of armory.metrics")
    if isinstance(function, type) and issubclass(function, object):
        # If a class is given, instantiate it
        function = function()
    assert callable(function), f"function {name} is not callable"
    return function
