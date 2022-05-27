"""
Supporting tools for metrics
"""

import functools

import numpy as np

from armory.logs import log


class MetricNameSpace:
    """
    Used to keep track of metrics and make them easily discoverable and enumerable
    """

    def __setattr__(self, name, function):
        if name.startswith("_"):
            raise ValueError(f"Function name '{name}' cannot start with '_'")
        if hasattr(self, name):
            raise ValueError(f"Cannot overwrite existing function {name}")
        if not callable(function):
            raise ValueError(f"{name} function {function} is not callable")
        super().__setattr__(name, function)

    def __delattr__(self, name):
        raise ValueError("Deletion not allowed")

    def _names(self):
        return sorted(x for x in self.__dict__ if not x.startswith("_"))

    def __contains__(self, name):
        return name in self._names()

    def __repr__(self):
        """
        Show the existing non-underscore names
        """
        return str(self._names())

    def __iter__(self):
        for name in self._names():
            yield name, self[name]

    def __getitem__(self, name):
        if not hasattr(self, name):
            raise KeyError(name)
        return getattr(self, name)

    def __setitem__(self, name, function):
        setattr(self, name, function)


def set_namespace(namespace, metric, name=None):
    """
    Set the namespace, getting the metric name if none given, and return the metric
    """
    if name is None:
        name = metric.__name__
    setattr(namespace, name, metric)
    return metric


def as_batch(element_metric):
    """
    Return a batchwise metric function from an elementwise metric function
    """

    @functools.wraps(element_metric)
    def wrapper(x_batch, x_adv_batch, **kwargs):
        x_batch = list(x_batch)
        x_adv_batch = list(x_adv_batch)
        if len(x_batch) != len(x_adv_batch):
            raise ValueError(
                f"len(a_batch) {len(x_batch)} != len(b_batch) {len(x_adv_batch)}"
            )
        y = []
        for x, x_adv in zip(x_batch, x_adv_batch):
            y.append(element_metric(x, x_adv, **kwargs))
        try:
            y = np.array(y)
        except ValueError:
            # Handle ragged arrays
            y = np.array(y, dtype=object)
        return y

    if wrapper.__doc__ is None:
        log.warning(f"{element_metric.__name__} has no doc string")
        wrapper.__doc__ = ""
    wrapper.__doc__ = "Batch version of:\n" + wrapper.__doc__
    wrapper.__name__ = "batch_" + wrapper.__name__
    # note: repr(wrapper) defaults to the element_metric, not __name__
    # See: https://stackoverflow.com/questions/10875442/possible-to-change-a-functions-repr-in-python
    return wrapper
