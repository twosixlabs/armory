"""
Metrics for scenarios

Outputs are lists of python variables amenable to JSON serialization:
    e.g., bool, int, float
    numpy data types and tensors generally fail to serialize
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def categorical_accuracy(y, y_pred):
    """
    Return the categorical accuracy of the predictions
    """
    if y.shape == y_pred.shape:
        return [int(x) for x in list(y == y_pred)]
    elif y.ndim + 1 == y_pred.ndim:
        return [int(x) for x in list(y == np.argmax(y_pred, axis=1))]
    else:
        raise ValueError(f"{y} and {y_pred} have mismatched dimensions")


def top_5_categorical_accuracy(y, y_pred):
    """
    Return the top 5 categorical accuracy of the predictions
    """
    return top_n_categorical_accuracy(y, y_pred, 5)


def top_n_categorical_accuracy(y, y_pred, n):
    if n < 1:
        raise ValueError(f"n must be a positive integer, not {n}")
    n = int(n)
    if len(y) != len(y_pred):
        raise ValueError("y and y_pred are of different length")
    if y.shape == y_pred.shape:
        raise ValueError("Must supply multiple predictions for top 5 accuracy")
    elif y.ndim + 1 == y_pred.ndim:
        y_pred_top5 = np.argsort(y_pred, axis=1)[:, -n:]
        return [int(y[i] in y_pred_top5[i]) for i in range(len(y))]
    else:
        raise ValueError(f"{y} and {y_pred} have mismatched dimensions")


def norm(x, x_adv, ord):
    """
    Return the given norm over a batch, outputting a list of floats
    """
    # cast to float first to prevent overflow errors
    diff = (x.astype(float) - x_adv.astype(float)).reshape(x.shape[0], -1)
    values = np.linalg.norm(diff, ord=ord, axis=1)
    return list(float(x) for x in values)


def linf(x, x_adv):
    """
    Return the L-infinity norm over a batch of inputs as a float
    """
    return norm(x, x_adv, np.inf)


def l2(x, x_adv):
    """
    Return the L2 norm over a batch of inputs as a float
    """
    return norm(x, x_adv, 2)


def l1(x, x_adv):
    """
    Return the L1 norm over a batch of inputs as a float
    """
    return norm(x, x_adv, 1)


def lp(x, x_adv, p):
    """
    Return the Lp norm over a batch of inputs as a float
    """
    if p <= 0:
        raise ValueError(f"p must be positive, not {p}")
    return norm(x, x_adv, p)


def l0(x, x_adv):
    """
    Return the L0 'norm' over a batch of inputs as a float
    """
    return norm(x, x_adv, 0)


class MetricCounter:
    """
    Keeps track of all results from a single metric
    """

    def __init__(self, name):
        try:
            self.function = globals()[name]
        except KeyError:
            raise KeyError(f"{name} is not part of armory.utils.metrics")
        self.name = name
        self.clear()

    def clear(self):
        self._values = []

    def update(self, *args, **kwargs):
        value = self.function(*args, **kwargs)
        self._values.extend(value)

    def values(self):
        return self._values

    def mean(self):
        return sum(float(x) for x in self._values) / len(self._values)


class MetricsListCounter:
    """
    Uses the set of task and perturbation metrics given to it.
    """

    def __init__(self, task=None, perturbation=None, means=True, full=False):
        """
        task - single metric or list of metrics
        perturbation - single metric or list of metrics
        means - whether to return the mean values for each metric
        full - whether to return the full values for each metric
        """
        self.tasks = self._generate_counters(task)
        self.adversarial_tasks = self._generate_counters(task)
        self.perturbations = self._generate_counters(perturbation)
        self.means = bool(means)
        self.full = bool(full)

    def _generate_counters(self, names):
        if names is None:
            names = []
        elif isinstance(names, str):
            names = [names]
        elif not isinstance(names, list):
            raise ValueError(
                f"{names} must be one of (None, str, list), not {type(names)}"
            )
        return [MetricCounter(x) for x in names]

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def clear(self):
        for metric in self.tasks + self.adversarial_tasks + self.perturbations:
            metric.clear()

    def update_task(self, y, y_pred, adversarial=False):
        tasks = self.adversarial_tasks if adversarial else self.tasks
        for task in tasks:
            task.update(y, y_pred)

    def update_perturbation(self, x, x_adv):
        for perturbation in self.perturbations:
            perturbation.update(x, x_adv)

    def log_task(self, adversarial=False):
        if adversarial:
            metrics = self.adversarial_tasks
            task_type = "adversarial"
        else:
            metrics = self.tasks
            task_type = "benign"

        for metric in metrics:
            logger.info(
                f"Average {metric.name} on {task_type} test examples: "
                f"{metric.mean():.2%}"
            )

    def results(self, prefix=""):
        """
        Return dict of results

        prefix - string to prefix metric name with, e.g., "benign" or "adversarial"
            will insert a "_" character before prefix if present
        """
        if not isinstance(prefix, str):
            raise ValueError(f"prefix must be a string, not {prefix}")
        if len(prefix) > 1:
            prefix = prefix + "_"

        results = {}
        results["task"] = task_results = {}
        for metric in self.tasks:
            prefix = "benign_"
            if self.full:
                task_results[f"{prefix}{metric.name}"] = metric.values()
            if self.means:
                task_results[f"{prefix}mean_{metric.name}"] = metric.mean()
        for metric in self.adversarial_tasks:
            prefix = "adversarial_"
            if self.full:
                task_results[f"{prefix}{metric.name}"] = metric.values()
            if self.means:
                task_results[f"{prefix}mean_{metric.name}"] = metric.mean()
        results["perturbation"] = perturbation_results = {}
        for metric in self.perturbations:
            prefix = "perturbation_"
            if self.full:
                perturbation_results[f"{prefix}{metric.name}"] = metric.values()
            if self.means:
                perturbation_results[f"{prefix}mean_{metric.name}"] = metric.mean()
        return results


class AverageMeter:
    """
    Computes and stores the average and current value

    Taken from https://github.com/craston/MARS/blob/master/utils.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
