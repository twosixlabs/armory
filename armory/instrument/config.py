"""
Set up the meters from a standard config file
"""

import numpy as np

from armory import metrics
from armory.instrument.instrument import (
    GlobalMeter,
    Meter,
    ResultsLogWriter,
    ResultsWriter,
    get_hub,
)
from armory.logs import log


class MetricsLogger:
    """
    Uses the set of task and perturbation metrics given to it.
    """

    def __init__(
        self,
        task=None,
        task_kwargs=None,
        perturbation=None,
        means=True,
        include_benign=True,
        include_adversarial=True,
        include_targeted=True,
        record_metric_per_sample=False,
        max_record_size=2**20,
    ):
        """
        task - single metric or list of metrics
        task_kwargs - a single dict or list of dicts (same length as task) or None
        perturbation - single metric or list of metrics
        means - whether to return the mean value for each metric
        record_metric_per_sample - whether to return metric values for each sample
        include_benign - whether to include benign task metrics
        include_adversarial - whether to include adversarial task metrics
        include_targeted - whether to include targeted task metrics
        max_record_size - maximum number of bytes in a record (for ResultsWriter)
        """
        self.means = means
        self.include_benign = include_benign
        self.include_adversarial = include_adversarial
        self.include_targeted = include_targeted
        self.record_final_only = not bool(record_metric_per_sample)
        self.task = self._ensure_list_of_strings(task)
        if task:
            if isinstance(task_kwargs, dict):
                self.task_kwargs = [task_kwargs]
            else:
                self.task_kwargs = task_kwargs

            if self.include_benign:
                self.add_benign_tasks()
            if self.include_adversarial:
                self.add_adversarial_tasks()
            if self.include_targeted:
                self.add_targeted_tasks()
        self.perturbation = self._ensure_list_of_strings(perturbation)
        if perturbation and self.include_adversarial:
            self.add_perturbations()

        self.results_writer = ResultsWriter(
            sink=self._sink, max_record_size=max_record_size
        )
        get_hub().connect_writer(self.results_writer, default=True)

        self.metric_results = None

    @staticmethod
    def _ensure_list_of_strings(arg):
        if arg is None:
            return []
        if isinstance(arg, str):
            return [arg]
        arg_list = list(arg)
        for x in arg_list:
            if not isinstance(x, str):
                raise ValueError(f"{x} is not a str")
        return arg_list

    def connect(self, meters, writer=None):
        hub = get_hub()
        for m in meters:
            hub.connect_meter(m)
        if writer is not None:
            hub.connect_writer(writer, meters=meters)

    def add_perturbations(self):
        meters = perturbation_meters(
            self.perturbation,
            use_mean=self.means,
            record_final_only=self.record_final_only,
        )
        self.connect(meters)

    def add_benign_tasks(self):
        meters = task_meters(
            self.task,
            "benign_",
            "scenario.y",
            "scenario.y_pred",
            use_mean=self.means,
            record_final_only=self.record_final_only,
            task_kwargs=self.task_kwargs,
        )
        writer = ResultsLogWriter(
            format_string="{name} on benign examples w.r.t. ground truth labels: {result}"
        )
        self.connect(meters, writer)

    def add_adversarial_tasks(self):
        meters = task_meters(
            self.task,
            "adversarial_",
            "scenario.y",
            "scenario.y_pred_adv",
            use_mean=self.means,
            record_final_only=self.record_final_only,
            task_kwargs=self.task_kwargs,
        )
        writer = ResultsLogWriter(
            format_string="{name} on adversarial examples w.r.t. ground truth labels: {result}"
        )
        self.connect(meters, writer)

    def add_targeted_tasks(self):
        meters = task_meters(
            self.task,
            "targeted_",
            "scenario.y_target",
            "scenario.y_pred_adv",
            use_mean=self.means,
            record_final_only=self.record_final_only,
            task_kwargs=self.task_kwargs,
        )
        writer = ResultsLogWriter(
            format_string="{name} on adversarial examples w.r.t. target labels: {result}"
        )
        self.connect(meters, writer)

    def add_tasks_wrt_benign_predictions(self):
        """
        Measure adversarial predictions w.r.t. benign predictions
            Convenience method for CARLA object detection scenario
        """
        if self.task is not None and self.include_adversarial:
            meters = task_meters(
                self.task,
                "adversarial_",
                "scenario.y_pred",
                "scenario.y_pred_adv",
                use_mean=self.means,
                record_final_only=self.record_final_only,
                task_kwargs=self.task_kwargs,
                suffix="_wrt_benign_preds",
            )
            writer = ResultsLogWriter(
                format_string="{name} on adversarial examples w.r.t. benign predictions as labels: {result}"
            )
            self.connect(meters, writer)

    @classmethod
    def from_config(
        cls,
        config,
        include_benign=True,
        include_adversarial=True,
        include_targeted=True,
    ):
        return cls(
            include_benign=include_benign,
            include_adversarial=include_adversarial,
            include_targeted=include_targeted,
            **{k: v for k, v in config.items() if k != "profiler_type"},
        )

    def _sink(self, results_dict):
        """
        sink for results_writer to write to
        """
        self.metric_results = results_dict

    def results(self):
        get_hub().close()
        if self.metric_results is None:
            log.warning("No metric results received from ResultsWriter")
            return {}
        return self.metric_results


def perturbation_meters(names, use_mean=True, record_final_only=True):
    meters = []
    if use_mean:
        final = np.mean
    else:
        final = None

    for name in names:
        metric = metrics.get(name)
        meter = Meter(
            f"perturbation_{name}",
            metric,
            "scenario.x",
            "scenario.x_adv",
            final=final,
            final_name=f"perturbation_mean_{name}",
            record_final_only=record_final_only,
        )
        meters.append(meter)
    return meters


def task_meter(
    name,
    prefix,
    metric_kwargs,
    y,
    y_pred,
    use_mean=True,
    record_final_only=True,
    suffix="",
):
    """
    Return meter generated for this specific task
    """
    # this needs to happen before a name change since metrics.get(name) triggers the load for a custom function
    # the custom function is part of a module, so name must include the entire .-separated path of the function,
    # while the custom function that is registered within armory is only referenced by its actual name rather than the entire path
    # there is no immediate issue with this unless the user wishes to modify the behavior of the meter created with the custom function wrapped in a decorator
    # should the user use a decorator (e.g. @populationwise), this leads to a disconnect if the name is not processed after metrics.get to drop the path before creating a meter;
    # the custom function is registered to the namespace (which also happens during metrics.get) of the decorator (e.g. metrics.task.population) simply by its actual name,
    # which means that "if name in metrics.task.population:" will always be false because name will include the entire path, and thus, not be in metrics.task.population
    metric = metrics.get(name)
    # now follow through with an if statement to change the name...
    if "." in name:  # no need to escape period with in operator
        name = name.split(".")[-1]
    result_formatter = metrics.get_result_formatter(name)
    final_kwargs = {}
    if name in metrics.task.population:
        return GlobalMeter(
            f"{prefix}{name}{suffix}",
            metric,
            y,
            y_pred,
            final_kwargs=metric_kwargs,
            final_result_formatter=result_formatter,
        )

    aggregator = metrics.task.get_aggregator_name(name)
    if aggregator is not None:
        final_name = aggregator
        final = metrics.get(final_name)
    elif use_mean:
        final_name = f"mean_{name}"
        final = metrics.get("safe_mean")
    else:
        final_name = ""
        final = None
        record_final_only = False
    final_result_formatter = metrics.get_result_formatter(final_name)

    return Meter(
        f"{prefix}{name}{suffix}",
        metric,
        y,
        y_pred,
        metric_kwargs=metric_kwargs,
        result_formatter=result_formatter,
        final=final,
        final_name=f"{prefix}{final_name}{suffix}",
        final_kwargs=final_kwargs,
        final_result_formatter=final_result_formatter,
        record_final_only=record_final_only,
    )


def task_meters(
    names,
    prefix,
    y,
    y_pred,
    task_kwargs=None,
    use_mean=True,
    record_final_only=True,
    suffix="",
):
    """
    Return list of meters for the given setup
    """
    meters = []
    if task_kwargs is None:
        task_kwargs = [None] * len(names)
    elif len(names) != len(task_kwargs):
        raise ValueError(f"{len(names)} tasks but {len(task_kwargs)} task_kwargs")

    for name, metric_kwargs in zip(names, task_kwargs):
        meter = task_meter(
            name,
            prefix,
            metric_kwargs,
            y,
            y_pred,
            use_mean=use_mean,
            record_final_only=record_final_only,
            suffix=suffix,
        )
        meters.append(meter)
    return meters
