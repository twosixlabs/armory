"""
Set up the meters from a standard config file
"""

import numpy as np

from armory.instrument import (
    get_hub,
    LogWriter,
    Meter,
    ResultsWriter,
)
from armory.logs import log
from armory.utils import metrics


# Major TODO:
#    1) add probe.update() calls in scenarios
#    2) replace old MetricsLogger with this one
#    3) remove old MetricsLogger deprecated calls
#    4) remove old code from armory.utils.metrics
#    5) add test suite for measurement
class MetricsLogger:
    """
    Uses the set of task and perturbation metrics given to it.
    """

    def __init__(
        self, task=None, task_kwargs=None, perturbation=None, means=True, **kwargs,
    ):
        """
        task - single metric or list of metrics
        task_kwargs - a single dict or list of dicts (same length as task) or None
        perturbation - single metric or list of metrics
        means - whether to return the mean value for each metric
        record_metric_per_sample - whether to return metric values for each sample
        """
        if kwargs.pop("record_metric_per_sample", None) is not None:
            log.warning(
                "record_metric_per_sample is deprecated: now always treated as True"
            )
        if kwargs.pop("profiler_type", None) is not None:
            log.warning(
                "ignoring profiler_type in MetricsLogger instantiation. Use metrics.resource_context to log computational resource usage"
            )
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs}")

        if task is not None:
            if isinstance(task, str):
                task = [task]
            if isinstance(task_kwargs, dict):
                task_kwargs = [task_kwargs]

            task_metrics(
                task, use_mean=means, include_target=True, task_kwargs=task_kwargs
            )
        if perturbation is not None:
            if isinstance(perturbation, str):
                perturbation = [perturbation]
            perturbation_metrics(perturbation, use_mean=means)

        get_hub().connect_writer(ResultsWriter(sink=self._sink))

        self.metric_results = None
        self.computational_resource_dict = {}

    @classmethod
    def from_config(cls, config, skip_benign=None, skip_attack=None, targeted=None):
        del skip_benign
        del skip_attack
        del targeted
        return cls(**config)

    def _computational_results(self):
        results = {}
        for name in self.computational_resource_dict:
            entry = self.computational_resource_dict[name]
            if "execution_count" not in entry or "total_time" not in entry:
                raise ValueError(
                    "Computational resource dictionary entry corrupted, missing data."
                )
            total_time = entry["total_time"]
            execution_count = entry["execution_count"]
            average_time = total_time / execution_count
            results[
                f"Avg. CPU time (s) for {execution_count} executions of {name}"
            ] = average_time
            if "stats" in entry:
                results[f"{name} profiler stats"] = entry["stats"]
        return results

    def _sink(self, results_dict):
        """
        sink for results_writer to write to
        """
        self.metric_results = results_dict

    def _metric_results(self):
        get_hub().close()
        if self.metric_results is None:
            log.warning("No metric results received from ResultsWriter")
            return {}
        return self.metric_results

    def results(self):
        results = {}
        results.update(self._computational_results())
        results.update(self._metric_results())
        return results

    # DEPRECATED METHODS
    def _deprecation_error(self, name):
        log.exception(
            f"Deprecation. Using old armory.utils.metrics.MetricsLogger.{name} API. "
            "Ignoring. Will cause error in version 0.16. Please update code!"
        )

    def clear(self):
        self._deprecation_error("clear")

    def update_perturbation(self, x, x_adv):
        self._deprecation_error("update_perturbation")

    def log_task(self, adversarial=False, targeted=False, used_preds_as_labels=False):
        self._deprecation_error("log_task")

    # END DEPRECATED METHODS


def perturbation_metrics(names, use_mean=True):
    if use_mean:
        final = np.mean
    else:
        final = None

    hub = get_hub()
    for name in names:
        metric = metrics.get_supported_metric(name)
        hub.connect_meter(
            Meter(
                name,
                metric,
                "scenario.x",
                "scenario.x_adv",
                final=final,
                final_name=f"mean_{name}",
            )
        )


MEAN_AP_METRICS = [
    "object_detection_AP_per_class",
    "apricot_patch_targeted_AP_per_class",
    "dapricot_patch_targeted_AP_per_class",
    "carla_od_AP_per_class",
]
# quanity_metrics only impacts output printing
QUANTITY_METRICS = [
    "object_detection_hallucinations_per_image",
    "carla_od_hallucinations_per_image",
]
# TODO: fix used_preds_as_labels usage (only used in CARLA scenario)


# TODO: move to armory.utils.metrics
def total_wer(sample_wers):
    """
    Aggregate a list of per-sample word error rate tuples (edit_distance, words)
        Return global_wer, (total_edit_distance, total_words)
    """
    # checks if all values are tuples from the WER metric
    if all(isinstance(wer_tuple, tuple) for wer_tuple in sample_wers):
        total_edit_distance = 0
        total_words = 0
        for wer_tuple in sample_wers:
            total_edit_distance += wer_tuple[0]
            total_words += wer_tuple[1]
        if total_words:
            global_wer = float(total_edit_distance / total_words)
        else:
            global_wer = float("nan")
        return global_wer, (total_edit_distance, total_words)
    else:
        raise ValueError("total_wer() only for WER metric aggregation")


# TODO: move to armory.utils.metrics
def identity(*args):
    return args


# TODO: move to armory.utils.metrics
class MeanAP:
    def __init__(self, ap_metric):
        self.ap_metric = ap_metric

    def __call__(self, *args, **kwargs):
        ap = self.ap_metric(*args, **kwargs)
        mean_ap = np.fromiter(ap.values(), dtype=float).mean()
        return {"mean": mean_ap, "class": ap}


class ResultsLogWriter(LogWriter):
    """
    Logs successful results (designed for task metrics)
    """

    def __init__(
        self,
        adversarial=False,
        targeted=False,
        used_preds_as_labels=False,
        log_level: str = "SUCCESS",
    ):
        super().__init__(log_level=log_level)
        if targeted:
            if adversarial:
                self.wrt = "target"
                self.task_type = "adversarial"
            else:
                raise ValueError("benign task cannot be targeted")
        elif adversarial:
            if used_preds_as_labels:
                self.wrt = "benign predictions as"
            else:
                self.wrt = "ground truth"
            self.task_type = "adversarial"
        else:
            self.wrt = "ground truth"
            self.task_type = "benign"

    def _write(self, name, batch, result):
        if name == "word_error_rate":
            total, (num, denom) = total_wer(result)
            f_result = f"total={total:.2%}, {num}/{denom}"
        elif name in MEAN_AP_METRICS:
            f_result = f"{result}"
        elif name in QUANTITY_METRICS:
            # Don't include % symbol
            f_result = f"{np.mean(result):.2}"
        else:
            f_result = f"{np.mean(result):.2%}"
        log.success(
            f"{name} on {self.task_type} examples w.r.t. {self.wrt} labels: {f_result}"
        )


def _task_metric(name, metric_kwargs, use_mean=True, include_target=True):
    """
    Return list of meters generated for this specific task
    """
    meters = []
    metric = metrics.get_supported_metric(name)
    final_kwargs = {}
    if name in MEAN_AP_METRICS:
        final_suffix = name
        final = MeanAP(metric)
        final_kwargs = metric_kwargs

        name = "input_to_{name}"
        metric = identity
        metric_kwargs = None
    elif name == "word_error_rate":
        final = total_wer
        final_suffix = "total_word_error_rate"
    elif use_mean:
        final = np.mean
        final_suffix = "mean_{name}"
    else:
        final = None
        final_suffix = ""

    meters.append(
        Meter(
            f"benign_{name}",
            metric,
            "scenario.y",
            "scenario.y_pred",
            metric_kwargs=metric_kwargs,
            final=final,
            final_name=f"benign_{final_suffix}",
            final_kwargs=final_kwargs,
        )
    )
    meters.append(
        Meter(
            f"adversarial_{name}",
            metric,
            "scenario.y",
            "scenario.y_pred_adv",
            metric_kwargs=metric_kwargs,
            final=final,
            final_name=f"adversarial_{final_suffix}",
            final_kwargs=final_kwargs,
        )
    )
    if include_target:
        meters.append(
            Meter(
                f"targeted_{name}",
                metric,
                "scenario.y_target",
                "scenario.y_pred_adv",
                metric_kwargs=metric_kwargs,
                final=final,
                final_name=f"targeted_{final_suffix}",
                final_kwargs=final_kwargs,
            )
        )
    return meters


def task_metrics(names, use_mean=True, include_target=True, task_kwargs=None):
    if task_kwargs is None:
        task_kwargs = [None] * len(names)
    elif len(names) != len(task_kwargs):
        raise ValueError(f"{len(names)} tasks but {len(task_kwargs)} task_kwargs")
    hub = get_hub()

    tuples = []
    for name, metric_kwargs in zip(names, task_kwargs):
        task = _task_metric(
            name, metric_kwargs, use_mean=use_mean, include_target=include_target
        )
        tuples.append(task)

    if include_target:
        benign, adversarial, targeted = zip(*tuples)
    else:
        benign, adversarial = zip(*tuples)
    meters = [m for tup in tuples for m in tup]  # unroll list of tuples

    for m in meters:
        hub.connect_meter(m)

    hub.connect_writer(ResultsLogWriter(), meters=benign)
    hub.connect_writer(ResultsLogWriter(adversarial=True), meters=adversarial)
    if include_target:
        hub.connect_writer(
            ResultsLogWriter(adversarial=True, targeted=True), meters=targeted
        )
