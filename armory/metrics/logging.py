import logging

import numpy as np

# TODO: Fix
from armory.metrics import (
    SUPPORTED_METRICS,
    apricot_patch_targeted_AP_per_class,
    object_detection_AP_per_class,
)


logger = logging.getLogger(__name__)


class MetricList:
    """
    Keeps track of all results from a single metric
    """

    def __init__(self, name, function=None):
        if function is None:
            try:
                # TODO: FIX
                self.function = SUPPORTED_METRICS[name]
            except KeyError:
                raise KeyError(f"{name} is not part of armory.utils.metrics")
        elif callable(function):
            self.function = function
        else:
            raise ValueError(f"function must be callable or None, not {function}")
        self.name = name
        self._values = []
        self._inputs = []

    def clear(self):
        self._values.clear()

    def append(self, *args, **kwargs):
        value = self.function(*args, **kwargs)
        self._values.extend(value)

    def __iter__(self):
        return self._values.__iter__()

    def __len__(self):
        return len(self._values)

    # TODO: functions should return json-able outputs?

    def values(self):
        return list(self._values)

    def mean(self):
        return sum(float(x) for x in self._values) / len(self._values)

    def append_inputs(self, *args):
        self._inputs.append(args)

    def total_wer(self):
        # checks if all values are tuples from the WER metric
        if all(isinstance(wer_tuple, tuple) for wer_tuple in self._values):
            total_edit_distance = 0
            total_words = 0
            for wer_tuple in self._values:
                total_edit_distance += wer_tuple[0]
                total_words += wer_tuple[1]
            return float(total_edit_distance / total_words)
        else:
            raise ValueError("total_wer() only for WER metric")

    def AP_per_class(self):
        # Computed at once across all samples
        y_s = [i[0] for i in self._inputs]
        y_preds = [i[1] for i in self._inputs]
        return object_detection_AP_per_class(y_s, y_preds)

    def apricot_patch_targeted_AP_per_class(self):
        # Computed at once across all samples
        y_s = [i[0] for i in self._inputs]
        y_preds = [i[1] for i in self._inputs]
        return apricot_patch_targeted_AP_per_class(y_s, y_preds)


class MetricsLogger:
    """
    Uses the set of task and perturbation metrics given to it.
    """

    def __init__(
        self,
        task=None,
        perturbation=None,
        means=True,
        record_metric_per_sample=False,
        profiler_type=None,
        computational_resource_dict=None,
        skip_benign=None,
        skip_attack=None,
        targeted=False,
        **kwargs,
    ):
        """
        task - single metric or list of metrics
        perturbation - single metric or list of metrics
        means - whether to return the mean value for each metric
        record_metric_per_sample - whether to return metric values for each sample
        """
        self.tasks = [] if skip_benign else self._generate_counters(task)
        self.adversarial_tasks = [] if skip_attack else self._generate_counters(task)
        self.targeted_tasks = (
            self._generate_counters(task) if targeted and not skip_attack else []
        )
        self.perturbations = (
            [] if skip_attack else self._generate_counters(perturbation)
        )
        self.means = bool(means)
        self.full = bool(record_metric_per_sample)
        self.computational_resource_dict = {}
        if not self.means and not self.full:
            logger.warning(
                "No per-sample metric results will be produced. "
                "To change this, set 'means' or 'record_metric_per_sample' to True."
            )
        if (
            not self.tasks
            and not self.perturbations
            and not self.adversarial_tasks
            and not self.targeted_tasks
        ):
            logger.warning(
                "No metric results will be produced. "
                "To change this, set one or more 'task' or 'perturbation' metrics"
            )

    def _generate_counters(self, names):
        if names is None:
            names = []
        elif isinstance(names, str):
            names = [names]
        elif not isinstance(names, list):
            raise ValueError(
                f"{names} must be one of (None, str, list), not {type(names)}"
            )
        return [MetricList(x) for x in names]

    @classmethod
    def from_config(cls, config, skip_benign=None, skip_attack=None, targeted=None):
        if skip_benign:
            config["skip_benign"] = skip_benign
        if skip_attack:
            config["skip_attack"] = skip_attack
        return cls(**config, targeted=targeted)

    def clear(self):
        for metric in self.tasks + self.adversarial_tasks + self.perturbations:
            metric.clear()

    def update_task(self, y, y_pred, adversarial=False, targeted=False):
        if targeted and not adversarial:
            raise ValueError("benign task cannot be targeted")
        tasks = (
            self.targeted_tasks
            if targeted
            else self.adversarial_tasks
            if adversarial
            else self.tasks
        )
        for metric in tasks:
            if metric.name in [
                "object_detection_AP_per_class",
                "apricot_patch_targeted_AP_per_class",
            ]:
                metric.append_inputs(y, y_pred)
            else:
                metric.append(y, y_pred)

    def update_perturbation(self, x, x_adv):
        for metric in self.perturbations:
            metric.append(x, x_adv)

    def log_task(self, adversarial=False, targeted=False):
        if targeted:
            if adversarial:
                metrics = self.targeted_tasks
                wrt = "target"
                task_type = "adversarial"
            else:
                raise ValueError("benign task cannot be targeted")
        elif adversarial:
            metrics = self.adversarial_tasks
            wrt = "ground truth"
            task_type = "adversarial"
        else:
            metrics = self.tasks
            wrt = "ground truth"
            task_type = "benign"

        for metric in metrics:
            # Do not calculate mean WER, calcuate total WER
            if metric.name == "word_error_rate":
                logger.info(
                    f"Word error rate on {task_type} examples relative to {wrt} labels: "
                    f"{metric.total_wer():.2%}"
                )
            elif metric.name == "object_detection_AP_per_class":
                average_precision_by_class = metric.AP_per_class()
                logger.info(
                    f"object_detection_mAP on {task_type} examples relative to {wrt} labels: "
                    f"{np.fromiter(average_precision_by_class.values(), dtype=float).mean():.2%}."
                    f" object_detection_AP by class ID: {average_precision_by_class}"
                )
            elif metric.name == "apricot_patch_targeted_AP_per_class":
                apricot_patch_targeted_AP_by_class = (
                    metric.apricot_patch_targeted_AP_per_class()
                )
                logger.info(
                    f"apricot_patch_targeted_mAP on {task_type} examples: "
                    f"{np.fromiter(apricot_patch_targeted_AP_by_class.values(), dtype=float).mean():.2%}."
                    f" apricot_patch_targeted_AP by class ID: {apricot_patch_targeted_AP_by_class}"
                )
            else:
                logger.info(
                    f"Average {metric.name} on {task_type} test examples relative to {wrt} labels: "
                    f"{metric.mean():.2%}"
                )

    def results(self):
        """
        Return dict of results
        """
        results = {}
        for metrics, prefix in [
            (self.tasks, "benign"),
            (self.adversarial_tasks, "adversarial"),
            (self.targeted_tasks, "targeted"),
            (self.perturbations, "perturbation"),
        ]:
            for metric in metrics:
                if metric.name == "object_detection_AP_per_class":
                    average_precision_by_class = metric.AP_per_class()
                    results[f"{prefix}_object_detection_mAP"] = np.fromiter(
                        average_precision_by_class.values(), dtype=float
                    ).mean()
                    results[f"{prefix}_{metric.name}"] = average_precision_by_class
                    continue

                if metric.name == "apricot_patch_targeted_AP_per_class":
                    apricot_patch_targeted_AP_by_class = (
                        metric.apricot_patch_targeted_AP_per_class()
                    )
                    results[f"{prefix}_apricot_patch_targeted_mAP"] = np.fromiter(
                        apricot_patch_targeted_AP_by_class.values(), dtype=float
                    ).mean()
                    results[
                        f"{prefix}_{metric.name}"
                    ] = apricot_patch_targeted_AP_by_class
                    continue

                if self.full:
                    results[f"{prefix}_{metric.name}"] = metric.values()
                if self.means:
                    try:
                        results[f"{prefix}_mean_{metric.name}"] = metric.mean()
                    except ZeroDivisionError:
                        raise ZeroDivisionError(
                            f"No values to calculate mean in {prefix}_{metric.name}"
                        )
                if metric.name == "word_error_rate":
                    try:
                        results[f"{prefix}_total_{metric.name}"] = metric.total_wer()
                    except ZeroDivisionError:
                        raise ZeroDivisionError(
                            f"No values to calculate WER in {prefix}_{metric.name}"
                        )

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
