"""
Set up the meters from a standard config file
"""

import numpy as np

from armory.instrument.instrument import (
    LogWriter,
    Meter,
    ResultsWriter,
    get_hub,
)
from armory.logs import log
from armory.utils import metrics


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
        self.task = task
        self.task_kwargs = task_kwargs
        self.perturbation = perturbation
        self.means = means
        self.include_benign = include_benign
        self.include_adversarial = include_adversarial
        self.include_targeted = include_targeted
        self.record_final_only = not bool(record_metric_per_sample)
        if task is not None:
            if isinstance(task, str):
                self.task = [task]
            if isinstance(task_kwargs, dict):
                self.task_kwargs = [task_kwargs]

            construct_meters_for_task_metrics(
                self.task,
                use_mean=means,
                include_benign=self.include_benign,
                include_adversarial=self.include_adversarial,
                include_targeted=self.include_targeted,
                task_kwargs=self.task_kwargs,
                record_final_only=self.record_final_only,
            )
        if perturbation is not None and self.include_adversarial:
            if isinstance(perturbation, str):
                perturbation = [perturbation]
            construct_meters_for_perturbation_metrics(
                perturbation, use_mean=means, record_final_only=self.record_final_only
            )

        self.results_writer = ResultsWriter(
            sink=self._sink, max_record_size=max_record_size
        )
        get_hub().connect_writer(self.results_writer, default=True)

        self.metric_results = None

    def add_tasks_wrt_benign_predictions(self):
        """
        Measure adversarial predictions w.r.t. benign predictions
            Convenience method for CARLA object detection scenario
        """
        if self.task is not None and self.include_adversarial:
            construct_meters_for_task_metrics_wrt_benign_predictions(
                self.task,
                use_mean=self.means,
                task_kwargs=self.task_kwargs,
                record_final_only=self.record_final_only,
            )

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


def construct_meters_for_perturbation_metrics(
    names, use_mean=True, record_final_only=True
):
    if use_mean:
        final = np.mean
    else:
        final = None

    hub = get_hub()
    for name in names:
        metric = metrics.get_supported_metric(name)
        hub.connect_meter(
            Meter(
                f"perturbation_{name}",
                metric,
                "scenario.x",
                "scenario.x_adv",
                final=final,
                final_name=f"perturbation_mean_{name}",
                record_final_only=record_final_only,
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
        # TODO: once metrics have also been updated, rewrite to be less error prone
        #    E.g., if someone renames this from "benign_word_error_rate" to "benign_wer"
        if "word_error_rate" in name:
            if "total_word_error_rate" not in name:
                result = metrics.get_supported_metric("total_wer")(result)
            total, (num, denom) = result
            f_result = f"total={total:.2%}, {num}/{denom}"
        elif "entailment" in name:
            if "total_entailment" not in name:
                result = metrics.get_supported_metric("total_entailment")(result)
            total = sum(result.values())
            f_result = (
                f"contradiction: {result['contradiction']}/{total}, "
                f"neutral: {result['neutral']}/{total}, "
                f"entailment: {result['entailment']}/{total}"
            )
        elif any(m in name for m in MEAN_AP_METRICS):
            if "input_to" in name:
                for m in MEAN_AP_METRICS:
                    if m in name:
                        metric = metrics.get_supported_metric(m)
                        result = metrics.MeanAP(metric)(result)
                        break
            f_result = f"{result}"
        elif any(m in name for m in QUANTITY_METRICS):
            # Don't include % symbol
            f_result = f"{np.mean(result):.2}"
        else:
            f_result = f"{np.mean(result):.2%}"
        log.success(
            f"{name} on {self.task_type} examples w.r.t. {self.wrt} labels: {f_result}"
        )


def _task_metric(
    name,
    metric_kwargs,
    use_mean=True,
    include_benign=True,
    include_adversarial=True,
    include_targeted=True,
    record_final_only=True,
):
    """
    Return list of meters generated for this specific task
    """
    meters = []
    metric = metrics.get_supported_metric(name)
    final_kwargs = {}
    if name in MEAN_AP_METRICS:
        final_suffix = name
        final = metrics.MeanAP(metric)
        final_kwargs = metric_kwargs

        name = f"input_to_{name}"
        metric = metrics.get_supported_metric("identity_unzip")
        metric_kwargs = None
        record_final_only = True
    elif name == "entailment":
        final = metrics.get_supported_metric("total_entailment")
        final_suffix = "total_entailment"
    elif name == "word_error_rate":
        final = metrics.get_supported_metric("total_wer")
        final_suffix = "total_word_error_rate"
    elif use_mean:
        final = np.mean
        final_suffix = f"mean_{name}"
    else:
        final = None
        final_suffix = ""
        record_final_only = False

    if include_benign:
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
                record_final_only=record_final_only,
            )
        )
    else:
        meters.append(None)
    if include_adversarial:
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
                record_final_only=record_final_only,
            )
        )
    else:
        meters.append(None)
    if include_targeted:
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
                record_final_only=record_final_only,
            )
        )
    else:
        meters.append(None)
    return meters


def construct_meters_for_task_metrics(
    names,
    use_mean=True,
    include_benign=True,
    include_adversarial=True,
    include_targeted=True,
    record_final_only=True,
    task_kwargs=None,
):
    if not any([include_benign, include_adversarial, include_targeted]):
        return
    if task_kwargs is None:
        task_kwargs = [None] * len(names)
    elif len(names) != len(task_kwargs):
        raise ValueError(f"{len(names)} tasks but {len(task_kwargs)} task_kwargs")
    hub = get_hub()

    tuples = []
    for name, metric_kwargs in zip(names, task_kwargs):
        task = _task_metric(
            name,
            metric_kwargs,
            use_mean=use_mean,
            include_benign=include_benign,
            include_adversarial=include_adversarial,
            include_targeted=include_targeted,
            record_final_only=record_final_only,
        )
        tuples.append(task)

    benign, adversarial, targeted = zip(*tuples)
    meters = [
        m for tup in tuples for m in tup if m is not None
    ]  # unroll list of tuples

    for m in meters:
        hub.connect_meter(m)

    if include_benign:
        hub.connect_writer(ResultsLogWriter(), meters=benign)
    if include_adversarial:
        hub.connect_writer(ResultsLogWriter(adversarial=True), meters=adversarial)
    if include_targeted:
        hub.connect_writer(
            ResultsLogWriter(adversarial=True, targeted=True), meters=targeted
        )


def _task_metric_wrt_benign_predictions(
    name, metric_kwargs, use_mean=True, record_final_only=True
):
    """
    Return the meter generated for this specific task
    Return list of meters generated for this specific task
    """
    metric = metrics.get_supported_metric(name)
    final_kwargs = {}
    if name in MEAN_AP_METRICS:
        final_suffix = name
        final = metrics.MeanAP(metric)
        final_kwargs = metric_kwargs

        name = f"input_to_{name}"
        metric = metrics.get_supported_metric("identity_unzip")
        metric_kwargs = None
        record_final_only = True
    elif name == "entailment":
        final = metrics.get_supported_metric("total_entailment")
        final_suffix = "total_entailment"
    elif name == "word_error_rate":
        final = metrics.get_supported_metric("total_wer")
        final_suffix = "total_word_error_rate"
    elif use_mean:
        final = np.mean
        final_suffix = f"mean_{name}"
    else:
        final = None
        final_suffix = ""
        record_final_only = False

    return Meter(
        f"adversarial_{name}_wrt_benign_preds",
        metric,
        "scenario.y_pred",
        "scenario.y_pred_adv",
        metric_kwargs=metric_kwargs,
        final=final,
        final_name=f"adversarial_{final_suffix}_wrt_benign_preds",
        final_kwargs=final_kwargs,
        record_final_only=record_final_only,
    )


def construct_meters_for_task_metrics_wrt_benign_predictions(
    names, use_mean=True, task_kwargs=None, record_final_only=True
):
    if task_kwargs is None:
        task_kwargs = [None] * len(names)
    elif len(names) != len(task_kwargs):
        raise ValueError(f"{len(names)} tasks but {len(task_kwargs)} task_kwargs")
    hub = get_hub()

    meters = []
    for name, metric_kwargs in zip(names, task_kwargs):
        meter = _task_metric_wrt_benign_predictions(
            name,
            metric_kwargs,
            use_mean=use_mean,
            record_final_only=record_final_only,
        )
        meters.append(meter)

    for m in meters:
        hub.connect_meter(m)

    hub.connect_writer(
        ResultsLogWriter(
            adversarial=True,
            used_preds_as_labels=True,
        ),
        meters=meters,
    )
