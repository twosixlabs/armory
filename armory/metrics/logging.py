import logging

from armory.metrics import task, query


logger = logging.getLogger(__name__)


# armory.probe("x", <np.array>)  # TODO: register these probes
# armory.probe("x_adv", np.array)
# armory.measure("l2", "x", "x_adv")
# armory.measure(<metric>, <a>, <b>)
# functools.partial


# metrics_logger.log(x=x, x_adv=x_adv)
# metrics_logger.update()  # sample? batch?
# metrics_logger.finalize()  # any final updates
# metrics_logger.results()

# from armory import metrics
# metrics_logger = metrics.logging.getMetricsLogger(__name__)
# # ...
# metrics_logger.probe(name, value, step=None)


class MetricsHandler:
    def __init__(self, probes=("x", "y", "y_pred", "y_target", "x_adv", "y_pred_adv")):
        """
        These probes are defaults, but can be restricted when needed
        """
        self.probes = set()
        self.add_probes(*probes)
        self.values = {}
        self.updated = False
        self.metric_dict = {}

    @classmethod
    def from_config(cls, metrics_config):
        return cls.from_kwargs(**metrics_config)

    @classmethod
    def from_kwargs(
        cls,
        means=True,
        perturbation="linf",
        record_metric_per_sample=True,
        task=("categorical_accuracy",),
        skip_benign=None,
        skip_attack=None,
        targeted=False,
    ):
        m = cls()
        if isinstance(perturbation, str):
            perturbation = [perturbation]
        for metric in perturbation:
            m.add_metric(metric, f"perturbation_{metric}", "x", "x_adv")
        for metric in task:
            if not skip_benign:
                m.add_metric(metric, f"benign_{metric}", "y", "y_pred")
            if not skip_attack:
                m.add_metric(metric, f"adversarial_{metric}", "y", "y_pred_adv")
                if targeted:
                    m.add_metric(metric, f"targeted_{metric}", "y_target", "y_pred_adv")

        return m

    def add_probes(self, *names):
        """
        Add probes with the following names
            add_probes("model_layer_3", "model_logits")
        """
        if len(names) != len(set(names)):
            raise ValueError(f"Names must be unique: {names}")
        for n in names:
            if n in self.probes:
                raise ValueError(f"Probe {n} already set")
        for n in names:
            self.probes.add(n)

    def add_metric(self, description, metric, *probe_names):
        """
        Add a metric. Example usage:
            NewMetricsLogger.add_metric("benign_categorical_accuracy", "categorical_accuracy", "y", "y_pred")
        """
        for p in probe_names:
            if p not in self.probes:
                raise ValueError(f"Probe named {p} not defined")
        if description in self.metric_dict:
            raise ValueError(f"description {description} already added; must be unique")
        if not isinstance(metric, MetricList):
            metric = MetricList(metric)
        self.metrics_dict[description] = (metric, probe_names)

    def update(self, **sensor_values):
        """
        Update sensor values to the latest values
        """
        for k in sensor_values:
            if k not in self.probes:
                raise ValueError(f"{k} is not in probes {self.probes}")
        for k, v in sensor_values.items():
            # TODO: defensive copy?
            self.values[k] = v
        self.updated = True

    def measure(self):
        """
        Measure metrics with the current set of values
        """
        if not self.updated:
            raise ValueError("Must call update before measuring again")
        for _, probe_names in self.metrics_dict.values():
            for p in probe_names:
                if p not in self.values:
                    raise ValueError(f"probe {p} has not been set")

        for metric, probe_names in self.metrics_dict.values():
            metric.update(*[self.values[x] for x in probe_names])

        self.updated = False

    def finalize(self):
        """
        Finalize all metric computations
        """
        for metric, _ in self.metrics_dict.values():
            metric.finalize()

    def results(self):
        """
        Return a dictionary of results
        """
        results = {}
        for description, (metric, _) in self.metrics_dict.items():
            sub_results = metric.results()
            if not results.keys().isdisjoint(sub_results):
                raise ValueError(
                    f"Duplicate keys in {list(results)} and {list(sub_results)}"
                )
            results.update(sub_results)

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


class MetricList:
    """
    Keeps track of all results from a single metric
    """

    def __init__(self, name, function=None, elementwise=True, aggregator="mean"):
        if function is None:
            self.function = query.get_metric(name)
            if self.function is None:
                raise KeyError(f"{name} is not a recognized metric function")
        elif callable(function):
            self.function = function
        else:
            raise ValueError(f"function must be callable or None, not {function}")

        self.name = name
        self.elementwise = bool(elementwise)
        if not self.elementwise and aggregator == "mean":
            raise ValueError("Cannot use 'mean' with non-elementwise metric")
        if name == "word_error_rate":
            self.aggregator = task.aggregrate.total_wer
            self.aggregator_name = "total_wer"
        elif name in (
            "object_detection_AP_per_class",
            "apricot_patch_targeted_AP_per_class",
            "dapricot_patch_targeted_AP_per_class",
        ):
            self.aggregator = task.aggregate.mean_ap
            self.aggregator_name = "mean_" + self.name
        elif aggregator == "mean":
            self.aggregator = task.aggregrate.mean
            self.aggregator_name = "mean_" + self.name
        elif not aggregator:
            self.aggregator = None
        else:
            raise ValueError(f"Aggregator {aggregator} not recognized")

        self._values = []
        self._results = {}

    def update(self, *probes):
        # TODO: handle sample-wise vs batch-wise inputs
        if self.elementwise:
            self._values(self.function(*probes))
        else:
            self._values.append(probes)  # NOTE: no intermediate values

    def finalize(self):
        if self.elementwise:
            r = {self.name: list(self._values)}
            if self.aggregator is not None:
                r[self.aggregator_name] = self.aggregator(self._values)
        else:
            r_dict = self.function(self._values)  # NOTE: may need unzipping
            r = {self.name: r_dict}
            if self.aggregator is not None:
                r[self.aggregator_name] = self.aggregator(r_dict)
        self._results = r

    def results(self):
        return self._results
