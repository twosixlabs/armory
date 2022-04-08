"""
OOP structure for Armory logging

This roughly follows the python logging (Logger / Handler) framework,
with some key differences
"""

from armory import log


_PROBES = {}


class Probe:
    def __init__(self, name):
        self.name = name
        self.meters = []
        self._hooks = {}
        self._warned = False

    def connect(self, meter):
        self.meters.append(meter)

    def update(self, *preprocessing, **named_values):
        """
        Measure values, applying preprocessing if a meter is available

        Example: probe.update(lambda x: x.detach().cpu().numpy(), a=layer_3_output)

        named_values can be any object, tuple, dict, etc.
            To add attributes, you could do:
                probe.update(data_point=(x_i, is_poisoned))
        """
        if not self.meters and not self._warned:
            log.warning(f"No Meter set up for probe {self.name}!")
            self._warned = True
            return

        # Determine what values are being measured
        measured = []
        for name in named_values:
            for meter in self.meters:
                if meter.is_measuring(name):
                    measured.append(name)
                    break

        # Preprocess measured values
        output_values = {}
        for name in measured:
            value = named_values[name]
            for p in preprocessing:
                value = p(value)
            output_values[name] = value

        # Output values to meters
        for meter in self.meters:
            meter.update(
                **{
                    name: value
                    for name, value in output_values
                    if meter.is_measuring(name)
                }
            )

    def hook(self, module, *preprocessing, input=None, output=None, mode="pytorch"):
        if mode == "pytorch":
            return self.hook_torch(module, *preprocessing, input=input, output=output)
        elif mode == "tf":
            return self.hook_tf(module, *preprocessing, input=input, output=output)
        raise ValueError(f"mode {mode} not in ('pytorch', 'tf')")

    def hook_tf(self, module, *preprocessing, input=None, output=None):
        raise NotImplementedError("hooking not ready for tensorflow")
        # NOTE:
        # https://discuss.pytorch.org/t/get-the-activations-of-the-second-to-last-layer/55629/6
        # TensorFlow hooks
        # https://www.tensorflow.org/api_docs/python/tf/estimator/SessionRunHook
        # https://github.com/tensorflow/tensorflow/issues/33478
        # https://github.com/tensorflow/tensorflow/issues/33129
        # https://stackoverflow.com/questions/48966281/get-intermediate-output-from-keras-tensorflow-during-prediction
        # https://stackoverflow.com/questions/59493222/access-output-of-intermediate-layers-in-tensor-flow-2-0-in-eager-mode/60945216#60945216

    def hook_torch(self, module, *preprocessing, input=None, output=None):
        if not hasattr(module, "register_forward_hook"):
            raise ValueError(
                f"module {module} does not have method 'register_forward_hook'. Is it a torch.nn.Module?"
            )
        if input == "" or (input is not None and not isinstance(input, str)):
            raise ValueError(f"input {input} must be None or a non-empty string")
        if output == "" or (output is not None and not isinstance(output, str)):
            raise ValueError(f"output {output} must be None or a non-empty string")
        if input is None and output is None:
            raise ValueError("input and output cannot both be None")
        if module in self._hooks:
            raise ValueError(f"module {module} is already hooked")

        def hook_fn(hook_module, hook_input, hook_output):
            del hook_module
            key_values = {}
            if input is not None:
                key_values[input] = hook_input
            if output is not None:
                key_values[output] = hook_output
            self.update(*preprocessing, **key_values)

        hook = module.register_forward_hook(hook_fn)
        self._hooks[module] = (hook, "pytorch")

    def unhook(self, module):
        hook, mode = self._hooks[module]
        if mode == "pytorch":
            hook.remove()
        elif mode == "tf":
            raise NotImplementedError()
        else:
            raise ValueError(f"mode {mode} not in ('pytorch', 'tf')")
        self._hooks.pop(module)
        self._hooks[module].remove()  # TODO: this seems wrong


class Meter:
    def is_measuring(self, name):
        """
        Return whether the meter is measuring the given name
        """
        return False

    def update(self, **named_values):
        pass

    def measure(self, *args, **kwargs):
        raise NotImplementedError("Implement in subclasses of Meter if needed")


class NullMeter(Meter):
    """
    Ensure probe preprocessing is all done, but otherwise do nothing.
    """

    def is_measuring(self, name):
        return True


class LogMeter(Meter):
    """
    Log all probed values
    """

    def __init__(self):
        super().__init__()
        self.values = {}

    def is_measuring(self, name):
        return True

    def update(self, **named_values):
        for name, value in named_values.items():
            self.values[name] = value
            log.info(
                f"LogMeter: step {self.step}, {self.stage}_{name} = {value}, type = {type(value)}"
            )


# class Procedure:  # instead of "Process" or "Experiment", which are overloaded
#    """
#    Provides context for meter and probes
#    """
#
#    def __init__(self, *, stage="", step=0):
#        self.set_stage(stage)
#        self.set_step(step)
#
#    def set_stage(self, stage: str):
#        if not isinstance(stage, str):
#            raise ValueError(f"'stage' must be a str, not {type(stage)}")
#        self.stage = stage
#
#    def set_step(self, step: int):
#        if not isinstance(step, int):
#            raise ValueError(f"'step' must be an int, not {type(step)}")
#        self.step = step
#
#
# #Measure the L2 distance at the preprocessor output between the benign and adversarial instances
# #In model code:
#  from armory import instrument
#  probe = instrument.get_probe("model")
# # ...
#  output = preprocessor(x_data)
#  probe.update(lambda x: x.detach().cpu().numpy(), prep_output=output)
#
# # In own code?
#  meter = instrument.AdvMeter("model.x", "model.y")  # --> "model.adv_x", "model.adv_y"
#  meter = instrument.LogMeter("benign:model.prep_output", "adv:model.prep_output", np.linalg.norm2)
# # where does it go?
#
#
# class AdvMeter(Meter):
#    STAGES = "benign", "adversary"
#    def __init__(self, *names):
#        self.names = names
#        self.values = {}
#        self.stage = None
#        for name in self.names:
#            for stage in self.STAGES:
#                self.values[self._full_name(name, stage)] = None
#
#    def _full_name(self, name, stage=None):
#        if stage is None:
#            stage = self.stage
#            if stage is None:
#                raise ValueError("must call set_stage")
#        return f"{stage}:{name}"
#
#    def is_measuring(self, name):
#        return name in self.names
#
#    def set_stage(self, stage):
#        if stage not in self.STAGES:
#            raise ValueError(f"stage {stage} not in {self.STAGES}")
#        self.stage = stage
#
#    def update(self, **named_values):
#        for name, value in named_values.items():
#            self.values[self._full_name(name)] = value
#
#
# "model.x[stage=adv], model.x[stage=ben]"
# probe.connect(meter)
#
#
#
# # outputs:
# (sample=i, metric_name)
#
#
# class GlobalMeter(Meter):
#    def __init__(self):
#        self.values = []
#
#    def update(self):
#        self.values.append()


# class MetricsMeter(Meter):
#    def __init__(self, **kwargs):
#        super().__init__(**kwargs)
#        self.probes = set()
#        self.values = {}
#        self.metrics_dict = {}
#        self.stages_mapping = {}
#
#    @classmethod
#    def from_config(cls, config, skip_benign=False, skip_attack=False, targeted=False):
#        return cls.from_kwargs(**config)
#
#    @classmethod
#    def from_kwargs(
#        cls,
#        means=True,
#        perturbation="linf",
#        profiler_type=None,  # Ignored
#        record_metric_per_sample=True,
#        task=("categorical_accuracy",),
#        skip_benign=None,
#        skip_attack=None,
#        targeted=False,
#    ):
#        if not record_metric_per_sample:
#            log.warning("record_metric_per_sample overridden to True")
#
#        m = cls()
#        kwargs = dict(aggregator="mean" if bool(means) else None,)
#
#        if isinstance(perturbation, str):
#            perturbation = [perturbation]
#        for metric in perturbation:
#            m.add_metric(
#                f"perturbation_{metric}",
#                metric,
#                "x",
#                "x_adv",
#                stages="perturbation",
#                **kwargs,
#            )
#        for metric in task:
#            if not skip_benign:
#                m.add_metric(
#                    f"benign_{metric}",
#                    metric,
#                    "y",
#                    "y_pred",
#                    stages="benign_task",
#                    **kwargs,
#                )
#            if not skip_attack:
#                m.add_metric(
#                    f"adversarial_{metric}",
#                    metric,
#                    "y",
#                    "y_pred_adv",
#                    stages="adversarial_task",
#                    **kwargs,
#                )
#                if targeted:
#                    m.add_metric(
#                        f"targeted_{metric}",
#                        metric,
#                        "y_target",
#                        "y_pred_adv",
#                        stages="adversarial_task",
#                        **kwargs,
#                    )
#
#        return m
#
#    def is_measuring(self, name):
#        return name in self.probes
#
#    def update(self, **probe_named_values):
#        for probe_name in probe_named_values:
#            if not self.is_measuring(probe_name):
#                raise ValueError(f"{probe_name} is not being measured")
#
#        for probe_name, value in probe_named_values.items():
#            # NOTE: ignore step for now
#            self.values[probe_name] = value
#
#    def add_metric(
#        self,
#        description,
#        metric,
#        *probe_names,
#        stages=None,
#        aggregator="mean",
#        record_metric_per_sample=True,
#        elementwise=True,
#    ):
#        """
#        Add a metric. Example usage:
#            metrics_meter = MetricsMeter()
#            metrics_meter.add_metric("benign_categorical_accuracy", "categorical_accuracy", "y", "y_pred")
#
#            metrics_meter.add_metric("mean_output", lambda x: x.mean(), "y_pred")
#            metrics_meter.add_metric("l2 perturbation", metrics.perturbation.l2, "x", "adv_x")
#            metrics_meter.add_metric("constant", lambda: 1)
#        """
#        if description in self.metrics_dict:
#            raise ValueError(f"Metric description '{description}' already exists")
#        if not isinstance(metric, MetricList):
#            metric = MetricList(description, function=metric, aggregator=aggregator,)
#
#        if stages is None:
#            stages = []
#        elif isinstance(stages, str):
#            stages = [stages]
#        else:
#            stages = list(stages)
#        stages.append(description)
#
#        self.probes.update(probe_names)
#        self.metrics_dict[description] = (metric, probe_names)
#        for stage in stages:
#            self.stages_mapping[stage] = description
#
#    def measure(self, *names):
#        """
#        Measure the metrics based on the current values
#
#        names - list of named metrics to measure
#            if empty, measure all metrics
#        """
#
#        if not names:
#            descriptions = list(self.metrics_dict)
#            probes = self.probes
#        else:
#            descriptions = []
#            probes = set()
#            for name in names:
#                if name not in self.stages_mapping:
#                    raise ValueError(f"metric or stage {name} has not been added")
#                description = self.stages_mapping[name]
#                descriptions.append(description)
#                _, probe_names = self.metrics_dict[description]
#                probes.update(probe_names)
#
#        for p in probes:
#            if p not in self.values:
#                raise ValueError(f"probe {p} value has not been set")
#
#        for name in descriptions:
#            metric, probe_names = self.metrics_dict[name]
#            metric.update(*(self.values[x] for x in probe_names))
#
#    def finalize(self):
#        """
#        Finalize all metric computations
#        """
#        for metric, _ in self.metrics_dict.values():
#            metric.finalize()
#
#    def results(self):
#        results = {}
#        for description, (metric, _) in self.metrics_dict.items():
#            sub_results = metric.results()
#            if not results.keys().isdisjoint(sub_results):
#                log.error(
#                    f"Overwritting duplicate keys in {list(results)} and {list(sub_results)}"
#                )
#            results.update(sub_results)
#        return results
#
#
# class MetricList:
#    """
#    Keeps track of all results from a single metric
#    """
#
#    def __init__(self, name, function=None, aggregator="mean"):
#        if callable(function):
#            self.function = function
#        elif isinstance(function, str):
#            self.function = metrics.get(function)
#        elif function is None:
#            self.function = metrics.get(name)
#        else:
#            raise ValueError(f"function must be callable or None, not {function}")
#
#        self.name = name
#        self.elementwise = True
#        if name == "word_error_rate":
#            self.aggregator = metrics.task.aggregate.total_wer
#            self.aggregator_name = "total_wer"
#        elif name in (
#            "object_detection_AP_per_class",
#            "apricot_patch_targeted_AP_per_class",
#            "dapricot_patch_targeted_AP_per_class",
#        ):
#            self.aggregator = metrics.task.aggregate.mean_ap
#            self.aggregator_name = "mean_" + self.name
#            self.elementwise = False
#        elif aggregator == "mean":
#            self.aggregator = metrics.task.aggregate.mean
#            self.aggregator_name = "mean_" + self.name
#        elif not aggregator:
#            self.aggregator = None
#        else:
#            raise ValueError(f"Aggregator {aggregator} not recognized")
#        self._values = []
#        self._results = {}
#
#    def update(self, *function_args):
#        if self.elementwise:
#            self._values.extend(self.function(*function_args))
#        else:
#            self._values.extend(zip(*function_args))
#
#    def finalize(self):
#        r = {}
#        if self.elementwise:
#            r[self.name] = list(self._values)
#            if self.aggregator is not None:
#                r[self.aggregator_name] = self.aggregator(self._values)
#        else:
#            computed_values = self.function(*zip(*self._values))
#            r[self.name] = computed_values
#            if self.aggregator is not None:
#                r[self.aggregator_name] = self.aggregator(computed_values)
#        self._results = r
#
#    def results(self):
#        return self._results
#
#    def results_keys(self):
#        if self.aggregator is None:
#            return set([self.name, self.aggregator_name])
#        return set([self.name])


def get_probe(name=None):
    if not name:
        name = None
    if name is not None:
        raise NotImplementedError(
            "name hierarchy not implemented yet. Set name to None"
        )
    global _PROBES
    if name not in _PROBES:
        _PROBES[name] = Probe()
    return _PROBES[name]


def connect(meter, name=None):
    probe = get_probe(name=name)
    probe.connect(meter)
