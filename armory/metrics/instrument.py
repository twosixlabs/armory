"""
OOP structure for Armory logging

This roughly follows the python logging (Logger / Handler) framework,
with some key differences
"""

import logging

logger = logging.getLogger(__name__)

# Workflow in a different module
# from armory.metrics import logging as mlog

#from armory.metrics import instrumentation
#instrument = instrumentation.getInstrument()  # __name__
#instrument.probe(a=x.detach().cpu().numpy(), b=y)
# ...

#from armory.metrics import instrumentation
#meter = instrumentation.MetricsHandler()
#instrument = instrumentation.getInstrument()  # gets root
#instrument.add_meter(meter)

# Instrument?

# ...
# mlog.update(**key_values)
# Optional alternative:
# mlogger = mlog.getLogger()
# mlogger.update(**key_values)

_PROBES = {}


#from armory import metrics
#probe = metrics.getProbe()
#probe.measure(lambda x: x.detach().cpu().numpy(), a=layer_3_output)
#probe.hook(convnet.layer1[0].conv2, input=None, output="b")


class Probe:
    def __init__(self):
        self.meters = []
        self._hooks = {}
        self._warned = False
        self.measured = set()

    def add_meter(self, meter):
        self.meters.append(meter)

    def update(self, *preprocessing, **named_values):
        """
        Measure values, applying preprocessing if a meter is available
        
        Example: probe.measure(lambda x: x.detach().cpu().numpy(), a=layer_3_output)
        """
        self.measured.update(named_values)
        if not self.meters and not self._warned:
            logger.warning("No Meter set up!")
            self._warned = True
            return
        
        for name, value in named_values.items():
            if not any(meter.is_measuring(name) for meter in self.meters):
                continue

            for p in preprocessing:
                value = p(value)
            
            for meter in self.meters:
                if meter.is_measuring(name):
                    meter.update(**{name: value})

    def hook(self, module, *preprocessing, input=None, output=None, mode="pytorch"):
        if mode == "pytorch":
            return self.hook_torch(module, *preprocessing, input=input, output=output)
        elif mode == "tf":
            return self.hook_tf(module, *preprocessing, input=input, output=output)
        raise ValueError(f"mode {mode} not in ('pytorch', 'tf')")

    def hook_tf(self, module, *preprocessing, input=None, output=None):
        raise NotImplementedError("hooking not ready for tensorflow")

    # https://discuss.pytorch.org/t/get-the-activations-of-the-second-to-last-layer/55629/6
# TensorFlow hooks
# https://www.tensorflow.org/api_docs/python/tf/estimator/SessionRunHook

    def hook_torch(self, module, *preprocessing, input=None, output=None):
        if not hasattr(module, "register_forward_hook"):
            raise ValueError(f"module {module} does not have method 'register_forward_hook'. Is it a torch.nn.Module?")
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
                key_values[output]= hook_output
            self.measure(*preprocessing, **key_values)

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
        self._hooks[module].remove()


class MetricsLogger:
    def __init__(self):
        self.meters = []
        self._warned = False

    def add_handler(self, handler):
        self.handlers.append(handler)

    def update(self, **key_values):
        if not self.handlers and not self._warned:
            logger.warning("No MetricsHandler set up!")
            self._warned = True

        for handler in self.handlers:
            handler.update(**key_values)


class Meter:
    def __init__(self, *, stage="", step=0):
        self.set_stage(stage)
        self.set_step(step)

    def is_measuring(self, name):
        return False

    def update(self, **named_values):
        pass

    def set_stage(self, stage: str):
        if not isinstance(stage, str):
            raise ValueError(f"'stage' must be a str, not {type(stage)}")
        self.stage = stage

    def set_step(self, step: int):
        if not isinstance(step, int):
            raise ValueError(f"'step' must be an int, not {type(step)}")
        self.step = step


class NullMeter(Meter):
    """
    Ensure preprocessing is all done, but otherwise do nothing.
    """

    def is_measuring(self, name):
        return True


class PrintMeter(Meter):
    def is_measuring(self, name):
        return True

    def update(self, **named_values):
        for name, value in named_values.items():
            print(f"PrintMeter: step {self.step}, {self.stage}_{name} = {value}, type = {type(value)}")


class HoldMeter(Meter):
    """
    Hold values as they come in
    """

    def is_measuring(self, name):
        return True

    def update(self, **named_values):
        for name, value in named_values.items():
            print(f"PrintMeter: step {self.step}, {self.stage}_{name} = {value}, type = {type(value)}")


class MetricsMeter(Meter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.probes = set()
        self.values = {}
        self.metrics_dict = {}

    @classmethod
    def from_config(cls, config, skip_benign=False, skip_attack=False, targeted=False):
        return cls()  # TODO

    def with_stage(self, name):
        if self.stage:
            return self.stage + "_" + name
        return name

    def is_measuring(self, name):
        return self.with_stage(name) in self.probes

    def update(self, **probe_named_values):
        for probe_name in probe_named_values:
            if not self.is_measuring(probe_name):
                raise ValueError(f"{probe_name} is not being measured")

        for probe_name, value in probe_named_values.items():
            # NOTE: ignore step for now
            self.values[self.with_stage(probe_name)] = value

    def add_metric(self, description, metric, *probe_names):  # TODO: add name?
        """
        Add a metric. Example usage:
            metrics_meter = MetricsMeter()
            metrics_meter.add_metric("benign_categorical_accuracy", "categorical_accuracy", "y", "y_pred")

            metrics_meter.add_metric("mean_output", lambda x: x.mean(), "y_pred")
            metrics_meter.add_metric("l2 perturbation", metrics.perturbation.l2, "x", "adv_x")
            metrics_meter.add_metric("constant", lambda: 1)
        """
        self.probes.update(probe_names)
        if description in self.metrics_dict:
            raise ValueError(f"Metric {description} already exists with that name")
        self.metrics_dict[description] = (metric, probe_names)
        
    def measure(self, *names):
        """
        Measure the metrics based on the current values

        names - list of named metrics to measure
            if empty, measure all metrics
        """

        if not names:
            names = list(self.metrics_dict)
            probes = self.probes
        else:
            names = list(names)
            probes = set()
            for name in names:
                if name not in self.metrics_dict:
                    raise ValueError(f"metric {name} has not been added")
                _, probe_names = self.metrics_dict[name]
                probes.update(probe_names)
            
        for p in probes:
            if p not in self.values:
                raise ValueError(f"probe {p} value has not been set")

        for name in names:
            metric, probe_names = self.metrics_dict[name]
            result = metric(*(self.values[x] for x in probe_names))
            logger.info(f"result: step {self.step} - {name} {result}")
            # TODO: store results

    def finalize(self):
        raise NotImplementedError

    def results(self):
        raise NotImplementedError


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


def add_meter(meter, name=None):
    probe = get_probe(name=name)
    probe.add_meter(meter)
