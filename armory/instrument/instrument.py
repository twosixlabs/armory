"""
OOP structure for Armory logging

Example 1:
    Use case - measure L2 distance of post-preprocessing for benign and adversarial
    Code:
        # in model
        from armory import instrument
        probe = instrument.get_probe("model")
        ...
        x_post = model_preprocessor(x)
        probe.update(lambda x: x.detach().cpu().numpy(), x_post=x_post)

        # outside of model code
        probe.hook(model, lambda x: x.detach().cpu().numpy(), x_post=x_post)

        # elsewhere (could be reasonably defined in a config file as well)
        from armory import instrument
        from armory import metrics
        meter = instrument.MetricMeter("l2_dist_postprocess", metrics.L2, "model.x_post[benign]", "model.x_post[adversarial]")
        instrument.add_meter(meter)
        instrument.add_writer(PrintWriter())

Design goals:
    probe - very lightweight, minimal or no code (hooking) in target model
        namespace addressable

Functionalities
    Probe - pull info from data source
    ProbeMapper - map from probe output, using provided context, to meter inputs
    Meter - measure quantity at specified intervals
    Writer - takes measured outputs from meters and pushed them to print/file/etc.
    Context - stores context and state for meters and writers, essentially a Hub
"""

import copy

from armory import log


_CONTEXT = None
_PROBES = {}
_METERS = []


class Probe:
    def __init__(self, name="", sink=None):
        self.name = name
        self.sink = sink
        self._hooks = {}
        self._warned = False

    def set_sink(self, sink):
        """
        Sink must implement 'is_measuring' and 'update' APIs
        """
        self.sink = sink

    def update(self, *preprocessing, **named_values):
        """
        Measure values, applying preprocessing if a meter is available

        Example: probe.update(lambda x: x.detach().cpu().numpy(), a=layer_3_output)

        named_values can be any object, tuple, dict, etc.
            To add attributes, you could do:
                probe.update(data_point=(x_i, is_poisoned))
        """
        if self.sink is None and not self._warned:
            log.warning(f"No sink set up for probe {self.name}!")
            self._warned = True
            return

        # Prepend probe name
        if self.name != "":
            named_values = {f"{self.name}.{k}": v for k, v in named_values.items()}

        for name, value in named_values.items():
            if self.sink.is_measuring(name):
                # Apply value preprocessing
                for p in preprocessing:
                    value = p(value)
                # Push to sink
                self.sink.update(name, value)

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
        hook, mode = self._hooks.pop(module)
        if mode == "pytorch":
            hook.remove()
        elif mode == "tf":
            raise NotImplementedError()
        else:
            raise ValueError(f"mode {mode} not in ('pytorch', 'tf')")


def process_meter_arg(arg: str):
    """
    Return the probe variable and stage_filter

    Example strings: 'model.x2[adversarial]', 'scenario.y_pred'
    """
    if "[" in arg:
        if arg.count("[") != 1 and arg.count("]") != 1:
            raise ValueError(f"arg {arg} must have a single matching [] or none")
        arg, filt = arg.split("[")
        if filt[-1] != "]":
            raise ValueError(f"arg {arg} cannot have chars after final ']'")
        stage_filter = filt[:-1].strip()
        # tokens = [x.strip() for x in filt.split(",")]
    else:
        stage_filter = None
    probe_variable = arg

    return probe_variable, stage_filter


class ProbeMapper:
    """
    Map from probe outputs to meters
    """

    def __init__(self, context=None):
        # nested dicts - {probe_variable: {stage_filter: [(meter, arg)]}}
        self.probe_filter_meter_arg = {}
        self.meters = []

    def connect_meter(self, meter):
        for arg in meter.get_arg_names():
            probe_variable, stage_filter = process_meter_arg(arg)
            if probe_variable not in self.probe_filter_meter_arg:
                self.probe_filter_meter_arg[probe_variable] = {}
            filter_map = self.probe_filter_meter_arg[probe_variable]
            if stage_filter not in filter_map:
                filter_map[stage_filter] = []
            meters_args = filter_map[stage_filter]
            if (meter, arg) in meters_args:
                log.warning(
                    f"(meter, arg) pair ({meter}, {arg}) already connected, not adding"
                )
            else:
                meters_args.append((meter, arg))
        if meter not in self.meters:
            self.meters.append(meter)

    def get_meters_args(self, probe_variable, stage):
        """
        Return a list of (meter, arg) that are using the current variable
        """
        filter_map = self.probe_filter_meter_arg.get(probe_variable, {})
        meters = filter_map.get(stage, [])
        meters.extend(filter_map.get(None, []))  # no stage filter (default)
        return meters

    def get_meters(self):
        return self.meters[:]


class Context:  # NOTE: may need to rename to something like Experiment or Procedure
    def __init__(self):
        self.batch = -1
        self.stage = ""
        self.mapper = ProbeMapper()
        self.writers = []
        self.closed = False

    def set_stage(self, stage: str):
        self.stage = stage

    def set_batch(self, batch: int):
        # NOTE: batch could be a set of sample indices
        self.batch = batch

    def increment_batch(self):
        self.batch += 1

    def is_measuring(self, probe_variable):
        return bool(self.mapper.get_meters_args(probe_variable, self.stage))

    def update(self, probe_variable, value):
        meters_args = self.mapper.get_meters_args(probe_variable, self.stage)
        if not meters_args:
            raise ValueError("No meters are measuring")
        for meter, arg in meters_args:
            meter.set(arg, value, self.batch)

    def connect_meter(self, meter):
        self.mapper.connect_meter(meter)
        for writer in self.writers:
            meter.add_writer(writer)

    def add_writer(self, writer):
        """
        Convenience method to add writer to all meters in this context
        """
        for meter in self.mapper.get_meters():
            meter.add_writer(writer)
        if writer not in self.writers:
            self.writers.append(writer)

    def close(self):
        if self.closed:
            return

        for meter in self.mapper.get_meters():
            meter.finalize()
        for writer in self.writers():
            writer.close()

        self.closed = True


class Meter:
    def __init__(
        self,
        name,
        metric,
        *metric_arg_names,
        auto_measure=True,
        final="mean",
        keep_results=True,
    ):
        """
        StandardMeter(metrics.l2, "model.x_post[benign]", "model.x_post[adversarial]")

        auto_measure - whether to measure when all of the variables are present
            if False, 'measure()' must be called externally
        """
        self.name = str(name)
        self.metric = metric
        if not len(metric_arg_names):
            raise ValueError("metric_arg_names cannot be empty list")
        self.arg_index = {arg: i for i, arg in enumerate(metric_arg_names)}
        self.clear()
        self._results = []
        self.writers = []
        self.auto_measure = bool(auto_measure)
        self._warned = False
        self.keep_results = bool(keep_results)
        # TODO: final metric (mean) [and global metrics]
        # TODO: do we need to handle metric kwargs?
        # TODO:
        # TODO: Add some sort of sampling rate (every batch, every x iterations, etc.)

    def get_arg_names(self):
        return list(self.arg_index)

    def clear(self):
        """
        Clear all of the current values
        """
        self.values = [None] * len(self.arg_index)
        self.values_set = [False] * len(self.arg_index)
        self.batches = [None] * len(self.arg_index)

    def add_writer(self, writer):
        """
        Add a writer to emit outputs to.
        """
        if writer not in self.writers:
            self.writers.append(writer)

    def set(self, name, value, batch):
        """
        Set the value for the arg name
        """
        if name not in self.arg_index:
            raise ValueError(f"{name} not a valid arg name from {self.arg_index}")
        i = self.arg_index[name]
        self.values[i] = value
        self.values_set[i] = True
        self.batches[i] = batch
        if self.auto_measure and self.is_ready():
            self.measure()

    def is_ready(self, raise_error=False):
        """
        Return True if all values have been set and batch numbers match
            if raise_error is True, raise ValueError instead of returning False
        """
        if not all(self.values_set):
            if raise_error:
                raise ValueError(f"Not all values have been set: {self.values_set}")
            return False
        if any(self.batches[0] != batch for batch in self.batches):
            if raise_error:
                raise ValueError("Batch numbers are mismatched: {self.batches}")
            return False
        return True

    def measure(self, clear_values=True):
        self.is_ready(raise_error=True)
        result = self.metric(*self.values)
        record = (self.name, self.batches[0], result)
        if self.keep_results:
            self._results.append(record)
        for writer in self.writers:
            writer.write(record)
        if not self.keep_results and not self.writers and not self._warned:
            log.warning(
                f"Meter {self.name} has no writer added and keep_results is False"
            )
            self._warned = True
        if clear_values:
            self.clear()

    def finalize(self):
        """
        Primarily intended for metrics of metrics, like mean or stdev of results
        """
        pass  # TODO
        # raise NotImplementedError()

    def results(self):
        if not self.keep_results:
            raise ValueError("keep_results is False")
        return copy.deepcopy(self._results)


# NOTE: Writer could be subclassed to directly push to TensorBoard or MLFlow
class Writer:
    def write(self, record):
        name, batch, result = record
        return self._write(name, batch, result)

    def _write(self, name, batch, result):
        raise NotImplementedError("Implement _write or override write in subclass")

    def close(self):
        pass


class NullWriter(Writer):
    def write(self, record):
        pass


class PrintWriter(Writer):
    def _write(self, name, batch, result):
        print(f"Meter Record: name={name}, batch={batch}, result={result}")


class LogWriter(Writer):
    def __init__(self, log_level: str = "INFO"):
        """
        log_level - one of the uppercase log levels allowed by armory.logs.log
        """
        log.log(log_level, f"LogWriter set to armory.logs.log at level {log_level}")
        self.log_level = log_level

    def _write(self, name, batch, result):
        log.log(
            self.log_level, f"Meter Record: name={name}, batch={batch}, result={result}"
        )


class ResultsWriter(Writer):
    def __init__(self, filepath, file_format="json"):
        self.filepath = filepath
        self.file_format = file_format
        # TODO: checking

    def _write(self, name, batch, result):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()


# GLOBAL CONTEXT METHODS #


def get_context():
    """
    Get the context state object for the experimental procedure
    """
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = Context()
    return _CONTEXT


def get_probe(name: str = ""):
    """
    Get a probe with specified name, creating it if needed
    """
    if name != "" and not str.isidentifier(name):
        raise ValueError(f"name {name} should be an identifier or the empty string")

    if name not in _PROBES:
        probe = Probe(name)
        probe.set_sink(get_context())
        _PROBES[name] = probe
    return _PROBES[name]


def connect_meter(meter):
    context = get_context()
    context.connect_meter(meter)
    _METERS.append(meter)


def add_meter(*args, **kwargs):
    meter = Meter(*args, **kwargs)
    connect_meter(meter)


def get_meters():
    """
    return all meters that have been instantiated and connected
    """
    return _METERS


def add_writer(writer):
    context = get_context()
    context.add_writer(writer)


def main():
    """
    Just for current WIP demonstration
    """
    # Begin model file
    import numpy as np

    # from armory.instrument import get_probe
    probe = get_probe("model")

    class Model:
        def __init__(self, input_dim=100, classes=10):
            self.input_dim = input_dim
            self.classes = classes
            self.preprocessor = np.random.random(self.input_dim)
            self.predictor = np.random.random((self.input_dim, self.classes))

        def predict(self, x):
            x_prep = self.preprocessor * x
            # if pytorch Tensor: probe.update(lambda x: x.detach().cpu().numpy(), prep_output=x_prep)
            probe.update(lambda x: np.expand_dims(x, 0), prep_output=x_prep)
            logits = np.dot(self.predictor.transpose(), x_prep)
            return logits

    # End model file

    # Begin metric setup (could happen anywhere)

    # from armory.instrument import add_writer, add_meter
    from armory.utils import metrics

    add_writer(PrintWriter())
    add_meter(
        "postprocessed_l2_distance",
        metrics.l2,
        "model.prep_output[benign]",
        "model.prep_output[adversarial]",
    )
    add_meter(  # TODO: enable adding context for iteration number of attack
        "sum of x_adv", np.sum, "attack.x_adv",  # could also do "attack.x_adv[attack]"
    )
    add_meter(
        "categorical_accuracy",
        metrics.categorical_accuracy,
        "scenario.y",
        "scenario.y_pred",
    )
    add_meter(  # Never measured, since 'y_target' is never set
        "targeted_categorical_accuracy",
        metrics.categorical_accuracy,
        "scenario.y_target",
        "scenario.y_pred",
    )

    # End metric setup

    # Update stages and batches in scecnario loop (this would happen in main scenario file)
    context = get_context()
    model = Model()
    # Normally, model, attack, and scenario probes would be defined in different files
    #    and therefore just be called 'probe'
    attack_probe = get_probe("attack")
    scenario_probe = get_probe("scenario")
    not_connected_probe = Probe("not_connected")
    for i in range(10):
        context.set_stage("get_batch")
        context.set_batch(i)
        x = np.random.random(100)
        y = np.random.randint(10)
        scenario_probe.update(x=x, y=y)
        not_connected_probe.update(x)  # should send a warning once

        context.set_stage("benign")
        y_pred = model.predict(x)
        scenario_probe.update(y_pred=y_pred)

        context.set_stage("attack")
        x_adv = x
        for j in range(5):
            model.predict(x_adv)
            x_adv = x_adv + np.random.random(100) * 0.1
            attack_probe.update(x_adv=x_adv)

        context.set_stage("adversarial")
        y_pred_adv = model.predict(x_adv)
        scenario_probe.update(x_adv=x_adv, y_pred_adv=y_pred_adv)


if __name__ == "__main__":
    main()
