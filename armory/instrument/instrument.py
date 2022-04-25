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

import json

from armory import log


_PROBES = {}
_HUB = None


class Probe:
    def __init__(self, name: str = "", sink=None):
        if name:
            if not all(token.isidentifier() for token in name.split(".")):
                raise ValueError(f"name {name} must be '' or '.'-separated identifiers")
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

        named_values:
            names must be valid python identifiers
            values can be any object, tuple, dict, etc.

        probe.update(data_point=(x_i, is_poisoned)) would enable downstream meters
            to measure (x_i, is_poisoned) from f"{probe.name}.data_point"
        """
        if self.sink is None:
            if not self._warned:
                log.warning(f"No sink set up for probe {self.name}!")
                self._warned = True
            return

        for name in named_values:
            if not name.isidentifier():
                raise ValueError(
                    f"named_values must be valid python identifiers, not {name}"
                )

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


class MockSink:
    """
    Measures all probe inputs and prints to screen
    """

    def is_measuring(self, probe_variable):
        return True

    def update(self, probe_variable, value):
        print(f"update probe variable {probe_variable} to {value}")


def process_meter_arg(arg: str):
    """
    Return the probe variable and stage_filter

    Example strings: 'model.x2[adversarial]', 'scenario.y_pred'
    """
    if "[" in arg:
        if arg.count("[") != 1 or arg.count("]") != 1:
            raise ValueError(
                f"arg '{arg}' must have a single matching '[]' pair or none"
            )
        arg, filt = arg.split("[")
        if filt[-1] != "]":
            raise ValueError(f"arg '{arg}' cannot have chars after final ']'")
        stage_filter = filt[:-1].strip()
        # tokens = [x.strip() for x in filt.split(",")]
    else:
        stage_filter = None
    probe_variable = arg

    return probe_variable, stage_filter


class ProbeMapper:
    """
    Map from probe outputs to meters. For internal use by Hub object.
    """

    def __init__(self):
        # nested dicts - {probe_variable: {stage_filter: [(meter, arg)]}}
        self.probe_filter_meter_arg = {}

    def __len__(self):
        """
        Return the number of (meter, arg) pairs
        """
        count = 0
        for probe_variable, filter_map in self.probe_filter_meter_arg.items():
            for stage_filter, meters_args in filter_map.items():
                count += len(meters_args)
        return count

    def __str__(self):
        return f"{type(self)} : {self.probe_filter_meter_arg}"

    def connect_meter(self, meter):
        """
        Connect meter to probes; idempotent
        """
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

    def disconnect_meter(self, meter):
        """
        Disconnect meter from probes; idempotent
        """
        for arg in meter.get_arg_names():
            probe_variable, stage_filter = process_meter_arg(arg)
            if probe_variable not in self.probe_filter_meter_arg:
                continue
            filter_map = self.probe_filter_meter_arg[probe_variable]

            if stage_filter not in filter_map:
                continue
            meters_args = filter_map[stage_filter]

            if (meter, arg) in meters_args:
                meters_args.remove((meter, arg))
                if not meters_args:
                    filter_map.pop(stage_filter)
                if not filter_map:
                    self.probe_filter_meter_arg.pop(probe_variable)

    def map_probe_update_to_meter_input(self, probe_variable, stage):
        """
        Return a list of (meter, arg) that are using the current probe_variable
        """
        filter_map = self.probe_filter_meter_arg.get(probe_variable, {})
        meters = filter_map.get(stage, [])
        meters.extend(filter_map.get(None, []))  # no stage filter (default)
        return meters


class Hub:
    """
    Map between probes, meters, and writers

    Maintains context of overall experiment in hub.context
    """

    def __init__(self):
        self.context = dict(batch=-1, stage="",)
        self.mapper = ProbeMapper()
        self.meters = []
        self.writers = []
        self.default_writers = []
        self.closed = False

    def set_context(self, **kwargs):
        for k in kwargs:
            if k not in ("stage", "batch"):
                log.warning(f"set_context kwarg {k} not currently used by Hub")
            if not k.isidentifier():
                raise ValueError(
                    f"set_context kwargs must be valid identifiers, not {k}"
                )
        self.context.update(kwargs)

    # is_measuring and update implement the sink interface
    def is_measuring(self, probe_variable):
        return bool(
            self.mapper.map_probe_update_to_meter_input(
                probe_variable, self.context["stage"]
            )
        )

    def update(self, probe_variable, value):
        meters_args = self.mapper.map_probe_update_to_meter_input(
            probe_variable, self.context["stage"]
        )
        if not meters_args:
            raise ValueError("No meters are measuring")
        for meter, arg in meters_args:
            meter.set(arg, value, self.context["batch"])

    def connect_meter(self, meter, use_default_writers=True):
        """
        Connect meter. If use_default_writers, connect to all default writers as well
        """
        if use_default_writers:
            for writer in self.default_writers:
                meter.add_writer(writer)

        if meter in self.meters:
            return

        self.meters.append(meter)
        self.mapper.connect_meter(meter)

    def connect_writer(self, writer, meters=None, default=False):
        """
        Convenience method to add writer to all (or a subset of meters)

        meters - if None, add to all current meters
            otherwise, meters should be a list of names and/or Meter objects
                the writer is connected to those meters
            if meters is an empty list, the writer will be added but no meters connected

        default - if True, writer is automatically added to each new meter added
        """
        if meters is None:
            meters_list = self.meters
        else:
            # parse meters list
            meters_list = []
            for m in meters:
                if isinstance(m, str):
                    pass
                elif isinstance(m, Meter):
                    if m in self.meters:
                        meters_list.append(m)
                else:
                    raise ValueError(
                        f"meters includes {m}, which is neither a str nor Meter"
                    )

        for meter in meters_list:
            meter.add_writer(writer)

        if default and writer not in self.default_writers:
            self.default_writers.append(writer)
        if writer not in self.writers:
            self.writers.append(writer)

    def record(self, name, result):
        """
        Push a record to the default writers
        """
        if not self.default_writers:
            log.warning(f"No default writers to record {name}:{result} to")
        for writer in self.default_writers:
            writer.write((name, self.context["batch"], result))

    def close(self):
        if self.closed:
            return

        for meter in self.meters:
            meter.finalize()

        for writer in self.writers:
            writer.close()

        self.closed = True


class Meter:
    def __init__(
        self,
        name,
        metric,
        *metric_arg_names,
        metric_kwargs=None,
        auto_measure=True,
        final=None,
        final_name=None,
        final_kwargs=None,
        keep_results=True,
        record_final_only=False,
    ):
        """
        StandardMeter(metrics.l2, "model.x_post[benign]", "model.x_post[adversarial]")

        probe_name.probe_variable[stage]

        metric_kwargs - kwargs that are constant across measurements
        auto_measure - whether to measure when all of the variables are present
            if False, 'measure()' must be called externally

        final - metric that takes in list of results as input
            Example: np.mean
        final_name - if final is not None, this is the name associated with the record
            if not specified, it defaults to f'{final}_{name}'

        keep_results - whether to locally store results
            if final is not None, keep_results is set to True

        record_final_only - if True, do not record results to writers except in finalize
        """
        self.name = str(name)
        if not callable(metric):
            raise ValueError(f"metric {metric} must be callable")
        self.metric = metric
        if not len(metric_arg_names):
            log.warning("metric_arg_names is an empty list")
        self.arg_index = {arg: i for i, arg in enumerate(metric_arg_names)}
        self.metric_kwargs = metric_kwargs or {}
        if not isinstance(self.metric_kwargs, dict):
            raise ValueError(
                f"metric_kwargs must be None or a dict, not {metric_kwargs}"
            )
        self.clear()
        self._results = []
        self.writers = []
        self.auto_measure = bool(auto_measure)
        self._warned = False

        keep_results = bool(keep_results)
        if final is None:
            if final_name is not None:
                final_name = str(final_name)
            if record_final_only:
                raise ValueError("record_final_only cannot be True if final is None")
        else:
            if not callable(final):
                raise ValueError(f"final {final} must be callable")
            if final_name is None:
                final_name = f"{final}_{name}"
            else:
                final_name = str(final_name)
            if not keep_results:
                log.warning("final is not None. Overriding keep_results to True")
                keep_results = True
        self.final = final
        self.final_name = final_name
        self.final_kwargs = final_kwargs or {}
        self.final_result = None
        if not isinstance(self.final_kwargs, dict):
            raise ValueError(f"final_kwargs must be None or a dict, not {final_kwargs}")
        self.keep_results = keep_results
        self.record_final_only = record_final_only
        self.never_measured = True

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
        result = self.metric(*self.values, **self.metric_kwargs)
        record = (self.name, self.batches[0], result)
        if self.keep_results:
            # Assume metric is sample-wise, but computed on a batch of samples
            try:
                self._results.extend(result)
            except TypeError:
                self._results.append(result)
        if not self.record_final_only:
            for writer in self.writers:
                writer.write(record)
        if not self.keep_results and not self.writers and not self._warned:
            log.warning(
                f"Meter '{self.name}' has no writer added and keep_results is False"
            )
            self._warned = True
        if clear_values:
            self.clear()
        self.never_measured = False

    def finalize(self):
        """
        Primarily intended for metrics of metrics, like mean or stdev of results
        """
        if self.never_measured:
            unset = [arg for arg, i in self.arg_index.items() if not self.values_set[i]]
            log.warning(
                f"Meter '{self.name}' was never measured. "
                f"The following args were never set: {unset}"
            )
            return  # Do not compute final if never_measured

        if self.final is None:
            return

        result = self.final(self._results, **self.final_kwargs)
        record = (self.final_name, None, result)
        self.final_result = result
        for writer in self.writers:
            writer.write(record)

    def results(self):
        if not self.keep_results:
            raise ValueError("keep_results is False")
        return self._results


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


class FileWriter(Writer):
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = open(self.filepath, "w")

    def _write(self, name, batch, result):
        record_json = json.dumps([name, batch, result])
        # TODO: fix numpy output conversion issues
        self.file.write(record_json + "\n")


class ResultsWriter(Writer):
    """
    Write results to dictionary
    """

    def __init__(self, sink=None):
        """
        sink is a callable that takes the output results dict as input
            if sink is None, call get_output after close to get results dict
        """
        self.sink = sink
        self.records = []
        self.output = None

    def _write(self, name, batch, result):
        self.records.append((name, batch, result))

    def collate_results(self):
        """
        Return a map from name to output, in original order.
        """
        output = {}
        for name, batch, result in self.records:
            if name not in output:
                output[name] = []
            output[name].append(result)
        return output

    def close(self):
        output = self.collate_results()
        if self.sink is None:
            self.output = output
        else:
            self.sink(output)

    def get_output(self):
        if self.output is None:
            raise ValueError("must call `close` before `get_output`")
        return self.output


# GLOBAL CONTEXT METHODS #


def get_hub():
    """
    Get the global hub and context object for the measurement procedure
    """
    global _HUB
    if _HUB is None:
        _HUB = Hub()
    return _HUB


def get_probe(name: str = ""):
    """
    Get a probe with specified name, creating it if needed
    """
    if name not in _PROBES:
        probe = Probe(name, sink=get_hub())
        _PROBES[name] = probe
    return _PROBES[name]


def del_globals():
    """
    Remove hub and probes from global context
        Subsequent calls to `get_hub` and `get_probe` will return new objects
        Must also delete local references and close objects as needed

    NOTE: primarily intended for creating clean contexts for testing
    """
    global _PROBES
    global _HUB
    _PROBES = {}
    _HUB = None


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

    hub = get_hub()
    hub.connect_meter(
        Meter(
            "postprocessed_l2_distance",
            metrics.l2,
            "model.prep_output[benign]",
            "model.prep_output[adversarial]",
        )
    )
    hub.connect_meter(
        Meter(  # TODO: enable adding context for iteration number of attack
            "sum of x_adv",
            np.sum,
            "attack.x_adv",  # could also do "attack.x_adv[attack]"
        )
    )
    hub.connect_meter(
        Meter(
            "categorical_accuracy",
            metrics.categorical_accuracy,
            "scenario.y",
            "scenario.y_pred",
        )
    )
    hub.connect_meter(
        Meter(  # Never measured, since 'y_target' is never set
            "targeted_categorical_accuracy",
            metrics.categorical_accuracy,
            "scenario.y_target",
            "scenario.y_pred",
        )
    )
    hub.connect_writer(PrintWriter())

    # End metric setup

    # Update stages and batches in scenario loop (this would happen in main scenario file)
    model = Model()
    # Normally, model, attack, and scenario probes would be defined in different files
    #    and therefore just be called 'probe'
    attack_probe = get_probe("attack")
    scenario_probe = get_probe("scenario")
    not_connected_probe = Probe("not_connected")
    for i in range(10):
        hub.set_context(stage="get_batch", batch=i)
        x = np.random.random(100)
        y = np.random.randint(10)
        scenario_probe.update(x=x, y=y)
        not_connected_probe.update(x)  # should send a warning once

        hub.set_context(stage="benign")
        y_pred = model.predict(x)
        scenario_probe.update(y_pred=y_pred)

        hub.set_context(stage="benign")
        x_adv = x
        for j in range(5):
            model.predict(x_adv)
            x_adv = x_adv + np.random.random(100) * 0.1
            attack_probe.update(x_adv=x_adv)

        hub.set_context(stage="benign")
        y_pred_adv = model.predict(x_adv)
        scenario_probe.update(x_adv=x_adv, y_pred_adv=y_pred_adv)
    hub.close()


if __name__ == "__main__":
    main()
