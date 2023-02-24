"""
Flexible measurement instrumentation

Functionalities
    Probe - pull info from data source
    Hub - stores context and state for meters and writers
    Meter - measure quantity at specified intervals
    Writer - takes measured outputs from meters and pushed them to print/file/etc.

Example:
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
        meter = instrument.Meter("l2_dist_postprocess", metrics.l2, "model.x_post[benign]", "model.x_post[adversarial]")
        hub = instrument.get_hub()
        hub.connect_meter(meter)
        hub.connect_writer(instrument.PrintWriter())
"""

import json
from typing import Callable

import armory.paths

try:
    # If numpy is available, enable NumpyEncoder for json export
    from armory.utils import json_utils
except ImportError:
    json_utils = None

from armory import log

_PROBES = {}
_HUB = None


class Probe:
    """
    Probes are used to capture values and route them to the provided sink.
        If probes are constructed via the global `get_probe(...)` method,
            the sink provided will be the global hub from `get_hub()`

    Example:
        >>> from armory.instrument import Probe, MockSink
        >>> probe = Probe("test_probe", sink=MockSink())
        >>> probe.update(variable_name=17)
        update probe variable test_probe.variable_name to 17
    """

    def __init__(self, name: str = "", sink=None):
        """
        name - the name of the probe, which must be "" or '.'-separated identifiers
            If not empty, will prepend the update name via f"{name}.{update_name}"
        sink - sink object for probe updates; must implement 'is_measuring' and 'update'
            Currently, the Hub and MockSink objects satisfy this interface
        """
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


class MockSink:
    """
    Measures all probe inputs and prints to screen

    Primarily intended for testing whether probes are measuring:
        probe.set_sink(MockSink())
        probe.update(my_variable=17)
    """

    def is_measuring(self, probe_variable):
        return True

    def update(self, probe_variable, value):
        print(f"update probe variable {probe_variable} to {value}")


def process_meter_arg(arg: str):
    """
    Helper function for ProbeMapper

    Return the probe variable and stage_filter

    Example strings: 'model.x2[adversarial]', 'scenario.y_pred'
    """
    if "[" in arg or "]" in arg:
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
        if stage is not None:
            meters = meters + filter_map.get(None, [])  # no stage filter (default)
        return meters


class Hub:
    """
    Map between probes, meters, and writers

    Maintains context of overall experiment in terms of 'stage' and 'batch'
        Example: hub.set_context(batch=1, stage="benign")
        In current scenarios, stage is a str and batch is an int
        These are used for filtering probe_variable updates and recording measurements
        These may be extended in subsequent releases

    `connect_meter` ands `connect_writer` are used for connecting Meter and Writer
    probes are connected via setting their sink argument to the Hub
        By default, probes are connected to the hub accessible via the global method `get_hub`

    `record` pushes a single record to the default writers
        This is useful if only a single measurement is needed for something

    `close` finalizes all meters and subsequently closes all writers
        This needs to be called before exiting, or some measurements may not be recorded
    """

    def __init__(self):
        self.context = dict(batch=-1, stage="")
        self.mapper = ProbeMapper()
        self.meters = []
        self.writers = []
        self.default_writers = []
        self.closed = False
        self.export_subdir = "saved_samples"
        self._set_output_dir(armory.paths.runtime_paths().output_dir)

    def _set_output_dir(self, name):
        self.output_dir = name
        self._set_export_dir(self.export_subdir)

    def get_output_dir(self):
        return self.output_dir

    def _set_export_dir(self, output_subdir):
        self.export_dir = f"{self.output_dir}/{output_subdir}"
        self.export_subdir = output_subdir

    def get_export_dir(self):
        return self.export_dir

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

    def disconnect_meter(self, meter):
        self.mapper.disconnect_meter(meter)
        if meter in self.meters:
            self.meters.remove(meter)

    def connect_writer(self, writer, meters=None, default=False):
        """
        Convenience method to add writer to all (or a subset of meters)

        meters - if None, add to all current meters
            otherwise, meters should be a list of Meter objects
                the writer is connected to those meters
                if the meters are not in the hub, they must first be connected
            if meters is an empty list, the writer will be added but no meters connected

        default - if True, writer is automatically added to each new meter added
        """
        if meters is None:
            meters = self.meters
        else:
            for m in meters:
                if not isinstance(m, Meter):
                    raise ValueError(f"meters includes {m}, which is not a Meter")
                elif m not in self.meters:
                    raise ValueError(
                        f"meter {m} is not connected to hub. "
                        "If desired, first call `hub.connect_meter`"
                    )

        for meter in meters:
            meter.add_writer(writer)

        if default and writer not in self.default_writers:
            self.default_writers.append(writer)
        if writer not in self.writers:
            self.writers.append(writer)

    def record(
        self, name, result, writers=None, use_default_writers=True, **write_kwargs
    ):
        """
        Push a record to the default writers
        writers - None, a Writer, or an iterable of Writer
            if not None, write to all given writers
        use_default_writers - whether to write to the default writers
        """
        if writers is None:
            writers = []
        elif isinstance(writers, Writer):
            writers = [writers]
        else:
            try:
                writers = list(writers)
            except TypeError:
                raise TypeError(
                    f"Received 'writers' input of type {type(writers)}, "
                    "expected one of (None, Writer, iterable of Writers)"
                )
            if not all(isinstance(writer, Writer) for writer in writers):
                raise ValueError("writers are not all of type Writer")

        if use_default_writers:
            writers.extend(self.default_writers)
        if not writers:
            log.warning(f"No writers to record {name}:{result} to")
        record = (name, self.context["batch"], result)
        for writer in writers:
            writer.write(record, **write_kwargs)

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
        result_formatter=None,
        auto_measure=True,
        final=None,
        final_name=None,
        final_kwargs=None,
        final_result_formatter=None,
        record_final_only=False,
    ):
        """
        A meter measures a function over specified input probe_variables for each update
            If final is not None, it also measures a function over those measurements
            Records are pushed to Writers for output

        name - str name of meter, used when writing records
        metric - callable function
        metric_arg_names - str names of probe_variables corresponding to args passed into the metric function
            Meter(..., "model.x_post[benign]", "model.x_adv_post", ...)
            Follows the pattern of `probe_name.probe_variable[stage]` (stage is optional)
        metric_kwargs - kwargs for the metric function that are constant across measurements
        result_formatter - a function (or None) that takes a result and formats it for logging

        auto_measure - whether to measure when all of the variables have ben set
            if False, 'measure()' must be called externally

        final - metric function that takes in the list of results as input (e.g., np.mean)
        final_name - if final is not None, this is the name associated with the record
            if not specified, it defaults to f'{final}_{name}'
        final_kwargs - kwargs for the final function that are constant across measurements
        final_result_formatter - like result_formatter, but for the final result
        record_final_only - if True, do not record the standard metric, only final
            if record_final_only is True and final is None, no records are emitted
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
        self._final_result = None
        self.writers = []
        self.auto_measure = bool(auto_measure)

        self.result_formatter = result_formatter

        if final is not None:
            if not callable(final):
                raise ValueError(f"final {final} must be callable")
            if final_name is None:
                final_name = f"final_{name}"
            else:
                final_name = str(final_name)
        self.final = final
        self.final_name = final_name
        self.final_kwargs = final_kwargs or {}
        self.final_result_formatter = final_result_formatter
        if not isinstance(self.final_kwargs, dict):
            raise ValueError(f"final_kwargs must be None or a dict, not {final_kwargs}")
        self.record_final_only = bool(record_final_only)
        self.never_measured = True

    def get_arg_names(self):
        return list(self.arg_index)

    def clear(self):
        """
        Clear all of the current values
        """
        self.values = [None] * len(self.arg_index)
        self.values_set = [False] * len(self.arg_index)
        self.arg_batch_indices = [None] * len(self.arg_index)

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
        self.arg_batch_indices[i] = batch
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
        if any(self.arg_batch_indices[0] != batch for batch in self.arg_batch_indices):
            if raise_error:
                raise ValueError(
                    "Batch numbers are mismatched: {self.arg_batch_indices}"
                )
            return False
        return True

    def measure(self, clear_values=True):
        self.is_ready(raise_error=True)
        result = self.metric(*self.values, **self.metric_kwargs)
        record = (self.name, self.arg_batch_indices[0], result)
        # Assume metric is sample-wise, but computed on a batch of samples
        try:
            self._results.extend(result)
        except TypeError:
            self._results.append(result)
        if not self.record_final_only:
            for writer in self.writers:
                writer.write(record, result_formatter=self.result_formatter)
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
        self._final_result = result
        for writer in self.writers:
            writer.write(record, result_formatter=self.final_result_formatter)

    def results(self):
        return self._results

    def final_result(self):
        return self._final_result


class GlobalMeter(Meter):
    """
    Meter that only produces a final result after finalize.
    Concatenates batches of arg inputs for input to final_metric

    This would simplify the following (from poison.py):
        Meter(
            "accuracy_on_benign_test_data_per_class",
            metrics.get_supported_metric("identity_unzip"),
            "scenario.y",
            "scenario.y_pred",
            final=lambda x: per_class_mean_accuracy(*metrics.identity_zip(x)),
            final_name="accuracy_on_benign_test_data_per_class",
            record_final_only=True,
        )
    To:
        GlobalMeter(
            "accuracy_on_benign_test_data_per_class",
            per_class_mean_accuracy,
            "scenario.y",
            "scenario.y_pred",
        )
    """

    def __init__(
        self,
        final_name,
        final_metric,
        *final_metric_arg_names,
        final_kwargs=None,
        final_result_formatter=None,
    ):
        final_name = str(final_name)
        if not callable(final_metric):
            raise ValueError(f"final_metric {final_metric} is not callable")
        final_kwargs = final_kwargs or {}
        if not isinstance(final_kwargs, dict):
            raise ValueError(f"final_kwargs must be None or a dict, not {final_kwargs}")

        from armory import metrics

        identity_unzip = metrics.get("identity_unzip")
        identity_zip = metrics.get("identity_zip")

        super().__init__(
            f"input_to_{final_name}",
            identity_unzip,
            *final_metric_arg_names,
            metric_kwargs=None,
            auto_measure=True,
            final=lambda x: final_metric(*identity_zip(x), **final_kwargs),
            final_name=final_name,
            final_kwargs=None,
            record_final_only=True,
            final_result_formatter=final_result_formatter,
        )


# NOTE: Writer could be subclassed to directly push to TensorBoard or MLFlow
class Writer:
    """
    Writers take records from Meters in the form (meter_name, batch_num, result)
        and write them to the desired output (e.g., print to screen, log to file)

    Subclasses implement the writing functionality and should override `_write`
        If subclasses manage resources like files that need to be closed,
            `_close` should be overridden to implement that functionality

    If subclasses can take kwargs in their `_write` method,
        these should be provided in the init here as a tuple of allowed kwargs
        Only these kwargs will be passed to the `_write` method
    """

    def __init__(self, _write_kwargs=None):
        self.closed = False
        if _write_kwargs is None:
            _write_kwargs = ()
        elif isinstance(_write_kwargs, str):
            raise ValueError("_write_kwargs must be an iterable of str, not a str")
        _write_kwargs = tuple(_write_kwargs)
        for kwarg in _write_kwargs:
            if not isinstance(kwarg, str):
                raise ValueError(f"kwarg {kwarg} is not of type str")
        self._write_kwargs = _write_kwargs

    def write(self, record, **kwargs):
        if self.closed:
            raise ValueError("Cannot write to closed Writer")
        name, batch, result = record
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in self._write_kwargs}
        return self._write(name, batch, result, **filtered_kwargs)

    def _write(self, name, batch, result):
        raise NotImplementedError("Implement _write or override write in subclass")

    def close(self):
        if self.closed:
            return
        self._close()
        self.closed = True

    def _close(self):
        pass


class NullWriter(Writer):
    def _write(self, name, batch, result):
        pass


class PrintWriter(Writer):
    def _write(self, name, batch, result):
        print(f"Meter Record: name={name}, batch={batch}, result={result}")


class LogWriter(Writer):
    LOG_LEVELS = (
        "TRACE",
        "DEBUG",
        "PROGRESS",
        "INFO",
        "SUCCESS",
        "METRIC",
        "WARNING",
        "ERROR",
        "CRITICAL",
    )

    def __init__(self, log_level: str = "INFO", _write_kwargs=None):
        """
        log_level - one of the uppercase log levels allowed by armory.logs.log
        """
        super().__init__(_write_kwargs=_write_kwargs)

        if log_level not in self.LOG_LEVELS:
            raise ValueError(f"log_level {log_level} not in {self.LOG_LEVELS}")
        self.log_level = log_level

    def _write(self, name, batch, result):
        log.log(
            self.log_level, f"Meter Record: name={name}, batch={batch}, result={result}"
        )


class ResultsLogWriter(LogWriter):
    """
    Logs successful results (designed for task metrics)
    """

    def __init__(
        self,
        format_string: str = "{name} on benign examples w.r.t. ground truth labels: {result}",
        log_level: str = "METRIC",
    ):
        """
        format_string - string to format results for output to logging
            Can contain 'name' and 'result' as keys for formatting
            Other keys will cause an error
        """
        super().__init__(log_level=log_level, _write_kwargs=("result_formatter",))
        try:
            format_string.format(name="name", result=1)
        except KeyError as e:
            raise KeyError(f"format_string key {e} is not in ('name', 'result')")
        self.format_string = format_string

    def _write(self, name, batch, result, result_formatter: Callable = None):
        """
        result_formatter - function that maps a result into a smaller formatted string
            for less verbose logging to stout
        """
        if result_formatter is not None:
            result = result_formatter(result)
        log.log(self.log_level, self.format_string.format(name=name, result=result))


class FileWriter(Writer):
    """
    Writes a txt file with line-separated json encoded outputs
    """

    def __init__(self, filepath, use_numpy_encoder=True):
        super().__init__()
        if use_numpy_encoder:
            if json_utils is None:
                raise ValueError("Cannot import numpy. Set use_numpy_encoder to False")
            self.numpy_encoder = json_utils.NumpyEncoder
        else:
            self.numpy_encoder = None
        self.filepath = filepath
        self.file = open(self.filepath, "w")

    def _write(self, name, batch, result):
        record = [name, batch, result]
        self.file.write(
            json.dumps(record, cls=self.numpy_encoder, separators=(",", ":")) + "\n"
        )

    def _close(self):
        self.file.close()


class ResultsWriter(Writer):
    """
    Write results to dictionary
    """

    def __init__(self, sink=None, max_record_size=None):
        """
        sink is a callable that takes the output results dict as input
            if sink is None, call get_output after close to get results dict
        max_record_size - if not None, the maximum size in bytes for a single record
            Records exceeding that size log a warning and are dropped
        """
        super().__init__()
        self.sink = sink
        self.records = []
        self.output = None
        if max_record_size is not None:
            if max_record_size < -1:
                raise ValueError(f"max_record_size {max_record_size} cannot be < -1")
            if json_utils is None:
                raise ValueError("Cannot import numpy. Set max_record_size to None")
            max_record_size = int(max_record_size)
        self.max_record_size = max_record_size

    def _write(self, name, batch, result):
        record = (name, batch, result)
        if self.max_record_size is not None:
            try:
                json_utils.check_size(record, self.max_record_size)
            except ValueError:
                log.warning(
                    f"record (name={name}, batch={batch}, result=...) size > "
                    f"max_record_size {self.max_record_size}. Dropping."
                )
                return
        self.records.append(record)

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

    def _close(self):
        output = self.collate_results()
        if self.sink is None:
            self.output = output
        else:
            self.sink(output)

    def get_output(self):
        if self.output is None:
            raise ValueError("must call `close` before `get_output`")
        if self.sink is not None:
            raise ValueError("output only kept if sink is None")
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
