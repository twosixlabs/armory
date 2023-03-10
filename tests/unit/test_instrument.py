"""
Test cases for armory.instrument measurement instrumentation
"""

import json

import pytest

from armory import metrics
from armory.instrument import instrument

pytestmark = pytest.mark.unit


def test_del_globals():
    instrument.del_globals()
    assert not (instrument._HUB)
    assert not (instrument._PROBES)
    instrument.get_hub()
    instrument.get_probe()
    assert instrument._HUB
    assert instrument._PROBES
    instrument.del_globals()
    assert not (instrument._HUB)
    assert not (instrument._PROBES)


def test_get_probe():
    instrument.del_globals()
    a = instrument.get_probe()
    also_a = instrument.get_probe("")
    assert also_a is a
    not_a = instrument.get_probe("name")
    assert not_a is not a
    with pytest.raises(ValueError):
        instrument.get_probe("Not a valid identifier")
    also_valid_name = instrument.get_probe("dot.separated.name")
    hub = instrument.get_hub()
    for probe in a, not_a, also_valid_name:
        assert probe.sink is hub


def test_get_hub():
    instrument.del_globals()
    a = instrument.get_hub()
    also_a = instrument.get_hub()
    assert also_a is a
    instrument.del_globals()
    not_a = instrument.get_hub()
    assert not_a is not a


class HelperSink:
    def __init__(self, measuring=True):
        self._is_measuring = measuring
        self.probe_variables = {}

    def is_measuring(self, probe_variable):
        return self._is_measuring

    def update(self, probe_variable, value):
        if not self.is_measuring(probe_variable):
            raise ValueError("update should not be called when not_measuring")
        self.probe_variables[probe_variable] = value


def test_probe(caplog):
    instrument.del_globals()

    # Ensure single warning message
    name = "NoSink"
    probe = instrument.Probe(name)
    probe.update(x=1)
    warning_message = f"No sink set up for probe {name}!"
    assert warning_message in caplog.text
    probe.update(x=2)
    assert caplog.text.count(warning_message) == 1

    sink = HelperSink()
    probe = instrument.Probe("my.probe_name", sink)
    with pytest.raises(ValueError):
        probe.update(**{"~!=Not Valid Identifier": 3})

    probe = instrument.Probe(sink=sink)
    probe.update(x=1)
    assert sink.probe_variables["x"] == 1
    probe.update(x=2, y=3)
    assert sink.probe_variables["x"] == 2
    assert sink.probe_variables["y"] == 3

    sink._is_measuring = False
    probe.update(x=3)
    assert sink.probe_variables["x"] != 3
    sink._is_measuring = True

    probe2 = instrument.Probe("my.name", sink=sink)
    probe2.update(x=-5)
    assert sink.probe_variables["x"] != -5
    assert sink.probe_variables["my.name.x"] == -5

    probe.update(lambda x: x + 1, lambda x: 2 * x, z=3, w=4)
    assert sink.probe_variables["z"] == (3 + 1) * 2
    assert sink.probe_variables["w"] == (4 + 1) * 2

    def not_implemented(*args, **kwargs):
        raise NotImplementedError()

    with pytest.raises(NotImplementedError):
        probe.update(not_implemented, x=1)

    sink._is_measuring = False
    probe.update(not_implemented, x=1)
    sink._is_measuring = True


def get_pytorch_model():
    # Taken from https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 20, 5)
            self.conv2 = nn.Conv2d(20, 20, 5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            return F.relu(self.conv2(x))

    return Model()


def test_process_meter_arg():
    for arg in ("x", "scenario.x"):
        assert instrument.process_meter_arg(arg) == (arg, None)

    for arg, (probe_variable, stage_filter) in [
        ("x[benign]", ("x", "benign")),
        ("scenario.x[attack]", ("scenario.x", "attack")),
    ]:
        assert instrument.process_meter_arg(arg) == (probe_variable, stage_filter)

    for arg in ("missing[", "missing]", "trailing[]   ", "toomany[[]"):
        with pytest.raises(ValueError):
            probe_variable, stage_filter = instrument.process_meter_arg(arg)


class MockMeter(instrument.Meter):
    """
    For testing ProbeMapper
    """

    def __init__(self, *args):
        self.args = args
        self.writers = []
        self.num_add_writers = 0
        self.num_finalizes = 0
        self.num_sets = 0

    def get_arg_names(self):
        return self.args

    def add_writer(self, writer):
        self.num_add_writers += 1
        if writer not in self.writers:
            self.writers.append(writer)

    def set(self, name, value, batch):
        self.num_sets += 1

    def finalize(self):
        self.num_finalizes += 1


def test_probe_mapper(caplog):
    probe_mapper = instrument.ProbeMapper()
    assert len(probe_mapper) == 0
    assert str(probe_mapper).endswith(" : {}")

    meter1 = MockMeter("scenario.x", "scenario.x_adv")
    probe_mapper.connect_meter(meter1)
    assert len(probe_mapper) == 2

    warning_substring = "already connected, not adding"
    probe_mapper.connect_meter(meter1)
    assert warning_substring in caplog.text
    assert len(probe_mapper) == 2

    meters = probe_mapper.map_probe_update_to_meter_input("scenario.x", None)
    print(meters)
    assert len(meters) == 1
    meter, arg = meters[0]
    assert meter is meter1

    probe_mapper.disconnect_meter(meter1)
    assert len(probe_mapper) == 0
    probe_mapper.disconnect_meter(meter1)

    meter2 = MockMeter("scenario.x[benign]", "scenario.y[benign]")
    meter3 = MockMeter()
    meter4 = MockMeter("x", "scenario.x", "scenario.x[benign]")
    for m in meter1, meter2, meter3, meter4:
        probe_mapper.connect_meter(m)
    assert len(probe_mapper) == 7
    meters = probe_mapper.map_probe_update_to_meter_input("scenario.x", None)
    assert len(meters) == 2
    assert (meter1, "scenario.x") in meters
    assert (meter4, "scenario.x") in meters

    meters = probe_mapper.map_probe_update_to_meter_input("scenario.x", "benign")
    assert len(meters) == 4
    for arg_meter in (
        (meter1, "scenario.x"),
        (meter2, "scenario.x[benign]"),
        (meter4, "scenario.x"),
        (meter4, "scenario.x[benign]"),
    ):
        assert arg_meter in meters

    meters = probe_mapper.map_probe_update_to_meter_input("invalid", None)
    assert len(meters) == 0

    for m in meter1, meter2, meter3, meter4:
        probe_mapper.disconnect_meter(m)
    assert len(probe_mapper) == 0


class LastRecordWriter(instrument.Writer):
    """
    Mock test interface for Writer
    """

    def __init__(self):
        self.record = None
        self.closed = False
        self.num_writes = 0
        self.num_closes = 0

    def write(self, record, **kwargs):
        self.record = record
        self.num_writes += 1

    def close(self):
        self.closed = True
        self.num_closes += 1


def test_hub(caplog):
    Hub = instrument.Hub
    hub = Hub()

    hub.set_context(not_used=0)
    assert "set_context kwarg not_used not currently used by Hub" in caplog.text

    hub.set_context(stage="adversarial")
    hub.set_context(stage="benign", batch=0)
    with pytest.raises(ValueError):
        hub.set_context(**{"not a valid identifier": 10})

    assert not hub.is_measuring("x")
    with pytest.raises(ValueError):
        hub.update("x", 0)

    name = "no default writer"
    result = 17
    hub.record(name, result)
    assert f"No writers to record {name}:{result} to" in caplog.text

    w1 = LastRecordWriter()
    w2 = LastRecordWriter()
    hub.connect_writer(w1, default=True)
    hub.connect_writer(w1, default=True)
    hub.connect_writer(w2)
    batch = 7
    hub.set_context(batch=batch)
    hub.record(name, result)
    assert w1.record == (name, batch, result)
    assert w1.num_writes == 1
    assert w2.num_writes == 0

    with pytest.raises(TypeError):
        hub.record(name, result, writers=2343)
    with pytest.raises(ValueError):
        hub.record(name, result, writers=(w1, w2, "hi"))
    name = "use_default_writers set to False"
    hub.record(name, result, use_default_writers=False)
    assert f"No writers to record {name}:{result} to" in caplog.text
    w3 = LastRecordWriter()
    hub.record(name, result, writers=w3, use_default_writers=False)
    assert w3.record == (name, batch, result)
    assert w1.num_writes == 1
    assert w2.num_writes == 0
    assert w3.num_writes == 1
    hub.record(name, result, writers=w3, use_default_writers=True)
    assert w3.record == (name, batch, result)
    assert w1.num_writes == 2
    assert w2.num_writes == 0
    assert w3.num_writes == 2

    m1 = MockMeter("a[benign]")
    m2 = MockMeter("a", "b[benign]")
    hub.connect_meter(m1)
    hub.connect_meter(m1)
    hub.set_context(stage="attack")
    hub.connect_meter(m2, use_default_writers=False)
    assert m1.num_add_writers == 2
    assert len(m1.writers) == 1
    assert m2.num_add_writers == 0
    assert hub.is_measuring("a")
    assert not hub.is_measuring("b")
    value = 5
    hub.update("a", value)
    assert m1.num_sets == 0
    assert m2.num_sets == 1
    hub.set_context(stage="benign")
    hub.update("a", value)
    assert m1.num_sets == 1
    assert m2.num_sets == 2

    hub.connect_writer(LastRecordWriter())
    assert m1.num_add_writers == 3
    assert m2.num_add_writers == 1
    hub.connect_writer(LastRecordWriter(), meters=[m2])
    assert m1.num_add_writers == 3
    assert m2.num_add_writers == 2

    with pytest.raises(ValueError):
        hub.connect_writer(w1, meters=["not a meter"])
    not_connected_meter = MockMeter("c", "d")
    with pytest.raises(ValueError):
        hub.connect_writer(w1, meters=[not_connected_meter])

    hub.close()
    hub.close()
    for m in (m1, m2):
        assert m.num_finalizes == 1
    for w in (w1, w2):
        assert w.num_closes == 1
    assert hub.closed

    hub.disconnect_meter(m1)
    hub.disconnect_meter(m2)
    assert len(hub.meters) == 0


def test_meter(caplog):
    Meter = instrument.Meter
    with pytest.raises(ValueError):
        Meter("name", "not callable")

    def f(*x):
        return sum(x)

    def g(x):
        return sum(x)

    Meter("name", f)
    assert "metric_arg_names is an empty list" in caplog.text

    with pytest.raises(ValueError):
        Meter("name", f, metric_kwargs=["not a dict"])

    with pytest.raises(ValueError):
        Meter("name", f, "a", final="not callable")

    with pytest.raises(ValueError):
        Meter("name", f, "a", final=g, final_kwargs="not a dict")

    m = Meter("name", f, "a", "b", auto_measure=False)
    assert sorted(m.get_arg_names()) == ["a", "b"]

    with pytest.raises(ValueError):
        m.set("invalid arg name", 0, 0)

    assert not m.is_ready()
    with pytest.raises(ValueError):
        m.is_ready(raise_error=True)
    m.set("a", 15, 0)
    assert not m.is_ready()
    with pytest.raises(ValueError):
        m.is_ready(raise_error=True)

    m.finalize()
    assert "The following args were never set:" in caplog.text

    m.set("b", 2, 0)
    assert m.is_ready()
    m.is_ready(raise_error=True)

    assert m.results() == []
    m.measure(clear_values=False)
    assert m.is_ready()
    assert m.results() == [17]
    m.finalize()

    m.set("b", 4, 1)
    assert not m.is_ready()
    with pytest.raises(ValueError):
        m.is_ready(raise_error=True)

    m.clear()
    m.set("a", 6, 1)
    with pytest.raises(ValueError):
        m.measure()
    m.set("b", 4, 1)
    m.measure(clear_values=True)
    assert not m.is_ready()
    assert m.results() == [17, 10]

    def f_requires_kwargs(*x, **kwargs):
        if kwargs == {}:
            raise ValueError("no kwargs passed in!")
        return sum(x)

    m = Meter("name", f_requires_kwargs, "a", metric_kwargs=None)
    with pytest.raises(ValueError):
        m.set("a", 1, 0)
    m = Meter("name", f_requires_kwargs, "a", metric_kwargs={"kwarg": "anything"})
    m.set("a", 1, 0)

    # test automeasure and writer
    m = Meter("name", f, "a", "b")
    writer = LastRecordWriter()
    assert writer.num_writes == 0
    m.add_writer(writer)
    m.set("a", 4, 0)
    m.set("b", 6, 0)
    assert writer.num_writes == 1
    assert writer.record == ("name", 0, 10)
    m.add_writer(writer)
    m.set("a", 4, 1)
    m.set("b", 6, 1)
    assert writer.num_writes == 2
    assert writer.record == ("name", 1, 10)

    def f_list(*x):
        return [sum(i) for i in list(zip(*x))]

    m = Meter("list", f_list, "a", "b")
    m.add_writer(writer)
    m.set("a", [1, 2, 3], 0)
    m.set("b", [2, 3, 4], 0)
    assert writer.num_writes == 3
    assert writer.record == ("list", 0, [3, 5, 7])
    # test finalize

    # use somethign with final_kwargs
    for final_name, record_name in [
        (None, "final_list"),
        ("different_final_name", "different_final_name"),
    ]:
        m = Meter("list", f_list, "a", "b", final=g, final_name=final_name)
        writer = LastRecordWriter()
        m.add_writer(writer)
        m.set("a", [1, 2, 3], 0)
        m.set("b", [2, 3, 4], 0)
        assert writer.num_writes == 1
        m.finalize()
        assert writer.num_writes == 2
        assert writer.record == (record_name, None, 15)
        assert m.final_result() == 15

    m = Meter(
        "list", f_list, "a", "b", final=g, final_name="final", record_final_only=True
    )
    writer = LastRecordWriter()
    m.add_writer(writer)
    m.set("a", [1, 2, 3], 0)
    m.set("b", [2, 3, 4], 0)
    assert writer.num_writes == 0
    m.finalize()
    assert writer.num_writes == 1
    assert writer.record == ("final", None, 15)
    assert m.final_result() == 15


def test_global_meter():
    GlobalMeter = instrument.GlobalMeter
    with pytest.raises(ValueError):
        GlobalMeter("name", "not callable")
    with pytest.raises(ValueError):
        GlobalMeter("name", lambda x: x, final_kwargs="not a dict")

    def sum_rows(a, b, c, **kwargs):
        if not kwargs:
            raise ValueError("please pass in kwargs")
        d = []
        for a_i, b_i, c_i in zip(a, b, c):
            d.append(a_i + b_i + c_i)
        return d

    a = list(range(10))
    b = list(range(2, 12))
    c = list(range(5, 15))
    d = sum_rows(a, b, c, kwargs={})
    m = GlobalMeter("sum", sum_rows, "a", "b", "c", final_kwargs={"any_key": "value"})
    writer = LastRecordWriter()
    m.add_writer(writer)
    step = 3
    for i in range(0, 10, step):
        for array_name, array in zip(["a", "b", "c"], [a, b, c]):
            m.set(array_name, array[i : i + step], i)
    assert writer.num_writes == 0
    m.finalize()
    assert writer.num_writes == 1
    assert writer.record == ("sum", None, d)
    assert m.final_result() == d

    m = GlobalMeter("sum", sum_rows, "a", "b", "c")
    for i in range(0, 10, step):
        for array_name, array in zip(["a", "b", "c"], [a, b, c]):
            m.set(array_name, array[i : i + step], i)
    # ValueError raised by sum_rows since kwargs == {}
    with pytest.raises(ValueError):
        m.finalize()


def test_writer():
    writer = instrument.Writer()
    with pytest.raises(ValueError):
        writer.write("invalid record - not a (name, batch, result) tuple")
    empty_record = ("empty", 0, None)
    with pytest.raises(NotImplementedError):
        writer.write(empty_record)
    writer.close()
    with pytest.raises(ValueError):
        writer.write(empty_record)
    writer.close()


def test_null_writer(capsys):
    writer = instrument.NullWriter()
    writer.write(("empty", 0, None))
    captured = capsys.readouterr()
    assert not captured.out
    assert not captured.err


def test_print_writer(capsys):
    writer = instrument.PrintWriter()
    writer.write(("empty", 0, None))
    captured = capsys.readouterr()
    assert captured.out == "Meter Record: name=empty, batch=0, result=None\n"
    assert not captured.err
    writer.write(("name", 1, [1, 2]))
    captured = capsys.readouterr()
    assert captured.out == "Meter Record: name=name, batch=1, result=[1, 2]\n"
    assert not captured.err


def test_log_writer(caplog):
    writer = instrument.LogWriter(log_level="ERROR")
    writer.write(("empty", 0, None))
    for record in caplog.records:
        assert record.levelname == "ERROR"
    assert "Meter Record: name=empty, batch=0, result=None" in caplog.text

    with pytest.raises(ValueError):
        writer = instrument.LogWriter(log_level="NOT A LEVEL")


@pytest.mark.docker_required
def test_file_writer(tmp_path):
    filepath = tmp_path / "file_writer_output.txt"
    writer = instrument.FileWriter(filepath, use_numpy_encoder=False)
    a = ["line", 4, [1, 2, 3]]
    b = ["l2", 5, 7.45]
    writer.write(a)
    writer.write(b)
    writer.close()
    with open(filepath) as f:
        text = f.read()
        assert text == '["line",4,[1,2,3]]\n["l2",5,7.45]\n'
        line_a, line_b = text.strip().split("\n")
        assert json.loads(line_a) == a
        assert json.loads(line_b) == b

    import numpy as np

    c = ["c", 0, np.array([1, 2])]
    writer = instrument.FileWriter(filepath, use_numpy_encoder=True)
    writer.write(a)
    writer.write(c)
    writer.close()
    with open(filepath) as f:
        text = f.read()
        assert text == '["line",4,[1,2,3]]\n["c",0,[1,2]]\n'
        line_a, line_c = text.strip().split("\n")
        assert json.loads(line_a) == a
        assert json.loads(line_c) == c[:2] + [list(c[2])]

    # mock a numpy import error
    json_utils = instrument.json_utils
    instrument.json_utils = None
    with pytest.raises(ValueError):
        writer = instrument.FileWriter(filepath, use_numpy_encoder=True)
    instrument.json_utils = json_utils


class WriterSink:
    def __call__(self, output):
        self.output = output


def test_results_writer(caplog):
    sink = WriterSink()
    writer = instrument.ResultsWriter(sink=sink)
    with pytest.raises(ValueError):
        writer.get_output()
    writer.write(("a", 0, None))
    writer.write(("b", 1, -2))
    writer.write(("a", 2, -1))
    writer.close()
    with pytest.raises(ValueError):
        writer.get_output()
    output = sink.output
    assert sorted(output.keys()) == ["a", "b"]
    assert len(output) == 2
    assert output["a"] == [None, -1]
    assert output["b"] == [-2]

    writer = instrument.ResultsWriter(sink=None)
    writer.close()
    assert writer.get_output() == {}

    writer = instrument.ResultsWriter()
    writer.write(("a", 0, None))
    writer.write(("b", 1, -2))
    writer.write(("a", 2, -1))
    writer.close()
    output = writer.get_output()
    assert sorted(output.keys()) == ["a", "b"]
    assert len(output) == 2
    assert output["a"] == [None, -1]
    assert output["b"] == [-2]

    with pytest.raises(ValueError):
        writer = instrument.ResultsWriter(sink=None, max_record_size=-7)

    writer = instrument.ResultsWriter(sink=None, max_record_size=50)
    writer.write(("string", 0, ["a"] * 50))
    assert "max_record_size" in caplog.text
    writer.write(("a", 2, -1))
    writer.close()
    results = writer.get_output()
    assert results == {"a": [-1]}


@pytest.mark.docker_required
def test_integration():
    instrument.del_globals()
    # Begin model file
    import numpy as np

    # from armory.instrument import get_probe
    probe = instrument.get_probe("model")

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

    hub = instrument.get_hub()
    hub.connect_meter(
        instrument.Meter(
            "postprocessed_l2_distance",
            metrics.perturbation.batch.l2,
            "model.prep_output[benign]",
            "model.prep_output[adversarial]",
        )
    )
    hub.connect_meter(
        instrument.Meter(  # TODO: enable adding context for iteration number of attack
            "sum of x_adv",
            np.sum,
            "attack.x_adv",  # could also do "attack.x_adv[attack]"
        )
    )
    hub.connect_meter(
        instrument.Meter(
            "categorical_accuracy",
            metrics.task.batch.categorical_accuracy,
            "scenario.y",
            "scenario.y_pred",
        )
    )
    hub.connect_meter(
        instrument.Meter(  # Never measured, since 'y_target' is never set
            "targeted_categorical_accuracy",
            metrics.task.batch.categorical_accuracy,
            "scenario.y_target",
            "scenario.y_pred",
        )
    )
    hub.connect_writer(instrument.PrintWriter())

    # End metric setup

    # Update stages and batches in scenario loop (this would happen in main scenario file)
    model = Model()
    # Normally, model, attack, and scenario probes would be defined in different files
    #    and therefore just be called 'probe'
    attack_probe = instrument.get_probe("attack")
    scenario_probe = instrument.get_probe("scenario")
    not_connected_probe = instrument.Probe("not_connected")
    for i in range(10):
        hub.set_context(stage="get_batch", batch=i)
        x = np.random.random(100)
        y = np.random.randint(10)
        scenario_probe.update(x=[x], y=[y])
        not_connected_probe.update(x)  # should send a warning once

        hub.set_context(stage="benign")
        y_pred = model.predict(x)
        scenario_probe.update(y_pred=[y_pred])

        hub.set_context(stage="benign")
        x_adv = x
        for j in range(5):
            model.predict(x_adv)
            x_adv = x_adv + np.random.random(100) * 0.1
            attack_probe.update(x_adv=[x_adv])

        hub.set_context(stage="benign")
        y_pred_adv = model.predict(x_adv)
        scenario_probe.update(x_adv=[x_adv], y_pred_adv=[y_pred_adv])
    hub.close()
