"""
Test cases for armory.instrument measurement instrumentation
"""

import pytest

from armory.instrument import instrument
from armory.utils import metrics

pytestmark = pytest.mark.unit


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


@pytest.mark.docker_required
def test_probe_pytorch_hook():
    import torch

    instrument.del_globals()
    sink = HelperSink()
    probe = instrument.Probe("model", sink=sink)
    model = get_pytorch_model()
    probe.hook(model.conv1, lambda x: x.detach().cpu().numpy(), output="b")

    key = "model.b"
    assert key not in sink.probe_variables

    x1 = torch.rand((1, 1, 28, 28))
    model(x1)
    b1 = sink.probe_variables["model.b"]
    assert b1.shape == (1, 20, 24, 24)
    x2 = torch.rand((1, 1, 28, 28))
    model(x2)
    b2 = sink.probe_variables["model.b"]
    assert b2.shape == (1, 20, 24, 24)
    assert not (b1 == b2).all()
    probe.unhook(model.conv1)
    # probe is unhooked, no update should occur
    model(x1)
    b3 = sink.probe_variables["model.b"]
    assert b3 is b2


def test_meter():
    metrics.get_supported_metric("tpr_fpr")
