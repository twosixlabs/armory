import io

import numpy as np
import pytest

from armory.utils import json_utils

pytestmark = pytest.mark.unit

JSON_INPUT = {"hi": np.array([1, 2, 3, 4]), "there": np.float32(5.0), "you": [1, 2, 3]}
JSON_OUTPUT = '{\n    "hi": [\n        1,\n        2,\n        3,\n        4\n    ],\n    "there": 5.0,\n    "you": [\n        1,\n        2,\n        3\n    ]\n}\n'


def test_dumps(inp=JSON_INPUT, out=JSON_OUTPUT):
    assert json_utils.dumps(inp) == out


def test_dump(inp=JSON_INPUT, out=JSON_OUTPUT):
    f = io.StringIO()
    json_utils.dump(inp, f)
    f.seek(0)
    assert f.read() == out


def test_size(inp=JSON_INPUT, out=JSON_OUTPUT):
    assert json_utils.size(inp) == len(bytes(JSON_OUTPUT, encoding="utf-8"))


def test_check_size(inp=JSON_INPUT, out=JSON_OUTPUT):
    json_utils.check_size(inp, len(out))
    with pytest.raises(ValueError):
        json_utils.check_size(inp, len(out) - 1)
