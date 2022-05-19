import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) or isinstance(obj, np.generic):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def dumps(obj, *, sort_keys=True, indent=4, cls=NumpyEncoder, **kwargs):
    """
    Pretty prints and handles numpy arrays and numpy values
        Also, adds a '\n' to the end.
    """
    return json.dumps(obj, sort_keys=sort_keys, indent=indent, cls=cls, **kwargs) + "\n"


def dump(obj, f, *, sort_keys=True, indent=4, cls=NumpyEncoder, **kwargs):
    """
    Pretty prints and handles numpy arrays and numpy values
        Also, adds a '\n' to the end of the file.
    """
    json.dump(obj, f, sort_keys=sort_keys, indent=indent, cls=cls, **kwargs)
    f.write("\n")


class NullFile:
    """
    Records the total size of written encoded content (in bytes)
        If encoding is None, will use python string length instead of byte length
        if max_size is not None, will raise ValueError if exceeded
    """

    def __init__(self, encoding="utf-8", max_size=None):
        self.encoding = encoding
        self.max_size = max_size
        self.size = 0

    def write(self, b):
        if isinstance(b, str) and self.encoding:
            b = bytes(b, encoding=self.encoding)
        self.size += len(b)
        if self.max_size is not None and self.size > self.max_size:
            raise ValueError(f"Exceeds max size of {self.max_size}")


def check_size(obj, max_size, *, sort_keys=True, indent=4, cls=NumpyEncoder, **kwargs):
    """
    Raises a ValueError if size exceeds max_size. Otherwise is silent.
        Will fail fast.
    """
    f = NullFile(max_size=max_size)
    dump(obj, f, sort_keys=sort_keys, indent=indent, cls=cls, **kwargs)


def size(obj, *, sort_keys=True, indent=4, cls=NumpyEncoder, **kwargs):
    """
    Provides the size of the JSON-encoded object (in bytes)
    """
    f = NullFile()
    dump(obj, f, sort_keys=sort_keys, indent=indent, cls=cls, **kwargs)
    return f.size
