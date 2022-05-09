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
    """

    def __init__(self, encoding="utf-8"):
        self.encoding = encoding
        self.size = 0

    def write(self, b):
        if isinstance(b, str) and self.encoding:
            b = bytes(b, encoding=self.encoding)
        self.size += len(b)


def size(obj, *, sort_keys=True, indent=4, cls=NumpyEncoder, **kwargs):
    """
    Provides the size of the JSON-encoded object
    """
    f = NullFile()
    dump(obj, f, sort_keys=sort_keys, indent=indent, cls=cls, **kwargs)
    return f.size
