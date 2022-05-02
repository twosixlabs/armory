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
    f.write(dumps(obj, sort_keys=sort_keys, indent=indent, cls=cls, **kwargs))
