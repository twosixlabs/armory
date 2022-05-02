import functools
import collections



def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rhasattr(obj, attr):
    try:
        left, right = attr.split(".", 1)
    except Exception:
        if isinstance(obj, dict):
            return attr in obj
        else:
            return hasattr(obj, attr)
    if hasattr(obj, left):
        return rhasattr(getattr(obj, left), right)
    else:
        return False


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def parse_overrides(overrides):
    if isinstance(overrides, str):
        output = {i.split()[0]: i.split[1] for i in overrides.split(" ")}
    elif isinstance(overrides, dict):
        output = flatten(overrides)
    elif isinstance(overrides, list) or isinstance(overrides, tuple):
        tmp = [i.split("=") for i in overrides]
        output = {i[0]: i[1] for i in tmp}
    else:
        raise ValueError(f"unexpected format for Overides: {type(overrides)}")
    return output


def set_overrides(obj, overrides):
    overrides = parse_overrides(overrides)
    for k, v in overrides.items():
        if rhasattr(obj, k):
            rsetattr(obj, k, v)
