import functools


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
        return hasattr(obj, attr)
    if hasattr(obj, left):
        return rhasattr(getattr(obj, left), right)
    else:
        return False
