"""Armory Utilities
General purpose utilities that span subpackages of Armory
namespace.

"""
import functools
import collections
from pydoc import locate


def rsetattr(obj, attr, val):
    """Recursively set attribute of object using dot notation
    Parameters:
        obj:        Object containing, potentially nested, attribute
        attr:       name of attribute to set in dot notation (e.g. location.name)
        val:        value to use to set the attribute

    Returns:
        `setattr` of the deepest level attribute
    """
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """Recursively find attribute of object

    Parameters:
        obj:        Object containing, potentially nested, attribute
        attr:       name of attribute to get in dot notation (e.g. location.name)

    Returns:
        value:      The value of the attribute
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rhasattr(obj, attr):
    """Check, recursively to see if object has attribute

    Parameters:
        obj:        Object containing, potentially nested, attribute
        attr:       name of attribute to get in dot notation (e.g. location.name)
    """
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


def flatten(obj, parent_key="", sep="."):
    """Flatten nested object to object with dot notation keys

    Parameters:
        obj:            Object containing, potentially nested, values/objects
        parent_key:     Specify the top level key to use
        sep:            Specify the separator to use in the flattened keys
    """
    items = []
    for k, v in obj.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def parse_overrides(overrides):
    """Parse overrides list into a dict of flattened key/value pairs

    Parameters:
        overrides (list):           List of overrides specified in dot notation
                                    (e.g. ["model.fit_kwargs.nb_epochs=1"])

    Returns:
        output (dict):              Dictionary of key/value pairs
                                    (e.g. {'model.fit_kwargs.nb_epochs':1})

    """
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
    """Set key values of object given list of overrides

    Parameters:
        obj:                        Object containing, potentially nested, values/objects
                                    that will be set given overrides
        overrides (list):           List of overrides specified in dot notation
                                    (e.g. ["model.fit_kwargs.nb_epochs=1"])

    """
    overrides = parse_overrides(overrides)
    for k, v in overrides.items():
        if rhasattr(obj, k):
            old_val = rgetattr(obj, k)
            # Set override value based on type of original value
            tp = locate(type(old_val).__name__)
            if tp is not None:
                new_val = tp(v)  # Casting to correct type
            else:
                new_val = v
            rsetattr(obj, k, new_val)
