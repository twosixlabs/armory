import types


def patch_method(obj):
    """
    Patch method for given class or object instance.
    If a class is passed in, patches ALL instances of class.
    If an object is passed in, only patches the given instance.
    """

    def decorator(method):
        if not isinstance(obj, object):
            raise ValueError(f"patch_method input {obj} is not a class or object")
        if isinstance(obj, type):
            cls = obj
            setattr(cls, method.__name__, method)
        else:
            setattr(obj, method.__name__, types.MethodType(method, obj))
        return method

    return decorator
