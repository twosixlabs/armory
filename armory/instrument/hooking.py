import types


def hook_wrapper(method, pre_method_hook=None, post_method_hook=None):
    def wrapped(*args, **kwargs):
        if pre_method_hook is not None:
            pre_method_hook(*args, **kwargs)
        return_value = method(*args, **kwargs)
        if post_method_hook is not None:
            post_method_hook(return_value)
        return return_value

    return wrapped


def method_hook(obj, method_name, pre_method_hook=None, post_method_hook=None):
    """
    Hook target method and return the original method

    If a class is passed in, hooks ALL instances of class.
    If an object is passed in, only hooks the given instance.
    """
    if not isinstance(obj, object):
        raise ValueError(f"obj {obj} is not a class or object")
    method = getattr(obj, method_name)
    if not callable(method):
        raise ValueError(f"obj.{method_name} attribute {method} is not callable")
    wrapped = hook_wrapper(
        method, pre_method_hook=pre_method_hook, post_method_hook=post_method_hook
    )

    if isinstance(obj, type):
        cls = obj
        setattr(cls, method_name, wrapped)
    else:
        setattr(obj, method_name, types.MethodType(wrapped, obj))

    return method


def method_unhook(obj, method_name, original_method):
    """
    Unhook target method by replacing with the original
    """
    if not isinstance(obj, object):
        raise ValueError(f"obj {obj} is not a class or object")
    method = getattr(obj, method_name)
    if not callable(method):
        raise ValueError(f"obj.{method_name} attribute {method} is not callable")
    if not callable(original_method):
        raise ValueError(f"original_method {original_method} is not callable")
    setattr(obj, method_name, original_method)
