import importlib


def get_metric(string):
    tokens = string.split(".")
    if not all(token.isidentifier() for token in tokens):
        raise ValueError(f"{string} is not a valid '.'-separated set of identifiers")
    if len(tokens) < 2:
        raise ValueError(f"{string} not a valid module and function path")

    errors = []
    for i in range(len(tokens) - 1, 0, -1):
        module_name = ".".join(tokens[:i])
        metric_name = ".".join(tokens[i:])
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            errors.append(f"ImportError: import {module_name}")
            continue
        try:
            return getattr(module, metric_name)
        except AttributeError:
            errors.append(
                f"AttributeError: module {module_name} has no attribute {metric_name}"
            )

    error_string = "\n".join(errors)
    raise ValueError(
        f"Could not import metric {string}. "
        "The following errors occurred:\n"
        f"{error_string}"
    )
