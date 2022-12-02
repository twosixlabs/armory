import copy

REGISTERED_KEY_MAPS = {}
DEFAULT = "DEFAULT"


def register(name: str, key_map: dict):
    if not isinstance(name, str):
        raise ValueError(f"name {name} is not a str")
    check_key_map(key_map)
    global REGISTERED_KEY_MAPS
    REGISTERED_KEY_MAPS[name] = key_map


def list_registered():
    return list(REGISTERED_KEY_MAPS)


def get(name):
    if name not in REGISTERED_KEY_MAPS:
        raise KeyError(f"key_map {name} not registered. Use one of {list_registered()}")
    # dicts are malleable, so return a copy
    return copy.deepcopy(REGISTERED_KEY_MAPS[name])


def has(name):
    return name in REGISTERED_KEY_MAPS


def check_key_map(key_map: dict):
    if not isinstance(key_map, dict):
        raise ValueError(f"key_map {key_map} must be None or a dict")
    for k, v in key_map.items():
        if not isinstance(k, str):
            raise ValueError(f"key {k} in key_map is not a str")
        if not isinstance(v, str):
            raise ValueError(f"value {v} in key_map is not a str")
    if len(key_map.values()) != len(set(key_map.values())):
        raise ValueError("key_map values must be unique")


for name in "mnist", "cifar10":
    register(name, {"image": "x", "label": "y"})

register("digit", {"audio": "x", "label": "y"})
