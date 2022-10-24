"""
Utilities for handling the global armory configuration file
"""

from collections import defaultdict
import json
import os
import warnings  # armory.logs initialization depends on this module, use warnings instead


def get_verify_ssl():
    return os.getenv("VERIFY_SSL") == "true" or os.getenv("VERIFY_SSL") is None


# TODO: Validate with JSON schema
def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise TypeError(f"config is a {type(config)}, not a dict")
    keys = (
        "dataset_dir",
        "local_git_dir",
        "saved_model_dir",
        "output_dir",
        "tmp_dir",
        "verify_ssl",
    )
    for key in keys:
        if key not in config:
            raise KeyError(
                f"config is missing key {key}. config may be out of date. Please run 'armory configure'"
            )

    inverse_dir = defaultdict(list)

    for key, value in config.items():
        if key not in keys:
            # warning instead of error to make forward compatible
            warnings.warn(f"config has additional key {key}")

        if key in ("verify_ssl") and not isinstance(value, bool):
            raise ValueError(f"{key} value {value} is not a bool")

        if key not in ("verify_ssl") and not isinstance(value, str):
            raise ValueError(f"{key} value {value} is not a string")

        if key not in ("verify_ssl"):
            inverse_dir[value].append(key)

    for value, keys in inverse_dir.items():
        if len(keys) > 1:
            raise ValueError(
                f"Configuration paths must be unique; {keys} are set to {value}"
            )


def load_global_config(config_path: str, validate: bool = True) -> dict:
    try:
        with open(config_path) as f:
            config = json.load(f)
    except json.decoder.JSONDecodeError:
        warnings.warn(f"Armory config file {config_path} could not be decoded")
        raise
    except OSError:
        warnings.warn(f"Armory config file {config_path} could not be read")
        raise

    if validate:
        try:
            validate_config(config)
        except (TypeError, KeyError, ValueError):
            warnings.warn(
                "Error parsing config.json. Please run `armory configure`.\n"
                "    If you previously ran an older version of armory, you may\n"
                f"    need to remove the {os.path.dirname(config_path)} directory due to changes"
            )
            raise

    return config


def save_config(config: dict, output_dir: str) -> None:
    validate_config(config)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(json.dumps(config, sort_keys=True, indent=4) + "\n")
